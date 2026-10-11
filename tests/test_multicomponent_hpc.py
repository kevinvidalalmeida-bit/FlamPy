"""Portable byte-level regression checks against the frozen pre-HPC kernels.

No timing assertions, Cantera dependency, or CPU instruction-set assumptions.
The oracle in _multicomponent_reference must remain independent of production.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numba import njit

import _multicomponent_reference as reference
from kflame.chemistry import multicomponent_kernels as chemistry
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.multicomponent import NativeMulticomponentTransport
from kflame.chemistry.transport import NativeTransport
from kflame.flame import diffusion_kernels as faces


TEMPERATURES = (200., 300., 650., 999.999, 1000., 1000.001, 2000., 3500.)


def _assert_bytes(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    # Comparing the bits also distinguishes +0.0 from -0.0.
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint64),
        np.ascontiguousarray(expected).view(np.uint64),
    )


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _snapshots(arrays):
    return [array.tobytes(order='C') for array in arrays]


@pytest.fixture(scope='module', params=('h2o2.yaml', 'gri30.yaml'))
def transport_data(request):
    mechanism = load_mechanism(resolve_mechanism(request.param))
    base = NativeTransport(mechanism)
    multi = NativeMulticomponentTransport(mechanism, request.param, base)
    mw = _readonly(mechanism.molecular_weights)
    inv_w = _readonly(mechanism.inv_molecular_weights)
    ratio = mw[None, :] / mw[:, None]
    mixture = tuple(_readonly(array) for array in (
        base._cond_poly, base._diff_poly, base._visc_poly, ratio**.25,
        np.sqrt(8.*(1.+1./ratio)), base._eps_pair, base._cstar_poly,
    )) + (float(base._tiny),)
    multi_data = tuple(_readonly(array) for array in (
        multi._log_eps_pair, multi.epsilon_k, multi.crot, multi.zrot,
        multi.frot_298, multi._viscosity_fit, multi._diffusion_fit,
        multi._astar_poly, multi._bstar_poly, multi._cstar_poly,
    )) + (False,) + tuple(_readonly(array) for array in (
        mechanism.nasa_low, mechanism.nasa_high, mechanism.nasa_Tmid,
    ))
    return SimpleNamespace(names=mechanism.species_names, mw=mw, inv_w=inv_w,
                           mixture=mixture, multi=multi_data)


def _compositions(data):
    n = data.mw.size
    fuel = 'CH4' if 'CH4' in data.names else 'H2'
    sparse = np.zeros(n)
    for name, amount in ((fuel, 1.), ('O2', 2.), ('N2', 7.52)):
        sparse[data.names.index(name)] = amount*data.mw[data.names.index(name)]
    sparse /= sparse.sum()
    trace = sparse.copy()
    for name in ('H', 'OH', 'H2O'):
        trace[data.names.index(name)] += 1e-12
    trace /= trace.sum()
    dense = np.arange(1., n+1.)
    dense /= dense.sum()
    # Deliberate non-normalization tests the ordered Wmix reduction, too.
    return tuple(_readonly(y) for y in (sparse, trace, dense*1.0001))


def _poison_workspace(workspace):
    for array in workspace:
        array.fill(-777 if np.issubdtype(array.dtype, np.integer) else np.nan)


@njit(cache=True)
def _cp_reference(t, lo, hi, mid, out):
    for k in range(out.size):
        c = hi[k] if t > mid[k] else lo[k]
        out[k] = c[0]+t*(c[1]+t*(c[2]+t*(c[3]+t*c[4])))


@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_viscosity_temperature_helper_preserves_root_phi_bytes(transport_data, temperature):
    data = transport_data
    n = data.mw.size
    cond, diffusion, visc, quarter, den, eps, cstar, _ = data.mixture
    inputs = (data.mw, cond, diffusion, visc, quarter, den, eps, cstar)
    before = _snapshots(inputs)
    expected_root = np.full(n, np.nan)
    expected_phi = np.full((n, n), np.nan)
    reference.face_temperature_data(
        temperature, 101325., data.mw, cond, diffusion, visc, quarter, den,
        False, eps, cstar, np.full(n, np.nan), np.full((n, n), np.nan),
        np.full(n, np.nan), expected_root, expected_phi, np.full((n, n), np.nan),
    )
    # Reuse the same output buffers with different poisons: every live entry
    # must be written independently of its previous contents.
    root, phi = np.empty(n), np.empty((n, n))
    for poison in (np.nan, -np.inf):
        root.fill(poison)
        phi.fill(poison)
        faces.face_viscosity_temperature_data(temperature, visc, quarter, den, root, phi)
        _assert_bytes(root, expected_root)
        _assert_bytes(phi, expected_phi)
        for y in _compositions(data):
            x = _readonly(y*data.inv_w/np.sum(y*data.inv_w))
            _assert_bytes(faces.face_viscosity(x, root, phi),
                          reference.face_viscosity(x, expected_root, expected_phi))
    assert _snapshots(inputs) == before


def _layout(value, layout, readonly=False):
    value = np.asarray(value, dtype=np.float64)
    if layout == 'strided':
        backing = np.full(tuple(2*s for s in value.shape), np.nan)
        result = backing[tuple(slice(None, None, 2) for _ in value.shape)]
        result[...] = value
    else:
        result = np.array(value, dtype=np.float64, order=layout, copy=True)
    if readonly:
        result.setflags(write=False)
    return result


def _guarded_output(n, poison):
    backing = np.full(2*n+1, 1234567.125)
    result = backing[1::2]
    result.fill(poison)
    return backing, result


@pytest.mark.parametrize('n', (1, 10, 23, 24, 53, 56))
@pytest.mark.parametrize('layout', ('C', 'F', 'strided'))
def test_lu_preserves_pivots_and_matrix_bytes(n, layout):
    rng = np.random.default_rng(183+n)
    matrix = rng.standard_normal((n, n))
    matrix[np.diag_indices(n)] += n
    # Ensure the first pivot may require a row exchange, including zero
    # multipliers in later columns, without introducing singular systems.
    if n > 1:
        matrix[[0, n-1]] = matrix[[n-1, 0]]
        matrix[n-1, 0] = -0.
    expected, actual = _layout(matrix, layout), _layout(matrix, layout)
    piv_expected, piv_actual = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    reference._lu(expected, piv_expected)
    chemistry._lu(actual, piv_actual)
    _assert_bytes(actual, expected)
    np.testing.assert_array_equal(piv_actual, piv_expected)
    rhs = rng.standard_normal(n)
    old_rhs, new_rhs = rhs.copy(), rhs.copy()
    reference._lu_solve(expected, piv_expected, old_rhs)
    chemistry._lu_solve(actual, piv_actual, new_rhs)
    _assert_bytes(new_rhs, old_rhs)


def test_lu_singular_failure_matches_reference():
    for kernel in (reference._lu, chemistry._lu):
        with pytest.raises(np.linalg.LinAlgError, match='Singular multicomponent face matrix'):
            kernel(np.zeros((24, 24)), np.empty(24, dtype=np.int64))


@pytest.mark.parametrize('n', (1, 10, 21, 24, 53, 56))
@pytest.mark.parametrize('layout', ('C', 'F', 'strided'))
@pytest.mark.parametrize('readonly', (False, True))
def test_flux_preserves_bytes_layouts_tail_lanes_and_signed_zero(n, layout, readonly):
    rng = np.random.default_rng(173+n)
    mw = np.linspace(2., 80., n)
    inv_w = 1./mw
    left = rng.uniform(.01, 1., n)
    right = rng.uniform(.01, 1., n)
    left /= left.sum()
    right /= right.sum()
    # Signed mixed terms and varying magnitudes expose reduction reordering.
    matrix = rng.standard_normal((n, n))*np.logspace(-10., 2., n)[None, :]
    thermal = rng.standard_normal(n)*1e-5
    if n > 1:
        left[0], right[0] = 0., -0.
    for mode in ('mixed', 'isothermal', 'thermal_only', 'signed_zero'):
        yl, yr = left.copy(), right.copy()
        m, dt = matrix.copy(), thermal.copy()
        tl, tr = 700., 1500.
        if mode == 'isothermal':
            tr = tl
        elif mode == 'thermal_only':
            yr = yl.copy()
        elif mode == 'signed_zero':
            yr = yl.copy()
            tl, tr = -0., 0.
            m.fill(0.)
            m[:, ::2] = -0.
            dt.fill(0.)
            dt[::2] = -0.
        arrays = tuple(_layout(array, layout, readonly) for array in
                       (yl, yr, m, dt, inv_w, mw))
        before = _snapshots(arrays)
        yl, yr, m, dt, iw, w = arrays
        expected = np.full(n, np.nan)
        expected_scratch = np.full(n, np.nan)
        reference.multicomponent_flux(yl, yr, tl, tr, .0017, 1100., .31, 28.7,
                                      m, dt, iw, w, expected, expected_scratch)
        for poison in (np.nan, -np.inf):
            out_backing, out = _guarded_output(n, poison)
            scratch_backing, scratch = _guarded_output(n, poison)
            # Output/input aliasing is outside the kernel contract.
            assert not np.shares_memory(out, scratch)
            for array in arrays:
                assert not np.shares_memory(out, array)
                assert not np.shares_memory(scratch, array)
            faces.multicomponent_flux(yl, yr, tl, tr, .0017, 1100., .31, 28.7,
                                      m, dt, iw, w, out, scratch)
            _assert_bytes(out, expected)
            _assert_bytes(scratch, expected_scratch)
            assert np.isfinite(out).all() and np.isfinite(scratch).all()
            np.testing.assert_array_equal(out_backing[::2], 1234567.125)
            np.testing.assert_array_equal(scratch_backing[::2], 1234567.125)
        assert _snapshots(arrays) == before


@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_schur_composition_preserves_frozen_bytes_with_reused_workspaces(transport_data, temperature):
    data = transport_data
    n = data.mw.size
    log_eps, eps, crot, zrot, frot, visc, diffusion, astar, bstar, cstar = data.multi[:10]
    constants = (data.mw, data.inv_w) + data.multi[:10] + data.multi[11:]
    before = _snapshots(constants)
    old_terms = [np.full(n, np.nan), np.full(n, np.nan)] + [np.full((n, n), np.nan) for _ in range(4)]
    new_terms = [np.full_like(array, np.nan) for array in old_terms]
    arguments = (temperature, data.mw, log_eps, eps, zrot, frot, visc, diffusion, astar, bstar, cstar)
    reference.temperature_terms(*arguments, *old_terms)
    chemistry.temperature_terms(*arguments, *new_terms)
    for actual, expected in zip(new_terms, old_terms):
        _assert_bytes(actual, expected)
    # The same frozen temperature data go to both composition solvers, keeping
    # temperature and Schur regressions independently attributable.
    mu, rot, binary, a, b, c = tuple(_readonly(array) for array in old_terms)
    cp = np.empty(n)
    _cp_reference(temperature, *data.multi[11:], cp)
    cp = _readonly(cp)
    compositions = _compositions(data)
    external = (cp, mu, rot, binary, a, b, c) + compositions
    external_before = _snapshots(external)
    old_ws = reference.composition_workspace(n)
    workspaces = [chemistry.composition_workspace(n), chemistry.composition_workspace(n)]
    _poison_workspace(old_ws)
    for workspace in workspaces:
        _poison_workspace(workspace)
    # Revisit the first state after other compositions to expose stale state.
    for index in (0, 1, 2, 0):
        y = compositions[index]
        pressure = (101325., 202650., 75000.)[index]
        expected_multi = np.full((n, n), np.nan)
        expected_thermal = np.full(n, np.nan)
        args = (temperature, pressure, y, cp, data.mw, data.inv_w, crot, mu, rot, binary, a, b, c)
        expected = reference.composition_terms_ws(*args, expected_multi, expected_thermal, old_ws)
        for workspace in workspaces:
            actual_multi = np.full((n, n), -np.inf)
            actual_thermal = np.full(n, -np.inf)
            actual = chemistry.composition_terms_ws(*args, actual_multi, actual_thermal, workspace)
            assert np.isfinite(actual).all()
            assert np.isfinite(actual_multi).all() and np.isfinite(actual_thermal).all()
            _assert_bytes(actual, expected)
            _assert_bytes(actual_multi, expected_multi)
            _assert_bytes(actual_thermal, expected_thermal)
    assert _snapshots(constants) == before
    assert _snapshots(external) == external_before


@njit(cache=True)
def _reference_face(tl, tr, yl, yr, spacing, pressure, mw, inv_w, mixture, multi_data, soret):
    """Test-local assembly from frozen transport kernels and original formulas."""
    n = mw.size
    tf = .5*(tl+tr)
    yf = np.empty(n)
    for k in range(n):
        yf[k] = .5*(yl[k]+yr[k])
    cond, inv_binary, diagonal = np.empty(n), np.empty((n, n)), np.empty(n)
    root, phi, scale = np.empty(n), np.empty((n, n)), np.zeros((n, n))
    reference.face_temperature_data(tf, pressure, mw, mixture[0], mixture[1], mixture[2],
                                    mixture[3], mixture[4], False, mixture[5], mixture[6],
                                    cond, inv_binary, diagonal, root, phi, scale)
    mu, rot = np.empty(n), np.empty(n)
    binary, a, b, c = np.empty((n, n)), np.empty((n, n)), np.empty((n, n)), np.empty((n, n))
    reference.temperature_terms(tf, mw, multi_data[0], multi_data[1], multi_data[3], multi_data[4],
                                multi_data[5], multi_data[6], multi_data[7], multi_data[8],
                                multi_data[9], mu, rot, binary, a, b, c)
    cp = np.empty(n)
    _cp_reference(tf, multi_data[11], multi_data[12], multi_data[13], cp)
    multi, thermal = np.empty((n, n)), np.empty(n)
    rho, lam, wm = reference.composition_terms_ws(tf, pressure, yf, cp, mw, inv_w, multi_data[2],
                                                 mu, rot, binary, a, b, c, multi, thermal,
                                                 reference.composition_workspace(n))
    if not soret:
        thermal[:] = 0.
    flux, x = np.empty(n), np.empty(n)
    reference.multicomponent_flux(yl, yr, tl, tr, spacing, tf, rho, wm, multi, thermal,
                                  inv_w, mw, flux, x)
    wsum = 0.
    for k in range(n):
        wsum += yf[k]*inv_w[k]
    wsum = 1./max(wsum, 1e-300)
    for k in range(n):
        x[k] = yf[k]*wsum*inv_w[k]
    viscosity = reference.face_viscosity(x, root, phi)
    return lam, viscosity, flux


@pytest.mark.parametrize('soret', (False, True))
def test_face_range_assembly_matches_frozen_protocol_bytes(transport_data, soret):
    data = transport_data
    compositions = _compositions(data)
    temperature = _readonly([200., 650., 1350., 2200., 3500.])
    y = _readonly(np.column_stack([compositions[i] for i in (0, 1, 2, 0, 1)]))
    spacing = _readonly([.001, .0017, .0009, .0021])
    nfaces = spacing.size
    expected_lam, expected_mu = np.empty(nfaces), np.empty(nfaces)
    expected_flux = np.empty((data.mw.size, nfaces))
    for f in range(nfaces):
        expected_lam[f], expected_mu[f], expected_flux[:, f] = _reference_face(
            temperature[f], temperature[f+1], y[:, f], y[:, f+1], spacing[f],
            101325., data.mw, data.inv_w, data.mixture, data.multi, soret)
    before = _snapshots((temperature, y, spacing))
    # Different serial partitions test independent workspace ownership without
    # hardware-dependent thread-count assumptions.
    for ranges in (((0, nfaces),), ((0, 1), (1, 3), (3, nfaces))):
        lam, mu = np.full(nfaces, np.nan), np.full(nfaces, np.nan)
        flux = np.full_like(expected_flux, np.nan)
        for start, stop in ranges:
            faces.face_range(start, stop, temperature, y, spacing, 101325., data.inv_w,
                             data.mw, data.mixture, data.multi, True, soret, lam, mu, flux)
        _assert_bytes(lam, expected_lam)
        _assert_bytes(mu, expected_mu)
        _assert_bytes(flux, expected_flux)
    assert _snapshots((temperature, y, spacing)) == before
