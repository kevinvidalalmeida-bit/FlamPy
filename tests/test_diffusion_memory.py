"""Diffusion memory regressions with independent, repository-owned oracles."""
import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion, diffusion_kernels


def _problem(*, transport='mixture-averaged', soret=False):
    mech = load_mechanism(resolve_mechanism('h2o2.yaml'))
    feeds = []
    for text in ('H2', 'O2:1,N2:3.76'):
        y = _composition(text, mech.species_names)*mech.molecular_weights
        feeds.append(y/y.sum())
    case = diffusion.DiffusionCase(resolve_mechanism('h2o2.yaml'), *feeds,
        fuel_mass_flux=.05, oxidizer_mass_flux=.15, transport_model=transport,
        soret_enabled=soret, radiation_enabled=True, radiation_scale=2.,
        boundary_emissivities=(.4, .8))
    return diffusion.DiffusionFlameProblem(case, mech, np.array([0., .003, .009, .016, .02]))


def _bitwise_equal(actual, expected):
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint64),
                                  np.ascontiguousarray(expected).view(np.uint64))


def _assert_columns_close(actual, expected):
    scale = abs(expected).max(axis=1, keepdims=True)
    np.testing.assert_array_less(abs(actual-expected),
        np.broadcast_to(3e-5*scale+1e-12, expected.shape))


@pytest.mark.parametrize('transport,soret', [
    ('mixture-averaged', False), ('mixture-averaged', True),
    ('multicomponent', False), ('multicomponent', True),
])
def test_local_jacobian_matches_public_residual_columns(transport, soret):
    p = _problem(transport=transport, soret=soret)
    x = p.pack(p.initial_fields())
    rng = np.random.default_rng(19)
    x[:-1] *= 1.+1e-3*rng.standard_normal(x.size-1)
    x[2*p.nv] = 1e-8  # inside the unchanged stagnation blend
    temperature = p.unpack(x)['T'].copy()
    modes = ('exact', 'frozen') if transport == 'multicomponent' else ('exact',)
    for fixed in (None, temperature):
        residual = lambda value: p.residual(value, fixed_temperature=fixed)
        # Public, full-grid finite differences do not use the local workspace.
        expected = p.jacobian(x, residual, residual(x)).toarray()
        for mode in modes:
            p.multicomponent_jacobian = mode
            for analytic in (False, True):
                p.analytic_chemistry = analytic
                actual = p.jacobian_blocks(x, fixed_temperature=fixed)
                columns = (np.arange(x.size) if mode == 'exact' else
                    np.array([j*p.nv+c for j in range(p.n) for c in range(3)]+[x.size-1]))
                # Frozen transport deliberately keeps base composition coefficients;
                # flow and temperature columns still match the full residual.
                _assert_columns_close(actual.tocsc().toarray()[:, columns], expected[:, columns])
        p.multicomponent_jacobian = 'exact'
        transient = lambda value: p.residual(value, previous=x, dt=1e-5, fixed_temperature=fixed)
        expected_transient = p.jacobian(x, transient, transient(x)).toarray()
        _assert_columns_close(p.jacobian_blocks(x, fixed_temperature=fixed).tocsc(1e5).toarray(),
                              expected_transient)


@pytest.mark.parametrize('order', ['C', 'F'])
def test_face_inputs_are_borrowed_read_only_and_results_are_independent(order):
    p = _problem(soret=True)
    fields = p.initial_fields()
    Y = np.array(fields['Y'], order=order)
    original = Y.copy()
    Y.flags.writeable = False
    assert diffusion._species_layout(Y) is Y
    fields['Y'] = Y
    first = p.properties(fields)
    assert first[5].flags.c_contiguous
    saved = [array.copy() for array in first]
    second = p.properties(dict(fields, T=fields['T']*1.001))
    for a, saved_a in zip(first, saved):
        _bitwise_equal(a, saved_a)
        assert all(not np.shares_memory(a, b) for b in second)
        assert not np.shares_memory(a, Y)
    np.testing.assert_array_equal(Y, original)
    # C and F input arrays must take the same arithmetic path.
    alternate = dict(fields, Y=np.array(Y, order='F' if order == 'C' else 'C'))
    for a, b in zip(first, p.properties(alternate)):
        _bitwise_equal(a, b)


def test_returned_residual_and_blocks_survive_later_calls():
    p = _problem(soret=True)
    x = p.pack(p.initial_fields())
    residual = p.residual(x)
    saved_residual = residual.copy()
    blocks = p.jacobian_blocks(x)
    names = ('lower', 'diag', 'upper', 'border', 'mass')
    saved = {name: getattr(blocks, name).copy() for name in names}
    shifted = x.copy()
    shifted[2*p.nv+2] *= 1.001
    other_residual = p.residual(shifted)
    other_blocks = p.jacobian_blocks(shifted)
    _bitwise_equal(residual, saved_residual)
    assert not np.shares_memory(residual, other_residual)
    for name in names:
        a = getattr(blocks, name)
        _bitwise_equal(a, saved[name])
        assert all(not np.shares_memory(a, getattr(other_blocks, key)) for key in names)


def test_reused_soret_scratch_clears_absent_species():
    p = _problem(soret=True)
    data = p._compiled_data()
    mixture = data['mixture']
    mw, invW = data['mw'], data['invW']
    td = diffusion_kernels._tdata(mw.size)
    diffusion_kernels.face_temperature_data(1000., p.P, mw, *mixture[:5], True,
        mixture[5], mixture[6], *td)
    ws = np.full((5, mw.size), np.nan)
    # Alternate populated, pure and signed mixtures using the very same scratch.
    compositions = [p.initial_fields()['Y'][:, 2], p.case.fuel_Y, p.case.oxidizer_Y]
    signed = compositions[0].copy()
    signed[0] = -1e-12
    compositions.append(signed)
    for Y in compositions:
        X, D, DT = (np.empty(mw.size) for _ in range(3))
        expected_X, expected_D, expected_DT = (np.empty(mw.size) for _ in range(3))
        actual = diffusion_kernels._face_mixture_ws(1000., Y, p.P, invW, mw, *td[:5],
            True, td[5], mixture[7], X, D, DT, ws)
        expected = _allocating_face_mixture(1000., Y, p.P, invW, mw, *td[:5],
            True, td[5], mixture[7], expected_X, expected_D, expected_DT)
        _bitwise_equal(np.array(actual), np.array(expected))
        for a, b in ((X, expected_X), (D, expected_D), (DT, expected_DT)):
            _bitwise_equal(a, b)


def _allocating_face_mixture(Tf, Yf, P, invW, mw, cond, invB, diagB, root, phi, soret, scale, tiny, X, D, DT):
    """Fixed allocating oracle, independent of the reusable scratch path."""
    ns = Yf.size
    inv_wmix = 0.
    for k in range(ns):
        inv_wmix += Yf[k]*invW[k]
    wm = 1./max(inv_wmix, 1e-300)
    SD = np.empty(ns)
    Smu = np.empty(ns)
    sum1, sum2 = diffusion_kernels.mixture_sums(wm, Yf, invW, cond, invB, phi, X, SD, Smu)
    rho, lam, mu = diffusion_kernels.mixture_from_sums(Tf, P, wm, X, sum1, sum2, SD, Smu, mw, diagB, root, D)
    if soret:
        # Same clipped mole fractions and mixture diffusivities as the
        # fused mixture-averaged Soret kernel.
        denom = 0.
        for k in range(ns):
            denom += Yf[k]/mw[k]
        wmix = 1./max(denom, 1e-300)
        Xs = np.empty(ns)
        for k in range(ns):
            Xs[k] = max(Yf[k]*wmix/mw[k], 1e-20)
        a = np.zeros(ns)
        diffusion = np.empty(ns)
        for k in range(ns):
            total = 0.
            for j in range(ns):
                if j != k:
                    total += invB[j, k]*max(Xs[j], tiny)
            if total <= 0.:
                diffusion[k] = diagB[k]/P
            else:
                diffusion[k] = (wmix-max(Xs[k], tiny)*mw[k])/(P*wmix*max(total, 1e-300))
            if Yf[k] >= 1e-20:
                total = 0.
                for j in range(ns):
                    if j != k:
                        total += Xs[j]*phi[k, j]
                a[k] = (15./4.)*root[k]**2/mw[k]/(1+1.065*total/Xs[k])
        total = 0.
        for k in range(ns):
            value = 0.
            for j in range(ns):
                value += scale[k, j]*(Yf[k]*a[j]-Yf[j]*a[k])
            DT[k] = value*diffusion[k]*mw[k]*wmix
            total += DT[k]
        for k in range(ns):
            DT[k] -= Yf[k]*total
    return rho, wm, lam, mu



def test_residual_boundary_rows_are_fully_written():
    p = _problem(soret=True)
    p.set_grid(np.array([0., .01, .02]))
    x = p.pack(p.initial_fields())
    fields = p.unpack(x)
    properties = p.properties(fields)
    residual = p.residual(x, cached_properties=properties, fixed_temperature=fields['T'])
    expected = np.zeros_like(residual)
    rows = expected[:-1].reshape(p.n, p.nv)
    m, V, T, Y = (fields[key] for key in ('m', 'V', 'T', 'Y'))
    J = properties[5]
    for j, target, feed, Tfeed in ((0, p.case.fuel_mass_flux, p.case.fuel_Y, p.case.fuel_temperature),
                                   (p.n-1, -p.case.oxidizer_mass_flux, p.case.oxidizer_Y, p.case.oxidizer_temperature)):
        rows[j, 0] = (m[j]-target)/p.mref
        rows[j, 1] = V[j]/p.aref
        rows[j, 2] = (T[j]-Tfeed)/p.Tref
        total = 0.
        for k in range(p.ns):
            rows[j, 3+k] = (m[j]*(Y[k, j]-feed[k])+J[k, 0 if j == 0 else p.n-2])/p.mref
            total += Y[k, j]
        rows[j, 3+p.excess[j]] = total-1.
    rho = properties[0]
    expected[-1] = (m[-1]-m[-2]+p.dz[-1]*(rho[-2]*V[-2]+rho[-1]*V[-1]))/p.mref
    assert np.isfinite(residual).all()
    boundary_rows = np.r_[np.arange(p.nv), np.arange((p.n-1)*p.nv, residual.size)]
    _bitwise_equal(residual[boundary_rows], expected[boundary_rows])
    assert residual[p.nv+2] == 0.
