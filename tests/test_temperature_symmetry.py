"""Byte-exact snapshot symmetry tests against an independent original kernel."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numba import njit

import _multicomponent_reference as original
from kflame.chemistry import multicomponent_kernels as kernels
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism, R_UNIV
from kflame.chemistry.multicomponent import (
    NativeMulticomponentTransport, TemperatureMechanismSnapshot,
    _certifiable_temperature_inputs,
)

# Recompile the independent frozen original without production helper calls.
REFERENCE = njit(cache=False, fastmath=False)(original.temperature_terms.py_func)
TEMPERATURES = (200., 250., 298., 300., 600., 900., 1200., 1800., 2500., 3000., 3500.)
PAIRS = (1, 6, 7, 8, 9)


def buffers(n):
    return tuple(np.full(n if i < 2 else (n, n), -123.75) for i in range(6))


def assert_bytes(actual, expected):
    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        assert got.tobytes() == want.tobytes()


@pytest.fixture(scope='module', params=('h2o2.yaml', 'gri30.yaml'))
def transport(request):
    mech = load_mechanism(resolve_mechanism(request.param))
    return NativeMulticomponentTransport(mech, request.param)


@pytest.fixture(scope='module')
def small():
    mech = load_mechanism(resolve_mechanism('h2o2.yaml'))
    return NativeMulticomponentTransport(mech, 'h2o2.yaml')


def test_native_six_outputs_extrapolation_and_diagonal(transport):
    snapshot = transport._temperature_snapshot
    assert snapshot.certified
    n = transport.n_species
    for t in sorted(set((*TEMPERATURES, transport.mech.min_temperature,
                         transport.mech.max_temperature))):
        expected, got = buffers(n), buffers(n)
        REFERENCE(t, *snapshot.inputs, *expected)
        assert snapshot.evaluate(t, got)
        assert_bytes(got, expected)
        for k in range(n):
            diagonal = 1.2 * R_UNIV * t * got[0][k] * got[3][k, k] / snapshot.inputs[0][k]
            assert got[2][k, k].tobytes() == diagonal.tobytes()
        for out in got[2:]:
            assert out.tobytes() == out.T.tobytes()


@pytest.mark.parametrize('index', PAIRS)
def test_one_ulp_asymmetry_falls_back_and_mirroring_would_change_bytes(small, index):
    inputs = [x.copy() for x in small._temperature_snapshot.inputs]
    coordinate = (1, 0) if index == 1 else (1, 0, 0)
    inputs[index][coordinate] = np.nextafter(inputs[index][coordinate], np.inf)
    snapshot = TemperatureMechanismSnapshot(tuple(inputs))
    assert not snapshot.certified
    changed = False
    for t in TEMPERATURES:
        expected, got, unsafe, raw = (buffers(small.n_species) for _ in range(4))
        REFERENCE(t, *inputs, *expected)
        assert not snapshot.evaluate(t, got)
        kernels.temperature_terms(t, *inputs, *raw)
        assert_bytes(got, expected)
        assert_bytes(raw, expected)
        # Deliberately violate private entry preconditions to show guard value.
        kernels._temperature_terms_symmetric(t, *snapshot.inputs, *unsafe)
        changed |= any(x.tobytes() != y.tobytes() for x, y in zip(unsafe, expected))
    assert changed


@pytest.mark.parametrize('index', PAIRS)
def test_every_pair_coefficient_is_certified(small, index):
    inputs = [x.copy() for x in small._temperature_snapshot.inputs]
    degrees = range(1) if index == 1 else range(inputs[index].shape[2])
    for degree in degrees:
        coordinate = (1, 0) if index == 1 else (1, 0, degree)
        previous = inputs[index][coordinate]
        inputs[index][coordinate] = np.nextafter(previous, np.inf)
        assert not TemperatureMechanismSnapshot(tuple(inputs)).certified
        inputs[index][coordinate] = previous


@pytest.mark.parametrize('index', PAIRS)
def test_signed_zero_symmetry_is_bitwise(small, index):
    inputs = [x.copy() for x in small._temperature_snapshot.inputs]
    upper = (0, 1) if index == 1 else (0, 1, 0)
    lower = (1, 0) if index == 1 else (1, 0, 0)
    inputs[index][upper], inputs[index][lower] = 0., -0.
    assert np.array_equal(inputs[index], inputs[index].swapaxes(0, 1))
    snapshot = TemperatureMechanismSnapshot(tuple(inputs))
    assert not snapshot.certified
    expected, got = buffers(small.n_species), buffers(small.n_species)
    REFERENCE(900., *inputs, *expected)
    assert not snapshot.evaluate(900., got)
    assert_bytes(got, expected)
    inputs[index][upper] = -0.
    snapshot = TemperatureMechanismSnapshot(tuple(inputs))
    assert snapshot.certified
    REFERENCE(900., *inputs, *expected)
    assert snapshot.evaluate(900., got)
    assert_bytes(got, expected)


def test_snapshot_mutation_isolation_and_irreversible_readonly(small):
    # All source arrays are read-only views with writable external owners.
    owners = [x.copy() for x in small._temperature_snapshot.inputs]
    views = [x.view() for x in owners]
    for x in views:
        x.setflags(write=False)
    snapshot = TemperatureMechanismSnapshot(tuple(views))
    saved = tuple(x.tobytes() for x in snapshot.inputs)
    for x in owners:
        x.flat[0] += 1.
    assert tuple(x.tobytes() for x in snapshot.inputs) == saved
    assert snapshot.certified
    for x, source in zip(snapshot.inputs, owners):
        assert not np.shares_memory(x, source)
        with pytest.raises(ValueError):
            x.setflags(write=True)
        with pytest.raises(ValueError):
            x.base.setflags(write=True)
        with pytest.raises(ValueError):
            x.flat[0] = 0.
    with pytest.raises(FrozenInstanceError):
        snapshot.certified = False
    expected, got = buffers(small.n_species), buffers(small.n_species)
    REFERENCE(1200., *snapshot.inputs, *expected)
    assert snapshot.evaluate(1200., got)
    assert_bytes(got, expected)


def test_exposed_metadata_cannot_change_private_certificate(small):
    snapshot = small._temperature_snapshot
    exposed = snapshot.inputs
    exposed[1].shape = (small.n_species ** 2,)
    exposed[6].dtype = np.uint64
    assert snapshot.certified
    expected, got = buffers(small.n_species), buffers(small.n_species)
    REFERENCE(900., *snapshot.inputs, *expected)
    assert snapshot.evaluate(900., got)
    assert_bytes(got, expected)


def test_bound_metadata_edit_invalidates_transport_dispatch(small, monkeypatch):
    # Use a new immutable view so the shared module fixture stays unchanged.
    view = small._log_eps_pair.view()
    bound = list(small._temperature_snapshot_bindings)
    bound[1] = view
    monkeypatch.setattr(small, '_log_eps_pair', view)
    monkeypatch.setattr(small, '_temperature_snapshot_bindings', tuple(bound))
    assert small._temperature_snapshot._matches_bindings(small._temperature_inputs(), tuple(bound))
    view.shape = (small.n_species ** 2,)
    assert not small._temperature_snapshot._matches_bindings(small._temperature_inputs(), tuple(bound))


def test_raw_numba_api_remains_compatible_with_mutation(small):
    @njit(cache=False, fastmath=False)
    def caller(t, inputs, out):
        kernels.temperature_terms(t, *inputs, *out)
    inputs = tuple(x.copy() for x in small._temperature_snapshot.inputs)
    for changed in (False, True):
        if changed:
            inputs[7][1, 0, 0] += 0.125
        expected, got = buffers(small.n_species), buffers(small.n_species)
        REFERENCE(900., *inputs, *expected)
        caller(900., inputs, got)
        assert_bytes(got, expected)


def test_export_is_owned_immutable_and_bound_to_complete_native_snapshot(small):
    inputs, certified = small.export_temperature_snapshot()
    assert certified
    expected, got = buffers(small.n_species), buffers(small.n_species)
    REFERENCE(900., *inputs, *expected)
    kernels._temperature_terms_symmetric(900., *inputs, *got)
    assert_bytes(got, expected)
    for x in inputs:
        assert not x.flags.writeable
        with pytest.raises(ValueError):
            x.setflags(write=True)
    # Exported metadata belongs to these views, not the native snapshot.
    inputs[1].shape = (small.n_species ** 2,)
    fresh, certified = small.export_temperature_snapshot()
    assert certified and fresh[1].shape == (small.n_species, small.n_species)


@pytest.mark.parametrize('asymmetric', (False, True))
def test_rebound_mutable_arrays_export_uncertified_isolated_values(small, monkeypatch, asymmetric):
    changed = small._astar_poly.copy()
    if asymmetric:
        changed[1, 0, 0] = np.nextafter(changed[1, 0, 0], np.inf)
    monkeypatch.setattr(small, '_astar_poly', changed)
    inputs, certified = small.export_temperature_snapshot()
    assert not certified
    assert not np.shares_memory(changed, inputs[7])
    saved = inputs[7].tobytes()
    expected, got = buffers(small.n_species), buffers(small.n_species)
    REFERENCE(900., *small._temperature_inputs(), *expected)
    kernels.temperature_terms(900., *inputs, *got)
    assert_bytes(got, expected)
    changed[1, 0, 0] += 1.
    assert inputs[7].tobytes() == saved


@njit(cache=False, fastmath=False)
def _parent_style_temperature(t, mw, multi_data, outputs):
    (log_eps, epsilon, crot, zrot, frot, visc, diff, aa, bb, cc,
     dense, nasa_lo, nasa_hi, nasa_tmid) = multi_data[:14]
    if len(multi_data) == 15:
        if multi_data[14]:
            kernels._temperature_terms_symmetric(t, mw, log_eps, epsilon, zrot,
                    frot, visc, diff, aa, bb, cc, *outputs)
            return True
    kernels.temperature_terms(t, mw, log_eps, epsilon, zrot, frot,
                              visc, diff, aa, bb, cc, *outputs)
    return False


def test_parent_style_legacy14_and_certified15_contract(small):
    inputs, certified = small.export_temperature_snapshot()
    mw, logeps, epsilon, zrot, frot, visc, diff, aa, bb, cc = inputs
    legacy = (logeps, epsilon, small.crot, zrot, frot, visc, diff, aa, bb, cc,
              False, small.mech.nasa_low, small.mech.nasa_high, small.mech.nasa_Tmid)
    expected, got = buffers(small.n_species), buffers(small.n_species)
    REFERENCE(900., *inputs, *expected)
    assert not _parent_style_temperature(900., mw, legacy, got)
    assert_bytes(got, expected)
    assert _parent_style_temperature(900., mw, (*legacy, certified), got)
    assert_bytes(got, expected)
    changed = aa.copy()
    changed[1, 0, 0] += 0.125
    legacy = (*legacy[:7], changed, *legacy[8:])
    REFERENCE(900., mw, logeps, epsilon, zrot, frot, visc, diff, changed, bb, cc, *expected)
    assert not _parent_style_temperature(900., mw, legacy, got)
    assert_bytes(got, expected)
    assert not _parent_style_temperature(900., mw, (*legacy, False), got)
    assert_bytes(got, expected)


@pytest.mark.parametrize('layout', ('fortran', 'strided'))
def test_uncertified_input_layout_keeps_original(small, layout):
    inputs = list(small._temperature_snapshot.inputs)
    value = inputs[6]
    if layout == 'fortran':
        inputs[6] = np.asfortranarray(value)
    else:
        owner = np.empty((*value.shape[:-1], 2 * value.shape[-1]))
        inputs[6] = owner[..., ::2]
        inputs[6][...] = value
    assert not _certifiable_temperature_inputs(tuple(inputs))
    snapshot = TemperatureMechanismSnapshot(tuple(inputs))
    assert not snapshot.certified
    expected, got, raw = (buffers(small.n_species) for _ in range(3))
    REFERENCE(1000., *inputs, *expected)
    assert not snapshot.evaluate(1000., got)
    kernels.temperature_terms(1000., *inputs, *raw)
    assert_bytes(got, expected)
    assert_bytes(raw, expected)


def variant_buffers(n, variant):
    out = list(buffers(n))
    if variant == 'fortran':
        out[3] = np.asfortranarray(out[3])
    elif variant == 'strided':
        out[3] = np.full((n, 2 * n), -123.75)[:, ::2]
    elif variant == 'alias':
        out[4] = out[3]
    elif variant == 'overlap':
        owner = np.full((n, n + 1), -123.75)
        out[3], out[4] = owner[:, :-1], owner[:, 1:]
    elif variant == 'readonly':
        out[3].setflags(write=False)
    return tuple(out)


@pytest.mark.parametrize('variant', ('fortran', 'strided', 'alias', 'overlap', 'readonly'))
def test_output_layout_readonly_and_alias_fallback(small, variant):
    snapshot = small._temperature_snapshot
    expected, got = (variant_buffers(small.n_species, variant) for _ in range(2))
    if variant == 'readonly':
        with pytest.raises(Exception) as reference_error:
            REFERENCE(900., *snapshot.inputs, *expected)
        with pytest.raises(type(reference_error.value)):
            snapshot.evaluate(900., got)
    else:
        REFERENCE(900., *snapshot.inputs, *expected)
        assert not snapshot.evaluate(900., got)
        assert_bytes(got, expected)


def test_mutable_raw_input_output_alias_retains_original_write_order(small):
    first, second = ([x.copy() for x in small._temperature_snapshot.inputs] for _ in range(2))
    expected, got = (list(buffers(small.n_species)) for _ in range(2))
    # Mutation of a coefficient input by output stores is intentionally legal
    # at the old raw boundary: preserving ordered-pair traversal is essential.
    expected[3], got[3] = first[1], second[1]
    REFERENCE(900., *first, *expected)
    kernels.temperature_terms(900., *second, *got)
    assert_bytes(got, expected)
    assert_bytes(second, first)


@pytest.mark.parametrize('t', (0., -1., np.inf, -np.inf, np.nan,
                               np.nextafter(0., 1.), 1.e-200, 1.e200, np.finfo(float).max))
def test_domain_and_extreme_temperature_behavior(small, t):
    snapshot = small._temperature_snapshot
    expected, got = buffers(small.n_species), buffers(small.n_species)
    try:
        REFERENCE(t, *snapshot.inputs, *expected)
    except Exception as error:
        with pytest.raises(type(error)):
            snapshot.evaluate(t, got)
    else:
        selected = snapshot.evaluate(t, got)
        assert selected == (np.isfinite(t) and t > 0.)
        assert_bytes(got, expected)


@pytest.mark.parametrize('case', ('nan_payload', 'infinity', 'zero_mass', 'negative_epsilon', 'float32', 'shape'))
def test_conservative_certification_rejects_uncertain_inputs(small, case):
    inputs = [x.copy() for x in small._temperature_snapshot.inputs]
    if case == 'nan_payload':
        inputs[1].view(np.uint64)[0, 1] = 0x7ff8000000000001
        inputs[1].view(np.uint64)[1, 0] = 0x7ff8000000000002
    elif case == 'infinity':
        inputs[1][0, 1] = inputs[1][1, 0] = np.inf
    elif case == 'zero_mass':
        inputs[0][0] = 0.
    elif case == 'negative_epsilon':
        inputs[2][0] = -1.
    elif case == 'float32':
        inputs[5] = inputs[5].astype(np.float32)
    elif case == 'shape':
        inputs[5] = inputs[5][:, :4]
    assert not TemperatureMechanismSnapshot(tuple(inputs)).certified


def test_transport_mechanism_isolation_and_replacement_fallback(small, monkeypatch):
    assert not np.shares_memory(small.mw, small.mech.molecular_weights)
    assert not np.shares_memory(small.zrot, small.mech.rot_relax)
    assert not np.shares_memory(small._diffusion_fit, small.base_transport._diff_poly)
    saved = small._log_eps_pair
    monkeypatch.setattr(small, '_log_eps_pair', saved.copy())
    small._log_eps_pair[1, 0] = np.nextafter(saved[1, 0], np.inf)
    import kflame.chemistry.multicomponent as module
    called = []
    def fallback(*args):
        called.append(args)
        return 'original'
    monkeypatch.setattr(module, 'evaluate_faces', fallback)
    n = small.n_species
    assert small.eval_faces(np.array([900.]), 101325., np.ones((n, 1)) / n,
                            np.full((n, 1), 3.5)) == 'original'
    assert called[0][6] is small._log_eps_pair


def test_compiled_snapshot_faces_match_original(small):
    n = small.n_species
    t = np.array([250., 900., 3500.])
    y = np.arange(1., n + 1.)[:, None] * np.ones((1, t.size))
    y /= y.sum(axis=0)
    cp = np.full_like(y, 3.5)
    # Independent temperature kernel; production composition is held fixed
    # because this change does not modify any composition or Schur arithmetic.
    expected = []
    for i, temperature in enumerate(t):
        terms = buffers(n)
        REFERENCE(temperature, *small._temperature_snapshot.inputs, *terms)
        multi, thermal = np.empty((n, n)), np.empty(n)
        scalar = kernels.composition_terms(temperature, 101325., y[:, i], cp[:, i],
                     small.mw, small.inv_w, small.crot, *terms, False, multi, thermal)
        expected.append((*scalar, multi, thermal))
    got = small.eval_faces(t, 101325., y, cp)
    for i, ref in enumerate(expected):
        assert_bytes(tuple(np.asarray(x) for x in
                           (got[0][i], got[1][i], got[2][i], got[3][:, :, i], got[4][:, i])),
                     tuple(np.asarray(x) for x in ref))
