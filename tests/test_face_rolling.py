"""Bitwise regressions for rolling base-face reuse between adjacent nodes."""
import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion, diffusion_kernels


_BLOCK_NAMES = ('lower', 'diag', 'upper', 'border', 'mass')
_CUTS = (
    (0, 7),
    (0, 1, 3, 6, 7),
    (0, 2, 4, 5, 7),
    tuple(range(8)),
)
_CASES = (
    pytest.param(('h2o2.yaml', 'H2', 'mixture-averaged', False, 'exact'), id='mix'),
    pytest.param(('h2o2.yaml', 'H2', 'mixture-averaged', True, 'exact'), id='mix-soret'),
    pytest.param(('h2o2.yaml', 'H2', 'multicomponent', False, 'exact'), id='multi-exact'),
    pytest.param(('h2o2.yaml', 'H2', 'multicomponent', True, 'exact'), id='multi-exact-soret'),
    pytest.param(('h2o2.yaml', 'H2', 'multicomponent', False, 'frozen'), id='multi-frozen'),
    pytest.param(('h2o2.yaml', 'H2', 'multicomponent', True, 'frozen'), id='multi-frozen-soret'),
    pytest.param(('gri30.yaml', 'CH4', 'multicomponent', False, 'frozen'), id='gri-frozen'),
    pytest.param(('gri30.yaml', 'CH4', 'multicomponent', True, 'frozen'), id='gri-frozen-soret'),
)


@pytest.fixture(scope='module', params=_CASES)
def rolling_problem(request):
    mechanism, fuel, transport, soret, mode = request.param
    path = resolve_mechanism(mechanism)
    mech = load_mechanism(path)
    feeds = []
    for text in (fuel, 'O2:1,N2:3.76'):
        y = _composition(text, mech.species_names) * mech.molecular_weights
        feeds.append(y / y.sum())
    case = diffusion.DiffusionCase(
        path, *feeds, fuel_mass_flux=.05, oxidizer_mass_flux=.15,
        transport_model=transport, soret_enabled=soret,
        radiation_enabled=True, radiation_scale=2., boundary_emissivities=(.4, .8),
    )
    p = diffusion.DiffusionFlameProblem(
        case, mech, np.array([0., .001, .004, .009, .013, .018, .02]),
    )
    # Isolate rolling reuse; chemistry tiling has separate integration tests.
    p.chemistry_tile_nodes = 0
    p.multicomponent_jacobian = mode
    assert p.ns == (53 if mechanism == 'gri30.yaml' else 10)
    assert p._frozen_multicomponent() == (transport == 'multicomponent' and mode == 'frozen')
    return p


def _signed_state(p):
    fields = p.initial_fields()
    # Unequal face temperatures, including both sides of the NASA switch.
    fields['T'] = np.array([310., 620., 999.999, 1000., 1000.001, 1450., 450.])
    fields['Y'] = fields['Y'].copy()
    trace = p.mech_data.species_names.index('OH')
    # Do not normalize: retain signed Newton trials and absent-species entries.
    fields['Y'][trace, 1:-1] = [-1e-12, 0., -2e-12, 3e-13, -1e-13]
    assert np.any(fields['Y'] < 0.)
    assert np.all(p.backend.invW @ fields['Y'] > 1e-300)
    return p.pack(fields)


def _assert_blocks_bitwise(actual, expected, context):
    for name in _BLOCK_NAMES:
        a = np.ascontiguousarray(getattr(actual, name))
        b = np.ascontiguousarray(getattr(expected, name))
        assert a.shape == b.shape, (context, name)
        assert a.dtype == b.dtype == np.dtype(np.float64)
        assert np.isfinite(a).all(), (context, name)
        np.testing.assert_array_equal(
            a.view(np.uint64), b.view(np.uint64), err_msg=f'{context}: {name}',
        )


@pytest.mark.parametrize('analytic', [False, True], ids=['fd-chemistry', 'analytic-chemistry'])
@pytest.mark.parametrize('fixed', [False, True], ids=['energy', 'fixed-temperature'])
def test_rolling_faces_preserve_blocks_across_range_cuts(monkeypatch, rolling_problem, analytic, fixed):
    p = rolling_problem
    p.analytic_chemistry = analytic
    x = _signed_state(p)
    saved_x = x.copy()
    temperature = p.unpack(x)['T'].copy() if fixed else None
    saved_temperature = None if temperature is None else temperature.copy()
    dispatcher = diffusion_kernels.jacobian_range
    control = {'reuse': False, 'cuts': _CUTS[0]}
    calls = []

    def wrapped_range(start, stop, *arguments):
        calls.append((start, stop, control['reuse']))
        return dispatcher(start, stop, *arguments, reuse_faces=control['reuse'])

    def contiguous_split(function, count, *arguments, minimum=24):
        # Face properties have N-1 entries; Jacobian column ranges have N.
        # Execute serially so the only changed factor is range boundaries/reuse.
        if count == 0:
            return
        bounds = (0, *(cut for cut in control['cuts'][1:-1] if cut < count), count)
        for start, stop in zip(bounds, bounds[1:]):
            function(start, stop, *arguments)

    monkeypatch.setattr(diffusion_kernels, 'jacobian_range', wrapped_range)
    monkeypatch.setattr(diffusion, '_split', contiguous_split)

    reference = p.jacobian_blocks(x, fixed_temperature=temperature)
    assert calls == [(0, p.n, False)]
    saved_reference = {name: getattr(reference, name).copy() for name in _BLOCK_NAMES}
    for cuts in _CUTS:
        control['cuts'] = cuts
        for reuse in (False, True):
            # The full-range, disabled result above already supplies this case.
            if cuts == _CUTS[0] and not reuse:
                continue
            control['reuse'] = reuse
            calls.clear()
            actual = p.jacobian_blocks(x, fixed_temperature=temperature)
            assert calls == [(start, stop, reuse) for start, stop in zip(cuts, cuts[1:])]
            _assert_blocks_bitwise(actual, reference, f'cuts={cuts}, reuse_faces={reuse}')

    # A later build must not overwrite the oracle or any borrowed inputs.
    for name in _BLOCK_NAMES:
        np.testing.assert_array_equal(
            np.ascontiguousarray(getattr(reference, name)).view(np.uint64),
            np.ascontiguousarray(saved_reference[name]).view(np.uint64),
        )
    np.testing.assert_array_equal(x.view(np.uint64), saved_x.view(np.uint64))
    if temperature is not None:
        np.testing.assert_array_equal(temperature.view(np.uint64), saved_temperature.view(np.uint64))


@pytest.mark.parametrize('cuts', _CUTS)
def test_rolling_transition_count_includes_last_node(cuts):
    n = cuts[-1]
    q = len(cuts) - 1
    transitions = [j for start, stop in zip(cuts, cuts[1:]) for j in range(start + 1, stop)]
    starts = set(cuts[1:-1])
    assert transitions == [j for j in range(1, n) if j not in starts]
    assert len(transitions) == n - q
    # The last node has no right face, but can reuse its predecessor's right.
    for j in transitions:
        assert 0 <= j - 1 < n - 1
    assert ((n - 1) in transitions) == ((n - 1) not in starts)
    recomputed_left = q - 1  # Range starting at zero has no left face.
    recomputed_right = n - 1
    assert recomputed_left + recomputed_right == 2 * (n - 1) - (n - q)
