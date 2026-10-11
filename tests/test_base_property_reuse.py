"""The accepted residual's properties may feed only that Newton base."""
import time

import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion


def problem(mechanism, transport, soret):
    path = resolve_mechanism(mechanism)
    mech = load_mechanism(path)
    fuel = 'H2' if mechanism == 'h2o2.yaml' else 'CH4'
    feeds = []
    for text in (fuel, 'O2:1,N2:3.76'):
        y = _composition(text, mech.species_names) * mech.molecular_weights
        feeds.append(y / y.sum())
    case = diffusion.DiffusionCase(path, *feeds, transport_model=transport,
                                   soret_enabled=soret, radiation_enabled=True)
    return diffusion.DiffusionFlameProblem(case, mech, np.array([0., .001, .004, .013, .02]))


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes(order='C') == expected.tobytes(order='C')


@pytest.mark.parametrize('mechanism', ['h2o2.yaml', 'gri30.yaml'])
@pytest.mark.parametrize('transport,soret', [('mixture-averaged', False),
                                          ('multicomponent', False), ('multicomponent', True)])
@pytest.mark.parametrize('fused', [False, True])
def test_reused_base_matches_uncached_blocks_and_preserves_inputs(mechanism, transport, soret, fused):
    p = problem(mechanism, transport, soret)
    p.use_fused_numba_thermo_kinetics = fused
    fields = p.initial_fields()
    fields['T'] = np.array([300., 620., 999.999, 1000.001, 430.])
    fields['Y'][p.mech_data.species_names.index('OH'), 1:4] = [-1e-12, 0., 1e-13]
    x = p.pack(fields)
    properties = p.properties(p.unpack(x))
    saved = [value.copy() for value in properties]
    for fixed in (None, fields['T'].copy()):
        reference = p.jacobian_blocks(x, fixed_temperature=fixed)
        actual = p.jacobian_blocks(x, fixed_temperature=fixed, cached_properties=properties)
        for name in ('lower', 'diag', 'upper', 'border', 'mass'):
            exact(getattr(actual, name), getattr(reference, name))
        exact(p.residual(x, fixed_temperature=fixed, cached_properties=properties),
              p.residual(x, fixed_temperature=fixed))
    for value, original in zip(properties, saved):
        exact(value, original)


def test_newton_does_not_reuse_rejected_trial_properties(monkeypatch):
    p = problem('h2o2.yaml', 'multicomponent', True)
    # A controlled native problem exercises line search and an aged refresh;
    # real property/Jacobian bit equality is covered separately above.
    start = p.pack(p.initial_fields())
    start[0] = 1.
    visits = []
    bases = []
    def properties(fields):
        marker = float(fields['m'][0] / p.mref)
        visits.append(marker)
        return (marker,)
    def residual(value, **options):
        assert options['cached_properties'][0] == value[0]
        result = np.zeros_like(value)
        result[0] = value[0]
        return result
    class Factor:
        def solve(self, value):
            return 3. * value
    class Model:
        def __init__(self, problem, value, fixed, *, cached_properties=None):
            assert cached_properties[0] == value[0]
            bases.append(value[0])
            self.age = 0
            self.fixed = False
            self.blocks = type('Blocks', (), {'n': problem.n})()
        def factor(self, rdt):
            return Factor()
    monkeypatch.setattr(p, 'properties', properties)
    monkeypatch.setattr(p, 'residual', residual)
    monkeypatch.setattr(diffusion, '_LinearModel', Model)
    solved, accepted, reason = diffusion._newton(p, start, time.perf_counter()+10.,
                                                max_iterations=3, max_age=1)
    assert reason == 'iteration_limit'
    assert not accepted
    assert bases == [1., -.5, .25]
    assert visits == [1., -2., -.5, 1., .25, -.5, -.125]
    assert solved[0] == -.125


def test_noncompiled_and_subclass_paths_keep_original_evaluation(monkeypatch):
    p = problem('h2o2.yaml', 'mixture-averaged', False)
    p.compiled_faces = False
    x = p.pack(p.initial_fields())
    calls = []
    def residual(value, **options):
        calls.append(options)
        return np.zeros_like(value)
    monkeypatch.setattr(p, 'residual', residual)
    _, accepted, _ = diffusion._newton(p, x, time.perf_counter()+10.)
    assert accepted
    assert 'cached_properties' not in calls[0]


@pytest.mark.parametrize('asymmetric', [False, True])
def test_compiled_face_symmetry_is_bound_to_owned_inputs(asymmetric):
    import _multicomponent_reference as reference
    from kflame.flame import diffusion_kernels
    p = problem('h2o2.yaml', 'multicomponent', True)
    transport = p.backend.multicomponent_transport
    if asymmetric:
        modified = transport._astar_poly.copy()
        modified[1, 0, 0] = np.nextafter(modified[1, 0, 0], np.inf)
        transport._astar_poly = modified
    data = p._compiled_data()
    assert len(data['multi']) == 14
    multi = data['multi_certified']
    assert len(multi) == 15
    assert multi[14] == (not asymmetric)
    if not asymmetric:
        assert not data['mw'].flags.writeable
    expected = diffusion_kernels._mdata(p.ns)
    actual = diffusion_kernels._mdata(p.ns)
    for temperature in (200., 999.999, 1000.001, 3500.):
        reference.temperature_terms(temperature, data['mw'], multi[0], multi[1],
            multi[3], multi[4], multi[5], multi[6], multi[7], multi[8], multi[9], *expected[:6])
        diffusion_kernels._multi_temperature(temperature, multi, data['mw'], actual)
        for got, want in zip(actual[:6], expected[:6]):
            exact(got, want)
