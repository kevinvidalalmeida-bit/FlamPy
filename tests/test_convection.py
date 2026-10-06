"""Convective discretization of the native flame residual."""
import numpy as np
import pytest

from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.flame.config import FlameCase
from kflame.flame.convection import exponential_weights, scheme_code, upwind_fraction
from kflame.flame.equations import residual
from kflame.flame.problem import FreeFlameProblem


def test_upwind_fraction_limits_and_series_continuity():
    assert upwind_fraction(0.) == 0.
    # Both sides of the series switch; the closed form loses digits to cancellation.
    x = np.array([1e-3*(1-1e-12), 1e-3*(1+1e-12)])
    np.testing.assert_allclose(upwind_fraction(x), x/6-x**3/360+x**5/15120, rtol=1e-7)
    assert upwind_fraction(1e4) == pytest.approx(1.-2e-4)
    pe = np.geomspace(1e-6, 1e3, 200)
    assert np.all(np.diff(upwind_fraction(pe)) > 0.) and np.all(upwind_fraction(pe) < 1.)


def _derivative(weights, phi):
    return weights[1:-1, 1]*(phi[2:]-phi[1:-1]) + weights[1:-1, 0]*(phi[1:-1]-phi[:-2])


@pytest.mark.parametrize('direction', [1., -1.])
def test_weights_are_central_at_low_and_upwind_at_high_peclet(direction):
    z = np.cumsum(np.r_[0., np.random.default_rng(5).uniform(.5, 1.5, 40)])*1e-4
    n = len(z)
    phi = 3.+2.*z+7e3*z**2
    exact = 2.+1.4e4*z[1:-1]
    u = np.full(n, direction)
    low = exponential_weights(u, np.ones(n), np.full(n, 1e-9), np.ones(n-1), z)
    np.testing.assert_allclose(_derivative(low, phi), exact, rtol=1e-9)  # exact for quadratics
    high = exponential_weights(u, np.ones(n), np.full(n, 1e14), np.ones(n-1), z)
    dzm, dzp = np.diff(z)[:-1], np.diff(z)[1:]
    upwind = (phi[1:-1]-phi[:-2])/dzm if direction > 0 else (phi[2:]-phi[1:-1])/dzp
    np.testing.assert_allclose(_derivative(high, phi), upwind, rtol=1e-6)


def test_resolved_derivative_converges_at_second_order():
    errors = []
    for n in (41, 81, 161):
        z = np.linspace(0., 1e-3, n)**1.3/1e-3**.3  # nonuniform
        phi = np.tanh((z-5e-4)/1.5e-4)
        exact = (1.-phi[1:-1]**2)/1.5e-4
        # Cell Peclet proportional to dz, as for a fixed flow on a refined grid.
        w = exponential_weights(np.ones(n), np.ones(n), np.full(n, 1.), np.full(n-1, 1e-1), z)
        errors.append(np.max(abs(_derivative(w, phi)-exact)))
    order = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert np.all(order > 1.8), order


def test_scheme_option_changes_the_residual_and_is_validated():
    with pytest.raises(ValueError):
        scheme_code('central')
    results = {}
    for scheme in ('upwind', 'exponential'):
        case = FlameCase(mech='h2o2.yaml', fuel='H2', inlet_mass_flux=.08,
                         cantera_seed_grid=False, convection_scheme=scheme)
        problem = FreeFlameProblem(case)
        problem.backend = NativeSpeciesBackend(problem)
        state = problem.make_initial_guess()
        results[scheme] = residual(state, problem)
        assert problem.last_residual_error is None  # compiled path, no silent fallback
    difference = abs(results['upwind']-results['exponential'])
    assert np.max(difference) > 0.
    np.testing.assert_array_equal(difference[:2], 0.)  # inlet continuity and temperature rows
    assert np.max(difference[2:2+problem.n_species]) > 0.  # second-order inlet species balance


def test_cold_start_uses_upwind_then_solves_the_requested_scheme(monkeypatch):
    import kflame.flame.solver as solver
    calls = []

    def fake(problem, x0, options):
        calls.append((problem.convection_code, x0 is None))
        return np.zeros(3), True, {'final_accepted': True, 'grid_converged': True}

    monkeypatch.setattr(solver, '_solve_flame_auto', fake)
    problem = type('P', (), {'convection_code': 1, 'n_points': 3})()
    _, ok, report = solver.solve_free_flame(problem)
    assert ok and calls == [(0, True), (1, False)]
    assert problem.convection_code == 1 and report['startup_convection']['scheme'] == 'upwind'
    calls.clear()
    solver.solve_free_flame(problem, x0=np.zeros(3))  # continuation: requested scheme only
    assert calls == [(1, False)]


@pytest.mark.parametrize('seed', range(6))
def test_compiled_grid_analysis_matches_python_reference(monkeypatch, seed):
    import kflame.flame.solver as solver
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    z = np.sort(np.r_[0., rng.uniform(0., 3e-2, n-2), 3e-2])
    z = np.unique(z)
    n = len(z)
    centre = rng.uniform(2e-3, 2e-2)
    profiles = {f'p{k}': np.tanh((z-centre)/rng.uniform(1e-4, 3e-3))*rng.uniform(-3., 3.)+rng.normal(0., 1e-3, n)
                for k in range(int(rng.integers(1, 12)))}
    profiles['flat'] = np.ones(n)
    profiles['wrong_shape'] = np.ones(n+1)
    refiner = solver.AdaptiveRefiner(ratio=float(rng.uniform(2., 10.)), slope=.05, curve=.1,
                                     prune=float(rng.choice([0., .005, .02])), max_points=5000)
    j_fixed = None if seed % 2 else int(rng.integers(0, n))
    compiled = refiner.analyze(z, profiles, j_fixed)
    monkeypatch.setattr(solver, '_refine_analyze_numba_core', None)
    assert compiled == refiner.analyze(z, profiles, j_fixed)


@pytest.mark.parametrize('ladder', [1.0, 0.5, float('nan')])
def test_invalid_time_step_ladder_creates_no_output(tmp_path, ladder):
    from kflame import solve_flame
    with pytest.raises(ValueError, match='time_step_ladder'):
        solve_flame(time_step_ladder=ladder, output=tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_free_flame_guess_is_translated_to_the_burner_surface():
    from kflame.api import _burner_guess_from_free_flame
    z = np.linspace(0., .03, 301)
    T = 300.+1700./(1.+np.exp(-(z-.011)/4e-4))
    Y = np.vstack([1.-(T-300.)/1700., (T-300.)/1700.])
    seed = _burner_guess_from_free_flame(dict(z=z, T=T, Y=Y, species_names=np.array(['A', 'B'])), 300., .03)
    assert seed['z'][0] == 0. and seed['z'][-1] == .03 and np.all(np.diff(seed['z']) > 0.)
    assert seed['T'][0] == pytest.approx(300.+.02*(T.max()-300.), rel=1e-6)
    assert np.interp(1000., seed['T'], seed['z']) < 2e-3  # flame moved from 11 mm to the surface
    np.testing.assert_allclose(seed['Y'].sum(axis=0), 1., atol=1e-12)
