"""Physical closures of the reduced transport, independently of convergence."""
from pathlib import Path

import numpy as np
import pytest

from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import ReducedBurnerProblem


@pytest.fixture(scope='module')
def model():
    return NonAdiabaticFGM(Path(__file__).resolve().parents[1]/'docs/assets/nonadiabatic3d')


@pytest.fixture
def problem(model):
    feed = fresh_mixture(load_mechanism(model.metadata['mechanism']), 1., 'CH4', 'O2:1,N2:3.76')
    return ReducedBurnerProblem(model, np.linspace(0., .03, 13), .1, feed)


def test_logical_map_agrees_with_physical_lookup_and_thermodynamics(problem):
    problem = ReducedBurnerProblem(problem.model, problem.z, problem.mass_flux,
                                   problem.inlet_Y, interpolation='barycentric')
    x = np.random.default_rng(5134).uniform(.12, .87, (13, 3))
    result = problem.state(x)
    physical = problem.model.lookup_batch(Z=result['Z'], C=result['C'], h=result['h'])
    np.testing.assert_allclose(result['Y'].T, physical['Y'], atol=2e-12)
    np.testing.assert_allclose(result['omega_C'], physical['omega_C'], rtol=2e-10, atol=1e-8)
    np.testing.assert_allclose(result['T'], physical['T'], atol=1e-7)
    np.testing.assert_allclose(problem.model.thermo.enthalpy_mass(result['T'], result['Y']), result['h'], atol=1e-6)


def test_smooth_approximation_preserves_mass_elements_and_enthalpy(problem):
    x = np.random.default_rng(5331).uniform(.02, .98, (13, 3))
    state = problem.state(x)
    np.testing.assert_allclose(state['Y'].sum(axis=0), 1., atol=1e-12)
    assert np.min(state['Y']) > -1e-12
    np.testing.assert_allclose(problem.model.thermo.enthalpy_mass(state['T'], state['Y']), state['h'], atol=1e-6)
    np.testing.assert_allclose(problem.model.table['bilger_weights']@state['Y']+float(problem.model.table['bilger_offset']), state['Z'], atol=1e-14)


def test_compiled_progress_polygon_is_strictly_monotone_for_all_training_rows(problem):
    controls = (problem.fields['Y']@problem.model.table['progress_weights']).reshape(tuple(problem.shape))
    assert np.diff(controls, axis=2).min() > 0.


def test_closure_continuation_preserves_mass_and_recovers_guide_end(problem):
    guide = ReducedBurnerProblem(problem.model, problem.z, problem.mass_flux,
                                 problem.inlet_Y, interpolation='quadratic_progress')
    x = np.random.default_rng(1134).uniform(.2, .8, (13, 3))
    exact = problem.state(x)
    problem.initialization_problem = guide
    problem.closure_blend = 0.
    result = problem.state(x)
    for name in ['Y', 'h', 'T', 'omega_C']:
        np.testing.assert_allclose(result[name], guide.state(x)[name], rtol=1e-11, atol=1e-7)
    problem.closure_blend = .35
    np.testing.assert_allclose(problem.state(x)['Y'].sum(axis=0), 1., atol=1e-12)
    problem.closure_blend = 1.
    np.testing.assert_array_equal(problem.state(x)['Y'], exact['Y'])


def test_progress_direction_has_continuous_first_derivative(problem):
    x = np.tile([.363, .273, 51./(problem.shape[2]-2)], (13, 1))
    eps = 1e-7
    left, right = x.copy(), x.copy()
    left[:, 2] -= eps
    right[:, 2] += eps
    a, b, c = problem.state(left), problem.state(x), problem.state(right)
    for key in ['C', 'h', 'omega_C']:
        np.testing.assert_allclose((b[key]-a[key])/eps, (c[key]-b[key])/eps, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize('axis', [0, 1])
def test_tensor_derivatives_are_continuous_at_composition_and_loss_knots(problem, axis):
    x = np.tile([.36, .27, .55], (13, 1))
    x[:, axis] = problem.axis_knots[axis][6]
    eps = 1e-8
    left, right = x.copy(), x.copy()
    left[:, axis] -= eps
    right[:, axis] += eps
    a, b, c = problem.state(left), problem.state(x), problem.state(right)
    for key in ['Z', 'C', 'h', 'omega_C']:
        np.testing.assert_allclose((b[key]-a[key])/eps, (c[key]-b[key])/eps, rtol=3e-4, atol=3e-3)


def test_equal_species_and_thermal_diffusion_is_positive_for_any_manifold(problem, monkeypatch):
    """For Le=1 the projected principal diffusion must be rho*D*I."""
    diffusivity = 1e-5
    def equal_diffusion(T, P, Y, invW):
        rho = problem.model.thermo.density(T, P, Y)
        cp = problem.model.thermo.cp_mass(T, Y)
        W = 1./(invW@Y)
        return rho, np.full_like(Y, diffusivity), rho*diffusivity*cp, W
    monkeypatch.setattr(problem.transport, 'eval_faces_poly_fast', equal_diffusion)
    x = np.random.default_rng(914).uniform(.2, .8, (13, 3))
    state = problem.state(x)
    expected = problem.model.thermo.density(state['T'], problem.pressure, state['Y'])[1:-1].min()*diffusivity
    result = problem.diffusion_diagnostics(x)
    assert result['passed']
    assert result['minimum_real_eigenvalue_kg_m_s'] == pytest.approx(expected, rel=2e-4)


def test_progress_initializer_recovers_requested_physical_C(problem):
    row = problem.model.table['C'].reshape(tuple(problem.shape))[7, 5]
    endpoints = np.column_stack([np.full(13, 7./(problem.shape[0]-1)),
                                 np.full(13, 5./(problem.shape[1]-1)), np.linspace(0., 1., 13)])
    controls = problem.state(endpoints)['C']
    target = np.linspace(controls[0], controls[-1], 13)
    coordinate = problem.initial_progress(target, row, composition_index=7, loss_index=5)
    x = np.column_stack([np.full(13, 7./(problem.shape[0]-1)),
                          np.full(13, 5./(problem.shape[1]-1)), coordinate])
    np.testing.assert_allclose(problem.state(x)['C'], target, atol=1e-10)


def test_corrected_species_flux_has_zero_total_mass(problem):
    x = np.column_stack([np.linspace(.35, .39, 13), np.linspace(.25, .31, 13), np.linspace(.1, .9, 13)])
    flux = problem.fluxes(problem.state(x))
    np.testing.assert_allclose(np.sum(flux['species'], axis=0), 0., atol=1e-12)


def test_total_enthalpy_has_no_double_counted_chemical_source(problem, monkeypatch):
    x = np.tile([.36, .27, .55], (13, 1))
    baseline = problem.residual(x.ravel()).reshape(-1, 3)
    # The first cell also feels the prescribed feed enthalpy at the inlet.
    np.testing.assert_allclose(baseline[2:-1, 2], 0., atol=1e-12)
    assert np.max(abs(baseline[1:-1, 1])) > 0.
    monkeypatch.setitem(problem.model.table, 'qdot', np.full_like(problem.model.table['qdot'], 1e15))
    changed = problem.residual(x.ravel()).reshape(-1, 3)
    np.testing.assert_array_equal(changed, baseline)


def test_boundary_projection_and_outlet_zero_gradients(problem):
    x = np.tile([.36, .27, .7], (13, 1))
    x[0, 2] = 0.
    state = problem.state(x)
    assert state['T'][0] == pytest.approx(problem.T_burner, abs=1e-7)
    residual = problem.residual(x.ravel()).reshape(-1, 3)
    np.testing.assert_array_equal(residual[-1], np.zeros(3))
    flux = problem.fluxes(state)
    assert residual[0, 0]*problem.scales[0] == pytest.approx(
        flux['total'][0, 0]-problem.mass_flux*problem.Z_feed)
    assert residual[0, 1]*problem.scales[1] == pytest.approx(
        flux['total'][0, 1]-problem.mass_flux*problem.C_feed)
    assert flux['total'][0, 2] == pytest.approx(problem.mass_flux*problem.h_feed+flux['conduction'][0])


def test_colored_thermodynamic_cache_matches_full_residual(problem):
    x = np.random.default_rng(818).uniform(.2, .8, (13, 3)).ravel()
    base = problem.state(x)
    selected = np.arange(1, len(x), 9)
    trial = x.copy()
    trial[selected] += 1e-7
    cached = problem.residual(trial, geometry=base['geometry'], temperature_cache=base['T'], changed_nodes=selected//3)
    full = problem.residual(trial, geometry=base['geometry'])
    np.testing.assert_allclose(cached, full, rtol=1e-10, atol=1e-11)


def test_external_states_are_rejected_and_never_clipped(problem):
    x = np.tile([.36, .27, .7], (13, 1))
    x[7, 1] = 1.00001
    with pytest.raises(ValueError, match='outside'):
        problem.state(x)


@pytest.mark.parametrize('grid,flux', [(np.array([0., .01, .009, .02, .03]), .1),
                                      (np.linspace(0., .03, 5), -1.)])
def test_invalid_grid_or_flux_rejected(model, grid, flux):
    feed = fresh_mixture(load_mechanism(model.metadata['mechanism']), 1., 'CH4', 'O2:1,N2:3.76')
    with pytest.raises(ValueError):
        ReducedBurnerProblem(model, grid, flux, feed)
