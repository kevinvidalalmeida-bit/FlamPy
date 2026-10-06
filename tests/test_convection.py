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
    nv = 2+problem.n_species
    np.testing.assert_array_equal(difference[:nv], 0.)  # boundary rows unchanged
