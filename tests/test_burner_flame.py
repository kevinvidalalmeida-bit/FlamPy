"""Boundary, Jacobian and thermodynamic checks for the burner extension."""
import numpy as np
import pytest

from kflame import solve_burner_flame
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.flame.config import FlameCase
from kflame.flame.problem import FreeFlameProblem
from kflame.flame.analytic_jacobian import build_analytic_blocks
from kflame.flame.equations import (
    residual, build_local_jacobian_cache, residual_local_rows,
    residual_local_rows_batch, residual_local_rows_batch_perturbed,
)
from kflame.flame.solver import _apply_fixed_temperature_anchor, _domain_too_narrow
from kflame.flame.enthalpy import inlet_enthalpy_diagnostics


@pytest.fixture(params=['mixture-averaged', 'multicomponent'])
def burner(request):
    case = FlameCase(mech='h2o2.yaml', fuel='H2', inlet_mass_flux=.08,
                     transport_model=request.param, soret_enabled=True,
                     cantera_seed_grid=False)
    problem = FreeFlameProblem(case)
    problem.backend = NativeSpeciesBackend(problem)
    state = problem.make_initial_guess()
    problem.setup_fixed_temperature(T_profile=state.reshape(problem.n_points, -1)[:, 1])
    return problem, state


def test_all_residual_routes_have_the_same_burner_boundary(burner):
    p, x = burner
    compiled = residual(x, p)
    assert p.last_residual_error is None  # No silent compiled fallback.
    p.use_numba_residual = False
    np.testing.assert_allclose(residual(x, p), compiled, rtol=2e-12, atol=1e-7)
    cache = build_local_jacobian_cache(x, p)
    nv = p.n_vars_per_point
    for node in range(p.n_points):
        rows, local = residual_local_rows(x, p, node, cache)
        np.testing.assert_allclose(local, compiled[rows], rtol=2e-11, atol=1e-7)
        rows, batch = residual_local_rows_batch(x[None, :], p, node, cache)
        np.testing.assert_allclose(batch[0], local, rtol=2e-11, atol=1e-7)
        cols = np.arange(node * nv, (node + 1) * nv, dtype=np.int32)
        rows, numba_batch = residual_local_rows_batch_perturbed(x, p, node, cols, x[cols], cache)
        assert p.last_residual_error is None
        np.testing.assert_allclose(numba_batch, np.broadcast_to(local, numba_batch.shape), rtol=2e-11, atol=1e-7)
    assert p.j_fixed is None
    np.testing.assert_allclose(compiled[::nv], 0., atol=1e-12)


def test_analytic_jacobian_matches_local_differences_and_is_full_rank(burner):
    p, x = burner
    jac = build_analytic_blocks(x, p)
    cache = build_local_jacobian_cache(x, p)
    nv = p.n_vars_per_point
    for node in range(p.n_points):
        for var in [0, 1, 2 + p.species_names.index('O2')]:
            col = node * nv + var
            step = 1e-6 * max(abs(x[col]), .01)
            plus, minus = x.copy(), x.copy()
            plus[col] += step
            minus[col] -= step
            rows, fplus = residual_local_rows(plus, p, node, cache)
            _, fminus = residual_local_rows(minus, p, node, cache)
            direction = np.zeros_like(x)
            direction[col] = 1.
            np.testing.assert_allclose(jac.matvec(direction)[rows], (fplus - fminus) / (2 * step),
                                       rtol=2e-4, atol=2e-3)
    # No redundant outlet row or missing free-flame anchor.
    velocity_columns = []
    for node in range(p.n_points):
        e = np.zeros_like(x)
        e[node * nv] = 1.
        velocity_columns.append(jac.matvec(e)[::nv])
    assert np.linalg.matrix_rank(np.column_stack(velocity_columns)) == p.n_points


def test_physical_burner_gradient_does_not_trigger_domain_expansion(burner):
    p, x = burner
    old_z = p.z.copy()
    np.testing.assert_array_equal(_apply_fixed_temperature_anchor(p, x), x)
    np.testing.assert_array_equal(p.z, old_z)
    state = x.reshape(p.n_points, -1).copy()
    state[:, 1] = np.linspace(300., 1200., p.n_points)
    state[1:, 1] = 1200.
    narrow, metrics = _domain_too_narrow(p, state.ravel())
    assert metrics['m_left'] > 1.
    assert not narrow and not metrics['left_gradient_checked']
    state[-2, 1] = 1100.
    assert _domain_too_narrow(p, state.ravel())[0]


@pytest.mark.parametrize('bad_flux', [None, 0., -1., np.nan, np.inf])
def test_invalid_mass_flux_never_creates_output(tmp_path, bad_flux):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='mass_flux'):
        solve_burner_flame(mass_flux=bad_flux, output=output)
    assert not output.exists()


@pytest.mark.parametrize('mechanism,fuel', [('h2o2.yaml', 'H2'), ('gri30.yaml', 'CH4')])
def test_total_enthalpy_matches_independent_thermodynamics(mechanism, fuel):
    ct = pytest.importorskip('cantera')
    mech = load_mechanism(mechanism)
    thermo = NativeThermo(mech)
    Y = fresh_mixture(mech, 1., fuel, 'O2:1,N2:3.76')
    temperatures = np.array([300., 999., 1001., 2200.])
    gas = ct.Solution(mechanism)
    expected = []
    for T in temperatures:
        gas.TPY = T, ct.one_atm, Y
        expected.append(gas.enthalpy_mass)
    np.testing.assert_allclose(thermo.enthalpy_mass(temperatures, Y), expected, rtol=2e-12, atol=1e-7)
    assert thermo.enthalpy_mass(300., Y) == pytest.approx(expected[0], abs=1e-7)
    profiles = np.repeat(Y[:, None], temperatures.size, axis=1)
    np.testing.assert_allclose(thermo.enthalpy_mass(temperatures, profiles), expected, rtol=2e-12, atol=1e-7)
    np.testing.assert_allclose(thermo.enthalpy_mass(300., profiles), expected[0], rtol=2e-12, atol=1e-7)


@pytest.mark.parametrize('mechanism', ['h2o2.yaml', 'gri30.yaml'])
def test_fused_enthalpy_and_derivative_match_cantera_across_species_breakpoints(mechanism):
    ct = pytest.importorskip('cantera')
    mech = load_mechanism(mechanism)
    thermo = NativeThermo(mech)
    mid = np.unique(mech.nasa_Tmid)
    T = np.r_[300., mid-1e-6, mid, mid+1e-6, 2200.]
    Y = np.random.default_rng(143).dirichlet(np.ones(mech.n_species),len(T)).T
    h, cp = thermo.enthalpy_mass_cp_mass(T,Y)
    gas = ct.Solution(mechanism)
    expected = []
    for temperature,y in zip(T,Y.T):
        gas.TPY = temperature,ct.one_atm,y
        expected.append([gas.enthalpy_mass,gas.cp_mass])
    np.testing.assert_allclose(np.column_stack([h,cp]),expected,rtol=3e-12,atol=1e-7)
    # Broadcasting and strided profiles must retain the single-state result.
    for temperature,fractions in [(T,Y[:,0]),(300.,Y),(300.,Y[:,0]),(T[::2],Y[:,::2])]:
        hh,cc = thermo.enthalpy_mass_cp_mass(temperature,fractions)
        yy = fractions[:,None] if fractions.ndim == 1 else fractions
        np.testing.assert_allclose(hh,thermo.enthalpy_mass(temperature,fractions),rtol=3e-12,atol=1e-7)
        expected_cp = np.sum(yy*thermo.partial_molar_cp(temperature).reshape(mech.n_species,-1)
                             *mech.inv_molecular_weights[:,None],axis=0)
        np.testing.assert_allclose(np.atleast_1d(cc),expected_cp,rtol=3e-12,atol=1e-10)


@pytest.mark.parametrize('mechanism', ['h2o2.yaml', 'gri30.yaml'])
def test_compiled_species_polynomials_are_bitwise_the_numpy_path(mechanism):
    thermo = NativeThermo(load_mechanism(mechanism))
    mid = thermo._Tmid
    T = np.r_[np.linspace(250., 3500., 401), mid, np.nextafter(mid, 0.), np.nextafter(mid, 1e9)]
    for argument in (T, T[::3], 1000., np.float64(1500.3)):
        T_arr, scalar = thermo._to_1d(argument)
        c, t = thermo._get_coefs(T_arr), T_arr[:, None]
        cp = c[:,:,0]+t*(c[:,:,1]+t*(c[:,:,2]+t*(c[:,:,3]+t*c[:,:,4])))
        h = (c[:,:,0]+t*(c[:,:,1]/2.+t*(c[:,:,2]/3.+t*(c[:,:,3]/4.+t*c[:,:,4]/5.)))
             +c[:,:,5]/t)
        for actual, expected in ((thermo.cp_R(argument), cp), (thermo.h_RT(argument), h)):
            expected = expected[0] if scalar else expected.T
            # Identical bits and memory layout keep downstream reductions unchanged.
            assert np.array_equal(actual, expected)
            assert actual.strides == expected.strides


@pytest.mark.parametrize('mechanism,fuel', [('h2o2.yaml','H2'),('gri30.yaml','CH4')])
def test_inlet_enthalpy_defect_is_independent_of_prescribed_total_flux(mechanism,fuel):
    mech = load_mechanism(mechanism)
    thermo = NativeThermo(mech)
    feed = fresh_mixture(mech,1.,fuel,'O2:1,N2:3.76')
    Y = .95*feed+.05/len(feed)
    mdot = .1
    flux = mdot*(feed-Y)
    matching = inlet_enthalpy_diagnostics(thermo,300.,Y,feed,flux,mdot,energy_scale=100.)
    assert matching['inlet_enthalpy_flux_error_W_m2'] == pytest.approx(0.,abs=1e-10)
    flux[mech.species_names.index(fuel)] += 1e-4
    flux[mech.species_names.index('O2')] -= 1e-4
    different = inlet_enthalpy_diagnostics(thermo,300.,Y,feed,flux,mdot,energy_scale=100.)
    expected = mdot*thermo.enthalpy_mass(300.,Y)+thermo.partial_molar_enthalpies(300.)@(
        flux*mech.inv_molecular_weights)-mdot*thermo.enthalpy_mass(300.,feed)
    assert different['inlet_enthalpy_flux_error_W_m2'] == pytest.approx(expected,abs=1e-9)
    assert different['relative_inlet_enthalpy_flux_error'] == pytest.approx(abs(expected)/100.)
