"""PLOG admission and native rate/derivative paths against Cantera."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from kflame.chemistry.analytic import species_partials, species_partials_range
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.pressure_rates import pack_plog, plog_rate
from kflame.chemistry.thermo import NativeThermo


@pytest.fixture(scope='module', params=['plog', 'sri', 'troe-signed', 'fourth-order'])
def plog_mechanism(tmp_path_factory, request):
    data = yaml.safe_load(Path(resolve_mechanism('h2o2.yaml')).read_text(encoding='utf-8'))
    data['reactions'] = [{
        'equation': 'H + O <=> OH', 'type': 'pressure-dependent-Arrhenius',
        'rate-constants': [
            {'P': '0.01 atm', 'A': 1e8, 'b': .2, 'Ea': 1000.},
            {'P': '1 atm', 'A': 1.2e9, 'b': -.1, 'Ea': 2000.},
            {'P': '1 atm', 'A': -2e8, 'b': -.1, 'Ea': 2000.},
            {'P': '100 atm', 'A': 3e10, 'b': -.3, 'Ea': 3000.},
        ],
    }, {'equation': 'H2 + O => H + OH',
        'rate-constant': {'A': 1e7, 'b': .1, 'Ea': 500.}}]
    if request.param in ('sri', 'troe-signed'):
        data['reactions'][0] = {
            'equation': 'H + O (+M) <=> OH (+M)', 'type': 'falloff',
            'high-P-rate-constant': {'A': 1e12, 'b': -.2, 'Ea': 1000.},
            'low-P-rate-constant': {'A': 1e20, 'b': -1.2, 'Ea': 2000.},
            'SRI': {'A': .8, 'B': 120., 'C': 800., 'D': 1.1, 'E': .03},
            'efficiencies': {'H2': 2., 'H2O': 6.},
        }
        if request.param == 'troe-signed':
            del data['reactions'][0]['SRI']
            # FFCM-1 contains this signed empirical Troe scale. It must
            # survive every rate and derivative route without clamping.
            data['reactions'][0]['Troe'] = {
                'A': 1.569, 'T3': -9147., 'T1': 299., 'T2': 152.4,
            }
    elif request.param == 'fourth-order':
        data['reactions'][0] = {
            'equation': 'H + O + H2 + O2 <=> H2O + HO2',
            'rate-constant': {'A': 1e15, 'b': -.2, 'Ea': 1000.},
        }
    path = tmp_path_factory.mktemp('plog')/'plog.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return path


@pytest.mark.parametrize('mode', ['numpy', 'dense', 'sparse'])
@pytest.mark.parametrize('pressure_atm', [.001, .01, .1, 1., 10., 100., 1000.])
def test_plog_rates_and_sources_match_cantera(plog_mechanism, mode, pressure_atm):
    ct = pytest.importorskip('cantera')
    gas = ct.Solution(str(plog_mechanism))
    mech = load_mechanism(plog_mechanism)
    thermo, kinetics = NativeThermo(mech), NativeKinetics(mech)
    kinetics._use_numba = mode == 'dense'
    kinetics._use_sparse_numba = mode == 'sparse'
    packed = pack_plog(mech)
    if packed is not None:
        assert all(not array.flags.writeable for array in packed)
        assert len(packed[4]) == 4  # Same-pressure terms, including negative A, survive.
    temperatures = np.array([600., 1100., 2200.])
    concentrations, gibbs, expected = [], [], []
    for temperature in temperatures:
        gas.TPX = temperature, pressure_atm*ct.one_atm, 'H2:.3,O2:.2,N2:.48,H:.01,O:.005,OH:.005'
        concentrations.append(gas.concentrations)
        gibbs.append(thermo.g_RT(temperature))
        expected.append(gas.net_production_rates)
        if packed is not None:
            rate = plog_rate(temperature, gas.P, 0, packed)[0]
            np.testing.assert_allclose(rate, gas.forward_rate_constants[0], rtol=2e-12)
        actual = kinetics.net_production_rates(temperature, gas.concentrations, thermo.g_RT(temperature))
        np.testing.assert_allclose(actual, gas.net_production_rates, rtol=3e-11, atol=1e-12)
    actual = kinetics.net_production_rates(temperatures, np.array(concentrations).T,
                                         np.array(gibbs).T)
    np.testing.assert_allclose(actual, np.array(expected).T, rtol=3e-11, atol=1e-12)


def test_fused_grouped_and_analytic_plog_paths(plog_mechanism):
    ct = pytest.importorskip('cantera')
    mech = load_mechanism(plog_mechanism)
    gas = ct.Solution(str(plog_mechanism))
    gas.TPX = 1100., 2.*ct.one_atm, 'H2:.3,O2:.2,N2:.48,H:.01,O:.005,OH:.005'
    problem = SimpleNamespace(P=gas.P, case=SimpleNamespace(mech=str(plog_mechanism)),
                              assume_finite_y=True)
    backend = NativeSpeciesBackend(problem, mech)
    nv = mech.n_species + 2
    temperatures = np.full(nv, gas.T)
    temperatures[1] += .01
    ys = np.repeat(gas.Y[:, None], nv, axis=1)
    for species in range(mech.n_species):
        ys[species, species + 2] += 1e-7
    first, second = np.empty_like(ys), np.empty_like(ys)
    hfirst, hsecond = np.empty_like(ys), np.empty_like(ys)
    backend.eval_grid_thermo_kinetics_into(temperatures, ys, first, hfirst)
    backend.eval_jacobian_thermo_kinetics_into(temperatures, ys, second, hsecond)
    assert first.tobytes() == second.tobytes()
    assert hfirst.tobytes() == hsecond.tobytes()
    for node in range(nv):
        gas.set_unnormalized_mass_fractions(ys[:, node])
        gas.TP = temperatures[node], problem.P
        np.testing.assert_allclose(first[:, node], gas.net_production_rates*gas.molecular_weights,
                                   rtol=3e-11, atol=1e-12)
    T, Y = temperatures[:1].copy(), ys[:, :1].copy()
    jy, jt = species_partials(backend, T, Y, _temperature=True)
    tiled = species_partials_range(backend, T, Y, 0, 1)
    np.testing.assert_array_equal(jy, tiled[0])
    np.testing.assert_array_equal(jt, tiled[1])
    def reference(t, y):
        gas.set_unnormalized_mass_fractions(y)
        gas.TP = t, problem.P
        return gas.net_production_rates*gas.molecular_weights
    step = 1e-3
    fd_t = (reference(T[0]+step, Y[:, 0])-reference(T[0]-step, Y[:, 0]))/(2*step)
    np.testing.assert_allclose(jt[:, 0], fd_t, rtol=2e-7, atol=1e-9)
    for j in range(mech.n_species):
        step = 1e-6
        yp, ym = Y[:, 0].copy(), Y[:, 0].copy()
        yp[j] += step
        ym[j] -= step
        fd = (reference(T[0], yp)-reference(T[0], ym))/(2*step)
        np.testing.assert_allclose(jy[:, 0, j], fd, rtol=3e-6, atol=1e-9)


@pytest.mark.parametrize('plog_mechanism', ['plog'], indirect=True)
def test_plog_invalid_pressure_and_nonpositive_group_rejected(plog_mechanism):
    mech = load_mechanism(plog_mechanism)
    packed = pack_plog(mech)
    with pytest.raises(ValueError, match='positive finite'):
        plog_rate(1000., 0., 0, packed)
    mech.reactions[0].plog_rates[:, 1] *= -1
    with pytest.raises(ValueError, match='positive finite sum'):
        plog_rate(1000., 101325., 0, pack_plog(mech))


@pytest.mark.parametrize('plog_mechanism', ['sri'], indirect=True)
@pytest.mark.parametrize('parameters', [
    {'A': 1., 'C': 800.},
    {'A': 1., 'B': 120., 'C': 0.},
    {'A': 1., 'B': 120., 'C': 800., 'D': 0.},
    {'A': float('nan'), 'B': 120., 'C': 800.},
])
def test_invalid_sri_rejected_at_admission(plog_mechanism, tmp_path, parameters):
    data = yaml.safe_load(plog_mechanism.read_text(encoding='utf-8'))
    data['reactions'][0]['SRI'] = parameters
    path = tmp_path/'invalid-sri.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    with pytest.raises(ValueError, match='SRI requires'):
        load_mechanism(path)


def test_diffusion_grouped_fallback_uses_complete_pressure_rates(plog_mechanism):
    from kflame.flame.diffusion import DiffusionCase, DiffusionFlameProblem

    ct = pytest.importorskip('cantera')
    gas = ct.Solution(str(plog_mechanism))
    gas.TPX = 1100., 2.*ct.one_atm, 'H2:.3,O2:.2,N2:.48,H:.01,O:.005,OH:.005'
    composition = gas.Y.copy()
    mechanism = load_mechanism(plog_mechanism)
    fuel = np.zeros(mechanism.n_species)
    fuel[mechanism.species_names.index('H2')] = 1.
    gas.TPX = gas.T, gas.P, 'O2:1,N2:3.76'
    oxidizer = gas.Y.copy()
    case = DiffusionCase(str(plog_mechanism), fuel, oxidizer,
                         P=gas.P, fuel_temperature=gas.T, oxidizer_temperature=gas.T)
    problem = DiffusionFlameProblem(case, mechanism, np.array([0., .01, .02]))
    fields = problem.initial_fields()
    fields['T'][:] = gas.T
    fields['Y'][:] = composition[:, None]
    x = problem.pack(fields)
    problem.analytic_chemistry = False
    actual = problem.jacobian_blocks(x).tocsc().toarray()
    residual = problem.residual(x)
    expected = problem.jacobian(x, problem.residual, residual).toarray()
    scale = np.max(np.abs(expected), axis=1, keepdims=True)
    assert np.all(np.abs(actual-expected) <= 3e-5*scale+1e-10)


@pytest.mark.parametrize('plog_mechanism', ['fourth-order'], indirect=True)
def test_fourth_order_derivatives_preserve_signed_newton_continuation(plog_mechanism):
    ct = pytest.importorskip('cantera')
    mechanism = load_mechanism(plog_mechanism)
    gas = ct.Solution(str(plog_mechanism))
    gas.TPX = 1100., 2.*ct.one_atm, 'H2:.3,O2:.2,N2:.48,H:.01,O:.005,OH:.003,H2O:.001,HO2:.001'
    y = gas.Y.copy()
    y[gas.species_index('H')] = -1e-3
    backend = NativeSpeciesBackend(SimpleNamespace(P=gas.P,
        case=SimpleNamespace(mech=str(plog_mechanism)), assume_finite_y=True), mechanism)
    jy, jt = species_partials(backend, np.array([gas.T]), y[:, None], _temperature=True)
    def reference(t, composition):
        gas.set_unnormalized_mass_fractions(composition)
        gas.TP = t, backend.problem.P
        return gas.net_production_rates*gas.molecular_weights
    temperature = 1100.
    step = 1e-3
    fd = (reference(temperature+step, y)-reference(temperature-step, y))/(2*step)
    np.testing.assert_allclose(jt[:, 0], fd, rtol=3e-6, atol=1e-9)
    for j in range(mechanism.n_species):
        plus, minus = y.copy(), y.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        fd = (reference(temperature, plus)-reference(temperature, minus))/(2e-6)
        np.testing.assert_allclose(jy[:, 0, j], fd, rtol=3e-6, atol=1e-9)
