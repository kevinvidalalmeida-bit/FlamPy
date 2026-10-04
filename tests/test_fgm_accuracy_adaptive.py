"""Independent tolerances, budget accounting and complete cell searches."""
import numpy as np
import pytest

from kflame.fgm.accuracy import FGMTolerances, assess_profile, source_metrics
from kflame.fgm.adaptive_nonadiabatic import midpoint_indices, generate_adaptive_nonadiabatic_fgm, AdaptiveAccuracyError
from kflame.fgm.nonadiabatic3d import SimplexMesh, AmbiguousManifoldError
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo


def test_each_quantity_has_an_independent_user_limit(tmp_path):
    tolerance = FGMTolerances.from_dict({'temperature_K': 3., 'source_peak_relative': {'qdot': .01}})
    assert tolerance.source_peak_relative['omega_C'] == .05
    assert tolerance.source_integral_relative['qdot'] == .03
    z = np.array([0., .1, 1.])
    truth = dict(z=z, T=np.full(3, 1000.), Y=np.full((2,3), .5), omega_C=np.array([0.,2.,0.]), qdot=np.array([0.,4.,0.]))
    pred = dict(covered=np.ones(3,bool), T=truth['T']+2., Y=truth['Y'].T,
                omega_C=truth['omega_C']*1.02, qdot=truth['qdot']*1.02)
    result = assess_profile(truth, pred, tolerance)
    assert not result['passed']
    assert result['ratios']['temperature'] < 1.
    assert result['ratios']['omega_C_peak'] < 1.
    assert result['ratios']['qdot_peak'] == pytest.approx(2.)
    pred['qdot'] = truth['qdot']*1.005
    assert assess_profile(truth, pred, tolerance)['passed']
    pred['Y'] = pred['Y'].copy()
    pred['Y'][0,0] = np.nan
    with pytest.raises(ValueError,match='finite'):
        assess_profile(truth,pred,tolerance)


@pytest.mark.parametrize('settings', [{'temperature_K': 0.}, {'species_absolute': float('nan')},
    {'source_coverage_min': {'qdot': 1.1}}, {'source_peak_relative': {'typo': .1}}, {'typo': .01},
    {'temperature_K': '20'}, {'source_L1_relative': {'omega_C': None}}])
def test_invalid_tolerances_fail(settings):
    with pytest.raises(ValueError):
        FGMTolerances.from_dict(settings)


def test_zero_source_is_vacuously_covered_but_nonzero_prediction_is_rejected():
    metrics = source_metrics(np.array([0.,1.]), np.zeros(2), np.zeros(2))
    assert metrics['absolute_source_coverage'] == 1.
    assert metrics['L1_relative'] == 0.
    metrics = source_metrics(np.array([0.,1.]), np.zeros(2), np.ones(2))
    assert metrics['Linf_over_truth_peak'] > 1e200


def test_midpoints_use_physical_coordinates_and_budget_counts_probes(tmp_path):
    assert midpoint_indices(np.array([0.,.1,.2,.9,1.]), {0,4}) == [2]
    with pytest.raises(AdaptiveAccuracyError, match='budget'):
        generate_adaptive_nonadiabatic_fgm(phis=[.8,1.,1.2,1.4,1.6],
            mass_flux_fractions=[.6,.4,.2,.1,.05], output=tmp_path/'limited',
            initial_phi_count=2, initial_loss_count=2, max_flames=3)
    import json
    report = json.loads((tmp_path/'limited'/'adaptive_report.json').read_text())
    assert report['evaluated_flames'] == 0
    assert not report['passed']


def test_native_failure_is_recorded_and_propagated(tmp_path,monkeypatch):
    import json
    import kflame.api
    def rejected(**kwargs):
        raise RuntimeError('native flame rejected')
    monkeypatch.setattr(kflame.api,'generate_nonadiabatic_fgm',rejected)
    with pytest.raises(RuntimeError,match='native flame rejected'):
        generate_adaptive_nonadiabatic_fgm(phis=[.8,1.,1.2],mass_flux_fractions=[.6,.3,.1],
            output=tmp_path/'rejected',initial_phi_count=2,initial_loss_count=2)
    report=json.loads((tmp_path/'rejected'/'adaptive_report.json').read_text())
    assert not report['passed']
    assert report['reason']=='native_generation_failure'


@pytest.mark.parametrize('fractions', [.2, [[.6,.3,.1]], [.6,.3,.3], [np.nan,.3,.1]])
def test_invalid_adaptive_axis_is_rejected_before_output(tmp_path, fractions):
    output = tmp_path/'invalid'
    with pytest.raises(ValueError, match='candidates'):
        generate_adaptive_nonadiabatic_fgm(phis=[.8,1.,1.2],
            mass_flux_fractions=fractions, output=output)
    assert not output.exists()


def test_multiple_probe_banks_reuse_new_profiles_and_recheck_physics(tmp_path, monkeypatch):
    """A partial initial bank must not force repeated native solves."""
    import json
    import kflame.api as api
    from kflame.chemistry.initialization import fresh_mixture
    from kflame.chemistry.mechanism import resolve_mechanism
    mech = load_mechanism('h2o2.yaml')
    thermo = NativeThermo(mech)
    banks = []
    for i, phi in enumerate([.8,1.2]):
        bank = tmp_path/f'bank_{i}'
        bank.mkdir()
        banks.append(bank)
        y = fresh_mixture(mech, phi, 'H2', 'O2:1,N2:3.76')
        flux = float(thermo.density(300.,101325.,y))
        rows = []
        for j, fraction in enumerate([None,.4,.2]):
            folder = bank/f'flame_{j}'
            folder.mkdir()
            free = fraction is None
            meta = dict(accepted=True, backend='native_cpu',
                flow_type='free_flame' if free else 'isothermal_burner',
                report={'grid_converged':True}, mechanism=str(resolve_mechanism('h2o2.yaml')),
                species_names=mech.species_names, inlet_Y=y.tolist(),
                temperature=300., pressure=101325., initial_width=.03,
                transport='mixture-averaged', soret=False, refinement={}, tolerances={})
            if free:
                meta['Su'] = 1.
            else:
                meta.update(mass_flux=flux*fraction, inlet_velocity=fraction,
                    heat_loss={'relative_energy_closure_error':0.})
            (folder/'metadata.json').write_text(json.dumps(meta))
            np.savez(folder/'flame.npz', T=[300.,900.], Y=np.column_stack([y,y]), z=[0.,.03])
            rows.append(dict(kind='adiabatic_reference' if free else 'isothermal_burner',
                phi=phi, fraction=fraction, output=folder.name))
        (bank/'generation.json').write_text(json.dumps(dict(all_final_accepted=True, rows=rows)))
    def unexpected_solve(**kwargs):
        raise RuntimeError('unexpected native solve')
    # Preserve the inspected public defaults when the solver is replaced.
    import inspect
    unexpected_solve.__signature__ = inspect.signature(api.solve_flame)
    monkeypatch.setattr(api,'solve_flame',unexpected_solve)
    monkeypatch.setattr(api,'solve_burner_flame',unexpected_solve)
    folder = api.generate_nonadiabatic_fgm(phis=[.8,1.2],mass_flux_fractions=[.4,.2],
        mechanism='h2o2.yaml', fuel='H2', progress_species='H2O:1', reuse_from=banks,
        output=tmp_path/'combined',raw_only=True)
    report = json.loads((folder/'generation.json').read_text())
    assert report['reused_profiles'] == 6
    assert report['all_final_accepted']
    with pytest.raises(RuntimeError, match='unexpected native solve'):
        api.generate_nonadiabatic_fgm(phis=[.8,1.2],mass_flux_fractions=[.4,.2],
            mechanism='h2o2.yaml', fuel='H2', progress_species='H2O:1', pressure=202650., reuse_from=banks,
            output=tmp_path/'changed_pressure',raw_only=True)


def test_hierarchy_matches_exhaustive_barycentric_oracle_and_cross_leaf_overlaps():
    rng = np.random.default_rng(812)
    tet = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    points = np.vstack([tet+3*i for i in range(80)])
    cells = np.arange(len(points)).reshape(-1,4)
    mesh = SimplexMesh(points, cells)
    bary = rng.dirichlet(np.ones(4), 150)
    chosen = rng.integers(0,80,150)
    controls = np.einsum('ni,nij->nj', bary, points[cells[chosen]])
    controls = np.vstack([controls, [[1.5,1.5,1.5],[-1.,0.,0.]]])
    nodes, weights, covered = mesh.locate_batch(controls)
    assert covered.sum() == 150
    np.testing.assert_array_equal(nodes[:150], cells[chosen])
    np.testing.assert_allclose(np.einsum('ni,nij->nj', weights[:150], points[nodes[:150]]), controls[:150], atol=1e-12)
    duplicate = SimplexMesh(points, np.vstack([cells, cells[0]]))
    with pytest.raises(AmbiguousManifoldError):
        duplicate.locate_batch([[.2,.2,.2]])


def test_original_adiabatic_adaptation_remains_available():
    from kflame.fgm.adaptive import build_adaptive_fgm
    from kflame import generate_adaptive_nonadiabatic_fgm
    assert callable(build_adaptive_fgm) and callable(generate_adaptive_nonadiabatic_fgm)


@pytest.mark.parametrize('mechanism', ['h2o2.yaml', 'gri30.yaml'])
def test_accelerated_signed_concentrations_preserve_numpy_rates(mechanism):
    mech = load_mechanism(mechanism)
    thermo = NativeThermo(mech)
    fast, original = NativeKinetics(mech), NativeKinetics(mech)
    original._numba_available = False  # original dense Python continuation oracle
    rng = np.random.default_rng(829)
    concentrations = rng.lognormal(-9.,3.,(mech.n_species,5))
    concentrations[:3,1] *= -1.
    concentrations[0,2] *= -1.
    concentrations[:,3] *= -1e-10
    T = np.array([400.,700.,1100.,1600.,2200.])
    expected = original.net_production_rates(T, concentrations, thermo.g_RT(T))
    actual = fast.net_production_rates(T, concentrations, thermo.g_RT(T))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fast.net_production_rates(T[2],concentrations[:,2],thermo.g_RT(T[2])),
                                  original.net_production_rates(T[2],concentrations[:,2],thermo.g_RT(T[2])))
