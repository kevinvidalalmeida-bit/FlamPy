"""Physical-control and connected-mesh checks for the three-control FGM."""
import json

import numpy as np
import pytest

from kflame import generate_nonadiabatic_fgm, solve_burner_flame, solve_flame
from kflame.chemistry.initialization import fresh_mixture, hp_equilibrium
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.fgm.nonadiabatic3d import (
    bilger_coefficients, SimplexMesh, NonAdiabaticFGM,
    OutsideManifoldError, AmbiguousManifoldError,
)


def test_local_bilger_matches_independent_stream_convention():
    ct = pytest.importorskip('cantera')
    mech = load_mechanism('gri30.yaml')
    weights, offset = bilger_coefficients(mech, 'CH4', 'O2:1,N2:3.76')
    gas = ct.Solution('gri30.yaml')
    for phi in [.7, 1., 1.3]:
        fresh = fresh_mixture(mech, phi, 'CH4', 'O2:1,N2:3.76')
        _, burned, _ = hp_equilibrium(mech, 300., ct.one_atm, fresh)
        for y in [fresh, burned, .3 * fresh + .7 * burned]:
            gas.TPY = 900., ct.one_atm, y
            assert weights @ y + offset == pytest.approx(gas.mixture_fraction('CH4', 'O2:1,N2:3.76', basis='mole'), abs=1e-12)


@pytest.mark.parametrize('mechanism', ['h2o2.yaml','gri30.yaml'])
def test_temperature_inversion_keeps_converged_nodes_and_recovers_mixed_branches(mechanism):
    model = NonAdiabaticFGM.__new__(NonAdiabaticFGM)
    mech = load_mechanism(mechanism)
    model.thermo = NativeThermo(mech)
    model.max_temperature = 3500.
    target = np.array([201.,300.,999.9,1000.1,1700.,3400.])
    Y = np.random.default_rng(841).dirichlet(np.ones(mech.n_species),len(target)).T
    h = model.thermo.enthalpy_mass(target,Y)
    guess = np.array([target[0],3500.,200.,3500.,200.,200.])
    actual = model._recover_temperature(h,Y,guess)
    assert actual[0] == target[0]
    np.testing.assert_allclose(actual,target,rtol=0.,atol=1e-7)
    np.testing.assert_allclose(model.thermo.enthalpy_mass(actual,Y),h,rtol=0.,atol=1e-6)


def test_connected_mesh_does_not_fill_gaps_or_accept_overlaps():
    tetrahedron = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    mesh = SimplexMesh(np.vstack([tetrahedron, tetrahedron + 3.]), [[0, 1, 2, 3], [4, 5, 6, 7]])
    nodes, barycentric = mesh.locate([.1, .2, .3])
    np.testing.assert_allclose(barycentric @ mesh.points[nodes], [.1, .2, .3], atol=1e-12)
    with pytest.raises(OutsideManifoldError):
        mesh.locate([2., 2., 2.])
    with pytest.raises(ValueError, match='finite'):
        mesh.locate([np.nan, 0., 0.])
    overlap = SimplexMesh(tetrahedron, [[0, 1, 2, 3], [0, 1, 2, 3]])
    with pytest.raises(AmbiguousManifoldError):
        overlap.locate([.1, .1, .1])


def test_lookup_preserves_local_Z_progress_and_total_enthalpy(tmp_path):
    mech = load_mechanism('h2o2.yaml')
    thermo = NativeThermo(mech)
    species_weights = np.array([float(name == 'H2O') for name in mech.species_names])
    z_weights, z_offset = bilger_coefficients(mech, 'H2', 'O2:1,N2:3.76')
    values = []
    for phi, conversion, T in [(.8, .2, 500.), (1.2, .2, 500.), (.8, .8, 1500.), (.8, .2, 1200.)]:
        fresh = fresh_mixture(mech, phi, 'H2', 'O2:1,N2:3.76')
        _, burned, _ = hp_equilibrium(mech, 300., 101325., fresh)
        y = fresh + conversion * (burned - fresh)
        values.append((y, T, z_weights @ y + z_offset, species_weights @ y, thermo.enthalpy_mass(T, y)))
    Y = np.array([v[0] for v in values])
    controls = np.array([v[2:] for v in values])
    np.savez_compressed(tmp_path / 'nonadiabatic_fgm.npz', Y=Y, controls=controls,
                        cells=np.array([[0, 1, 2, 3]]), species_names=np.asarray(mech.species_names),
                        reference_points=controls[:3, :2], reference_cells=np.array([[0, 1, 2]]),
                        reference_h=controls[:3, 2], omega_C=np.zeros(4), qdot=np.zeros(4),
                        conductivity=np.full(4, .05))
    (tmp_path / 'metadata.json').write_text(json.dumps(dict(mechanism='h2o2.yaml', pressure_Pa=101325.)))
    model = NonAdiabaticFGM(tmp_path)
    barycentric = np.array([.2, .3, .1, .4])
    Z, C, h = barycentric @ controls
    result = model.lookup(Z=Z, C=C, h=h)
    np.testing.assert_allclose(result['Y'], barycentric @ Y, atol=1e-12)
    assert result['Y'].sum() == pytest.approx(1., abs=1e-12)
    assert z_weights @ result['Y'] + z_offset == pytest.approx(Z, abs=1e-12)
    assert species_weights @ result['Y'] == pytest.approx(C, abs=1e-12)
    assert thermo.enthalpy_mass(result['T'], result['Y']) == pytest.approx(h, abs=1e-5)
    rng = np.random.default_rng(218)
    weights = rng.dirichlet(np.ones(4), 45)
    queries = weights @ controls
    batch = model.lookup_batch(Z=queries[:,0], C=queries[:,1], h=queries[:,2])
    np.testing.assert_allclose(batch['Y'], weights @ Y, atol=1e-12)
    np.testing.assert_allclose(thermo.enthalpy_mass(batch['T'], batch['Y'].T), queries[:,2], atol=1e-5)
    for i in range(len(queries)):
        single = model.lookup(Z=queries[i,0], C=queries[i,1], h=queries[i,2])
        assert batch['T'][i] == pytest.approx(single['T'], abs=1e-7)
    masked = model.lookup_batch(Z=[Z,Z], C=[C,C], h=[h,1e12], outside='mask')
    np.testing.assert_array_equal(masked['covered'], [True,False])
    assert np.isnan(masked['Y'][1]).all()
    with pytest.raises(OutsideManifoldError):
        model.lookup_batch(Z=[Z,Z], C=C, h=[h,1e12])


@pytest.mark.parametrize('kwargs', [dict(phis=[1.]), dict(phis=[1., .8]),
                                 dict(mass_flux_fractions=[0.]), dict(mass_flux_fractions=[1.]),
                                 dict(mass_flux_fractions=[.2, .2]), dict(jobs=0), dict(jobs=1.5),
                                 dict(jobs=True)])
def test_invalid_variable_family_creates_no_output(tmp_path, kwargs):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        generate_nonadiabatic_fgm(output=output, **kwargs)
    assert not output.exists()


def test_continuation_rejects_unaccepted_or_different_feed_before_output(tmp_path):
    seed = tmp_path / 'seed'
    seed.mkdir()
    meta = dict(accepted=False, backend='native_cpu', flow_type='isothermal_burner')
    (seed / 'metadata.json').write_text(json.dumps(meta))
    output = tmp_path / 'new_flame'
    with pytest.raises(ValueError, match='accepted native flame'):
        solve_burner_flame(phi=1., mass_flux=.1, initial_solution=seed, output=output)
    assert not output.exists()
    mech = load_mechanism('gri30.yaml')
    meta.update(accepted=True, mechanism='gri30.yaml', temperature=300., pressure=101325.,
                inlet_Y=fresh_mixture(mech, .8, 'CH4', 'O2:1,N2:3.76').tolist(),
                report={'grid_converged': True})
    (seed / 'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='same feed'):
        solve_burner_flame(phi=1., mass_flux=.1, initial_solution=seed, output=output)
    assert not output.exists()


def test_continuation_rejects_a_burner_seed_for_a_free_flame_before_output(tmp_path):
    seed = tmp_path/'seed'
    seed.mkdir()
    meta = dict(accepted=True, backend='native_cpu', flow_type='isothermal_burner',
                report={'grid_converged':True})
    (seed/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='for a burner a burner or free flame'):
        solve_flame(initial_solution=seed, output=tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_burner_accepts_a_free_flame_seed_and_still_checks_its_feed(tmp_path):
    seed = tmp_path/'seed'
    seed.mkdir()
    mech = load_mechanism('gri30.yaml')
    meta = dict(accepted=True, backend='native_cpu', flow_type='free_flame', mechanism='gri30.yaml',
                temperature=300., pressure=101325., report={'grid_converged': True},
                inlet_Y=fresh_mixture(mech, .8, 'CH4', 'O2:1,N2:3.76').tolist())
    (seed/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='same feed'):  # passed the flow-type check
        solve_burner_flame(phi=1., mass_flux=.1, initial_solution=seed, output=tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_builder_rejects_missing_training_rows(tmp_path):
    from kflame.fgm.nonadiabatic3d import build_nonadiabatic_table
    (tmp_path / 'generation.json').write_text(json.dumps(dict(
        all_final_accepted=True, mechanism='gri30.yaml', progress_species={'H2O': 1.},
        fuel='CH4', oxidizer='O2:1,N2:3.76', phis=[.8, 1.], mass_flux_fractions=[.5], rows=[])))
    with pytest.raises(ValueError, match='exactly once'):
        build_nonadiabatic_table(tmp_path)
    assert not (tmp_path / 'nonadiabatic_fgm.npz').exists()


def test_trajectory_sampling_keeps_inlet_and_burned_plateau_endpoints():
    from kflame.fgm.nonadiabatic3d import _trajectory_samples
    profile = dict(Y=np.array([[.01, .01+1e-15, .5, .5], [.99, .99-1e-15, .5, .5]]),
                   T=np.array([300., 301., 1499., 1500.]), conductivity=np.arange(4., dtype=float))
    y, T, conductivity = _trajectory_samples(profile, np.array([1., 0.]), np.array([0., 1.]))
    np.testing.assert_allclose(y, profile['Y'][:, [0, 3]])
    np.testing.assert_allclose(T, [300., 1500.])
    np.testing.assert_allclose(conductivity, [0., 3.])


def test_training_enthalpy_budget_uses_weakest_loss_and_both_drift_signs():
    from kflame.fgm.nonadiabatic3d import training_enthalpy_quality
    feed = np.array([-1e5, -2e5])
    deficit = np.array([[300., 1e4, 1e5], [-300., 2e4, 1e5]])
    result = training_enthalpy_quality(feed, feed[:, None]-deficit, .02)
    assert not result['passed']
    assert [r['passed'] for r in result['rows']] == [False, True]
    assert result['rows'][0]['relative_adiabatic_enthalpy_error'] == pytest.approx(.03)
    assert result['rows'][1]['relative_adiabatic_enthalpy_error'] == pytest.approx(.015)
    shifted = training_enthalpy_quality(feed+7e5, feed[:, None]-deficit+7e5, .02)
    assert shifted == result  # Independent of the enthalpy reference offset.
    deficit[:, 0] = 0.
    deficit[0, 1] = 0.
    assert not training_enthalpy_quality(feed, feed[:, None]-deficit, .02)['rows'][0]['passed']


def test_builder_rejects_accepted_flames_with_contaminated_adiabatic_endpoint(tmp_path):
    from scipy.optimize import brentq
    from kflame.fgm.nonadiabatic3d import build_nonadiabatic_table
    mech = load_mechanism('h2o2.yaml')
    thermo = NativeThermo(mech)
    rows = []
    for i, phi in enumerate([.8, 1.2]):
        feed = fresh_mixture(mech, phi, 'H2', 'O2:1,N2:3.76')
        _, burned, _ = hp_equilibrium(mech, 300., 101325., feed)
        hfeed = thermo.enthalpy_mass(300., feed)
        for j in range(2):
            folder = tmp_path/f'flame_{i}_{j}'
            folder.mkdir()
            # A nominally accepted free flame carries a 1000 J/kg outlet
            # error, larger than 2% of the 20000 J/kg burner loss.
            target = hfeed+(1000. if j == 0 else -20000.)
            endpoint_T = brentq(lambda T: thermo.enthalpy_mass(T, burned)-target, 300., 3500.)
            Y = np.column_stack([feed, .5*(feed+burned), burned])
            np.savez(folder/'flame.npz', Y=Y, T=[300.,1000.,endpoint_T],
                     conductivity=[.03,.06,.1], species_names=np.asarray(mech.species_names))
            meta = dict(accepted=True, backend='native_cpu', report={'grid_converged':True},
                        heat_loss={'relative_energy_closure_error':0.})
            (folder/'metadata.json').write_text(json.dumps(meta))
            rows.append(dict(composition_index=i, loss_index=j, inlet_Y=feed.tolist(), output=folder.name))
    generation = dict(all_final_accepted=True, mechanism='h2o2.yaml', fuel='H2',
                      oxidizer='O2:1,N2:3.76', phis=[.8,1.2], mass_flux_fractions=[.5],
                      progress_species={'H2O':1.}, rows=rows, temperature_K=300.,
                      pressure_Pa=101325., max_energy_error=.02)
    (tmp_path/'generation.json').write_text(json.dumps(generation))
    with pytest.raises(ValueError, match='weakest-loss budget'):
        build_nonadiabatic_table(tmp_path, progress_points=3)
    assert not (tmp_path/'nonadiabatic_fgm.npz').exists()
    quality = json.loads((tmp_path/'training_enthalpy_quality.json').read_text())
    assert not quality['passed']
    assert quality['maximum_relative_adiabatic_enthalpy_error'] == pytest.approx(.05)
