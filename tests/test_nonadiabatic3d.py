"""Physical-control and connected-mesh checks for the three-control FGM."""
import json

import numpy as np
import pytest

from kflame import generate_nonadiabatic_fgm, solve_burner_flame
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


@pytest.mark.parametrize('kwargs', [dict(phis=[1.]), dict(phis=[1., .8]),
                                 dict(mass_flux_fractions=[0.]), dict(mass_flux_fractions=[1.]),
                                 dict(mass_flux_fractions=[.2, .2])])
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
    with pytest.raises(ValueError, match='accepted native burner'):
        solve_burner_flame(phi=1., mass_flux=.1, initial_solution=seed, output=output)
    assert not output.exists()
    mech = load_mechanism('gri30.yaml')
    meta.update(accepted=True, mechanism='gri30.yaml', temperature=300., pressure=101325.,
                inlet_Y=fresh_mixture(mech, .8, 'CH4', 'O2:1,N2:3.76').tolist())
    (seed / 'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='same feed'):
        solve_burner_flame(phi=1., mass_flux=.1, initial_solution=seed, output=output)
    assert not output.exists()


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
