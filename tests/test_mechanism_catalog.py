"""Catalog identity, complete native admission and independent rate checks."""
from pathlib import Path

import numpy as np
import pytest

from kflame.chemistry.catalog import CATALOG, available_mechanisms
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.thermo import NativeThermo


@pytest.mark.parametrize('entry', CATALOG, ids=[entry[0] for entry in CATALOG])
def test_catalog_identity_thermo_and_native_rates(entry):
    ct = pytest.importorskip('cantera')
    title, filename, species, reactions, location, _ = entry
    try:
        path = resolve_mechanism(title)
    except FileNotFoundError:
        if location == 'local':
            pytest.skip('External mechanism not installed in this environment')
        raise
    assert Path(path).name == filename
    mechanism = load_mechanism(path)
    assert (mechanism.n_species, mechanism.n_reactions) == (species, reactions)
    gas = ct.Solution(path, transport_model=None)
    assert gas.species_names == mechanism.species_names
    assert gas.n_reactions == reactions
    # Same atomic weights; the two implementations sum atoms in different
    # orders for a few species. This oracle permits at most a few ulps and
    # does not change the byte-exact gate against the native baseline.
    np.testing.assert_allclose(mechanism.molecular_weights, gas.molecular_weights,
                               rtol=5e-16, atol=0.)
    thermo, kinetics = NativeThermo(mechanism), NativeKinetics(mechanism)
    kinetics._use_sparse_numba = True
    for temperature, pressure_atm in [(500., .1), (1100., 1.), (2300., 10.)]:
        fractions = np.full(species, 1e-8)
        for name, value in {'H2': .15, 'CH4': .05, 'CO': .05, 'O2': .2,
                            'N2': .53, 'H': .005, 'O': .003, 'OH': .004,
                            'H2O': .008}.items():
            if name in gas.species_names:
                fractions[gas.species_index(name)] = value
        gas.TPX = temperature, pressure_atm*ct.one_atm, fractions
        np.testing.assert_allclose(thermo.cp_R(temperature), gas.standard_cp_R, rtol=2e-12)
        # NASA g/RT is at reference pressure. Cantera's standard-state
        # property includes log(P/Pref); kinetics adds that concentration
        # factor separately. Compare identical thermodynamic conventions.
        reference_gibbs = gas.standard_gibbs_RT - np.log(gas.P/gas.reference_pressure)
        np.testing.assert_allclose(thermo.g_RT(temperature), reference_gibbs,
                                   rtol=2e-12, atol=2e-12)
        actual = kinetics.net_production_rates(temperature, gas.concentrations,
                                              thermo.g_RT(temperature))
        expected = gas.net_production_rates
        np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=1e-10)
        scale = max(float(np.max(np.abs(actual))), 1.)
        np.testing.assert_allclose(mechanism.atom_matrix@actual/scale, 0., atol=5e-12)


def test_external_directory_and_missing_mechanism_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('KFLAME_MECHANISM_DIR', str(tmp_path))
    with pytest.raises(FileNotFoundError, match='not installed'):
        resolve_mechanism('AramcoMech 3.0')
    assert not next(row for row in available_mechanisms()
                    if row['name'] == 'AramcoMech 3.0')['installed']
    explicit = tmp_path/'input.yaml'
    explicit.write_text('input', encoding='utf-8')
    assert resolve_mechanism(explicit) == str(explicit.resolve())
    assert Path(resolve_mechanism('DRM22')).name == 'drm22.yaml'
    assert Path(resolve_mechanism('DRM19')).name == 'drm19.yaml'
