"""Trace-species accuracy must not be hidden by global major-species scales."""
import numpy as np
import pytest

from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.pollutants import pollutant_profiles, compare_pollutant_profiles


def test_wet_ppm_and_nox_molar_source_match_independent_conversion():
    ct = pytest.importorskip('cantera')
    mech = load_mechanism(resolve_mechanism('gri30.yaml'))
    gas = ct.Solution('gri30.yaml')
    gas.TPX = 1700., 101325., 'N2:.7,O2:.1,H2O:.1,CO2:.09,CO:.009,NO:.0008,NO2:.0002'
    sources = gas.net_production_rates * gas.molecular_weights
    result = pollutant_profiles(dict(species_names=mech.species_names,
        Y=gas.Y[:, None], omega_Y=sources[:, None]), mech.molecular_weights)
    for name in ('CO', 'NO', 'NO2'):
        i = gas.species_index(name)
        np.testing.assert_allclose(result[name]['ppm'], 1e6 * gas.X[i], rtol=2e-14)
        np.testing.assert_allclose(result[name]['source'], sources[i], rtol=2e-14)
    np.testing.assert_allclose(result['NOx']['ppm'], 1000., rtol=2e-14)
    indices = [gas.species_index('NO'), gas.species_index('NO2')]
    np.testing.assert_allclose(result['NOx']['source'], 1000. * gas.net_production_rates[indices].sum(), rtol=2e-14)
    assert result['NOx']['source_unit'] == 'mol/(m3 s)'


def test_missing_nitrogen_chemistry_is_rejected():
    mech = load_mechanism(resolve_mechanism('h2o2.yaml'))
    profile = dict(species_names=mech.species_names, Y=np.ones((mech.n_species, 2)) / mech.n_species,
                   omega_Y=np.zeros((mech.n_species, 2)))
    with pytest.raises(ValueError, match='absent'):
        pollutant_profiles(profile, mech.molecular_weights, species=('NO',))


def test_trace_source_error_fails_even_when_global_source_metric_would_pass():
    reference = {'NO': dict(ppm=np.array([0., 100., 10.]), source=np.array([0., 1e-4, 0.]),
                            source_unit='kg/(m3 s)')}
    predicted = {'NO': dict(ppm=np.array([0., 100., 10.]), source=np.array([0., 2e-4, 0.]),
                            source_unit='kg/(m3 s)')}
    # A major-species source of 10 kg/(m3 s) would hide this 100% NO error.
    assert abs(predicted['NO']['source'] - reference['NO']['source']).max() / 10. < .1
    report = compare_pollutant_profiles(predicted, reference)
    assert not report['passed']
    assert report['pollutants']['NO']['checks'] == dict(ppm=True, source=False)


def test_absolute_floor_handles_zero_reference_without_dividing_by_zero():
    reference = {'NO': dict(ppm=np.zeros(2), source=np.zeros(2), source_unit='kg/(m3 s)')}
    predicted = {'NO': dict(ppm=np.array([0., .5]), source=np.zeros(2), source_unit='kg/(m3 s)')}
    report = compare_pollutant_profiles(predicted, reference)
    assert report['passed']
    assert report['pollutants']['NO']['ppm']['error_relative_to_reference_peak'] is None
    assert not compare_pollutant_profiles(predicted, reference, concentration_atol_ppm=.1)['passed']


def test_nonfinite_or_mismatched_profiles_are_rejected():
    row = dict(ppm=np.array([np.nan]), source=np.zeros(1), source_unit='kg/(m3 s)')
    with pytest.raises(ValueError, match='finite'):
        compare_pollutant_profiles({'NO': row}, {'NO': row})
    with pytest.raises(ValueError, match='match'):
        compare_pollutant_profiles({'NO': row}, {'NO2': row})


def _saved_native_and_comparison(tmp_path, full_arrays):
    """A 4-node native flame and an FGM comparison covering its middle nodes.

    The (Z,C) validator stores covered states only; the (Z,C,h) validator
    stores every native node, with NaN where the table gives no state.
    """
    import hashlib
    import json
    from pathlib import Path
    mech = load_mechanism(resolve_mechanism('gri30.yaml'))
    native, comparison = tmp_path / 'native', tmp_path / 'comparison'
    native.mkdir(); comparison.mkdir()
    Y = np.zeros((mech.n_species, 4))
    Y[mech.species_names.index('NO')] = 1e-4
    Y[mech.species_names.index('N2')] = 1. - 1e-4
    profile = dict(z=np.linspace(0., .02, 4), Z=np.linspace(1., 0., 4), Y=Y,
                   omega_Y=np.zeros_like(Y), species_names=np.array(mech.species_names))
    metadata = dict(accepted=True, backend='native_cpu', flow_type='counterflow_diffusion',
        mechanism=str(resolve_mechanism('gri30.yaml')), source_sha256='test')
    metadata['mechanism_sha256'] = hashlib.sha256(Path(metadata['mechanism']).read_bytes()).hexdigest()
    (native / 'metadata.json').write_text(json.dumps(metadata))
    np.savez(native / 'flame.npz', **profile)
    (comparison / 'validation.json').write_text(json.dumps(dict(passed=True, native_metadata=metadata)))
    covered = np.array([False, True, True, False])
    if full_arrays:
        lookup_Y = np.where(covered[None, :], Y, np.nan)
        lookup_omega = np.where(covered[None, :], profile['omega_Y'], np.nan)
    else:
        lookup_Y, lookup_omega = Y[:, covered], profile['omega_Y'][:, covered]
    np.savez(comparison / 'comparison.npz', covered=covered, native_Z=profile['Z'],
             Y=lookup_Y, omega_Y=lookup_omega)
    return native, comparison


def test_fgm_pollutant_audit_accepts_full_arrays_of_the_heat_loss_validator(tmp_path):
    # Review P1: NaN outside the (Z,C,h) table used to raise ValueError.
    from kflame.pollutants import validate_diffusion_pollutants
    native, comparison = _saved_native_and_comparison(tmp_path, full_arrays=True)
    report = validate_diffusion_pollutants(native, comparison, tmp_path / 'audit', kind='fgm', species=('NO',))
    assert report['pollutants']['NO']['passed']
    assert report['coverage']['NO']['ppm']['fraction'] == .5


def test_fgm_pollutant_tails_require_coverage_even_with_exact_covered_values(tmp_path):
    from kflame.pollutants import validate_diffusion_pollutants
    mech = load_mechanism(resolve_mechanism('gri30.yaml'))
    assert mech.n_species > 0
    native, comparison = _saved_native_and_comparison(tmp_path, full_arrays=False)
    report = validate_diffusion_pollutants(native, comparison, tmp_path / 'audit', kind='fgm', species=('NO',))
    assert report['pollutants']['NO']['passed']
    assert report['coverage']['NO']['ppm']['fraction'] == .5
    assert not report['coverage_passed'] and not report['passed']
