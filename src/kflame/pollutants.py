"""Species-specific concentration/source audits, without detailed chemistry calls.

Concentrations are wet mole-basis ppm, without oxygen correction. These are
local flame profiles, not exhaust emission factors. Negative numerical values
are retained. NOx means NO + NO2 on a molar basis when both are selected.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from kflame.chemistry.mechanism import load_mechanism


def pollutant_profiles(profile, molecular_weights, species=('CO', 'NO', 'NO2')):
    """Extract selected species from species-by-point Y and omega_Y arrays.

    Molecular weights are kg/kmol; omega_Y is kg/(m3 s). NOx sources are
    mol/(m3 s), rather than a sum of unlike molecular mass sources.
    """
    names = list(profile['species_names'])
    species = tuple(species)
    if not species or len(set(species)) != len(species):
        raise ValueError('Select a nonempty sequence of distinct species')
    missing = set(species) - set(names)
    if missing:
        raise ValueError(f'Pollutant species absent from mechanism: {sorted(missing)}')
    Y = np.asarray(profile['Y'], dtype=float)
    omega = np.asarray(profile['omega_Y'], dtype=float)
    W = np.asarray(molecular_weights, dtype=float)
    if (Y.ndim != 2 or Y.shape[0] != len(names) or omega.shape != Y.shape
            or W.shape != (len(names),) or not np.isfinite(W).all() or np.any(W <= 0)
            or not np.isfinite(Y).all() or not np.isfinite(omega).all()):
        raise ValueError('Require finite aligned species profiles and positive molecular weights')
    molar = Y / W[:, None]
    total = molar.sum(axis=0)
    if np.any(total <= 0):
        raise ValueError('Mixture molar concentration must be positive')
    ppm = 1e6 * molar / total
    result = {}
    for name in species:
        i = names.index(name)
        result[name] = dict(ppm=ppm[i], source=omega[i], source_unit='kg/(m3 s)')
    if {'NO', 'NO2'}.issubset(species):
        indices = [names.index('NO'), names.index('NO2')]
        result['NOx'] = dict(ppm=ppm[indices].sum(axis=0),
            source=1000. * (omega[indices] / W[indices, None]).sum(axis=0),
            source_unit='mol/(m3 s)')
    return result


def compare_pollutant_profiles(predicted, reference, *, concentration_atol_ppm=1.,
        concentration_rtol=.05, source_atol=1e-8, source_rtol=.1):
    """Compare aligned profiles using each pollutant's own peak scale.

    The profile infinity error must be <= atol + rtol * max(abs(reference)).
    Absolute floors keep near-zero profiles meaningful. Source atol uses each
    row's stated source unit. These are screening limits, not certification.
    """
    limits = dict(concentration_atol_ppm=concentration_atol_ppm,
                  concentration_rtol=concentration_rtol,
                  source_atol=source_atol, source_rtol=source_rtol)
    if any(not np.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Audit limits must be finite and nonnegative')
    if set(predicted) != set(reference) or not reference:
        raise ValueError('Predicted and reference pollutant selections must match')
    rows = {}
    for name, expected in reference.items():
        actual = predicted[name]
        if actual['source_unit'] != expected['source_unit']:
            raise ValueError('Source units differ')
        row = dict(source_unit=expected['source_unit'])
        checks = {}
        for field, atol, rtol in (('ppm', concentration_atol_ppm, concentration_rtol),
                                 ('source', source_atol, source_rtol)):
            a, b = np.asarray(actual[field]), np.asarray(expected[field])
            if (a.ndim != 1 or a.shape != b.shape or not a.size
                    or not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError('Require nonempty finite aligned pollutant profiles')
            scale = float(abs(b).max())
            error = float(abs(a - b).max())
            row[field] = dict(predicted_peak=float(a.max()), reference_peak=float(b.max()),
                reference_absolute_scale=scale, maximum_absolute_error=error,
                error_relative_to_reference_peak=error / scale if scale else None,
                limit=atol + rtol * scale)
            checks[field] = error <= row[field]['limit']
        row.update(checks=checks, passed=all(checks.values()))
        rows[name] = row
    return dict(passed=all(row['passed'] for row in rows.values()), pollutants=rows,
        limits=limits, concentration_basis='wet mole ppm; no oxygen correction',
        criterion='Maximum profile error <= absolute floor + relative limit * own reference absolute peak')


def _load(folder, filename):
    with np.load(Path(folder) / filename, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def validate_diffusion_pollutants(native, comparison, output, *, kind='cantera',
                                  species=('CO', 'NO', 'NO2'), **limits):
    """Audit a saved independent Cantera comparison or withheld FGM lookup.

    FGM errors concern covered states only, with a separate per-pollutant
    coverage gate (99% of points above 1% of each full reference absolute
    concentration/source peak). Sources are read from the table, not recomputed.
    """
    native, comparison, output = Path(native), Path(comparison), Path(output)
    if kind not in ('cantera', 'fgm'):
        raise ValueError('kind must be cantera or fgm')
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((native / 'metadata.json').read_text(encoding='utf-8'))
    validation = json.loads((comparison / 'validation.json').read_text(encoding='utf-8'))
    if (not metadata.get('accepted') or metadata.get('flow_type') != 'counterflow_diffusion'
            or metadata.get('backend') != 'native_cpu'
            or validation.get('native_metadata') != metadata):
        raise ValueError('Require an accepted native flame and its matching saved comparison')
    profile = _load(native, 'flame.npz')
    if hashlib.sha256(Path(metadata['mechanism']).read_bytes()).hexdigest() != metadata['mechanism_sha256']:
        raise ValueError('Saved mechanism fingerprint differs')
    mech = load_mechanism(metadata['mechanism'])
    if list(profile['species_names']) != list(mech.species_names):
        raise ValueError('Mechanism species order differs')
    ours = pollutant_profiles(profile, mech.molecular_weights, species)
    if kind == 'cantera':
        other = _load(comparison, 'reference.npz')
        if list(other['species_names']) != list(profile['species_names']):
            raise ValueError('Reference species order differs')
        if (np.any(np.diff(other['z']) <= 0) or profile['z'][0] < other['z'][0]
                or profile['z'][-1] > other['z'][-1]):
            raise ValueError('Reference grid must cover the native grid without extrapolation')
        reference = pollutant_profiles(other, mech.molecular_weights, species)
        # Convert to mole fractions before interpolating: interpolated mass
        # fractions followed by nonlinear normalization is a different metric.
        reference = {name: dict(source_unit=row['source_unit'],
            **{key: np.interp(profile['z'], other['z'], row[key]) for key in ('ppm', 'source')})
            for name, row in reference.items()}
        predicted = ours
        covered = np.ones(len(profile['z']), dtype=bool)
    else:
        other = _load(comparison, 'comparison.npz')
        if (not np.array_equal(other['native_Z'], profile['Z'])
                or other['covered'].shape != profile['z'].shape):
            raise ValueError('FGM comparison coordinates differ')
        covered = other['covered'].astype(bool)
        if not covered.any():
            raise ValueError('No covered FGM points')
        predicted = pollutant_profiles(dict(other, species_names=profile['species_names']),
                                        mech.molecular_weights, species)
        reference = {name: dict(source_unit=row['source_unit'],
            **{key: row[key][covered] for key in ('ppm', 'source')}) for name, row in ours.items()}
    report = compare_pollutant_profiles(predicted, reference, **limits)
    coverage = {}
    for name, row in ours.items():
        # For Cantera all native points are covered. FGM must also resolve
        # pollutant tails outside the heat-release-based reaction zone.
        coverage[name] = {}
        for key in ('ppm', 'source'):
            scale = float(abs(row[key]).max())
            active = abs(row[key]) > .01 * scale
            coverage[name][key] = dict(active_points=int(active.sum()),
                covered_active_points=int(covered[active].sum()),
                fraction=float(covered[active].mean()) if active.any() else 1.)
    coverage_passed = all(item['fraction'] >= .99 for row in coverage.values() for item in row.values())
    report.update(kind=kind, native_metadata=metadata, comparison_passed=validation['passed'],
        coverage=coverage, coverage_passed=coverage_passed, covered_points=int(covered.sum()),
        total_points=len(covered), scope='Local flame concentrations and net sources; '
            'errors on aligned covered points only. Not exhaust emission factors.',
        source_sha256=metadata['source_sha256'])
    report['passed'] = report['passed'] and coverage_passed and validation['passed']
    output.mkdir(parents=True)
    arrays = dict(z=profile['z'][covered], covered=covered)
    columns = [arrays['z']]
    headings = ['z_m']
    for name in predicted:
        for label, rows in (('predicted', predicted), ('reference', reference)):
            for key in ('ppm', 'source'):
                column = f'{name}_{label}_{key}'
                arrays[column] = rows[name][key]
                headings.append(column)
                columns.append(arrays[column])
    np.savez_compressed(output / 'pollutants.npz', **arrays)
    with (output / 'profiles.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        writer.writerows(np.column_stack(columns))
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
