"""Offline refinement of connected nonadiabatic flamelet families.

Candidate coordinates, physical settings, tolerances and budgets are supplied
by the caller. Validation flames are never read by the refinement controller.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.fgm.accuracy import FGMTolerances, assess_profile
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM, build_nonadiabatic_table


class AdaptiveAccuracyError(RuntimeError):
    """The requested accuracy could not be certified within the budget/grid."""


def midpoint_indices(coordinates, selected):
    """One available candidate nearest the physical midpoint of each gap."""
    return [min(range(a+1, b), key=lambda i: abs(coordinates[i]-.5*(coordinates[a]+coordinates[b])))
            for a, b in zip(sorted(selected)[:-1], sorted(selected)[1:]) if b-a > 1]


def _subset_family(source, destination, phis, fractions):
    source, destination = Path(source), Path(destination)
    generation = json.loads((source/'generation.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=False)
    selected = []
    for row in generation['rows']:
        if row['phi'] not in phis or (row['kind'] != 'adiabatic_reference' and row['fraction'] not in fractions):
            continue
        record = dict(row, composition_index=phis.index(row['phi']),
                      loss_index=0 if row['kind'] == 'adiabatic_reference' else fractions.index(row['fraction'])+1)
        selected.append(record)
        target = destination/row['output']
        target.mkdir()
        for name in ('flame.npz', 'metadata.json'):
            shutil.copy2(source/row['output']/name, target/name)
    generation.update(phis=phis, mass_flux_fractions=fractions, rows=selected)
    (destination/'generation.json').write_text(json.dumps(generation, indent=2)+'\n', encoding='utf-8', newline='\n')


def generate_adaptive_nonadiabatic_fgm(*, phis, mass_flux_fractions, output,
        tolerances=None, max_flames=500, max_iterations=30, initial_phi_count=3,
        initial_loss_count=5, progress_points=181,
        indicator_safety_factor=.8,
        progress_species='CO2:1,H2O:1,CO:1,H2:0.5', max_energy_error=.02,
        reuse_from=None, **flame_settings):
    """Refine phi/loss coordinates using withheld native burner profiles.

    max_flames bounds all distinct evaluated flames, including references and
    probes, not only final table rows. Fractions remain rho_feed*Su multiples.
    The finite candidate grid bounds the available refinement resolution.
    The returned directory contains the table and adaptive_report.json.
    A failure writes a report and raises; tolerances are never relaxed.
    """
    from kflame.api import generate_nonadiabatic_fgm

    if isinstance(progress_species, dict):
        progress_species = ','.join(f'{name}:{value}' for name, value in progress_species.items())
    phis = np.asarray(phis, dtype=float)
    fractions = np.sort(np.asarray(mass_flux_fractions, dtype=float))[::-1]
    if (phis.ndim != 1 or len(phis) < 3 or not np.isfinite(phis).all()
            or np.any(phis <= 0.) or np.any(np.diff(phis) <= 0.)
            or fractions.ndim != 1 or len(fractions) < 3 or not np.isfinite(fractions).all()
            or np.any(fractions <= 0.) or np.any(fractions >= 1.) or np.any(np.diff(fractions) >= 0.)):
        raise ValueError('Supply at least three distinct finite candidates per axis; phis must increase')
    for name, value in [('max_flames', max_flames), ('max_iterations', max_iterations),
                        ('initial_phi_count', initial_phi_count), ('initial_loss_count', initial_loss_count)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 2:
            raise ValueError(f'{name} must be an integer >= 2')
    tolerances = (FGMTolerances() if tolerances is None else
                  FGMTolerances.from_dict(tolerances) if isinstance(tolerances, dict) else tolerances)
    if not isinstance(tolerances, FGMTolerances):
        raise ValueError('tolerances must be FGMTolerances or a settings dictionary')
    if not np.isfinite(indicator_safety_factor) or not 0. < indicator_safety_factor <= 1.:
        raise ValueError('indicator_safety_factor must lie in (0, 1]')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    def seed(coordinates, count, logarithmic=False):
        targets = (np.geomspace(coordinates[0], coordinates[-1], min(count, len(coordinates)))
                   if logarithmic else np.linspace(coordinates[0], coordinates[-1], min(count, len(coordinates))))
        return set(int(np.argmin(abs(coordinates-target))) for target in targets)
    selected_phi = seed(phis, initial_phi_count)
    selected_loss = seed(fractions, initial_loss_count, True)
    evaluated, history, cached_truth = set(), [], {}
    report = dict(passed=False, tolerances=tolerances.to_dict(), max_flames=max_flames,
                  indicator_safety_factor=float(indicator_safety_factor),
                  candidate_phis=phis.tolist(), candidate_fractions=fractions.tolist(), history=history,
                  reused_bank=bool(reuse_from), scope='Offline burner-profile interpolation indicators; independent validation required.')
    def checkpoint(reason=None):
        report.update(evaluated_flames=len(evaluated), selected_flames=len(selected_phi)*(len(selected_loss)+1))
        if reason:
            report['reason'] = reason
        (root/'adaptive_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    mech = load_mechanism(flame_settings.get('mechanism', 'gri30.yaml'))
    thermo, kinetics = NativeThermo(mech), NativeKinetics(mech)
    available = Path(reuse_from) if reuse_from else None
    checkpoint('initialized')
    for iteration in range(max_iterations):
        mid_phi, mid_loss = midpoint_indices(phis, selected_phi), midpoint_indices(fractions, selected_loss)
        probe_phi, probe_loss = sorted(selected_phi | set(mid_phi)), sorted(selected_loss | set(mid_loss))
        requested = {(i, j) for i in probe_phi for j in [-1]+probe_loss}
        if len(evaluated | requested) > max_flames:
            checkpoint('flame_budget_exhausted')
            raise AdaptiveAccuracyError('Probe budget exhausted; inspect adaptive_report.json')
        if not mid_phi and not mid_loss:
            checkpoint('candidate_resolution_exhausted_without_independent_probes')
            raise AdaptiveAccuracyError('No independent candidate probes remain; supply a finer candidate grid')
        start = time.perf_counter()
        try:
            pool = generate_nonadiabatic_fgm(phis=phis[probe_phi].tolist(),
                mass_flux_fractions=fractions[probe_loss].tolist(), progress_species=progress_species,
                progress_points=progress_points, max_energy_error=max_energy_error,
                output=root/f'probes_{iteration:02d}', raw_only=True, reuse_from=available, **flame_settings)
        except Exception as error:
            report['error'] = str(error)
            checkpoint('native_generation_failure')
            raise
        available = Path(reuse_from) if reuse_from else pool
        evaluated.update(requested)
        folder = root/f'table_{iteration:02d}'
        _subset_family(pool, folder, phis[sorted(selected_phi)].tolist(), fractions[sorted(selected_loss)].tolist())
        build_nonadiabatic_table(folder, progress_points=progress_points)
        model = NonAdiabaticFGM(folder)
        generation = json.loads((pool/'generation.json').read_text(encoding='utf-8'))
        records = []
        for row in generation['rows']:
            i = int(np.flatnonzero(phis == row['phi'])[0])
            if row['kind'] == 'adiabatic_reference':
                continue  # indicators concern the nonadiabatic burner interior
            j = int(np.flatnonzero(fractions == row['fraction'])[0])
            if i in selected_phi and j in selected_loss:
                continue
            key = (i, j)
            if key not in cached_truth:
                with np.load(pool/row['output']/'flame.npz', allow_pickle=False) as saved:
                    truth = {k: saved[k] for k in ('z', 'T', 'Y')}
                rho = thermo.density(truth['T'], generation['pressure_Pa'], truth['Y'])
                omega = kinetics.net_production_rates(truth['T'],
                    rho[None,:]*truth['Y']*mech.inv_molecular_weights[:,None], thermo.g_RT(truth['T']))
                truth.update(Z=model.table['bilger_weights']@truth['Y']+float(model.table['bilger_offset']),
                             C=model.table['progress_weights']@truth['Y'], h=thermo.enthalpy_mass(truth['T'], truth['Y']),
                             omega_C=model.table['progress_weights']@(omega*mech.molecular_weights[:,None]),
                             qdot=-np.sum(thermo.partial_molar_enthalpies(truth['T'])*omega, axis=0))
                cached_truth[key] = truth
            truth = cached_truth[key]
            prediction = model.lookup_batch(Z=truth['Z'], C=truth['C'], h=truth['h'], outside='mask')
            assessment = assess_profile(truth, prediction, tolerances)
            ratios = assessment.get('ratios', {})
            indicator_score = (max(value if 'coverage' in key else value/indicator_safety_factor
                                   for key, value in ratios.items()) if ratios else None)
            records.append(dict(assessment, phi_index=i, loss_index=j, phi=float(phis[i]), fraction=float(fractions[j]),
                                passed=indicator_score is not None and indicator_score<=1., indicator_score=indicator_score))
        history.append(dict(iteration=iteration, selected_phis=phis[sorted(selected_phi)].tolist(),
                            selected_fractions=fractions[sorted(selected_loss)].tolist(),
                            table_flames=len(model.metadata['rows']), evaluated_flames=len(evaluated),
                            probes=len(records), failed_probes=sum(not r['passed'] for r in records),
                            elapsed_s=time.perf_counter()-start, checks=records))
        failed = [r for r in records if not r['passed']]
        if records and not failed:
            report.update(passed=True, final_table=folder.name, reason='all_candidate_midpoint_indicators_passed')
            checkpoint()
            shutil.copy2(root/'adaptive_report.json', folder/'adaptive_report.json')
            return folder
        # Refine the worst failed probe on every unsampled axis it uses. This
        # includes interaction probes, so coupled errors cannot be ignored.
        worst = max(failed, key=lambda r: float('inf') if r['indicator_score'] is None else r['indicator_score'])
        selected_phi.add(worst['phi_index'])
        selected_loss.add(worst['loss_index'])
        checkpoint('refining')
    checkpoint('iteration_budget_exhausted')
    raise AdaptiveAccuracyError('Iteration budget exhausted; inspect adaptive_report.json')
