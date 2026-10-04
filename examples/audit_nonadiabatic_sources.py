"""Separate detailed chemistry, flame discretization and FGM source errors.

python examples/audit_nonadiabatic_sources.py --output runs/source_audit
Uses published withheld profiles. No simulation or detailed chemistry is
performed inside the production FGM lookup.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM, OutsideManifoldError


def detailed_sources(mech, T, Y, pressure):
    thermo = NativeThermo(mech)
    concentration = thermo.density(T, pressure, Y)[None, :] * Y * mech.inv_molecular_weights[:, None]
    omega_molar = NativeKinetics(mech).net_production_rates(T, concentration, thermo.g_RT(T))
    omega_mass = omega_molar * mech.molecular_weights[:, None]
    qdot = -np.sum(thermo.partial_molar_enthalpies(T) * omega_molar, axis=0)
    return omega_mass, qdot


def cantera_sources(mechanism, T, Y, pressure):
    import cantera as ct
    gas = ct.Solution(resolve_mechanism(mechanism))
    states = ct.SolutionArray(gas, len(T))
    states.TPY = T, pressure, Y.T
    return (states.net_production_rates * gas.molecular_weights[None, :]).T, states.heat_release_rate


def source_metrics(z, truth, prediction, covered=None):
    z, truth, prediction = (np.asarray(a,dtype=float) for a in (z,truth,prediction))
    if covered is None:
        covered = np.ones(len(z), dtype=bool)
    covered = np.asarray(covered,dtype=bool)
    if (z.ndim != 1 or len(z)<2 or truth.shape!=z.shape or prediction.shape!=z.shape
            or covered.shape!=z.shape or not covered.any()
            or not np.isfinite(z).all() or np.any(np.diff(z)<=0.)
            or not np.isfinite(truth).all() or not np.isfinite(prediction[covered]).all()):
        raise ValueError('Source metrics require increasing finite coordinates and finite covered sources')
    error = prediction[covered] - truth[covered]
    peak = max(float(np.max(abs(truth))),1e-300)
    integral_truth = float(np.trapezoid(truth, z))
    # Integrate only covered intervals, never bridge an unsupported gap.
    intervals = covered[:-1] & covered[1:]
    dz = np.diff(z)[intervals]
    local_truth = truth[covered]
    full_error = np.where(covered, prediction-truth, 0.)
    square_integral = float(np.sum(.5*(full_error[:-1][intervals]**2+full_error[1:][intervals]**2)*dz))
    square_truth = float(np.sum(.5*(truth[:-1][intervals]**2+truth[1:][intervals]**2)*dz))
    integral_error = float(np.sum(.5*(full_error[:-1][intervals]+full_error[1:][intervals])*dz))
    integral_abs_error = float(np.sum(.5*(abs(full_error[:-1][intervals])+abs(full_error[1:][intervals]))*dz))
    integral_abs = float(np.trapezoid(abs(truth), z))
    captured_abs = float(np.sum(.5*(abs(truth[:-1][intervals])+abs(truth[1:][intervals]))*dz))
    covered_indices = np.flatnonzero(covered)
    j = int(covered_indices[np.argmax(abs(error))])
    return dict(Linf_over_truth_peak=float(np.max(abs(error))/peak),
                L2_relative=float(np.sqrt(square_integral/max(square_truth, 1e-300))),
                L1_relative=integral_abs_error/max(float(np.trapezoid(abs(truth),z)),1e-300),
                integral_error_over_abs_integral=abs(integral_error)/max(integral_abs, 1e-300),
                integral_truth=integral_truth, absolute_source_coverage=captured_abs/max(integral_abs, 1e-300),
                worst_index=j, worst_z_mm=float(1000.*z[j]), truth_at_worst=float(truth[j]),
                prediction_at_worst=float(prediction[j]), truth_peak=peak,
                peak_amplitude_error=abs(float(np.max(prediction[covered]))-float(np.max(local_truth)))/peak)


def audit(bundle, output, table_folder=None):
    import cantera as ct
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = NonAdiabaticFGM(table_folder or bundle)
    mech = load_mechanism(model.metadata['mechanism'])
    weights = model.table['progress_weights']
    pressure = model.metadata['pressure_Pa']
    reports = [json.loads((bundle / n).read_text(encoding='utf-8')) for n in ('validation.json', 'validation_additional.json')]
    cases = []
    for group, report in zip(('case', 'additional'), reports):
        for i, case in enumerate(report['cases']):
            with np.load(bundle / f'{group}_{i:02d}_comparison.npz', allow_pickle=False) as saved:
                data = {k: saved[k] for k in saved.files}
            native_mass, native_q = detailed_sources(mech, data['T'], data['Y'], pressure)
            ct_mass, ct_q = cantera_sources(model.metadata['mechanism'], data['T'], data['Y'], pressure)
            native_C, ct_C = weights @ native_mass, weights @ ct_mass
            species_peak = np.max(abs(ct_mass), axis=1)
            active = species_peak > 1e-6 * np.max(species_peak)
            same_state = dict(
                omega_C_Linf_over_peak=float(np.max(abs(native_C-ct_C))/np.max(abs(ct_C))),
                qdot_Linf_over_peak=float(np.max(abs(native_q-ct_q))/np.max(abs(ct_q))),
                omega_mass_Linf_over_global_peak=float(np.max(abs(native_mass-ct_mass))/np.max(abs(ct_mass))),
                active_species_worst_relative_peak_error=float(np.max(np.max(abs(native_mass-ct_mass),axis=1)[active]/species_peak[active])),
                active_species_count=int(active.sum()))
            prediction = {k: np.full(len(data['z']), np.nan) for k in ('T','omega_C','qdot')}
            Y = np.full(data['Y'].shape, np.nan)
            mask = np.zeros(len(data['z']), dtype=bool)
            for j in range(len(mask)):
                try:
                    r = model.lookup(Z=data['Z'][j], C=data['C'][j], h=data['h'][j])
                except OutsideManifoldError:
                    continue
                mask[j] = True
                Y[:,j] = r['Y']
                for k in prediction:
                    prediction[k][j] = r[k]
            reconstructed_mass, reconstructed_q = detailed_sources(mech, prediction['T'][mask], Y[:,mask], pressure)
            reconstructed_C = np.full(len(mask), np.nan)
            reconstructed_q_full = np.full(len(mask), np.nan)
            reconstructed_C[mask], reconstructed_q_full[mask] = weights @ reconstructed_mass, reconstructed_q
            ct_interp_C = np.interp(data['z'], data['reference_z'], data['reference_omega_C'])
            ct_interp_q = np.interp(data['z'], data['reference_z'], data['reference_qdot'])
            metrics = dict(
                omega_C=source_metrics(data['z'], native_C, prediction['omega_C'], mask),
                qdot=source_metrics(data['z'], native_q, prediction['qdot'], mask),
                recomputed_from_fgm_state_C=source_metrics(data['z'], native_C, reconstructed_C, mask),
                recomputed_from_fgm_state_q=source_metrics(data['z'], native_q, reconstructed_q_full, mask),
                detailed_flame_vs_cantera_C=source_metrics(data['z'], ct_interp_C, native_C),
                detailed_flame_vs_cantera_q=source_metrics(data['z'], ct_interp_q, native_q))
            progress_feed = float(weights @ np.asarray(model.metadata['rows'][0]['inlet_Y']))
            # CH4-air fresh progress is zero for these weights; verify rather than
            # silently applying this boundary identity to a different fuel stream.
            if progress_feed != 0.:
                raise ValueError('This diagnostic expects zero fresh progress')
            expected_integral = case['mass_flux_kg_m2_s'] * (float(data['C'][-1])-progress_feed)
            mass_peak = float(np.max(abs(native_mass)))
            conservation = dict(
                chemical_mass_source_sum_over_peak=float(np.max(abs(native_mass.sum(axis=0)))/mass_peak),
                chemical_Z_source_over_mass_peak=float(np.max(abs(model.table['bilger_weights']@native_mass))/mass_peak),
                integrated_C_source=float(np.trapezoid(native_C,data['z'])),
                boundary_C_flux_difference=expected_integral,
                integrated_progress_balance_relative_error=abs(float(np.trapezoid(native_C,data['z']))-expected_integral)/abs(expected_integral))
            result = dict(case=f'{group}_{i:02d}', phi=case['phi'], fraction=case['fraction'],
                          same_state_chemistry=same_state, sources=metrics, conservation=conservation,
                          node_coverage=float(mask.mean()))
            result['detailed_passed']=(same_state['omega_C_Linf_over_peak']<1e-9
                                      and same_state['qdot_Linf_over_peak']<1e-9
                                      and metrics['detailed_flame_vs_cantera_C']['Linf_over_truth_peak']<.02
                                      and metrics['detailed_flame_vs_cantera_q']['Linf_over_truth_peak']<.02
                                      and conservation['integrated_progress_balance_relative_error']<.005
                                      and conservation['chemical_mass_source_sum_over_peak']<1e-10
                                      and conservation['chemical_Z_source_over_mass_peak']<1e-10)
            result['manifold_passed']=all(metrics[k]['Linf_over_truth_peak']<.05
                                          and metrics[k]['L1_relative']<.05
                                          and metrics[k]['integral_error_over_abs_integral']<.03
                                          and metrics[k]['absolute_source_coverage']>.99 for k in ('omega_C','qdot'))
            cases.append(result)
            print(json.dumps(dict(case=result['case'], chemistry=same_state,
                                 tabulation_C=metrics['omega_C']['Linf_over_truth_peak'],
                                 reconstructed_C=metrics['recomputed_from_fgm_state_C']['Linf_over_truth_peak'],
                                 flame_vs_cantera_C=metrics['detailed_flame_vs_cantera_C']['Linf_over_truth_peak'],
                                 progress_balance=conservation['integrated_progress_balance_relative_error'])),flush=True)
            np.savez_compressed(output/f'{group}_{i:02d}_sources.npz', z=data['z'], C=data['C'],
                                T_native=data['T'], omega_C_native=native_C, omega_C_cantera_same_state=ct_C,
                                omega_C_cantera_flame=ct_interp_C, omega_C_fgm=prediction['omega_C'],
                                omega_C_reconstructed=reconstructed_C, covered=mask,
                                qdot_native=native_q, qdot_cantera_flame=ct_interp_q, qdot_fgm=prediction['qdot'])
    result = dict(reference_version=ct.__version__, training_flames=len(model.metadata['rows']),
                  progress_points=model.metadata['progress_points'], cases=cases,
                  detailed_passed=all(c['detailed_passed'] for c in cases),
                  manifold_passed=all(c['manifold_passed'] for c in cases),
                  limits=dict(source_Linf_over_peak_max=.05,source_L1_max=.05,
                              integrated_source_error_max=.03,absolute_source_coverage_min=.99,
                              detailed_vs_cantera_source_Linf_max=.02,same_state_source_error_max=1e-9,
                              progress_balance_max=.005),
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  table_sha256=hashlib.sha256((Path(table_folder or bundle)/'nonadiabatic_fgm.npz').read_bytes()).hexdigest(),
                  mechanism_sha256=hashlib.sha256(Path(resolve_mechanism(model.metadata['mechanism'])).read_bytes()).hexdigest(),
                  profile_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle.glob('*_comparison.npz'))},
                  metric_definition='Maximum absolute source error divided by the full detailed source peak; '
                                    'not a pointwise relative error or a solver residual.',
                  integration_definition='Integrate error only on intervals with two covered endpoints; '
                                         'normalize by the full native absolute source integral and report covered source weight separately.',
                  scope='Detailed same-state chemistry and source/flux diagnostics; a priori manifold accuracy.')
    (output/'source_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path('docs/assets/nonadiabatic3d'))
    parser.add_argument('--table',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.bundle,args.output,args.table)


if __name__=='__main__':
    main()
