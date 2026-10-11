"""Validate the reduced burner on a table against second-order native references.

References come from examples.regenerate_native_references (finest level).
The heat loss compared is the conservative outlet loss mdot*(h_feed-h_out):
it equals the reduced model's wall flux exactly and the second-order native
wall flux to the reference's closure. Seeds are the nearest training rows;
references never enter the reduced solve.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import solve_reduced_burner_fgm, ReducedConvergenceError
from kflame.serialization import json_safe
from examples.audit_nonadiabatic_sources import detailed_sources

LIMITS = dict(temperature_K=20., species_absolute=5e-3, heat_loss_relative=.02, stand_off_distance_m=2e-5)
# Progress source and heat release, as in the earlier reduced-burner validation.
SOURCE_LIMITS = dict(peak_relative=.05, L1_relative=.05, integral_relative=.03)
_model = None


def _progress(model, Y):
    """Unscaled common progress C of the table for mass fractions Y (species, nodes)."""
    return np.asarray(model.table['progress_weights'])@Y


def _branches(model, profile, ref):
    """Reduced rising-branch nodes, reference branch indices and its C values."""
    Cr, Cref = _progress(model, profile['Y']), _progress(model, ref['Y'])
    top = int(np.argmax(Cref))
    branch = np.flatnonzero(np.diff(np.maximum.accumulate(Cref[:top+1]), prepend=-np.inf) > 0.)
    xp = Cref[branch]
    rising = np.arange(int(np.argmax(Cr))+1)
    rising = rising[(Cr[rising] >= xp[0]) & (Cr[rising] <= xp[-1])]
    return Cr, rising, branch, xp


def state_errors(model, profile, ref):
    """Largest T and species errors at equal progress C, plus the outlet state.

    The front position is measured separately (stand-off distance); comparing
    at equal C keeps a shift of the steep front from counting again as a state
    error. The reduced profile is compared on its rising branch with the
    reference state at the same C (reference rising branch, strictly increasing
    part), and the outlet states are compared directly.
    """
    Cr, rising, branch, xp = _branches(model, profile, ref)
    T_at_C = np.interp(Cr[rising], xp, ref['T'][branch])
    Y_at_C = np.array([np.interp(Cr[rising], xp, y[branch]) for y in ref['Y']])
    dT = max(float(np.max(abs(profile['T'][rising]-T_at_C), initial=0.)), abs(float(profile['T'][-1]-ref['T'][-1])))
    dY = max(float(np.max(abs(profile['Y'][:, rising]-Y_at_C), initial=0.)),
             float(np.max(abs(profile['Y'][:, -1]-ref['Y'][:, -1]))))
    return dT, dY


def source_errors(model, profile, ref, reference_sources):
    """Progress source and heat release of the reduced flame against the reference.

    Peak and L1 compare both flames at equal C (as the state errors); the net
    spatial integral is invariant to a front shift and uses each flame's own z.
    """
    Cr, rising, branch, xp = _branches(model, profile, ref)
    z = profile['z']
    errors = {}
    for name, truth in reference_sources.items():
        predicted = np.asarray(profile[name])
        at_C = np.interp(Cr[rising], xp, truth[branch])
        scale = float(np.trapezoid(abs(truth), ref['z']))
        difference = np.zeros_like(predicted)
        difference[rising] = predicted[rising]-at_C
        errors[name] = dict(
            peak_relative=float(np.max(abs(difference))/np.max(abs(truth))),
            L1_relative=float(np.trapezoid(abs(difference), z)/scale),
            integral_relative=float(abs(np.trapezoid(predicted, z)-np.trapezoid(truth, ref['z']))/scale))
    return errors


def _validate(task):
    global _model
    table, profiles, references, case, phi, mdot, save = task
    if _model is None:
        _model = NonAdiabaticFGM(table)
    model = _model
    mech = load_mechanism(model.metadata['mechanism'])
    feed = fresh_mixture(mech, phi, model.metadata['fuel'], model.metadata['oxidizer'])
    with np.load(Path(references)/case/'level_1'/'flame.npz') as saved:
        ref = {k: saved[k] for k in ('z', 'T', 'Y')}
    level0 = json.loads((Path(references)/case/'level_0'/'metadata.json').read_text(encoding='utf-8'))
    level1 = json.loads((Path(references)/case/'level_1'/'metadata.json').read_text(encoding='utf-8'))
    seed = min((row for row in model.metadata['rows'] if row['loss_index'] > 0),
               key=lambda row: (abs(row['phi']-phi), abs(row['mass_flux']-mdot)))
    started = time.perf_counter()
    try:
        profile, report = solve_reduced_burner_fgm(
            model, phi=phi, mass_flux=mdot, seed_profile=Path(profiles)/seed['output']/'flame.npz',
            seed_row=seed['output'], width=float(ref['z'][-1]), curvature_limit=1.0,
            progress_equation='adaptive_projection')
    except ReducedConvergenceError as error:
        profile, report = error.profile, error.report
    h_feed = float(model.thermo.enthalpy_mass(model.metadata['temperature_K'], feed))
    q_reduced = mdot*(h_feed-float(profile['h'][-1]))
    q_ref = level1['heat_loss']['outlet_enthalpy_loss_W_m2']
    q_ref_coarse = level0['heat_loss']['outlet_enthalpy_loss_W_m2']
    z = profile['z']
    dT, dY = state_errors(model, profile, ref)
    metrics = dict(
        temperature_K=dT,
        species_absolute=dY,
        heat_loss_relative=abs(q_reduced-q_ref)/abs(q_ref),
        stand_off_distance_m=abs(z[int(np.argmax(np.gradient(profile['T'], z)))]
                                 - ref['z'][int(np.argmax(np.gradient(ref['T'], ref['z'])))]))
    checks = {k: bool(v <= LIMITS[k]) for k, v in metrics.items()}
    mass, qdot = detailed_sources(mech, ref['T'], ref['Y'], model.metadata['pressure_Pa'])
    reference_sources = dict(omega_C=np.asarray(model.table['progress_weights'])@mass, qdot=qdot)
    sources = source_errors(model, profile, ref, reference_sources)
    for name, values in sources.items():
        for norm, value in values.items():
            checks[f'{name}_{norm}'] = bool(value <= SOURCE_LIMITS[norm])
    # Reported only: the same comparisons at equal position z.
    same_z = dict(
        temperature_K=float(np.max(abs(profile['T']-np.interp(z, ref['z'], ref['T'])))),
        species_absolute=float(np.max(abs(profile['Y']-np.array([np.interp(z, ref['z'], y) for y in ref['Y']])))))
    if save is not None:
        names = [str(name) for name in model.table['species_names']]
        keep = [names.index(k) for k in ('CO', 'OH', 'H2O', 'CO2', 'NO') if k in names]
        np.savez_compressed(Path(save)/f'{case}.npz', z=z, T=profile['T'], h=profile['h'],
                            C=_progress(model, profile['Y']), Y=profile['Y'][keep], omega_C=profile['omega_C'],
                            omega_C_ref=reference_sources['omega_C'],
                            z_ref=ref['z'], T_ref=ref['T'], C_ref=_progress(model, ref['Y']), Y_ref=ref['Y'][keep],
                            species=np.array([names[i] for i in keep]))
    return dict(case=case, phi=phi, mass_flux=mdot, seed_row=seed['output'], accepted=bool(report['accepted']),
                reason=report['reason'], passed=bool(report['accepted'] and all(checks.values())),
                metrics=metrics, sources=sources, checks=checks, same_z_metrics=same_z,
                q_reduced_W_m2=q_reduced, q_reference_W_m2=q_ref,
                reference_grid_uncertainty=abs(q_ref-q_ref_coarse)/abs(q_ref),
                reference_wall_outlet_closure=level1['heat_loss']['relative_energy_closure_error'],
                reduced_seconds=time.perf_counter()-started, reduced_nodes=len(z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', required=True)
    parser.add_argument('--profiles', required=True, help='training library with the seed rows')
    parser.add_argument('--references', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--profiles-out', help='folder for compact per-case profiles (for plotting)')
    args = parser.parse_args()
    if args.profiles_out:
        Path(args.profiles_out).mkdir(parents=True, exist_ok=False)
    summary = json.loads((Path(args.references)/'references.json').read_text(encoding='utf-8'))
    cases = sorted({(r['case'], r['phi'], r['mass_flux']) for r in summary['records']})
    tasks = [(args.table, args.profiles, args.references, *case, args.profiles_out) for case in cases]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_validate, tasks))
    output = dict(table=args.table, limits=LIMITS, source_limits=SOURCE_LIMITS, state_comparison='equal progress C (rising branch) and outlet state',
                  cases=len(results),
                  accepted=sum(r['accepted'] for r in results), passed=sum(r['passed'] for r in results),
                  wall_seconds=time.perf_counter()-started, results=results)
    Path(args.output).write_text(json.dumps(json_safe(output), indent=1)+'\n', encoding='utf-8')
    for r in results:
        m = r['metrics']
        print(f"{r['case']}: {'PASS' if r['passed'] else 'FAIL'} {r['reason']:>26} q_err={100*m['heat_loss_relative']:5.2f}% "
              f"dT(C)={m['temperature_K']:5.1f}K dT(z)={r['same_z_metrics']['temperature_K']:5.1f}K "
              f"wC_peak={100*r['sources']['omega_C']['peak_relative']:4.2f}% q_L1={100*r['sources']['qdot']['L1_relative']:4.2f}% "
              f"ref_unc={100*r['reference_grid_uncertainty']:.2f}%", flush=True)
    print(f"accepted {output['accepted']}/{len(results)}, passed {output['passed']}/{len(results)}, "
          f"{output['wall_seconds']:.1f} s", flush=True)


if __name__ == '__main__':
    main()
