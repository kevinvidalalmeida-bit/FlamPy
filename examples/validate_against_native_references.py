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

LIMITS = dict(temperature_K=20., species_absolute=5e-3, heat_loss_relative=.02, stand_off_distance_m=2e-5)
_model = None


def _validate(task):
    global _model
    table, profiles, references, case, phi, mdot = task
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
    metrics = dict(
        temperature_K=float(np.max(abs(profile['T']-np.interp(z, ref['z'], ref['T'])))),
        species_absolute=float(np.max(abs(profile['Y']-np.array([np.interp(z, ref['z'], y) for y in ref['Y']])))),
        heat_loss_relative=abs(q_reduced-q_ref)/abs(q_ref),
        stand_off_distance_m=abs(z[int(np.argmax(np.gradient(profile['T'], z)))]
                                 - ref['z'][int(np.argmax(np.gradient(ref['T'], ref['z'])))]))
    checks = {k: bool(v <= LIMITS[k]) for k, v in metrics.items()}
    return dict(case=case, phi=phi, mass_flux=mdot, seed_row=seed['output'], accepted=bool(report['accepted']),
                reason=report['reason'], passed=bool(report['accepted'] and all(checks.values())),
                metrics=metrics, checks=checks, q_reduced_W_m2=q_reduced, q_reference_W_m2=q_ref,
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
    args = parser.parse_args()
    summary = json.loads((Path(args.references)/'references.json').read_text(encoding='utf-8'))
    cases = sorted({(r['case'], r['phi'], r['mass_flux']) for r in summary['records']})
    tasks = [(args.table, args.profiles, args.references, *case) for case in cases]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_validate, tasks))
    output = dict(table=args.table, limits=LIMITS, cases=len(results),
                  accepted=sum(r['accepted'] for r in results), passed=sum(r['passed'] for r in results),
                  wall_seconds=time.perf_counter()-started, results=results)
    Path(args.output).write_text(json.dumps(json_safe(output), indent=1)+'\n', encoding='utf-8')
    for r in results:
        m = r['metrics']
        print(f"{r['case']}: {'PASS' if r['passed'] else 'FAIL'} {r['reason']:>26} q_err={100*m['heat_loss_relative']:5.2f}% "
              f"dT={m['temperature_K']:5.1f}K ref_unc={100*r['reference_grid_uncertainty']:.2f}%", flush=True)
    print(f"accepted {output['accepted']}/{len(results)}, passed {output['passed']}/{len(results)}, "
          f"{output['wall_seconds']:.1f} s", flush=True)


if __name__ == '__main__':
    main()
