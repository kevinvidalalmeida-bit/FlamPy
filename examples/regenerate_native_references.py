"""Independent second-order native references for saved reduced-burner cases.

For each case folder (case_*/reduced_report.json gives phi and mass flux),
solve the detailed burner with exponential convection and the second-order
inlet balance at two refinements. Each phi first solves its free flame,
which seeds the weakest-loss burner; the other burners continue in order of
decreasing mass flux. The finer level is the reference; the difference to
the coarser level estimates its own grid uncertainty. Phi groups run in
parallel processes. No table data enter these flames.
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kflame import solve_burner_flame, solve_flame

LEVELS = ((0.01, 0.02, 0.002), (0.005, 0.01, 0.001))
MAX_FLUX_RATIO = 1.5  # larger continuation jumps are bridged by unrecorded burners


def _accepted(folder):
    meta = folder/'metadata.json'
    return meta.exists() and json.loads(meta.read_text(encoding='utf-8')).get('accepted', False)


def _ensure(solver, folder, **options):
    """Solve unless an accepted result exists; a rejected one is recomputed."""
    if not _accepted(folder):
        shutil.rmtree(folder, ignore_errors=True)
        solver(output=folder, **options)
BASE = dict(initial_points=24, ratio=3., max_points=6000, max_time=900., convection='exponential',
            time_step_ladder=2**0.5)


def _solve_phi(task):
    phi, cases, output, physical = task
    output = Path(output)
    settings = dict(BASE, **physical)
    started = time.perf_counter()
    records = []
    previous = {}
    for level, (slope, curve, prune) in enumerate(LEVELS):
        options = dict(settings, slope=slope, curve=curve, prune=prune)
        free = output/f'free_phi_{phi:.6f}_level_{level}'
        _ensure(solve_flame, free, phi=phi, initial_solution=previous.get('free'), **options)
        previous['free'] = free
        seed, seed_flux = free, None
        for name, mdot in sorted(cases, key=lambda item: -item[1]):
            target = output/name/f'level_{level}'
            target.parent.mkdir(parents=True, exist_ok=True)
            initial, initial_flux = (previous[name], mdot) if name in previous else (seed, seed_flux)
            if initial_flux is not None and initial_flux/mdot > MAX_FLUX_RATIO:
                count = int(np.ceil(np.log(initial_flux/mdot)/np.log(MAX_FLUX_RATIO)))
                for b, bridge_flux in enumerate(np.geomspace(initial_flux, mdot, count+1)[1:-1]):
                    bridge = output/name/f'bridge_{level}_{b}'
                    _ensure(solve_burner_flame, bridge, phi=phi, mass_flux=float(bridge_flux),
                            initial_solution=initial, **options)
                    initial = bridge
            _ensure(solve_burner_flame, target, phi=phi, mass_flux=mdot, initial_solution=initial, **options)
            meta = json.loads((target/'metadata.json').read_text(encoding='utf-8'))
            records.append(dict(case=name, phi=phi, mass_flux=mdot, level=level, slope=slope, nodes=meta['nodes'],
                                q_wall=meta['heat_loss']['burner_heat_loss_W_m2'],
                                q_outlet=meta['heat_loss']['outlet_enthalpy_loss_W_m2'],
                                closure=meta['heat_loss']['relative_energy_closure_error'],
                                runtime_s=meta['runtime_s']))
            previous[name] = target
            seed, seed_flux = target, mdot
    return records, time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, help='folder with case_*/reduced_report.json')
    parser.add_argument('--like', required=True, help='table metadata.json with the physical model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    meta = json.loads(Path(args.like).read_text(encoding='utf-8'))
    physical = dict(mechanism=meta['mechanism'], fuel=meta['fuel'], oxidizer=meta['oxidizer'],
                    temperature=meta['temperature_K'], pressure=meta['pressure_Pa'],
                    transport=meta['transport'], soret=meta['soret'], width=0.03)
    groups = {}
    for report in sorted(Path(args.cases).glob('case_*/reduced_report.json')):
        values = json.loads(report.read_text(encoding='utf-8'))
        groups.setdefault(float(values['phi']), []).append((report.parent.name, float(values['mass_flux_kg_m2_s'])))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(phi, cases, str(output), physical) for phi, cases in sorted(groups.items())]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_solve_phi, tasks))
    records = [record for group, _ in results for record in group]
    summary = dict(levels=LEVELS, settings=BASE, cases=len({r['case'] for r in records}),
                   wall_seconds=time.perf_counter()-started, records=records)
    (output/'references.json').write_text(json.dumps(summary, indent=1)+'\n', encoding='utf-8')
    print(f"{summary['cases']} cases, {len(records)} flames in {summary['wall_seconds']:.1f} s", flush=True)


if __name__ == '__main__':
    main()
