"""Compare native diffusion results and warm timings between two source trees.

Run from the repository root, using a preserved pre-change source tree:
python examples/benchmark_memory_layout.py --baseline-src PATH/src --output runs/memory_layout
The two versions run in separate, sequential processes with identical settings.
Compilation is warmed separately. Timings are local samples, not a speed guarantee.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


CASES = {
    'H2': dict(mechanism='h2o2.yaml', fuel='H2', fuel_mass_flux=.05, oxidizer_mass_flux=.15),
    'CH4': dict(mechanism='gri30.yaml', fuel='CH4', fuel_mass_flux=.12, oxidizer_mass_flux=.24),
    'H2-MS': dict(mechanism='h2o2.yaml', fuel='H2', fuel_mass_flux=.05, oxidizer_mass_flux=.15,
                  transport='multicomponent', soret=True),
    'CH4-MS': dict(mechanism='gri30.yaml', fuel='CH4', fuel_mass_flux=.12, oxidizer_mass_flux=.24,
                   transport='multicomponent', soret=True),
    'H2-R': dict(mechanism='h2o2.yaml', fuel='H2', fuel_mass_flux=.05, oxidizer_mass_flux=.15,
                 radiation=True),
    'CH4-R': dict(mechanism='gri30.yaml', fuel='CH4', fuel_mass_flux=.12, oxidizer_mass_flux=.24,
                  radiation=True),
}


def worker(args):
    import time
    # Select the source tree before importing numerical modules or kflame.
    sys.path.insert(0, str(args.worker_src.resolve()))
    import kflame
    import numba
    import numpy as np
    import scipy
    from kflame import solve_diffusion_flame

    source = Path(kflame.__file__).resolve()
    if source.parent.parent != args.worker_src.resolve():
        raise RuntimeError(f'Unexpected source tree: {source}')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {}
    for name in args.cases:
        settings = CASES[name]
        # Compile/warm the exact mechanism, transport and radiation configuration.
        solve_diffusion_flame(**settings, refine=args.warm_refined, output=args.output / name / 'warmup')
        samples = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            flame = solve_diffusion_flame(**settings, output=args.output / name / f'sample_{repeat}')
            samples.append(dict(solve_s=flame['report']['runtime_s'], api_s=flame['runtime_s'],
                                wall_s=time.perf_counter()-started, nodes=len(flame['z']),
                                accepted=bool(flame['accepted']),
                                residual=flame['report']['maximum_scaled_residual']))
            print(name, repeat+1, json.dumps(samples[-1]), flush=True)
        meta = json.loads((flame['output'] / 'metadata.json').read_text(encoding='utf-8'))
        cases[name] = dict(samples=samples, source_sha256=meta['source_sha256'],
                           median_s=statistics.median(row['solve_s'] for row in samples),
                           minimum_s=min(row['solve_s'] for row in samples),
                           maximum_s=max(row['solve_s'] for row in samples))
        summary = dict(source=str(source), python=sys.version, numpy=np.__version__,
                       scipy=scipy.__version__, numba=numba.__version__,
                       numba_threads=numba.get_num_threads(), cases=cases,
                       warmup_refined=args.warm_refined,
                       environment={key: os.environ.get(key) for key in
                           ('NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
        (args.output / 'timings.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')


def compare(output, names, repeats):
    import numpy as np
    versions = {label: json.loads((output / label / 'timings.json').read_text(encoding='utf-8'))
                for label in ('baseline', 'candidate')}
    results = {}
    for name in names:
        pairs = []
        for repeat in range(repeats):
            relative = Path(name) / f'sample_{repeat}' / 'flame.npz'
            with np.load(output / 'baseline' / relative, allow_pickle=False) as a, \
                    np.load(output / 'candidate' / relative, allow_pickle=False) as b:
                same_grid = np.array_equal(a['z'], b['z'])
                fields = {}
                for key in a.files:
                    if key not in b.files:
                        fields[key] = dict(equal=False, reason='missing field')
                    elif a[key].shape != b[key].shape:
                        fields[key] = dict(equal=False, reason='different shape')
                    elif np.issubdtype(a[key].dtype, np.number):
                        fields[key] = dict(equal=bool(a[key].dtype == b[key].dtype
                                                     and a[key].tobytes(order='C') == b[key].tobytes(order='C')),
                                           maximum_absolute_difference=float(np.max(np.abs(a[key]-b[key]))))
                    else:
                        fields[key] = dict(equal=bool(np.array_equal(a[key], b[key])))
                pairs.append(dict(same_grid=same_grid, fields=fields,
                                  exactly_equal=all(row['equal'] for row in fields.values())
                                                and set(a.files) == set(b.files)))
        before, after = (versions[label]['cases'][name] for label in ('baseline', 'candidate'))
        results[name] = dict(baseline_median_s=before['median_s'], candidate_median_s=after['median_s'],
                             ratio_baseline_over_candidate=before['median_s']/after['median_s'],
                             all_samples_exactly_equal=all(row['exactly_equal'] for row in pairs),
                             comparisons=pairs)
    summary = dict(scope='Warm native solve timings; baseline process precedes candidate. '
                         'Same inputs, no continuation seeds. Exact arrays checked; '
                         'differences reported without relaxing solver acceptance.',
                   repeats=repeats, versions=versions, results=results)
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({name: {key: value for key, value in row.items() if key != 'comparisons'}
                      for name, row in results.items()}, indent=2))
    if not all(row['all_samples_exactly_equal'] for row in results.values()):
        raise SystemExit('Exact field comparison failed; inspect comparison.json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-src', type=Path)
    parser.add_argument('--candidate-src', type=Path, default=Path('src'))
    parser.add_argument('--worker-src', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warm-refined', action='store_true',
                        help='Warm the complete adaptive solve before timing, including refinement-specific paths')
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('repeats and threads must be positive')
    if args.worker_src:
        worker(args)
        return
    if args.compare_only:
        compare(args.output, args.cases, args.repeats)
        return
    if args.baseline_src is None:
        parser.error('--baseline-src is required')
    if args.output.exists():
        parser.error('--output must be a new directory')
    args.output.mkdir(parents=True)
    environment = dict(os.environ, NUMBA_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for label, source in [('baseline', args.baseline_src), ('candidate', args.candidate_src)]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-src', str(source.resolve()),
                        '--output', str((args.output / label).resolve()), '--repeats', str(args.repeats),
                        '--cases', *args.cases, *(['--warm-refined'] if args.warm_refined else [])],
                       env=environment, check=True)
    compare(args.output, args.cases, args.repeats)


if __name__ == '__main__':
    main()
