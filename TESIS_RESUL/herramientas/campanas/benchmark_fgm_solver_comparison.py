"""Eight adaptive base-state FGM: KFLAME and Cantera, four transports, no repeats."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'TESIS_RESUL/herramientas/campanas'))
import benchmark_fgm_adaptive_campaign as adaptive
from benchmark_fgm_campaign import digest


def design():
    cases = []
    for index in range(len(adaptive.TRANSPORTS)):
        base = adaptive.design(True)[index]
        order = ('kflame', 'cantera') if index % 2 == 0 else ('cantera', 'kflame')
        for solver in order:
            cases.append(dict(base, id=f'{base["id"]}_{solver}', solver=solver))
    return cases


def make_manifest(smoke=False):
    m = adaptive.make_manifest(smoke)
    m.update(kind='fgm-solver-comparison', conditions=design())
    m['environment']['versions']['cantera'] = importlib.metadata.version('cantera')
    m['settings']['loglevel'] = 0
    for path in (Path(__file__).resolve(), ROOT/'TESIS_RESUL/herramientas/campanas/fgm_cantera_adapter.py'):
        m['code'][path.relative_to(ROOT).as_posix()] = digest(path)
    m['protocol'].update(
        continuation='KFLAME native secant; Cantera set_initial_guess(previous SolutionArray); within-family lower-phi states only',
        comparison='same base physics and table algorithm; independent adaptive row sets; compare total workflow and final row count',
        cantera='Cantera FreeFlame, auto=True, refine_grid=True; steady rtol=1e-4 atol=1e-9; transient rtol=1e-4 atol=1e-11',
        order='alternate solver order across the four transports; one fresh construction each',
        scope='eight base-state tables; no pressure/temperature sweep, no repetition uncertainty, no isolated solver-kernel speedup')
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path,
                   default=Path('TESIS_RESUL/corridas/reproduccion/thesis_fgm_comparison'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Separate technical test: phi=.9,1,1.1 initially, Nc=61, indicator=10%%')
    p.add_argument('--case', action='append')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--worker', help=argparse.SUPPRESS)
    p.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    out = args.output.resolve()
    if args.worker:
        manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        c = next(c for c in manifest['conditions'] if c['id'] == args.worker)
        engine = None
        if c['solver'] == 'cantera':
            from fgm_cantera_adapter import CanteraEngine
            engine = CanteraEngine
        return adaptive.worker(out, args.worker, args.attempt, engine=engine)
    m = make_manifest(args.smoke)
    cases = m['conditions']
    if args.case:
        unknown = set(args.case)-{c['id'] for c in cases}
        if unknown: p.error(f'Unknown cases: {sorted(unknown)}')
        cases = [c for c in cases if c['id'] in args.case]
    print('4 transports x 2 flame solvers = 8 FGM; 300 K, 1 atm, CH4/GRI30; no repetitions.', flush=True)
    print(f'Initial phi={m["adaptation"]["initial_phi"]}; max insertions=10; target={m["adaptation"]["target_defect"]}; Nc={m["settings"]["n_c"]}')
    for c in cases: print(c['id'])
    if args.dry_run:
        if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != m:
            raise ValueError('Configuration/code/environment changed: use a new directory.')
        print('Dry run: no solves or writes.'); return 0
    adaptive.ensure_manifest(out, m, args.resume)
    lock = out/'campaign.lock'
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Campaign locked: check active processes before removing {lock}')
    with os.fdopen(fd, 'w') as f: f.write(str(os.getpid()))
    try:
        ok = True
        for i,c in enumerate(cases,1):
            print(f'[{i}/{len(cases)}] {c["id"]}', flush=True)
            ok = adaptive.run_case(out, c['id'], args.retry_failed, entrypoint=Path(__file__).resolve()) and ok
        return 0 if ok else 1
    finally: lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
