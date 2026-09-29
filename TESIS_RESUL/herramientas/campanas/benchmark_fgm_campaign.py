"""Reproducible FGM families and held-out flames; large runs are user launched.

Uses the existing native generator internals without changing the solver API.
Every family is a fresh, warmed process. Interrupted families restart as a new
attempt so their timing is never assembled from different sessions.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
THREADS = dict(NUMBA_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    import numpy as np
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def atomic(path, data, npz=False):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('wb') as f:
        if npz: np.savez_compressed(f, **data)
        else: f.write(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
        f.flush()
        os.fsync(f.fileno())
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 9: raise
            time.sleep(.1 * (i + 1))


def design(smoke=False):
    import numpy as np
    schedule = json.loads((ROOT / 'examples/fgm_adaptive_map_schedule.json').read_text(encoding='utf-8'))
    phi = np.array(schedule['phi_resolved'])
    indices = np.linspace(0, len(phi) - 2, 12).round().astype(int)
    holdouts = np.sqrt(phi[indices] * phi[indices + 1])
    if smoke:
        phi = np.array([.9, 1.0])
        holdouts = np.array([np.sqrt(.9)])
        schedule = dict(phi_resolved=phi.tolist(), requested=[True, True], bridge=[False, False])
    assert not np.any(np.isclose(holdouts[:, None], phi[None, :], rtol=0, atol=1e-12))
    return schedule, holdouts.tolist()


def make_manifest(smoke=False):
    from kflame.fgm.generate import build_argparser
    schedule, holdouts = design(smoke)
    settings = vars(build_argparser().parse_args([
        '--mech', 'gri30.yaml', '--fuel', 'CH4', '--oxidizer', 'O2:1,N2:3.76',
        '--parallel-workers', '1', '--numba-kinetics-threads', '4', '--disable-seed-cache',
        '--save-raw-profiles', '--profile-solver', '--max-flame-time-s', '300',
    ]))
    settings['numba_kinetics_threads_resolved'] = 4
    files = sorted((ROOT / 'src/kflame').rglob('*.py')) + sorted((ROOT / 'src/kflame/chemistry/data').rglob('*.yaml'))
    files += [Path(__file__).resolve(), ROOT / 'examples/fgm_adaptive_map_schedule.json']
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files}
    return dict(schema=1, smoke=smoke, fuel='CH4', mechanism='gri30.yaml', repeats=1 if smoke else 5,
                schedule=schedule, holdouts=holdouts, settings=settings, seed=20260927,
                environment=dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
                                 versions={p: importlib.metadata.version(p) for p in ('numpy','scipy','numba')},
                                 threads=THREADS), code=hashes,
                protocol=dict(family='fixed certified schedule, cold first flame then native secant continuation',
                              holdout='independent cold starts; excluded from construction and indicator',
                              timing='warmed process; setup, flame solve with property projection, table build; excludes saving',
                              spatial='FGM reference mesh: slope=.04, curve=.08, ratio=2.5, prune=.003; distinct from L3',
                              coordinate='Z_in: historical mass-basis reference streams; Zstar affine display coordinate',
                              query='linear interpolation of T, Y_CO2, Y_CO, omega_c; no full-flame speedup'))


def ensure_manifest(out, proposed, resume):
    path = out / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old != proposed:
            raise ValueError('Configuration/code/environment changed: use another output directory.')
        if not resume: raise ValueError('Campaign exists: use --resume.')
    else:
        if out.exists() and any(out.iterdir()): raise ValueError('Output must be empty for a new campaign.')
        atomic(path, proposed)
        mech = ROOT / 'src/kflame/chemistry/data/gri30.yaml'
        (out / 'inputs').mkdir(exist_ok=True)
        shutil.copy2(mech, out / 'inputs/gri30.yaml')
        atomic(out / 'inputs/schedule.json', proposed['schedule'])


def latest(job):
    attempts = sorted(job.glob('attempt-*'))
    return attempts[-1] if attempts else None


def completed(job):
    last = latest(job)
    if last and (last / 'state.json').exists():
        state = json.loads((last / 'state.json').read_text(encoding='utf-8'))
        if state.get('status') == 'interrupted': return None
    if last and (last / 'result.json').exists():
        r = json.loads((last / 'result.json').read_text(encoding='utf-8'))
        if r['status'] in ('accepted', 'failed'): return r
    return None


def reconstruct(path):
    import numpy as np
    from kflame.fgm.generate import FlameRecord
    with np.load(path, allow_pickle=False) as data:
        kwargs = {f.name: data[f.name] if data[f.name].ndim else data[f.name].item()
                  for f in dataclasses.fields(FlameRecord)}
    return FlameRecord(**kwargs)


def tabulate(records, names, settings, n_c=None):
    import numpy as np
    from kflame.fgm.common import build_global_indicator, build_adaptive_c_grid, validate_fgm_table
    from kflame.fgm.generate import build_tables
    cf = np.linspace(0, 1, settings['c_fine'])
    indicator = build_global_indicator(records, names, settings['indicator_species'].split(','), cf,
                                       settings['indicator_weight_grad'], settings['indicator_weight_conc'],
                                       settings['indicator_weight_temp'], settings['indicator_weight_qdot'])
    grid = build_adaptive_c_grid(cf, indicator, n_c or settings['n_c'], settings['refine_bias'])
    table = build_tables(records, names, grid)
    # Existing builder uses object dtype for predictor labels; persist plain strings.
    table['predictor_kind'] = np.asarray(table['predictor_kind'], dtype=str)
    contract = validate_fgm_table(table)
    table.update(species_names=np.array(names), indicator_fine_c=cf, indicator_fine_value=indicator)
    return table, contract


def worker(out, phase, attempt):
    import numpy as np
    from kflame.fgm import generate as g
    m = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    args = argparse.Namespace(**m['settings'])
    args.mech = str(out / 'inputs/gri30.yaml')
    g.configure_numba_kinetics_threads(4)
    mech = g.load_mechanism(args.mech)
    weights = g.parse_progress_weights(args.progress_species)
    opts = g.make_solve_options(args)
    # Exercise cold and continued routes outside scientific timing.
    warm = []
    prev = None
    for phi in (.9, 1.0):
        rec, state, trace = g.solve_flame_native(phi, args, mech, opts, weights, prev_solution=prev)
        warm.append(dict(phi=phi, accepted=bool(rec.solve_ok and rec.final_accepted), trace=trace))
        if not warm[-1]['accepted']: raise RuntimeError('Warmup flame failed acceptance.')
        prev = state
    atomic(attempt / 'warmup.json', warm)
    start = time.perf_counter()
    mech = g.load_mechanism(args.mech)
    opts = g.make_solve_options(args)
    setup_s = time.perf_counter() - start
    phis = m['schedule']['phi_resolved'] if phase == 'family' else m['holdouts']
    records, traces = [], []
    prev = prev_prev = None
    for i, phi in enumerate(phis):
        start = time.perf_counter()
        rec, state, trace = g.solve_flame_native(phi, args, mech, opts, weights,
                                                prev_solution=prev, prev_prev_solution=prev_prev)
        elapsed = time.perf_counter() - start
        if phase == 'family':
            rec.requested = m['schedule']['requested'][i]
            rec.bridge = m['schedule']['bridge'][i]
        span = float(rec.beta[-1] - rec.beta[0])
        raw_c = (rec.beta - rec.beta[0]) / span
        trace.update(phi=phi, index=i, measured_s=elapsed,
                     c_min=float(raw_c.min()), c_max=float(raw_c.max()),
                     c_min_step=float(np.diff(raw_c).min()), accepted=bool(rec.solve_ok and rec.final_accepted))
        traces.append(trace)
        atomic(attempt / 'profiles' / f'{i:03d}.npz',
               dict(dataclasses.asdict(rec), species_names=np.array(mech.species_names)), npz=True)
        atomic(attempt / 'trace.json', traces)
        print(f'{phase} {i+1}/{len(phis)} phi={phi:.8g} accepted={trace["accepted"]} t={elapsed:.3f}s', flush=True)
        if not trace['accepted']: raise RuntimeError(f'Unaccepted flame phi={phi}.')
        if trace['c_min_step'] < -1e-7 or trace['c_min'] < -1e-7 or trace['c_max'] > 1+1e-7:
            raise RuntimeError(f'Progress coordinate is not monotone/bounded at phi={phi}; inspect saved profile.')
        records.append(rec)
        if phase == 'family': prev_prev, prev = prev, state
    build_s, contract = None, None
    if phase == 'family':
        start = time.perf_counter()
        table, contract = tabulate(records, list(mech.species_names), m['settings'])
        build_s = time.perf_counter() - start
        atomic(attempt / 'fgm_table.npz', table, npz=True)
    files = list((attempt / 'profiles').glob('*.npz'))
    files += [attempt / 'trace.json', attempt / 'warmup.json']
    if phase == 'family': files.append(attempt / 'fgm_table.npz')
    atomic(attempt / 'result.json', dict(status='accepted', phase=phase, n_flames=len(records),
           setup_s=setup_s, family_s=sum(t['measured_s'] for t in traces), tabulation_s=build_s,
           compute_s=setup_s+sum(t['measured_s'] for t in traces)+(build_s or 0),
           validation=contract, artifacts={str(p.relative_to(attempt)): digest(p) for p in files}))


def verify_artifacts(attempt, result):
    for relative, expected in result.get('artifacts', {}).items():
        p = attempt / relative
        if not p.exists() or digest(p) != expected: raise ValueError(f'Changed or missing artifact: {p}')


def run_job(out, phase, rep, retry_failed=False):
    job = out / phase / f'rep-{rep:02d}'
    previous = completed(job)
    if previous and (previous['status'] == 'accepted' or not retry_failed):
        verify_artifacts(latest(job), previous)
        print(f'{phase}/{rep}: retained {previous["status"]}', flush=True)
        return previous['status'] == 'accepted'
    attempt = job / f'attempt-{len(list(job.glob("attempt-*")))+1:03d}'
    attempt.mkdir(parents=True)
    atomic(attempt / 'state.json', dict(status='running', started=time.time()))
    env = dict(os.environ, **THREADS, PYTHONIOENCODING='utf-8')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', phase,
               '--output', str(out), '--attempt', str(attempt)]
    process = None
    try:
        with (attempt / 'process.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith(('family ', 'holdout ')): print(line.strip(), flush=True)
            code = process.wait()
        if not (attempt / 'result.json').exists():
            atomic(attempt / 'result.json', dict(status='failed', exit_code=code, reason='See process.log'))
        result = json.loads((attempt / 'result.json').read_text(encoding='utf-8'))
        atomic(attempt / 'state.json', dict(status=result['status'], finished=time.time()))
        return result['status'] == 'accepted'
    except KeyboardInterrupt:
        if process and process.poll() is None: process.terminate(); process.wait()
        atomic(attempt / 'state.json', dict(status='interrupted', finished=time.time()))
        raise


def selected_jobs(phase, repeats, family_repetition=None):
    """Select execution jobs without changing the five-repeat scientific design."""
    if family_repetition is not None:
        if phase=='holdout': raise ValueError('--family-repetition requires phase family or all.')
        if not 1<=family_repetition<=repeats: raise ValueError('Family repetition outside campaign design.')
    repetitions=[family_repetition] if family_repetition is not None else range(1,repeats+1)
    jobs=[('family',r) for r in repetitions] if phase in ('family','all') else []
    if phase in ('holdout','all'): jobs.append(('holdout',1))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['family','holdout','all'], default='all')
    parser.add_argument('--output', type=Path,
                        default=Path('TESIS_RESUL/corridas/reproduccion/thesis_fgm'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Separate 2-flame + 1-holdout technical check')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--family-repetition', type=int,
                        help='Run only this family repetition; keep the complete campaign design for later resume')
    parser.add_argument('--worker', choices=['family','holdout'], help=argparse.SUPPRESS)
    parser.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if args.worker:
        start = time.perf_counter()
        try: worker(out, args.worker, args.attempt)
        except KeyboardInterrupt:
            atomic(args.attempt / 'state.json', dict(status='interrupted', finished=time.time()))
            return 130
        except BaseException as exc:
            atomic(args.attempt / 'result.json', dict(status='failed', reason=str(exc), process_until_failure_s=time.perf_counter()-start))
            traceback.print_exc()
            return 1
        return 0
    proposed = make_manifest(args.smoke)
    jobs=selected_jobs(args.phase,proposed['repeats'],args.family_repetition)
    print(f'{len(proposed["schedule"]["phi_resolved"])} construction flames x {proposed["repeats"]} repetitions; '
          f'{len(proposed["holdouts"])} independent holdouts; two warmups per process.', flush=True)
    count=sum(len(proposed['schedule']['phi_resolved']) if phase=='family' else len(proposed['holdouts'])
              for phase,_ in jobs)
    print(f'Selected jobs: {jobs}; {count} scientific solves + {2*len(jobs)} warmups before resume skips.',flush=True)
    if args.dry_run:
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text(encoding='utf-8')) != proposed:
                raise ValueError('Existing campaign incompatible.')
        print('Dry run: no flames and no output changes.')
        return 0
    ensure_manifest(out, proposed, args.resume)
    if digest(out/'inputs/gri30.yaml') != proposed['code']['src/kflame/chemistry/data/gri30.yaml']:
        raise ValueError('Mechanism copy changed.')
    lock = out / 'campaign.lock'
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Campaign is locked; check active processes before removing {lock}')
    os.close(fd)
    try:
        ok = True
        for phase,rep in jobs: ok = run_job(out,phase,rep,args.retry_failed) and ok
        return 0 if ok else 1
    finally: lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
