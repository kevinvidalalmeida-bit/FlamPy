"""80 adaptive CH4 FGM constructions: 20 thermodynamic states, four transports.

One fresh process per FGM, no statistical repetitions. Native flame solving,
secant continuation, adaptive c-grid and leave-one-out row indicator are reused.
Large runs are launched by the user; --smoke is a separate technical campaign.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'TESIS_RESUL/herramientas/campanas'))
sys.path.insert(0, str(ROOT / 'src'))
from benchmark_fgm_campaign import (THREADS, atomic, completed, digest, latest,
                                    tabulate, verify_artifacts)

TRANSPORTS = [('P', 'mixture-averaged', False), ('PS', 'mixture-averaged', True),
              ('M', 'multicomponent', False), ('MS', 'multicomponent', True)]


def design(smoke=False):
    states = [(300, 1, 'base')]
    if not smoke:
        states += [(300, p, 'pressure') for p in range(2, 11)]
        states += [(t, 1, 'temperature') for t in range(340, 701, 40)]
    return [dict(id=f'CH4_T{t}_p{p}_{tag}', fuel='CH4', T_in_K=t, p_atm=p,
                 P_Pa=p * 101325.0, sweep=sweep, transport=tag,
                 transport_model=model, soret_enabled=soret)
            for t, p, sweep in states for tag, model, soret in TRANSPORTS]


def make_manifest(smoke=False):
    from kflame.fgm.generate import build_argparser
    settings = vars(build_argparser().parse_args([
        '--mech', 'gri30.yaml', '--fuel', 'CH4', '--oxidizer', 'O2:1,N2:3.76',
        '--parallel-workers', '1', '--numba-kinetics-threads', '4',
        '--disable-seed-cache', '--save-raw-profiles', '--profile-solver']))
    settings['numba_kinetics_threads_resolved'] = 4
    if smoke: settings['n_c'] = 61
    files = sorted((ROOT / 'src/kflame').rglob('*.py'))
    files += sorted((ROOT / 'src/kflame/chemistry/data').rglob('*.yaml'))
    files += [Path(__file__).resolve(), ROOT / 'TESIS_RESUL/herramientas/campanas/benchmark_fgm_campaign.py']
    return dict(schema=1, kind='adaptive-fgm-sweeps', technical_check=smoke,
                conditions=design(smoke), repetitions=1, settings=settings,
                adaptation=dict(initial_phi=[.9, 1., 1.1] if smoke else [.7, .9, 1., 1.1, 1.4],
                                target_defect=.10 if smoke else .01, max_bridges=10,
                                max_rounds=64, max_flames=256),
                code={p.relative_to(ROOT).as_posix(): digest(p) for p in files},
                environment=dict(python=sys.version, platform=platform.platform(),
                                 cpu=platform.processor(), threads=THREADS,
                                 versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'numba')}),
                protocol=dict(
                    start='fresh family after three warmup flames; no saved seed profiles',
                    continuation='native predictor using nearest two already solved lower-phi states within this FGM',
                    adaptation='native leave-one-out indicator; logarithmic midpoints; shared adaptive c-grid rebuilt each round',
                    timing='sum of setup, native solves including properties, coordinate checks, table builds and selection; excludes saving and warmup; solver profiling enabled',
                    statistics='one observation per condition; no median, quartiles or confidence interval',
                    acceptance='all flames accepted, raw progress monotone/bounded, valid table, internal indicator reaches target',
                    scope='internal interpolation indicator; independent validation from base campaign applies only to that base',
                    units=dict(time='s', pressure='Pa', temperature='K', length='m')))


def settings_for(manifest, condition, out):
    mechanism = Path(manifest['settings']['mech']).name
    return dict(manifest['settings'], T_in=float(condition['T_in_K']), P=condition['P_Pa'],
                transport_model=condition['transport_model'], soret_enabled=condition['soret_enabled'],
                mech=str(out / 'inputs' / mechanism))


def ensure_manifest(out, proposed, resume):
    path = out / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != proposed:
            raise ValueError('Configuration/code/environment changed: use another output directory.')
        if not resume: raise ValueError('Campaign exists: use --resume.')
    else:
        if out.exists() and any(out.iterdir()): raise ValueError('New output directory must be empty.')
        (out / 'inputs').mkdir(parents=True)
        mechanism = Path(proposed['settings']['mech']).name
        shutil.copy2(ROOT / 'src/kflame/chemistry/data' / mechanism, out / 'inputs' / mechanism)
        atomic(path, proposed)
    mechanism = Path(proposed['settings']['mech']).name
    code_key = f'src/kflame/chemistry/data/{mechanism}'
    if digest(out / 'inputs' / mechanism) != proposed['code'][code_key]:
        raise ValueError('Mechanism copy changed.')


def select_bridges(phi, scores, target, max_bridges):
    """Same selection rule as kflame.fgm.refine; all inserted rows are solved."""
    import numpy as np
    phi, scores = np.asarray(phi), np.asarray(scores)
    if (len(phi) < 3 or scores.shape != (len(phi)-1,) or
            not np.isfinite(scores).all() or (np.diff(phi) <= 0).any()):
        raise ValueError('Invalid adaptive indicator or coordinates.')
    selected = np.flatnonzero(scores > target)
    if selected.size > max_bridges:
        selected = selected[np.argsort(scores[selected])[-max_bridges:]]
    selected = np.sort(selected)
    bridges = np.sqrt(phi[selected] * phi[selected+1])
    if np.any(bridges <= phi[selected]) or np.any(bridges >= phi[selected+1]):
        raise RuntimeError('Composition resolution exhausted before reaching target.')
    return bridges.tolist()


def raw_coordinate_audit(rec):
    import numpy as np
    beta = np.asarray(rec.beta)
    span = float(beta[-1] - beta[0])
    if not np.isfinite(beta).all() or abs(span) <= 1e-14:
        return dict(valid=False, reason='Degenerate/nonfinite progress normalization', span=span)
    c = (beta - beta[0]) / span
    valid = bool(np.diff(c).min() >= -1e-7 and c.min() >= -1e-7 and c.max() <= 1+1e-7)
    return dict(valid=valid, span=span, c_min=float(c.min()), c_max=float(c.max()),
                min_step=float(np.diff(c).min()), backward_steps=int((np.diff(c) < -1e-7).sum()))


class Timings:
    def __init__(self):
        self.values = dict(setup_s=0., family_s=0., coordinate_checks_s=0., tabulation_s=0., selection_s=0.)

    @contextmanager
    def measure(self, key):
        start = time.perf_counter()
        try: yield
        finally: self.values[key] += time.perf_counter() - start

    def summary(self):
        return dict(self.values, compute_s=sum(self.values.values()))


def construct(manifest, condition, out, attempt, timings, engine=None):
    import numpy as np
    from kflame.fgm import generate as native
    from kflame.fgm.refine import _defects
    g = native if engine is None else engine
    settings = settings_for(manifest, condition, out)
    args = argparse.Namespace(**settings)
    g.configure_numba_kinetics_threads(4)
    weights = g.parse_progress_weights(args.progress_species)
    mech = g.load_mechanism(args.mech)
    opts = g.make_solve_options(args)
    warm, previous, previous2 = [], None, None
    for phi in (.9, 1., 1.1):
        rec, state, trace = g.solve_flame_native(phi, args, mech, opts, weights,
                                                prev_solution=previous, prev_prev_solution=previous2)
        warm.append(dict(phi=phi, accepted=bool(rec.solve_ok and rec.final_accepted), trace=trace))
        atomic(attempt / 'warmup.json', warm)
        if not warm[-1]['accepted']: raise RuntimeError(f'Warmup rejected at phi={phi}.')
        previous2, previous = previous, state
    # No warmup state enters the measured family.
    with timings.measure('setup_s'):
        mech = g.load_mechanism(args.mech)
        opts = g.make_solve_options(args)
        weights = g.parse_progress_weights(args.progress_species)
    adapt = manifest['adaptation']
    records, states, traces, rounds = {}, {}, [], []
    pending = adapt['initial_phi']
    for round_index in range(adapt['max_rounds']+1):
        for phi in pending:
            lower = sorted(p for p in states if p < phi)
            prev = states[lower[-1]] if lower else None
            prev2 = states[lower[-2]] if len(lower) > 1 else None
            with timings.measure('family_s'):
                start = time.perf_counter()
                rec, state, trace = g.solve_flame_native(phi, args, mech, opts, weights,
                                                        prev_solution=prev, prev_prev_solution=prev2)
                elapsed = time.perf_counter()-start
            rec.requested, rec.bridge = round_index == 0, round_index > 0
            with timings.measure('coordinate_checks_s'): audit = raw_coordinate_audit(rec)
            trace.update(phi=phi, round=round_index, measured_s=elapsed, coordinate=audit,
                         previous_phi=lower[-1] if lower else None,
                         previous2_phi=lower[-2] if len(lower)>1 else None,
                         accepted=bool(rec.solve_ok and rec.final_accepted))
            index = len(traces)
            trace['profile'] = f'profiles/{index:03d}.npz'
            traces.append(trace)
            atomic(attempt / trace['profile'], dict(dataclasses.asdict(rec),
                   species_names=np.asarray(mech.species_names, dtype=str)), npz=True)
            atomic(attempt / 'trace.json', traces)
            atomic(attempt / 'progress.json', dict(status='running', n_solved=len(traces), round=round_index,
                                                  **timings.summary()))
            print(f'FGM {condition["id"]} round={round_index} n={len(traces)} '
                  f'phi={phi:.8g} t={elapsed:.3f}s accepted={trace["accepted"]}', flush=True)
            if not trace['accepted']: raise RuntimeError(f'Flame not accepted: phi={phi}.')
            if not audit['valid']: raise RuntimeError(f'Raw progress coordinate failed: phi={phi}; inspect profile.')
            records[phi], states[phi] = rec, state
        ordered = [records[p] for p in sorted(records)]
        with timings.measure('tabulation_s'):
            table, contract = tabulate(ordered, list(mech.species_names), settings)
        with timings.measure('selection_s'):
            scores, diagnostics = _defects(table)
            pending = select_bridges(table['phi_grid'], scores, adapt['target_defect'], adapt['max_bridges'])
            maximum = float(np.max(scores))
        rounds.append(dict(round=round_index, n_flames=len(records), phi=table['phi_grid'],
                           max_defect=maximum, interval_defect=scores, leave_one_out=diagnostics,
                           proposed_phi=pending, timings=timings.summary()))
        atomic(attempt / 'rounds.json', rounds)
        atomic(attempt / 'round_tables' / f'{round_index:03d}.npz', table, npz=True)
        print(f'ADAPT {condition["id"]}: {len(records)} flames, max indicator={100*maximum:.4f}%, '
              f'new rows={len(pending)}', flush=True)
        if not pending:
            atomic(attempt / 'fgm_table.npz', table, npz=True)
            return dict(status='accepted', condition=condition, n_flames=len(records),
                        n_inserted=len(records)-len(adapt['initial_phi']), rounds=round_index,
                        max_defect=maximum, validation=contract, **timings.summary())
        if round_index >= adapt['max_rounds'] or len(records)+len(pending) > adapt['max_flames']:
            raise RuntimeError('Adaptive budget exhausted before indicator target; partial tables retained.')
    raise AssertionError('Unreachable adaptive loop')


def worker(out, case, attempt, engine=None):
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    condition = next(c for c in manifest['conditions'] if c['id'] == case)
    timings, start = Timings(), time.perf_counter()
    try:
        result = construct(manifest, condition, out, attempt, timings, engine=engine)
    except KeyboardInterrupt:
        atomic(attempt / 'state.json', dict(status='interrupted', **timings.summary()))
        return 130
    except Exception as exc:
        traceback.print_exc()
        result = dict(status='failed', condition=condition, reason=str(exc), **timings.summary())
    result['process_wall_s'] = time.perf_counter()-start  # includes warmup and saving, explicitly separate
    files = [p for p in attempt.rglob('*') if p.is_file() and p.name not in
             ('state.json', 'result.json', 'process.log') and not p.name.endswith('.tmp')]
    result['artifacts'] = {p.relative_to(attempt).as_posix(): digest(p) for p in files}
    atomic(attempt / 'result.json', result)
    return 0 if result['status'] == 'accepted' else 1


def run_case(out, case, retry_failed=False, entrypoint=None):
    job = out / 'cases' / case
    previous = completed(job)
    if previous and (previous['status'] == 'accepted' or not retry_failed):
        verify_artifacts(latest(job), previous)
        print(f'{case}: retained {previous["status"]}', flush=True)
        return previous['status'] == 'accepted'
    attempt = job / f'attempt-{len(list(job.glob("attempt-*")))+1:03d}'
    attempt.mkdir(parents=True)
    atomic(attempt / 'state.json', dict(status='running', started=time.time()))
    command = [sys.executable, str(entrypoint or Path(__file__).resolve()), '--worker', case,
               '--output', str(out), '--attempt', str(attempt)]
    process, started = None, time.perf_counter()
    try:
        with (attempt / 'process.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **THREADS, PYTHONIOENCODING='utf-8'),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace')
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith(('FGM ', 'ADAPT ')): print(line.rstrip(), flush=True)
            code = process.wait()
        if code == 130:
            atomic(attempt / 'state.json', dict(status='interrupted', process_wall_s=time.perf_counter()-started))
            raise KeyboardInterrupt
        if not (attempt / 'result.json').exists():
            atomic(attempt / 'result.json', dict(status='failed', exit_code=code, reason='See process.log',
                                                process_wall_s=time.perf_counter()-started))
        result = json.loads((attempt / 'result.json').read_text(encoding='utf-8'))
        atomic(attempt / 'state.json', dict(status=result['status'], finished=time.time()))
        print(f'{case}: {result["status"]}; {result.get("reason", "")}', flush=True)
        return result['status'] == 'accepted'
    except KeyboardInterrupt:
        if process and process.poll() is None: process.terminate(); process.wait()
        atomic(attempt / 'state.json', dict(status='interrupted', process_wall_s=time.perf_counter()-started))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path,
                   default=Path('TESIS_RESUL/corridas/reproduccion/thesis_fgm_adaptive'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Separate technical protocol: base only, 3 initial rows, 10%% indicator, Nc=61')
    p.add_argument('--case', action='append', help='Select case IDs for this invocation; campaign design stays unchanged')
    p.add_argument('--retry-failed', action='store_true', help='Explicitly retry failed FGM in new attempts')
    p.add_argument('--worker', help=argparse.SUPPRESS)
    p.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    out = args.output.resolve()
    if args.worker: return worker(out, args.worker, args.attempt)
    proposed = make_manifest(args.smoke)
    conditions = proposed['conditions']
    if args.case:
        unknown = set(args.case)-{c['id'] for c in conditions}
        if unknown: p.error(f'Unknown cases: {sorted(unknown)}')
        conditions = [c for c in conditions if c['id'] in args.case]
    print(f'{len(proposed["conditions"])//4} states x 4 transports; one construction each; '
          f'{len(conditions)} selected. Final flame count is adaptive.', flush=True)
    for c in conditions: print(f'{c["id"]}: T={c["T_in_K"]} K, p={c["p_atm"]} atm, {c["transport_model"]}, Soret={c["soret_enabled"]}')
    if args.dry_run:
        if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text(encoding='utf-8')) != proposed:
            raise ValueError('Existing campaign incompatible.')
        print('Dry run: no solves or output changes. Three warmup flames per executed FGM.')
        return 0
    ensure_manifest(out, proposed, args.resume)
    lock = out / 'campaign.lock'
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Campaign locked: check active processes before removing {lock}')
    with os.fdopen(fd, 'w') as f: f.write(str(os.getpid()))
    try:
        ok = True
        for i, c in enumerate(conditions, 1):
            print(f'[{i}/{len(conditions)}] Starting {c["id"]}', flush=True)
            ok = run_case(out, c['id'], args.retry_failed) and ok
        return 0 if ok else 1
    finally: lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
