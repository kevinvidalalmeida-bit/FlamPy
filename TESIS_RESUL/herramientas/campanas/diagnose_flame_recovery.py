"""Replay the 104 L3 conditions once with KFLAME trajectory instrumentation.

Uses the established diagnostic wrapper, including interrupted domain passes and
the multicomponent bootstrap. Production results and their timing sample remain
unchanged. Each worker performs one excluded warmup and one diagnostic solve.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import diagnose_flame_campaign as diagnostic

runner = diagnostic.runner
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'TESIS_RESUL/herramientas/postprocesado'))
import postprocess_flame_sweeps as pp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT / 'TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'TESIS_RESUL/corridas/reproduccion/thesis_flames_L3_recovery')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--case-id', help='Execute only this condition; the manifest retains the full design.')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        os.environ.update(runner.THREADS)
        diagnostic.instrument('native')
        return runner.worker(args.worker)

    root, out = args.input.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        parser.error('Use a separate diagnostic directory outside the production campaign.')
    source = pp.load_manifest(root)
    records = [r for r in pp.load_records(root)
               if r['phase'] == 'main' and r['variant'] == 'native' and r.get('usable')]
    production = {}
    for case in source['cases']:
        matches = [r for r in records if r['case_id'] == case['id']]
        if not matches:
            raise ValueError(f'No usable production reference: {case["id"]}')
        production[case['id']] = min(matches, key=lambda r: r['repetition'])
    if len(production) != 104:
        raise ValueError(f'Expected 104 unique conditions; found {len(production)}')
    env = runner.environment()
    if env != source['environment']:
        raise ValueError('Code or environment differs from the original campaign; do not mix diagnostic protocols.')
    files = [Path(__file__), Path(diagnostic.__file__), ROOT / 'TESIS_RESUL/herramientas/campanas/benchmark_flame_sweeps.py']
    manifest = dict(kind='flame-recovery-104', diagnostic_only=True, parent=str(root),
        parent_manifest_sha256=runner.digest(root / 'manifest.json'),
        extension_sha256=runner.digest(root / 'transport_extension.json'),
        environment=env, code_sha256={str(p.relative_to(ROOT)): runner.digest(p) for p in files},
        cases=source['cases'], settings=runner.settings(3, max_seconds=source['max_seconds']),
        protocol='One excluded warmup and one instrumented KFLAME solve per condition. All phases including bootstrap and interrupted domain passes. No production timing observations added.',
        references={cid: dict(folder=r['folder'], repetition=r['repetition'],
                             profile_sha256=runner.digest(root / r['folder'] / 'profile.npz'),
                             result_sha256=runner.digest(root / r['folder'] / 'result.json'))
                    for cid, r in production.items()})
    planned = [c for c in manifest['cases'] if not args.case_id or c['id'] == args.case_id]
    if not planned:
        parser.error('Unknown --case-id')
    print(f'{len(planned)} conditions: {len(planned)} instrumented KFLAME solves + '
          f'{len(planned)} excluded warmups; serial execution. Output: {out}', flush=True)
    if args.dry_run:
        print('Environment and production references verified. No solves or output files created.')
        return 0

    out.mkdir(parents=True, exist_ok=True)
    with runner.campaign_lock(out):
        manifest_path = out / 'manifest.json'
        if manifest_path.exists():
            if not args.resume or runner.read(manifest_path) != manifest:
                raise ValueError('Use --resume with identical code/configuration, or a new output directory.')
        else:
            if any(p.name != '.campaign.lock' for p in out.iterdir()):
                raise ValueError('Output directory is not empty.')
            shutil.copytree(root / 'inputs', out / 'inputs')
            for p in files:
                dest = out / 'code' / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
            runner.atomic_json(manifest_path, manifest)
        for i, case in enumerate(planned, 1):
            folder = out / case['id'] / 'native'
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / 'result.json'
            if result_path.exists() and runner.read(result_path).get('diagnostic_committed'):
                print(f'{i}/{len(planned)} {case["id"]}: recorded; skipped.', flush=True)
                continue
            previous = [p for p in folder.iterdir() if p.is_file()]
            if previous:
                archive = folder / f'interrupted-{time.time_ns()}'
                archive.mkdir()
                for p in previous:
                    shutil.copy2(p, archive / p.name)
            req = folder / 'request.json'
            runner.atomic_json(req, dict(case=case, settings=manifest['settings'], backend='native',
                variant='native', mechanism=str(out / 'inputs' / case['mechanism'])))
            print(f'{i}/{len(planned)} {case["id"]}', flush=True)
            start = time.perf_counter()
            with (folder / 'console.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(req)],
                    env=dict(os.environ, **runner.THREADS), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=2 * source['max_seconds'] + 120)
                except (KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
                    process.kill(); process.wait()
                    runner.atomic_json(folder / 'interruption.json', dict(reason=type(exc).__name__,
                        elapsed_s=time.perf_counter()-start, utc=time.time(), diagnostic_committed=False))
                    raise
            result = runner.read(result_path) if result_path.exists() else dict(status='worker_failed', accepted=False)
            result.update(diagnostic_committed=True, diagnostic_only=True, returncode=code,
                          process_s=time.perf_counter()-start, folder=str(folder.relative_to(out)),
                          usable=bool(code == 0 and result.get('accepted') and result.get('diagnostics_complete')))
            runner.atomic_json(result_path, result)
            print(f'  {result["status"]}; diagnostic time={result.get("time_s")}', flush=True)
    from postprocess_flame_recovery import main as report
    return report(['--input', str(root), '--diagnostics', str(out),
                   '--output', str(root / 'report_expanded/revision_figures')])


if __name__ == '__main__':
    raise SystemExit(main())
