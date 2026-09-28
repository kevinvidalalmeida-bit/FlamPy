"""Continue an exhausted six-level verification without changing its solver code.

This is an explicit protocol extension, recorded separately from the original
manifest. Only the user launches the additional simulations.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('flame_sweep_runner', ROOT/'benchmarks/benchmark_flame_sweeps.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
TARGET = .002


def extended_settings(level, base_seconds, max_points, max_seconds, width=.03):
    s = runner.settings(min(level, 5), width, base_seconds)
    if level > 5:
        s.update(level=level, slope=runner.SLOPES[-1]/2**(level-5),
                 curve=2*runner.SLOPES[-1]/2**(level-5),
                 max_points=max_points, max_seconds=max_seconds)
    return s


def mesh_checks(history):
    checks = []
    for case_id, levels in history.items():
        for backend in ('native', 'cantera'):
            recent = [h.get(backend, {}) for h in levels[-3:]]
            usable = len(recent) == 3 and all(r.get('usable') for r in recent)
            changes = ([runner.relative_change(b['Su'], a['Su'])
                        for a, b in zip(recent, recent[1:])] if usable else [])
            checks.append(dict(case_id=case_id, backend=backend,
                               changes_percent=[100*d for d in changes],
                               passed=usable and all(d < TARGET for d in changes)))
    return checks


def stored_pair(out, phase, c, expected):
    directory = out/phase/c['id']/'pair-00'
    done = runner.completed_attempt(directory)
    # Incomplete attempts must also agree before pair_run restarts them.
    for path in directory.glob('attempt-*/pair.json'):
        if runner.read(path)['settings'] != expected:
            raise ValueError(f'Incompatible saved settings: {path}')
    if done is None:
        return None
    info = runner.read(done/'pair.json')
    rows = [runner.read(done/name/'result.json') for name in info['order']]
    if {r['backend'] for r in rows} != {'native', 'cantera'}:
        raise ValueError(f'Incomplete solver pair: {done}')
    return rows


def base_history(out, fuel, base_seconds):
    anchors = [runner.case(fuel), runner.case(fuel, pressure=10), runner.case(fuel, temperature=500)]
    history = {c['id']: [] for c in anchors}
    for level in range(6):
        for c in anchors:
            rows = stored_pair(out, f'verify-L{level}', c, runner.settings(level, max_seconds=base_seconds))
            if rows is None:
                raise ValueError(f'Finish the original verification first: verify-L{level}, {c["id"]}')
            history[c['id']].append({r['backend']: r for r in rows})
    return anchors, history


def print_checks(checks):
    for c in checks:
        changes = ', '.join(f'{d:.5f}%' for d in c['changes_percent']) or 'unusable pair'
        print(f'  {c["case_id"]} / {c["backend"]}: {changes}; '
              f'{"PASS" if c["passed"] else "PENDING"}', flush=True)


def continue_verification(out, args, base_seconds, selection):
    def settings(level, width=.03):
        return extended_settings(level, base_seconds, args.max_points, args.max_seconds, width)

    def pair(phase, c, s, ordinal):
        rows = stored_pair(out, phase, c, s)
        if rows is not None:
            return rows
        return runner.pair_run(out, phase, c, 0, s, ['native', 'cantera'], ordinal)

    for fuel in ('CH4', 'H2'):
        if selection[fuel]['passed']:
            print(f'{fuel}: verification already passed; retained.')
            continue
        anchors, history = base_history(out, fuel, base_seconds)
        selected = None
        domain_checks = []
        # Reconsider L5 first: its domain check may be the only missing step.
        for level in range(5, 8):
            if level > 5:
                for i, c in enumerate(anchors):
                    rows = pair(f'verify-L{level}', c, settings(level), i)
                    history[c['id']].append({r['backend']: r for r in rows})
            checks = mesh_checks(history)
            print(f'{fuel}, L{level}: two successive mesh changes (target <0.2%):')
            print_checks(checks)
            domain_checks = []
            if all(c['passed'] for c in checks):
                for i, c in enumerate(anchors):
                    current = history[c['id']][-1]
                    width = 2*max(r['width'] for r in current.values())
                    rows = pair(f'domain-L{level}', c, settings(level, width), i)
                    for r in rows:
                        change = (runner.relative_change(r['Su'], current[r['backend']]['Su'])
                                  if r.get('usable') else None)
                        domain_checks.append(dict(case_id=c['id'], backend=r['backend'],
                            change_percent=None if change is None else 100*change,
                            passed=change is not None and change < TARGET))
                if all(c['passed'] for c in domain_checks):
                    selected = settings(level)
            selection[fuel] = dict(passed=selected is not None, settings=selected,
                target_relative_change=TARGET, level_reached=level,
                mesh_checks=checks, domain_checks=domain_checks,
                protocol_extension='verification_extension.json')
            runner.atomic_json(out/'verification.json', selection)
            if selected is not None:
                print(f'{fuel}: mesh and domain checks passed at L{level}.', flush=True)
                break
    return all(v['passed'] for v in selection.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/thesis_flames'))
    parser.add_argument('--max-points', type=int, default=16000)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.max_points < 6000 or not 0 < args.max_seconds < float('inf'):
        parser.error('max-points must be >=6000 and max-seconds finite and positive')
    out = args.output.resolve()
    if not (out/'manifest.json').is_file() or not (out/'verification.json').is_file():
        parser.error('An existing scientific campaign with verification results is required')
    manifest = runner.read(out/'manifest.json')
    if manifest['smoke']:
        parser.error('Cannot extend a smoke test')
    # Preserve the original code/environment identity, inputs and time budget.
    base_args = SimpleNamespace(pairs=manifest['pairs'], phase='verify',
                               max_seconds=manifest['max_seconds'], resume=True)
    protocol = dict(schema=1, base_manifest_sha256=runner.digest(out/'manifest.json'),
        extension_source_sha256=runner.digest(Path(__file__)),
        target_relative_change=TARGET,
        reason='Six original mesh levels exhausted; explicit additional spatial verification',
        additional_settings=[extended_settings(l, manifest['max_seconds'], args.max_points,
                                               args.max_seconds) for l in (6, 7)],
        application='Selected settings apply to subsequent main and ablation phases; original runs remain unchanged')

    def validate():
        runner.ensure_manifest(out, base_args)
        path = out/'verification_extension.json'
        if path.exists() and runner.read(path) != protocol:
            raise ValueError('Incompatible extension code or settings; keep the original extension command')
        selection = runner.read(out/'verification.json')
        if set(selection) != {'CH4', 'H2'}:
            raise ValueError('Finish the original verification for both fuels first')
        if any((out/p).exists() for p in ('main', 'ablation')) and not all(v['passed'] for v in selection.values()):
            raise ValueError('Cannot change spatial settings after main or ablation has started')
        for fuel in ('CH4', 'H2'):
            if not selection[fuel]['passed']:
                _, history = base_history(out, fuel, manifest['max_seconds'])
                print(f'{fuel}: original L3->L4 and L4->L5 changes:')
                print_checks(mesh_checks(history))
        return selection

    if args.dry_run:
        validate()
        print(f'Additional levels: L6 slope=0.00125, L7 slope=0.000625; '
              f'max_points={args.max_points}; max_seconds={args.max_seconds:g} per solve.')
        print('At most 24 additional mesh solves and 36 domain solves (including L5 if applicable), '
              'plus warmups. Stops per fuel when checks pass. No simulations or files written.')
        return 0
    with runner.campaign_lock(out):
        selection = validate()
        path = out/'verification_extension.json'
        if path.exists() and not args.resume:
            raise ValueError('Extension exists; use --resume')
        if all(v['passed'] for v in selection.values()):
            print('Both fuels already passed verification; no simulations needed.')
            return 0
        if not path.exists():
            runner.atomic_json(out/'verification_before_extension.json', selection)
            # Archive the exact extension source in addition to its fingerprint.
            (out/'inputs/extend_flame_verification.py').write_bytes(Path(__file__).read_bytes())
            runner.atomic_json(path, protocol)
        elif runner.digest(out/'inputs/extend_flame_verification.py') != protocol['extension_source_sha256']:
            raise ValueError('Archived extension source was modified')
        passed = continue_verification(out, args, manifest['max_seconds'], selection)
        runner.refresh_index(out)
    if passed:
        print('Verification passed. Main and ablation will use verification.json settings.')
        return 0
    print('Target still unmet. Main remains blocked; inspect mesh_checks and domain_checks '
          'in verification.json. Completed failures are retained, not automatically retried.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Repeat the same command with --resume.', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
