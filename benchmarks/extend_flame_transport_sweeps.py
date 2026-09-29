"""Complete pressure/temperature transport sweeps without rerunning the L3 base.

Uses the unchanged paired worker, warmup, diagnostics and atomic resume protocol.
The extension manifest is separate; original manifest and results are retained.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import benchmark_flame_sweeps as runner


def additional_cases():
    existing = {c['id'] for c in runner.cases()}
    result = {}
    for fuel in ('CH4', 'H2'):
        for transport, soret in runner.MODES:
            for p, t in [(p, 300) for p in (1, 2, 3, 5, 10)] + [(1, t) for t in (300, 350, 400, 450, 500)]:
                c = runner.case(fuel, pressure=p, temperature=t, transport=transport, soret=soret)
                if c['id'] not in existing:
                    result[c['id']] = c
    assert len(result) == 48
    return list(result.values())


def preflight(root):
    base = runner.read(root/'manifest.json')
    if base.get('smoke') or base['pairs'] != 5 or base['cases'] != runner.cases():
        raise ValueError('Requires the original 56-condition, five-pair scientific campaign.')
    if base['spatial_protocol'].get('selected_level') != 3:
        raise ValueError('This extension preserves fixed L3.')
    current = runner.environment()
    if current != base['environment']:
        different = [k for k in current if current[k] != base['environment'].get(k)]
        raise ValueError('Code/environment differs from the base campaign: '+', '.join(different))
    for name in ('gri30.yaml', 'h2o2.yaml', 'collision_integrals_mm.json'):
        if runner.digest(root/'inputs'/name) != runner.digest(runner.ROOT/'src/kflame/chemistry/data'/name):
            raise ValueError('Physical input differs: '+name)
    s = runner.settings(3, max_seconds=base['max_seconds'])
    # Verify every reused pair and profile before extending its curves.
    for c in base['cases']:
        for rep in range(base['pairs']):
            done = runner.completed_attempt(root/'main'/c['id']/f'pair-{rep:02d}')
            if done is None:
                raise ValueError(f'Incomplete base pair: {c["id"]}/{rep}')
            info = runner.read(done/'pair.json')
            if info['settings'] != s or set(info['order']) != {'native', 'cantera'}:
                raise ValueError('Base pair settings differ: '+str(done))
            for backend in info['order']:
                r = runner.read(done/backend/'result.json')
                if not r.get('usable') or r['condition'] != c or r['repetition'] != rep:
                    raise ValueError('Base result is incomplete or incompatible: '+str(done/backend))
                if runner.digest(done/backend/'profile.npz') != r['profile_sha256']:
                    raise ValueError('Base profile was modified: '+str(done/backend))
    return base, s


def extension_manifest(root, base, s):
    return dict(schema=1, kind='pressure_temperature_all_transports',
                base_manifest_sha256=runner.digest(root/'manifest.json'),
                environment=base['environment'], pairs=base['pairs'], settings=s,
                cases=additional_cases(), added_conditions=48, added_measured_solves=480,
                combined_conditions=104, phase='main',
                generator_sha256=runner.digest(__file__),
                protocol='Same paired worker, independent warmups, alternating solver order; only new cases')


def validate_saved_extension(root, planned, pairs, settings):
    completed = 0
    for c in planned:
        for rep in range(pairs):
            directory = root/'main'/c['id']/f'pair-{rep:02d}'
            for path in directory.glob('attempt-*/pair.json'):
                info = runner.read(path)
                if info['settings'] != settings or set(info['order']) != {'native', 'cantera'}:
                    raise ValueError('Saved extension settings differ: '+str(path))
            done = runner.completed_attempt(directory)
            if done is None:
                continue
            completed += 1
            for backend in ('native', 'cantera'):
                row = runner.read(done/backend/'result.json')
                if row['condition'] != c or row['repetition'] != rep or row['backend'] != backend:
                    raise ValueError('Saved extension condition differs: '+str(done/backend))
                if row.get('usable') and runner.digest(done/backend/'profile.npz') != row.get('profile_sha256'):
                    raise ValueError('Saved extension profile differs: '+str(done/backend))
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path,
                        default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.campaign.resolve()
    base, s = preflight(root)
    expected = extension_manifest(root, base, s)
    path = root/'transport_extension.json'
    if path.exists():
        if runner.read(path) != expected:
            raise ValueError('Incompatible transport extension. Preserve this campaign and use its archived code/environment.')
        if not args.resume and not args.dry_run:
            raise ValueError('Extension exists; use --resume.')
    elif any((root/'main'/c['id']).exists() for c in expected['cases']):
        raise ValueError('Extension results exist without their manifest.')
    completed = validate_saved_extension(root, expected['cases'], base['pairs'], s)
    print(f'Base verificada: 56 condiciones, 560 resoluciones. L3; cinco pares por condicion.')
    print(f'Ampliacion: 48 condiciones (24 CH4 + 24 H2), 240 pares, 480 resoluciones medidas + calentamientos.')
    print(f'Pares completados: {completed}/240; pendientes: {240-completed}. Total final: 104 condiciones.')
    if args.dry_run:
        print('Comprobacion terminada; no se escribieron datos ni se resolvieron llamas.')
        return 0
    with runner.campaign_lock(root):
        if not path.exists():
            dest = root/'code/benchmarks'/Path(__file__).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(__file__, dest)
            runner.atomic_json(path, expected)
        completed = 0
        failed = 0
        for i, c in enumerate(expected['cases']):
            for rep in range(base['pairs']):
                rows = runner.pair_run(root, 'main', c, rep, s, ['native', 'cantera'], i)
                completed += 1
                failed += int(not all(r.get('usable') for r in rows))
                runner.atomic_json(root/'transport_extension_progress.json', dict(
                    completed_pairs=completed, total_pairs=240, unusable_pairs=failed,
                    remaining_pairs=240-completed))
                print(f'Ampliacion: {completed}/240 pares; no utilizables: {failed}', flush=True)
    print('Ampliacion finalizada. Los fallos completados se conservan y no se repiten automaticamente.')
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
