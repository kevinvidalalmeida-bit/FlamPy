"""Independent a-posteriori L3 spatial study; production observations remain unchanged.

Six anchors, L2/L3/L4/L5, plus doubled-domain L3. Twelve archived L3
solutions are reused; 48 new measured solves (one pair per test), plus warmups.
No convergence threshold blocks or changes the completed main campaign.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sweeps',ROOT/'benchmarks/benchmark_flame_sweeps.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def anchors():
    return [runner.case(fuel,pressure=p,temperature=t)
            for fuel in ('CH4','H2') for p,t in ((1,300),(10,300),(1,500))]


def baseline(root):
    manifest=runner.read(root/'manifest.json')
    if manifest['smoke'] or manifest['spatial_protocol']['selected_level']!=3:
        raise ValueError('Requires a scientific L3 campaign')
    if manifest['environment']!=runner.environment():
        raise ValueError('Code/environment differs from the production campaign')
    for name in ('gri30.yaml','h2o2.yaml','collision_integrals_mm.json'):
        if runner.digest(root/'inputs'/name)!=runner.digest(ROOT/'src/kflame/chemistry/data'/name):
            raise ValueError('Changed physical input: '+name)
    result=[]
    for c in anchors():
        found=None
        for p in sorted((root/'main'/c['id']).glob('pair-*')):
            attempt=runner.completed_attempt(p)
            if attempt is None:continue
            info=runner.read(attempt/'pair.json')
            rows=[runner.read(attempt/b/'result.json') for b in info['order']]
            if len(rows)==2 and {r['backend'] for r in rows}=={'native','cantera'} and all(r.get('usable') for r in rows):
                found=rows;break
        if found is None:raise ValueError('No accepted baseline pair for '+c['id'])
        for row in found:
            if row['settings']!=runner.settings(3,max_seconds=manifest['max_seconds']):
                raise ValueError('Baseline numerical settings differ')
            if runner.digest(root/row['folder']/'profile.npz')!=row['profile_sha256']:
                raise ValueError('Baseline profile integrity failure')
            result.append(row)
    return manifest,result


def study_manifest(root,manifest,rows):
    return dict(schema=1,kind='L3_spatial_sensitivity',parent=str(root),
        parent_manifest_sha256=runner.digest(root/'manifest.json'),
        environment=manifest['environment'],script_sha256=runner.digest(Path(__file__)),
        cases=anchors(),baseline=rows,levels=[2,3,4,5],domain_level=3,target_percent=.5,
        max_seconds=manifest['max_seconds'],new_measured_solves=48,
        protocol='One independent warm solve per solver/test; reuse earliest accepted L3 pair. '
        'Compare all refined velocities against L3; double each pair maximum final L3 domain. '
        'Record signed differences and successive changes; threshold is descriptive, no production policy mutation.')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3'))
    ap.add_argument('--output', type=Path,
                    default=Path('TESIS_RESUL/corridas/reproduccion/thesis_flames_L3_sensitivity'))
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv);root=args.input.resolve();out=args.output.resolve()
    if out==root or root in out.parents or out in root.parents:
        ap.error('Use a separate sibling directory for spatial tests')
    m,base=baseline(root);protocol=study_manifest(root,m,base)
    if (out/'manifest.json').exists():
        if runner.read(out/'manifest.json')!=protocol:raise ValueError('Incompatible spatial study; use another directory')
        if not args.resume and not args.dry_run:raise ValueError('Use --resume')
    elif out.exists() and any(p.name!='.campaign.lock' for p in out.iterdir()):
        raise ValueError('Output is nonempty without a manifest')
    print('6 states; L2, L3, L4, L5 + doubled L3 domain; both solvers.')
    print('Reuse 12 L3 profiles; 48 new measured solves + 48 excluded warmups. No five-fold repetition needed.')
    if args.dry_run:
        print('Identity, baselines and inputs checked. No simulations or writes.');return 0
    out.mkdir(parents=True,exist_ok=True)
    with runner.campaign_lock(out):
        if not (out/'manifest.json').exists():
            shutil.copytree(root/'inputs',out/'inputs')
            shutil.copy2(Path(__file__),out/'inputs'/Path(__file__).name)
            runner.atomic_json(out/'manifest.json',protocol)
        else:
            for name in ('gri30.yaml','h2o2.yaml','collision_integrals_mm.json'):
                if runner.digest(out/'inputs'/name)!=runner.digest(root/'inputs'/name):
                    raise ValueError('Modified archived spatial input: '+name)
        try:
            for i,c in enumerate(anchors()):
                for level in (2,4,5):
                    runner.pair_run(out,f'verify-L{level}',c,0,
                        runner.settings(level,max_seconds=m['max_seconds']),['native','cantera'],i+level)
                width=2*max(r['width'] for r in base if r['case_id']==c['id'])
                runner.pair_run(out,'domain-L3',c,0,
                    runner.settings(3,width,m['max_seconds']),['native','cantera'],i+3)
        except KeyboardInterrupt:
            runner.refresh_index(out)
            print('Interrupted; use the same command with --resume.',flush=True)
            raise
        finally:
            # Offline figure updates are useful even when a pair was interrupted.
            subprocess.run([sys.executable,str(ROOT/'tools/postprocess_flame_mesh_sensitivity.py'),
                '--input',str(out),'--output',str(root/'report')],cwd=ROOT,check=True)
    print('Spatial study complete; production L3 unchanged. Inspect sensitivity_report.tex and JSON.')
    return 0


if __name__=='__main__':raise SystemExit(main())
