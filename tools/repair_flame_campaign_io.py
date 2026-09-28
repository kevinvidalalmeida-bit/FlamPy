"""Audit an I/O-only runner hotfix and recover fully saved but uncommitted pairs.

No solver is invoked. Original manifests, source and recovery inputs are retained.
"""
import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sweeps',ROOT/'benchmarks/benchmark_flame_sweeps.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def computational_ast(source):
    """Only serializer, rename helper, tempfile import and rename calls may differ."""
    tree=ast.parse(source)
    tree.body=[node for node in tree.body if not (
        isinstance(node,ast.FunctionDef) and node.name in ('atomic_json','replace_file'))]
    tree.body=[node for node in tree.body if not (
        isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='tempfile')]
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='replace_file':
                node.func=ast.Attribute(value=ast.Name(id='os',ctx=ast.Load()),attr='replace',ctx=ast.Load())
            return node
    return ast.dump(Normalize().visit(tree),include_attributes=False)


def migrate(out):
    path=out/'manifest.json'
    old=runner.read(path)
    current=runner.environment()
    oldenv=old['environment']
    key=next(k for k in oldenv['source_sha256'] if k.replace('\\','/')=='benchmarks/benchmark_flame_sweeps.py')
    previous_hash=oldenv['source_sha256'][key]
    new_hash=current['source_sha256'][key]
    if oldenv==current:
        return
    expected=copy.deepcopy(oldenv)
    expected['source_sha256'][key]=new_hash
    if expected!=current:
        raise ValueError('Changes beyond the runner I/O hotfix: cannot migrate this campaign')
    revision=out/'storage_repairs'/new_hash
    oldsource=revision/'runner.before.py'
    archive=out/'code'/Path(key)
    if not oldsource.exists():oldsource=archive
    if runner.digest(oldsource)!=previous_hash:
        raise ValueError('Archived original runner hash does not match the campaign')
    newsource=Path(runner.__file__)
    if computational_ast(oldsource.read_text(encoding='utf-8'))!=computational_ast(newsource.read_text(encoding='utf-8')):
        raise ValueError('Computational runner code changed; this is not an I/O-only hotfix')
    revision.mkdir(parents=True,exist_ok=True)
    if not (revision/'runner.before.py').exists():shutil.copy2(oldsource,revision/'runner.before.py')
    if not (revision/'manifest.before.json').exists():shutil.copy2(path,revision/'manifest.before.json')
    shutil.copy2(newsource,revision/'runner.after.py')
    runner.atomic_json(revision/'audit.json',dict(old_runner_sha256=previous_hash,new_runner_sha256=new_hash,
        computational_ast_unchanged=True,environment_except_runner_unchanged=True,
        reason='Retry atomic replacement on transient Windows access/sharing errors; unique JSON temp names',
        existing_execution_provenance='Executions already present were produced by runner.before.py'))
    shutil.copy2(newsource,archive)
    migrated=copy.deepcopy(old)
    migrated['environment']=current
    runner.atomic_json(path,migrated)


def recover_pair(out, path):
    info=runner.read(path)
    if info.get('status') not in ('running','interrupted'):
        return False
    candidates=[]
    for name in info['order']:
        folder=path.parent/name
        result_path=folder/'result.json'
        temp=folder/'result.json.tmp'
        if temp.exists():
            candidate=runner.read(temp)
            # An interrupted JSON replacement must retain the worker's measurements.
            saved=runner.read(result_path) if result_path.exists() else {}
            if not saved or any(candidate.get(k)!=v for k,v in saved.items()):return False
        elif result_path.exists():
            candidate=runner.read(result_path)
        else:return False
        if not (candidate.get('usable') and candidate.get('accepted') and
                candidate.get('diagnostics_complete') and candidate.get('returncode')==0):return False
        req=runner.read(folder/'request.json')
        if (req['settings']!=info['settings'] or candidate['settings']!=info['settings'] or
            req['case']!=candidate['condition'] or req['backend']!=candidate['backend'] or
            candidate['case_id']!=info['case_id'] or candidate['repetition']!=info['repetition'] or
            candidate['order']!=info['order'] or candidate['variant']!=name or
            (out/candidate['folder']).resolve()!=folder.resolve() or
            candidate.get('phase')!=path.parents[3].name or
            runner.digest(folder/'profile.npz')!=candidate.get('profile_sha256')):return False
        candidates.append((folder,candidate,temp if temp.exists() else None))
    backup=out/'storage_repairs/recovered_pairs'/path.parent.relative_to(out)
    backup.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,backup/'pair.before.json')
    for folder,candidate,temp in candidates:
        if temp:
            shutil.copy2(temp,backup/(folder.name+'.result.pending.json'))
            shutil.copy2(folder/'result.json',backup/(folder.name+'.result.before.json'))
            runner.atomic_json(folder/'result.json',candidate)
    runner.atomic_json(path,dict(info,status='complete',storage_recovery='Validated completed worker results, coordinator metadata and profile SHA-256; no solve repeated'))
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(argv)
    out=args.input.resolve()
    if not (out/'manifest.json').exists():parser.error('Missing campaign manifest')
    with runner.campaign_lock(out):
        migrate(out)
        manifest=runner.read(out/'manifest.json')
        runner.ensure_manifest(out,SimpleNamespace(pairs=manifest['pairs'],phase='main',
            max_seconds=manifest['max_seconds'],resume=True,
            spatial_policy='fixed-L3' if manifest['spatial_protocol'].get('selected_level')==3 else 'verified-L4'))
        recovered=[]
        for path in sorted(out.glob('main/*/pair-*/attempt-*/pair.json')):
            if recover_pair(out,path):recovered.append(str(path.parent.relative_to(out)))
        runner.refresh_index(out)
        records=runner.read(out/'index.json')
        print(dict(recovered_pairs=recovered,recorded_runs=len(records),
                   accepted_runs=sum(bool(r.get('accepted')) for r in records),
                   usable_runs=sum(bool(r.get('usable')) for r in records)))
    return 0


if __name__=='__main__':raise SystemExit(main())
