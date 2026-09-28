"""Campaign integrity tests use synthetic records, never scientific outputs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


runner=module('sweeps','benchmarks/benchmark_flame_sweeps.py')
post=module('sweep_report','tools/postprocess_flame_sweeps.py')
extension=module('sweep_extension','tools/extend_flame_verification.py')
repair=module('sweep_repair','tools/repair_flame_campaign_io.py')


def test_atomic_json_retries_locks_and_preserves_original_on_permanent_error(tmp_path,monkeypatch):
    path=tmp_path/'result.json'
    runner.atomic_json(path,dict(version=1))
    rename=runner.os.replace
    attempts=[]
    def locked(source,target):
        attempts.append(source)
        assert runner.read(path)==dict(version=1)
        if len(attempts)<3:raise PermissionError('simulated Windows sharing lock')
        rename(source,target)
    monkeypatch.setattr(runner.os,'replace',locked)
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:None)
    runner.atomic_json(path,dict(version=2))
    assert len(attempts)==3 and runner.read(path)==dict(version=2)
    def denied(*args):raise PermissionError('persistent lock')
    monkeypatch.setattr(runner.os,'replace',denied)
    with pytest.raises(PermissionError):runner.atomic_json(path,dict(version=3))
    assert runner.read(path)==dict(version=2)
    assert len(list(tmp_path.glob('result.json.*.tmp')))==1


def test_io_audit_detects_computational_change():
    source=(ROOT/'benchmarks/benchmark_flame_sweeps.py').read_text(encoding='utf8')
    altered=source.replace('ratio=2.5','ratio=3.5')
    assert repair.computational_ast(source)!=repair.computational_ast(altered)
    assert repair.computational_ast(source)==repair.computational_ast(source.replace('time.sleep(min(.05 * 2**attempt, 1.0))','time.sleep(.1)'))


def test_recover_saved_pair_requires_profile_integrity(tmp_path):
    c=runner.case('H2');s=runner.settings(3)
    attempt=tmp_path/'main'/c['id']/'pair-00/attempt-001'
    info=dict(status='running',order=['native','cantera'],settings=s,case_id=c['id'],repetition=0)
    runner.atomic_json(attempt/'pair.json',info)
    for name in info['order']:
        folder=attempt/name;folder.mkdir()
        (folder/'profile.npz').write_bytes(b'synthetic profile')
        row=dict(usable=True,accepted=True,diagnostics_complete=True,returncode=0,
            settings=s,condition=c,backend=name,case_id=c['id'],repetition=0,order=info['order'],
            variant=name,folder=str(folder.relative_to(tmp_path)),phase='main',time_s=1.23,
            profile_sha256=runner.digest(folder/'profile.npz'))
        runner.atomic_json(folder/'request.json',dict(case=c,settings=s,backend=name))
        if name=='cantera':
            runner.atomic_json(folder/'result.json',dict(accepted=True,time_s=1.23))
            runner.atomic_json(folder/'result.json.tmp',row)
        else:runner.atomic_json(folder/'result.json',row)
    (attempt/'cantera/profile.npz').write_bytes(b'tampered')
    assert not repair.recover_pair(tmp_path,attempt/'pair.json')
    assert runner.read(attempt/'pair.json')['status']=='running'
    (attempt/'cantera/profile.npz').write_bytes(b'synthetic profile')
    assert repair.recover_pair(tmp_path,attempt/'pair.json')
    assert runner.read(attempt/'pair.json')['status']=='complete'
    assert runner.read(attempt/'cantera/result.json')['time_s']==1.23
    assert not repair.recover_pair(tmp_path,attempt/'pair.json')


def test_matrix_and_shared_points():
    cs=runner.cases()
    assert len(cs)==len({c['id'] for c in cs})==56
    assert all(c['native_strategy']==('backward-euler' if c['mechanism']=='h2o2.yaml' else 'ptc-ser') for c in cs)
    for fuel in ('CH4','H2'):
        rows=[c for c in cs if c['fuel']==fuel]
        assert len(rows)==28
        assert len([c for c in rows if post.in_sweep(c,'composition')])==20
        assert len([c for c in rows if post.in_sweep(c,'pressure')])==5
        assert len([c for c in rows if post.in_sweep(c,'temperature')])==5
        assert len([c for c in rows if c['phi']==1 and c['pressure_atm']==1 and c['temperature']==300])==4


def test_fixed_L3_runs_main_without_claiming_verification(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'environment',lambda:dict(test='synthetic',packages=dict(cantera='test')))
    calls=[]
    monkeypatch.setattr(runner,'pair_run',lambda out,phase,c,rep,s,variants,i:calls.append((phase,s,variants)))
    assert runner.main(['--phase','main','--spatial-policy','fixed-L3','--output',str(tmp_path),'--resume'])==0
    assert len(calls)==280
    assert all(p=='main' and s['level']==3 and s['slope']==.01 and s['curve']==.02 for p,s,v in calls)
    assert not (tmp_path/'verification.json').exists()
    assert runner.read(tmp_path/'manifest.json')['spatial_protocol']['verification_required'] is False
    assert all('passed' not in v for v in runner.read(tmp_path/'spatial_selection.json').values())
    args=SimpleNamespace(pairs=5,phase='main',max_seconds=600,resume=True,spatial_policy='verified-L4')
    with pytest.raises(ValueError,match='Incompatible'):runner.ensure_manifest(tmp_path,args)


def test_paired_bootstrap_and_missing_pairs():
    rows=[dict(repetition=i,variant=b,usable=True,time_s=(i+1)*(2 if b=='cantera' else 1))
          for i in range(5) for b in ('native','cantera')]
    stats=post.paired_statistics(rows)
    assert stats['n_pairs']==5 and stats['ratio']==2
    assert stats['ci95']==[2.,2.]
    rows[-1]['usable']=False
    assert post.paired_statistics(rows)['n_pairs']==4
    assert post.paired_statistics(rows[:2])['ci95'] is None


def fake_process(monkeypatch,interrupt_on=None,reject=False):
    counter=[]
    class Process:
        def __init__(self,argv,**kwargs):
            self.folder=Path(argv[-1]).parent
            req=runner.read(argv[-1]); self.n=len(counter)+1; counter.append(req)
            runner.atomic_json(self.folder/'result.json',dict(status='rejected' if reject else 'accepted',
                accepted=not reject,diagnostics_complete=True,time_s=1.,Su=1.,nodes=20,width=.03))
        def wait(self,timeout=None):
            if timeout and self.n==interrupt_on:raise KeyboardInterrupt()
            return 0
        def kill(self):pass
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    return counter


def test_resume_keeps_failure_without_retry(tmp_path,monkeypatch):
    counter=fake_process(monkeypatch,reject=True)
    c=runner.case('CH4')
    for _ in range(2):runner.pair_run(tmp_path,'main',c,0,runner.settings(),['native','cantera'],0)
    assert len(counter)==2
    rows=post.load_records(tmp_path)
    assert len(rows)==2 and all(not r['usable'] for r in rows)


def test_interrupted_pair_is_restarted_in_full(tmp_path,monkeypatch):
    counter=fake_process(monkeypatch,interrupt_on=2)
    c=runner.case('H2')
    with pytest.raises(KeyboardInterrupt):
        runner.pair_run(tmp_path,'main',c,0,runner.settings(),['native','cantera'],0)
    assert len(post.load_records(tmp_path))==0
    runner.pair_run(tmp_path,'main',c,0,runner.settings(),['native','cantera'],0)
    assert len(counter)==4
    assert len(list(tmp_path.glob('main/*/pair-00/attempt-*')))==2
    assert len(post.load_records(tmp_path))==2


def test_manifest_rejects_mismatch_and_changed_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'environment',lambda:dict(test='synthetic'))
    args=SimpleNamespace(pairs=5,phase='main',max_seconds=600,resume=True)
    runner.ensure_manifest(tmp_path,args)
    runner.ensure_manifest(tmp_path,args)
    args.pairs=4
    with pytest.raises(ValueError,match='Incompatible'):runner.ensure_manifest(tmp_path,args)
    args.pairs=5
    (tmp_path/'inputs/h2o2.yaml').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='modified'):runner.ensure_manifest(tmp_path,args)


def test_offline_profile_comparison_is_translation_invariant(tmp_path):
    for name,shift in [('a',0.),('b',2.)]:
        folder=tmp_path/name; folder.mkdir()
        z=np.linspace(0,1,101); T=300+1000*z
        np.savez(folder/'profile.npz',z=z+shift,T=T,Y=np.array([1-z,z]),
                 qdot=np.sin(np.pi*z),species_names=np.array(['H2','H2O']))
    error=post.comparison(tmp_path,dict(folder='a',Su=1.),dict(folder='b',Su=1.))
    assert error['T_L2_percent']<1.e-10
    assert error['max_species_L2_percent']<1.e-10


def test_postprocessor_does_not_import_solver():
    import ast
    tree=ast.parse((ROOT/'tools/postprocess_flame_sweeps.py').read_text(encoding='utf-8'))
    names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):names.extend(n.name for n in node.names)
        if isinstance(node,ast.ImportFrom):names.append(node.module or '')
    assert not any(n.startswith(('kflame','cantera','subprocess')) for n in names)


def test_lock_prevents_concurrent_writer(tmp_path):
    with runner.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another'):
            with runner.campaign_lock(tmp_path):pass


def test_verification_selects_L4_against_L5_and_domain(tmp_path,monkeypatch):
    calls=[]
    def run(out,phase,c,repetition,s,variants,ordinal):
        calls.append((phase,c['id']))
        value=1+.01/(2**s['level'])
        if phase.startswith('domain'):value+=1.e-6
        return [dict(backend=b,usable=True,Su=value,width=.03) for b in variants]
    monkeypatch.setattr(runner,'pair_run',run)
    runner.verify(tmp_path,SimpleNamespace(max_seconds=600))
    result=runner.read(tmp_path/'verification.json')
    assert all(result[f]['passed'] and result[f]['settings']['level']==4 for f in ('CH4','H2'))
    assert len([p for p,_ in calls if p.startswith('domain')])==6
    assert len(calls)==18
    assert {p for p,_ in calls}=={'verify-L4','verify-L5','domain-L4'}
    assert all(v['target_relative_change']==.005 for v in result.values())


def test_verification_rejects_domain_sensitivity_at_selected_level(tmp_path,monkeypatch):
    def run(out,phase,c,repetition,s,variants,ordinal):
        value=1.01 if phase.startswith('domain') else 1.
        return [dict(backend=b,usable=True,Su=value,width=.03) for b in variants]
    monkeypatch.setattr(runner,'pair_run',run)
    with pytest.raises(RuntimeError,match='remains blocked'):
        runner.verify(tmp_path,SimpleNamespace(max_seconds=600))
    assert all(not v['passed'] for v in runner.read(tmp_path/'verification.json').values())


def test_verification_blocks_main_when_no_grid_is_accepted(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'pair_run',lambda *a,**kw:[dict(backend=b,usable=False) for b in ('native','cantera')])
    with pytest.raises(RuntimeError,match='remains blocked'):
        runner.verify(tmp_path,SimpleNamespace(max_seconds=600))
    assert all(not v['passed'] for v in runner.read(tmp_path/'verification.json').values())


def seed_extension(tmp_path,monkeypatch):
    """Synthetic convergence sequences, with CH4 passing L6 and H2 passing L7."""
    ext=extension.runner
    monkeypatch.setattr(ext,'environment',lambda:dict(test='synthetic'))
    ext.ensure_manifest(tmp_path,SimpleNamespace(pairs=5,phase='verify',max_seconds=600,resume=True))
    calls=[]
    def run(out,phase,c,repetition,s,variants,ordinal):
        calls.append((phase,c['id'],dict(s)))
        values=([1.1,1.06,1.03,1.008,1.004,1.0025,1.0015,1.001]
                if c['fuel']=='CH4' else [1.1,1.06,1.03,1.012,1.006,1.003,1.0015,1.0008])
        value=values[s['level']]+(1.e-6 if phase.startswith('domain') else 0)
        folder=out/phase/c['id']/'pair-00/attempt-001'
        rows=[]
        for b in variants:
            row=dict(backend=b,variant=b,usable=True,accepted=True,Su=value,width=s['width'],
                     settings=s,phase=phase,case_id=c['id'],repetition=0)
            ext.atomic_json(folder/b/'result.json',row); rows.append(row)
        ext.atomic_json(folder/'pair.json',dict(status='complete',order=variants,settings=s))
        return rows
    for fuel in ('CH4','H2'):
        for level in range(6):
            for c in (ext.case(fuel),ext.case(fuel,pressure=10),ext.case(fuel,temperature=500)):
                run(tmp_path,f'verify-L{level}',c,0,ext.settings(level),['native','cantera'],0)
    ext.atomic_json(tmp_path/'verification.json',{f:dict(passed=False,settings=None,level_reached=5) for f in ('CH4','H2')})
    calls.clear()
    monkeypatch.setattr(ext,'pair_run',run)
    return calls


def test_extension_reuses_all_base_pairs_and_preserves_manifest(tmp_path,monkeypatch):
    calls=seed_extension(tmp_path,monkeypatch)
    before=(tmp_path/'manifest.json').read_bytes()
    argv=['--output',str(tmp_path),'--resume']
    assert extension.main(argv)==0
    selected=runner.read(tmp_path/'verification.json')
    assert selected['CH4']['settings']['level']==6
    assert selected['H2']['settings']['level']==7
    assert all(v['settings']['max_points']==16000 and v['settings']['max_seconds']==1800 for v in selected.values())
    assert len(calls)==15  # Nine new mesh pairs and six domain pairs.
    assert all(s['level']>=6 for _,_,s in calls)
    assert (tmp_path/'manifest.json').read_bytes()==before
    assert extension.main(argv)==0
    assert len(calls)==15
    with pytest.raises(ValueError,match='Incompatible extension'):
        extension.main(argv+['--max-points','18000'])


def test_extension_dry_run_never_writes_or_solves(tmp_path,monkeypatch):
    calls=seed_extension(tmp_path,monkeypatch)
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert extension.main(['--output',str(tmp_path),'--dry-run'])==0
    after={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after and not calls


def test_extension_retains_completed_failure_and_stays_blocked(tmp_path,monkeypatch):
    calls=seed_extension(tmp_path,monkeypatch)
    saved_run=extension.runner.pair_run
    def reject(out,phase,c,repetition,s,variants,ordinal):
        rows=saved_run(out,phase,c,repetition,s,variants,ordinal)
        for row in rows:
            row['usable']=False
            runner.atomic_json(out/phase/c['id']/'pair-00/attempt-001'/row['backend']/'result.json',row)
        return rows
    monkeypatch.setattr(extension.runner,'pair_run',reject)
    argv=['--output',str(tmp_path),'--resume']
    assert extension.main(argv)==2
    count=len(calls)
    assert extension.main(argv)==2 and len(calls)==count
    assert all(not v['passed'] for v in runner.read(tmp_path/'verification.json').values())


def test_extension_rejects_saved_pair_settings_change(tmp_path,monkeypatch):
    seed_extension(tmp_path,monkeypatch)
    path=next(tmp_path.glob('verify-L5/*/pair-00/attempt-001/pair.json'))
    value=runner.read(path); value['settings']['slope']=.1
    runner.atomic_json(path,value)
    with pytest.raises(ValueError,match='Incompatible saved settings'):
        extension.main(['--output',str(tmp_path),'--dry-run'])


def test_mesh_review_distinguishes_one_from_two_changes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    review=module('mesh_review','tools/plot_flame_verification.py')
    rows=[]
    for fuel in ('CH4','H2'):
        for p,t in review.STATES:
            for backend in ('native','cantera'):
                c=runner.case(fuel,pressure=p,temperature=t)
                for level,speed in enumerate([1.02,1.01,1.006,1.002,1.001]):
                    rows.append(dict(phase=f'verify-L{level}',case_id=c['id'],backend=backend,
                        condition=c,settings=runner.settings(level),usable=True,Su=speed))
    _,summary=review.analyze(rows)
    assert all(r['two_changes_below_0p5'] for r in summary if r['level']==3)
    # A missing/failed previous level must not be bypassed to claim two passes.
    rows[2]['usable']=False
    _,summary=review.analyze(rows)
    result=next(r for r in summary if r['fuel']=='CH4' and r['level']==4)
    assert result['max_last_change_percent']<.5
    assert not result['two_changes_below_0p5']
