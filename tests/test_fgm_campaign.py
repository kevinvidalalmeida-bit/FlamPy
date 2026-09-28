"""Campaign invariants and postprocessing using manufactured, labelled data."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'tools'))
import benchmark_fgm_campaign as campaign
import postprocess_fgm_campaign as post


def test_design_no_leakage():
    schedule,holdouts=campaign.design()
    phi=np.array(schedule['phi_resolved'])
    assert len(phi)==44 and sum(schedule['requested'])==5
    assert sum(schedule['bridge'])==39 and len(holdouts)==12
    assert np.all(np.diff(holdouts)>0)
    assert min(holdouts)>phi[0] and max(holdouts)<phi[-1]
    assert not np.isclose(np.array(holdouts)[:,None],phi[None,:],rtol=0,atol=1e-12).any()


def test_preflight_selects_one_family_and_holdouts_without_changing_design():
    assert campaign.selected_jobs('all',5,1)==[('family',1),('holdout',1)]
    assert campaign.selected_jobs('family',5)==[('family',i) for i in range(1,6)]
    assert campaign.selected_jobs('holdout',5)==[('holdout',1)]
    with pytest.raises(ValueError):campaign.selected_jobs('holdout',5,1)
    with pytest.raises(ValueError):campaign.selected_jobs('family',1,2)


def test_atomic_retries_windows_lock(tmp_path,monkeypatch):
    original=campaign.os.replace
    calls=[]
    def locked(a,b):
        calls.append(a)
        if len(calls)<3: raise PermissionError('simulated OneDrive transient lock')
        original(a,b)
    monkeypatch.setattr(campaign.os,'replace',locked)
    monkeypatch.setattr(campaign.time,'sleep',lambda _:None)
    campaign.atomic(tmp_path/'r.json',dict(v=np.float64(2),missing=np.nan))
    assert json.loads((tmp_path/'r.json').read_text())==dict(v=2,missing=None)
    assert not list(tmp_path.glob('*.tmp'))


def test_completed_failure_not_repeated(tmp_path,monkeypatch):
    folder=tmp_path/'family/rep-01/attempt-001'
    campaign.atomic(folder/'result.json',dict(status='failed'))
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *a,**kw: pytest.fail('Must not rerun completed failures'))
    assert not campaign.run_job(tmp_path,'family',1)
    assert len(list(folder.parent.glob('attempt-*')))==1


def test_interrupted_family_is_pending(tmp_path):
    job=tmp_path/'family/rep-01'
    campaign.atomic(job/'attempt-001/result.json',dict(status='failed'))
    campaign.atomic(job/'attempt-001/state.json',dict(status='interrupted'))
    assert campaign.completed(job) is None
    assert campaign.latest(job).name=='attempt-001'


def test_interrupted_family_restarts_in_new_attempt(tmp_path,monkeypatch):
    import io
    job=tmp_path/'family/rep-01'
    old=job/'attempt-001'
    campaign.atomic(old/'state.json',dict(status='interrupted'))
    (old/'partial_profile.npz').write_bytes(b'preserve this prior attempt')
    class Process:
        def __init__(self,command,**kwargs):
            folder=Path(command[command.index('--attempt')+1])
            assert folder.name=='attempt-002'
            campaign.atomic(folder/'result.json',dict(status='accepted'))
            self.stdout=io.StringIO('family 1/1 technical fixture\n')
        def wait(self):return 0
    monkeypatch.setattr(campaign.subprocess,'Popen',Process)
    assert campaign.run_job(tmp_path,'family',1)
    assert (old/'partial_profile.npz').read_bytes()==b'preserve this prior attempt'
    assert campaign.completed(job)['status']=='accepted'


def test_postprocess_before_campaign_leaves_output_empty(tmp_path):
    out=tmp_path/'new_campaign'
    post.main(['--input',str(out)])
    assert not out.exists()


def test_manifest_incompatible_rejected(tmp_path):
    campaign.atomic(tmp_path/'manifest.json',dict(config='old'))
    with pytest.raises(ValueError,match='changed'):
        campaign.ensure_manifest(tmp_path,dict(config='new'),True)
    assert json.loads((tmp_path/'manifest.json').read_text())['config']=='old'


def test_artifact_corruption_detected(tmp_path):
    p=tmp_path/'profile.npz';p.write_bytes(b'original')
    result=dict(artifacts={'profile.npz':campaign.digest(p)})
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed'): campaign.verify_artifacts(tmp_path,result)


def manufactured_table():
    c=np.linspace(0,1,15);z=np.array([.02,.04,.06])
    T=300+1000*c[None,:]+100*z[:,None]
    y=np.zeros((3,2,len(c)))
    y[:,0,:]=.1*c[None,:]+z[:,None]
    y[:,1,:]=.02*c[None,:]+z[:,None]/2
    return dict(c_grid=c,Z_grid=z,phi_grid=np.array([.7,1.,1.4]),T=T,Y=y,
                omega_c=20*c[None,:]+2*z[:,None],species_names=np.array(['CO2','CO']))


def test_linear_interpolation_holdout_and_normalization():
    table=manufactured_table();c=table['c_grid'];z=.03
    truth=SimpleNamespace(phi=.85,Z=z,c=c,beta=c,T=300+1000*c+100*z,
               Y=np.array([.1*c+z,.02*c+z/2]),omega_c=20*c+2*z,
               species_names=table['species_names'],solve_ok=True,final_accepted=True)
    data,metrics=post.fidelity(table,[truth])
    assert data['truth'].shape==(1,1001,4)
    assert max(r['maximum'] for r in metrics)<1e-10
    changed=dict(table,T=table['T']+10)
    _,metrics=post.fidelity(changed,[truth])
    assert metrics[0]['mean']==pytest.approx(1.)
    assert metrics[0]['p95']==pytest.approx(1.)
    truth.phi=1.
    with pytest.raises(ValueError,match='leakage'):post.fidelity(table,[truth])


def test_postprocess_no_solver_import():
    # Both modules expose workers, but importing offline postprocessing must
    # not import the native solve module or call a solve function.
    import subprocess
    code="import sys; sys.path[:0]=['tools','benchmarks']; import postprocess_fgm_campaign; assert 'kflame.fgm.generate' not in sys.modules; assert 'kflame.flame.solver' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def audit_fixture(beta):
    n=len(beta)
    return SimpleNamespace(phi=1.,Z=.05,beta=np.array(beta,dtype=float),
          T=np.linspace(300,2000,n),Y=np.array([np.linspace(0,.1,n)]),
          omega_c=np.ones(n),species_names=np.array(['CO2']))


def test_coordinate_audit_detects_ambiguous_plateau():
    row=post.coordinate_audit([audit_fixture([0,.5,.5,1])],'family',1)[0]
    assert row['backward_steps']==0
    assert row['plateau_segments']==1
    assert row['plateau_max_field_change_percent']==pytest.approx(100/3)


def test_coordinate_audit_uniform_tail_is_harmless():
    r=audit_fixture([0,.5,1,1])
    r.T[-1]=r.T[-2];r.Y[:,-1]=r.Y[:,-2]
    row=post.coordinate_audit([r],'family',1)[0]
    assert row['plateau_segments']==1
    assert row['plateau_max_field_change_percent']==0


def test_coordinate_audit_catches_raw_reversal_and_singular_scale():
    row=post.coordinate_audit([audit_fixture([0,.7,.6,1])],'holdout',1)[0]
    assert row['backward_steps']==1
    with pytest.raises(ValueError,match='Degenerate'):
        post.coordinate_audit([audit_fixture([1,1,1])],'family',1)


def test_cost_total_uses_paired_runs_and_query_throughput():
    # Median of sums differs from sum of medians for these artificial phases.
    results=[dict(setup_s=a,family_s=b,tabulation_s=1)
             for a,b in [(1,100),(100,1),(1,1)]]
    obs=[dict(batch=batch,us_per_state=value) for batch in (1,10000) for value in (2,4,8)]
    stats=post.cost_statistics(results,obs)
    assert stats['offline_total_s']['median']==102
    assert stats['query']['10000']['states_per_second']['median']==250000
    assert stats['query']['1']['us_per_state']['q25']==3
