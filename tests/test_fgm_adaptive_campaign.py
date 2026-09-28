"""Adaptive orchestration, persistence and offline reporting; fixtures are synthetic."""
import dataclasses
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_fgm_adaptive_campaign as campaign
import postprocess_fgm_adaptive_campaign as post


def test_twenty_states_four_transports_without_repetitions():
    conditions = campaign.design()
    assert len(conditions) == len({c['id'] for c in conditions}) == 80
    states = {(c['T_in_K'], c['p_atm']) for c in conditions}
    assert len(states) == 20
    assert sum(c['sweep']=='base' for c in conditions) == 4
    for t, p in states:
        group = [c for c in conditions if c['T_in_K']==t and c['p_atm']==p]
        assert {(c['transport_model'],c['soret_enabled']) for c in group} == {
            ('mixture-averaged',False),('mixture-averaged',True),
            ('multicomponent',False),('multicomponent',True)}
        assert t == 300 or p == 1
        for c in group:
            s = campaign.settings_for({'settings': {}}, c, Path('campaign'))
            assert s['T_in']==t and s['P']==p*101325
            assert s['transport_model']==c['transport_model'] and s['soret_enabled']==c['soret_enabled']


def test_bridge_selection_matches_native_priority_and_log_midpoints():
    phi = [.7,.9,1.,1.1,1.4]
    np.testing.assert_allclose(campaign.select_bridges(phi,[.02,.09,.01,.08],.01,2),
                               np.sqrt([.9*1.,1.1*1.4]))
    assert campaign.select_bridges(phi,[.01]*4,.01,4) == []
    with pytest.raises(ValueError): campaign.select_bridges(phi,[np.nan]*4,.01,4)


def test_ten_insertions_cap_and_fewer_when_only_some_intervals_fail():
    assert campaign.make_manifest()['adaptation']['max_bridges'] == 10
    phi = np.linspace(.7, 1.4, 16)
    scores = np.arange(1, 16, dtype=float)
    bridges = campaign.select_bridges(phi, scores, .01, 10)
    np.testing.assert_allclose(bridges, np.sqrt(phi[5:-1]*phi[6:]))
    assert len(campaign.select_bridges(phi, scores, 12., 10)) == 3


def test_raw_progress_rejects_reversal_and_degenerate_span():
    assert campaign.raw_coordinate_audit(SimpleNamespace(beta=np.array([0.,.5,1.])))['valid']
    for beta in ([0,.7,.6,1],[1,1,1],[0,float('nan'),1]):
        assert not campaign.raw_coordinate_audit(SimpleNamespace(beta=np.array(beta)))['valid']


@dataclasses.dataclass
class SyntheticRecord:
    phi: float
    beta: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.,.5,1.]))
    solve_ok: bool = True
    final_accepted: bool = True
    requested: bool = True
    bridge: bool = False


def install_fake_native(monkeypatch):
    from kflame.fgm import generate as g, refine
    calls = []
    monkeypatch.setattr(g,'configure_numba_kinetics_threads',lambda _: 4)
    monkeypatch.setattr(g,'load_mechanism',lambda _: SimpleNamespace(species_names=['CO2']))
    monkeypatch.setattr(g,'make_solve_options',lambda _: None)
    def solve(phi,args,mech,opts,weights,prev_solution=None,prev_prev_solution=None):
        calls.append((phi,prev_solution,prev_prev_solution))
        return SyntheticRecord(phi),dict(phi=phi),dict(predictor_kind='synthetic')
    monkeypatch.setattr(g,'solve_flame_native',solve)
    monkeypatch.setattr(campaign,'tabulate',lambda records,*args: (
        dict(phi_grid=np.array([r.phi for r in records])),dict(valid=True)))
    monkeypatch.setattr(refine,'_defects',lambda table: (
        np.array([.3,.2]) if len(table['phi_grid'])==3 else np.zeros(len(table['phi_grid'])-1), []))
    return calls


def fake_manifest():
    return dict(settings=dict(progress_species='CO2:1',n_c=61),
                adaptation=dict(initial_phi=[.9,1.,1.1],target_defect=.1,max_bridges=4,max_rounds=2,max_flames=20))


def test_construct_solves_insertions_reuses_only_internal_states_and_excludes_warmup(tmp_path,monkeypatch):
    calls = install_fake_native(monkeypatch)
    result = campaign.construct(fake_manifest(),campaign.design(True)[0],tmp_path,tmp_path/'attempt',campaign.Timings())
    assert result['status']=='accepted' and result['n_flames']==5 and result['n_inserted']==2
    assert len(calls)==8  # three warmup + three initial + two inserted, never 44
    assert calls[0][1] is None and calls[3][1] is None and calls[3][2] is None
    assert calls[4][1] is not calls[1][1]  # no retained warmup state
    assert calls[-1][2]['phi']==pytest.approx(np.sqrt(.9))
    traces=json.loads((tmp_path/'attempt/trace.json').read_text())
    assert len(traces)==5 and traces[0]['previous_phi'] is None
    assert len(list((tmp_path/'attempt/profiles').glob('*.npz')))==5
    assert len(list((tmp_path/'attempt/round_tables').glob('*.npz')))==2
    assert result['compute_s']==pytest.approx(sum(result[k] for k in campaign.Timings().values))


def test_budget_exhaustion_keeps_partial_profiles_but_publishes_no_final_table(tmp_path,monkeypatch):
    install_fake_native(monkeypatch)
    manifest=fake_manifest();manifest['adaptation']['max_rounds']=0
    with pytest.raises(RuntimeError,match='budget exhausted'):
        campaign.construct(manifest,campaign.design(True)[0],tmp_path,tmp_path,campaign.Timings())
    assert len(list((tmp_path/'profiles').glob('*.npz')))==3
    assert (tmp_path/'round_tables/000.npz').exists() and not (tmp_path/'fgm_table.npz').exists()


def test_failed_timed_operation_still_records_cost():
    timer=campaign.Timings()
    with pytest.raises(RuntimeError):
        with timer.measure('family_s'): raise RuntimeError('synthetic failure')
    assert timer.summary()['compute_s']>0


def test_resume_retains_failure_without_silent_retry(tmp_path,monkeypatch):
    attempt=tmp_path/'cases/test/attempt-001'
    campaign.atomic(attempt/'result.json',dict(status='failed',reason='synthetic'))
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *a,**kw: pytest.fail('Must not run'))
    assert not campaign.run_case(tmp_path,'test')


def test_restart_preserves_interrupted_attempt(tmp_path,monkeypatch):
    old=tmp_path/'cases/test/attempt-001'
    campaign.atomic(old/'state.json',dict(status='interrupted'))
    (old/'evidence.txt').write_text('keep')
    class Process:
        def __init__(self,command,**kwargs):
            folder=Path(command[command.index('--attempt')+1])
            assert folder.name=='attempt-002'
            campaign.atomic(folder/'result.json',dict(status='accepted'))
            self.stdout=io.StringIO('FGM synthetic fixture\n')
        def wait(self): return 0
    monkeypatch.setattr(campaign.subprocess,'Popen',Process)
    assert campaign.run_case(tmp_path,'test')
    assert (old/'evidence.txt').read_text()=='keep'


def test_incompatible_manifest_is_not_overwritten(tmp_path):
    campaign.atomic(tmp_path/'manifest.json',dict(configuration='old'))
    with pytest.raises(ValueError,match='changed'):
        campaign.ensure_manifest(tmp_path,dict(configuration='new'),True)
    assert json.loads((tmp_path/'manifest.json').read_text())==dict(configuration='old')


def test_partial_postprocessing_keeps_pending_and_never_solves(tmp_path,monkeypatch):
    from kflame.fgm import generate as g
    monkeypatch.setattr(g,'solve_flame_native',lambda *a,**kw: pytest.fail('Postprocessor invoked solver'))
    manifest=dict(kind='adaptive-fgm-sweeps',technical_check=True,conditions=campaign.design(True))
    campaign.atomic(tmp_path/'manifest.json',manifest)
    case=manifest['conditions'][0]
    campaign.atomic(tmp_path/'cases'/case['id']/'attempt-001/result.json',dict(status='accepted',
        n_flames=5,n_inserted=2,rounds=1,max_defect=.03,compute_s=1.2,family_s=1.1))
    assert post.main(['--input',str(tmp_path)])==0
    status=json.loads((tmp_path/'report/status.json').read_text())
    assert status['accepted']==1 and status['pending']==3 and status['repetitions']==1
    assert (tmp_path/'report/01_coste_adaptativo.pdf').exists()


def test_no_campaign_postprocess_writes_nothing(tmp_path):
    missing=tmp_path/'missing'
    post.main(['--input',str(missing)])
    assert not missing.exists()
