"""Reduced solver campaign and concordance; manufactured data clearly scoped."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'tools'))
import benchmark_fgm_solver_comparison as comparison
import benchmark_fgm_adaptive_campaign as adaptive
import postprocess_fgm_solver_comparison as post
from fgm_cantera_adapter import CanteraEngine


def test_eight_base_conditions_and_alternating_order():
    cases=comparison.design()
    assert len(cases)==len({c['id'] for c in cases})==8
    assert {c['T_in_K'] for c in cases}=={300} and {c['p_atm'] for c in cases}=={1}
    for i in range(4):
        pair=cases[2*i:2*i+2]
        assert len({c['transport'] for c in pair})==1
        assert {c['solver'] for c in pair}=={'kflame','cantera'}
        assert pair[0]['solver']==('kflame' if i%2==0 else 'cantera')


def test_manifest_records_both_engines_and_the_adaptive_rules():
    m=comparison.make_manifest()
    assert m['repetitions']==1 and len(m['conditions'])==8
    assert m['adaptation']['max_bridges']==10
    assert m['adaptation']['initial_phi']==[.7,.9,1.,1.1,1.4]
    assert m['settings']['n_c']==241 and 'cantera' in m['environment']['versions']
    assert 'benchmarks/fgm_cantera_adapter.py' in m['code']


def test_cantera_properties_match_progress_sources_and_inlet(monkeypatch):
    """Exercise the real adapter against a tiny deterministic FreeFlame fixture."""
    import cantera as ct
    from types import SimpleNamespace
    gas=ct.Solution(str(ROOT/'src/kflame/chemistry/data/gri30.yaml'))
    args=SimpleNamespace(mech='',fuel='CH4',oxidizer='O2:1,N2:3.76',T_in=300.,P=101325.,
        width=.03,initial_grid_points=8,transport_model='multicomponent',flux_gradient_basis='molar',
        soret_enabled=True,ratio=2.5,slope=.04,curve=.08,prune=.003,max_grid_points=1600,
        grid_min=0,max_flame_time_s=300,loglevel=0)
    class FakeFlame:
        def __init__(self,gas,grid):
            self.inlet=SimpleNamespace(T=gas.T,Y=gas.Y.copy());self.P=gas.P
            self.flame=self;self.grid=np.array([0.,.015,.03]);self.T=np.array([300.,1000.,2200.])
            self.Y=np.repeat(gas.Y[:,None],3,axis=1)
            co2=gas.species_index('CO2');n2=gas.species_index('N2')
            self.Y[co2]+=[0,.05,.1];self.Y[n2]-=[0,.05,.1]
            self.velocity=np.array([.4,1.,2.]);self.density=np.ones(3)
            self.cp_mass=np.ones(3)*1000;self.thermal_conductivity=np.ones(3)*.05
            self.heat_release_rate=np.array([0,100,0]);self.net_production_rates=np.ones_like(self.Y)
            for name in ('grid_size_stats','jacobian_count_stats','jacobian_time_stats','eval_count_stats','eval_time_stats','time_step_stats'):setattr(self,name,[1])
        def set_steady_tolerances(self,**kw):pass
        def set_transient_tolerances(self,**kw):pass
        def set_refine_criteria(self,**kw):self.criteria=kw
        def set_max_grid_points(self,*a):pass
        def set_interrupt(self,*a):pass
        def solve(self,**kw):assert kw['auto'] and kw['refine_grid']
        def to_array(self):return 'synthetic_solution'
        def get_refine_criteria(self):return self.criteria
    monkeypatch.setattr(ct,'FreeFlame',FakeFlame)
    rec,state,trace=CanteraEngine.solve_flame_native(1.,args,gas,CanteraEngine.make_solve_options(args),{'CO2':1.})
    assert rec.final_accepted and np.isnan(rec.residual_inf)
    np.testing.assert_allclose(rec.omega_c,gas.molecular_weights[gas.species_index('CO2')]/.1)
    assert trace['effective_options']['soret_enabled'] and state['solution']=='synthetic_solution'


def test_identical_tables_have_zero_concordance_error(tmp_path):
    z=np.array([.02,.05,.08]);c=np.linspace(0,1,9)
    table=dict(Z_grid=z,c_grid=c,species_names=np.array(['CO2','CO']),
               T=300+z[:,None]*100+c[None,:]*1000,
               Y=np.stack([np.tile(c,(3,1))*.1,np.tile(c,(3,1))*.01],axis=1),
               omega_c=np.tile(c,(3,1))*20)
    for name in ('k','c'):np.savez(tmp_path/f'{name}.npz',**table)
    assert max(r['maximum_percent'] for r in post.compare_tables(tmp_path/'k.npz',tmp_path/'c.npz'))==0


def test_offline_partial_report_never_invokes_solver(tmp_path,monkeypatch):
    monkeypatch.setattr(CanteraEngine,'solve_flame_native',lambda *a,**kw:pytest.fail('Solver invoked'))
    m=dict(kind='fgm-solver-comparison',conditions=comparison.design(),technical_check=True)
    adaptive.atomic(tmp_path/'manifest.json',m)
    first=m['conditions'][0]
    adaptive.atomic(tmp_path/'cases'/first['id']/'attempt-001/result.json',dict(status='accepted',
        n_flames=5,n_inserted=2,rounds=1,max_defect=.03,compute_s=2.,family_s=1.8,
        tabulation_s=.1,selection_s=.01))
    post.main(['--input',str(tmp_path)])
    status=json.loads((tmp_path/'report/status.json').read_text())
    assert status['accepted']==1 and status['pending']==7
    assert (tmp_path/'report/01_coste_comparado.pdf').exists()
