"""Small synthetic checks; no flame simulations or changes to scientific records."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from scipy.sparse import diags

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import postprocess_flame_diagnostics as post
spec=importlib.util.spec_from_file_location('conditioning',ROOT/'benchmarks/diagnose_flame_conditioning.py')
cond=importlib.util.module_from_spec(spec);spec.loader.exec_module(cond)


def test_condition_estimator_known_diagonal():
    a=diags([.01,1.,100.],format='csc')
    assert np.isclose(cond.estimate(a)['kappa1_lower_estimate'],10000.)


def test_block_sparse_conversion_preserves_neighbor_blocks():
    j=SimpleNamespace(n_blocks=2,shape=(4,4),diag=np.array([np.eye(2)*3,np.eye(2)*5]),
                      lower=np.array([np.eye(2)*2]),upper=np.array([np.eye(2)*7]))
    assert np.array_equal(cond.sparse_blocks(j).toarray(),np.block([[np.eye(2)*3,np.eye(2)*7],[np.eye(2)*2,np.eye(2)*5]]))


def test_cantera_callbacks_not_replaced_by_retained_stats():
    r=dict(case_id='synthetic',backend='cantera',nodes=3,time_s=10.,
        instrumentation=dict(phases=[],function_calls={},exclusive_root_seconds={},
                             cantera_events=[dict(kind='transient_accepted')]*7),
        report=dict(jacobian_count_stats=[2],jacobian_time_stats=[3.],eval_count_stats=[8],
                    eval_time_stats=[1.],time_step_stats=[4]),log_counts=dict(newton_attempts=5))
    m=post.metrics(r)
    assert m['euler_accepted']==7 and m['reported_cantera_steps']==4
    assert m['euler_rejected'] is None and m['unassigned_s']==6.


def test_native_counts_rejected_ptc_and_euler_separately():
    r=dict(case_id='synthetic',backend='native',nodes=3,time_s=10.,report={},
        instrumentation=dict(phases=[dict(history=[dict(phase='transient',scheme='PTC-SER',ok=False,iterations=[{}]),
        dict(phase='transient',scheme='BE-fallback',ok=True,iterations=[{},{}])])],
        function_calls=dict(jacobian=2,residual=8),exclusive_root_seconds=dict(jacobian=3.,residual=1.,factorize=2.,linear_solve=.5)))
    m=post.metrics(r)
    assert m['ptc_rejected']==1 and m['euler_accepted']==1 and m['newton_inner_iterations']==3
    assert m['unassigned_s']==3.5


def test_profiles_and_mesh_use_the_same_reference_selection():
    def row(transport,soret,repetition,usable=True):
        return dict(phase='main',backend='native',usable=usable,repetition=repetition,
            condition=dict(fuel='H2',phi=1,temperature=300,pressure_atm=1,transport=transport,soret=soret))
    wrong=row('mixture-averaged',False,0)
    failed=row('multicomponent',True,0,False)
    chosen=row('multicomponent',True,1)
    later=row('multicomponent',True,2)
    assert post.pp.reference_record([wrong,failed,later,chosen],'H2','native') is chosen
    assert post.pp.reference_record([wrong,chosen],'CH4','native') is None


def test_cantera_damping_rows_are_not_newton_iterations():
    log="""Attempt Newton solution of steady-state problem.
  0     1.0e-03   1.0e-03    6.390    4.104    4.104    1      1/20
  1     7.0e-04   1.0e-03    6.390    4.104    4.104    1      1/20
  Damping coefficient found (solution has not converged yet)
  No damped step can be taken without violating solution component bounds.
  No damping coefficient found (max damping iterations reached)
Timestep failed--> Reducing timestep
  Damping coefficient found (solution has converged)
"""
    v=post.cantera_log_metrics(log)
    assert (v['steady_newton_accepted'],v['steady_newton_rejected'],v['euler_rejected'])==(2,2,1)
    assert v['damping_trials']==2


def test_profile_totals_count_bootstrap_once():
    r=dict(profile={'chem':dict(time_s=1.,count=4)},
           transport_bootstrap=dict(profile={'chem':dict(time_s=2.,count=7),'flux':dict(time_s=3.,count=2)}))
    v=post.profile_totals(r)
    assert v['chem']==dict(time_s=3.,count=11)
    assert v['flux']==dict(time_s=3.,count=2)
