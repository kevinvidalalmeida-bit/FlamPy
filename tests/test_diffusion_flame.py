"""Counterflow geometry, native independence and full Jacobian locality."""
import builtins
from pathlib import Path

import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism,resolve_mechanism
from kflame.flame.diffusion import DiffusionCase,DiffusionFlameProblem,upwind_gradient


def problem():
    mech=load_mechanism(resolve_mechanism('h2o2.yaml'))
    feeds=[]
    for text in ('H2','O2:1,N2:3.76'):
        y=_composition(text,mech.species_names)*mech.molecular_weights;feeds.append(y/y.sum())
    case=DiffusionCase(resolve_mechanism('h2o2.yaml'),*feeds,fuel_mass_flux=.05,oxidizer_mass_flux=.15)
    return DiffusionFlameProblem(case,mech,np.linspace(0,.02,7))


def test_donor_switches_with_axial_velocity():
    z=np.array([0.,.1,.3,.7,1.])
    m=np.array([1.,1.,0.,-1.,-1.])
    np.testing.assert_allclose(upwind_gradient(z**2,z,m),[.1,.4,1.7])


def test_colored_jacobian_matches_independent_columns():
    p=problem();x=p.pack(p.initial_fields());base=p.residual(x)
    actual=p.jacobian(x,p.residual,base).toarray()
    for column in (0,2,3,p.nv+4,3*p.nv+5,(p.n-1)*p.nv+2,len(x)-1):
        component=column%p.nv
        step=(1e-10+1e-5*abs(x[column]) if column<len(x)-1 and component>=3
              else 1e-7*max(abs(x[column]),1.))
        shifted=x.copy();shifted[column]+=step
        expected=(p.residual(shifted)-base)/step
        np.testing.assert_allclose(actual[:,column],expected,rtol=1e-9,atol=1e-7)


def test_native_diffusion_does_not_import_cantera(tmp_path,monkeypatch):
    original=builtins.__import__
    def no_reference(name,*args,**kwargs):
        if name=='cantera' or name.startswith('cantera.'):
            raise AssertionError('Native workflow imported Cantera')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_reference)
    from kflame import solve_diffusion_flame
    f=solve_diffusion_flame(mechanism='h2o2.yaml',fuel={'H2':1.},fuel_mass_flux=.05,
                           oxidizer_mass_flux=.15,refine=False,max_time=60.,output=tmp_path/'native')
    assert f['accepted'] and f['report']['maximum_scaled_residual']<1e-7
    np.testing.assert_allclose(f['m'][[0,-1]],[.05,-.15],atol=1e-10)
    np.testing.assert_allclose(f['Y'].sum(axis=0),1.,atol=1e-10)
    np.testing.assert_allclose(f['species_diffusive_flux'].sum(axis=0),0.,atol=1e-12)
    assert f['u'][0]>0. and f['u'][-1]<0. and f['Lambda']<0.
    assert 'Su' not in f
    assert (f['output']/'profiles.csv').is_file()
    with pytest.raises(ValueError,match='Continuation'):
        solve_diffusion_flame(mechanism='h2o2.yaml',fuel='H2',fuel_temperature=400.,
            initial_solution=f['output'],output=tmp_path/'wrong')


def test_invalid_counterflow_inputs_do_not_create_outputs(tmp_path):
    from kflame import solve_diffusion_flame
    with pytest.raises(ValueError,match='fuel_mass_flux'):
        solve_diffusion_flame(fuel_mass_flux=-.1,output=tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_newton_allows_a_physical_decrease_of_positive_trace_species():
    # Analytic linear problem isolates the bounds treatment. A trace species
    # must be able to decrease while remaining positive; otherwise steady
    # continuation can stall indefinitely despite a correct Newton direction.
    import time
    from types import SimpleNamespace
    from scipy.sparse import diags
    from kflame.flame.diffusion import _newton
    target=np.r_[np.tile([1.,.1,.3,1.-1e-12,1e-12],5),-1.]
    x=target.copy();x[4:-1:5]=2e-11;x[3:-1:5]=1.-2e-11
    scale=np.ones(len(x));scale[4:-1:5]=1e10
    p=SimpleNamespace(n=5,nv=5,Tref=1000.,
        residual=lambda value,**kwargs:(value-target)*scale,
        jacobian=lambda value,fun,base:diags(scale).tocsc())
    solved,ok,reason=_newton(p,x,time.perf_counter()+5.)
    assert ok,reason
    np.testing.assert_allclose(solved,target,atol=1e-15,rtol=0.)


@pytest.mark.parametrize('mechanism,composition',[('h2o2.yaml','H2:2,O2:1,N2:4'),('gri30.yaml','CH4:1,O2:2,N2:7.52')])
def test_fused_viscosity_matches_cantera(mechanism,composition):
    ct=pytest.importorskip('cantera')
    from kflame.chemistry.transport import NativeTransport
    gas=ct.Solution(resolve_mechanism(mechanism));gas.X=composition
    mech=load_mechanism(resolve_mechanism(mechanism));transport=NativeTransport(mech)
    T=np.array([300.,650.,1300.,2000.])
    X=np.tile(gas.X[:,None],(1,len(T)))
    expected=[]
    for temp in T:
        gas.TP=temp,101325.;expected.append(gas.viscosity)
    np.testing.assert_allclose(transport.viscosity(T,X),expected,rtol=2e-11)
    np.testing.assert_allclose(transport.viscosity(T,X[:,:1]),expected,rtol=2e-11)
    np.testing.assert_allclose(transport.viscosity(T,X[:,0]),expected,rtol=2e-11)
    assert transport.viscosity(T[0],X[:,0])==pytest.approx(expected[0],rel=2e-11)
