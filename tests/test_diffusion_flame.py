"""Counterflow geometry, native independence and full Jacobian locality."""
import builtins
from pathlib import Path

import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism,resolve_mechanism
from kflame.flame.diffusion import DiffusionCase,DiffusionFlameProblem,upwind_gradient


def problem(transport='mixture-averaged',soret=False):
    mech=load_mechanism(resolve_mechanism('h2o2.yaml'))
    feeds=[]
    for text in ('H2','O2:1,N2:3.76'):
        y=_composition(text,mech.species_names)*mech.molecular_weights;feeds.append(y/y.sum())
    case=DiffusionCase(resolve_mechanism('h2o2.yaml'),*feeds,fuel_mass_flux=.05,oxidizer_mass_flux=.15,
                       transport_model=transport,soret_enabled=soret)
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


@pytest.mark.parametrize('transport,soret',[('mixture-averaged',False),('multicomponent',True)])
def test_premixed_thermal_reuse_preserves_diffusion_jacobian(transport,soret):
    p=problem(transport,soret);x=p.pack(p.initial_fields());base=p.residual(x)
    cached=p.properties(p.unpack(x))
    actual=p.jacobian(x,p.residual,base,
        flow_fun=lambda value:p.residual(value,cached_properties=cached),
        thermo_fun=lambda value,thermal:p.residual(value,nodal_thermo=thermal),base_thermo=cached[:4])
    expected=p.jacobian(x,p.residual,base)
    np.testing.assert_allclose(actual.toarray(),expected.toarray(),rtol=2e-8,atol=2e-6)


def test_thermal_banks_release_previous_color_and_budget_fallback_is_exact(monkeypatch):
    import weakref
    from kflame.flame import diffusion
    p=problem();x=p.pack(p.initial_fields());base=p.residual(x);cached=p.properties(p.unpack(x))
    original=p._perturbed_node_thermo;references=[]
    def checked_bank(value,nodes):
        assert all(reference() is None for reference in references)
        bank=original(value,nodes)
        references[:]=[weakref.ref(array) for array in bank]
        return bank
    monkeypatch.setattr(p,'_perturbed_node_thermo',checked_bank)
    options=dict(flow_fun=lambda value:p.residual(value,cached_properties=cached),
        thermo_fun=lambda value,thermal:p.residual(value,nodal_thermo=thermal),base_thermo=cached[:4])
    grouped=p.jacobian(x,p.residual,base,**options)
    assert all(reference() is None for reference in references)
    monkeypatch.setattr(diffusion,'_THERMO_BATCH_BUDGET_BYTES',0)
    ordinary=p.jacobian(x,p.residual,base,**options)
    np.testing.assert_allclose(grouped.toarray(),ordinary.toarray(),rtol=2e-8,atol=2e-6)


@pytest.mark.parametrize('z',[np.array([0.,.5,1.]),np.array([0.,.1,.4,.8,1.])])
def test_enthalpy_diagnostic_is_compatible_with_continuity_and_its_reference(z):
    from kflame.flame.diffusion import enthalpy_face_balance
    rho=np.ones(len(z));V=np.sin(np.pi*z)
    m=np.r_[.5,.5-np.cumsum(np.diff(z)*(V[:-1]+V[1:]))]
    flux=np.zeros(len(z)-1)
    for offset in (0.,1.,1e6):
        _,_,_,_,defect=enthalpy_face_balance(z,m,rho,V,np.full(len(z),offset),flux)
        assert abs(defect)<1e-9
    h=1000.*z**2
    ordinary=enthalpy_face_balance(z,m,rho,V,h,flux)[-1]
    shifted=enthalpy_face_balance(z,m,rho,V,h+1e6,flux)[-1]
    assert shifted==pytest.approx(ordinary,abs=1e-9)


@pytest.mark.parametrize('transient',[False,True])
def test_newton_accepts_a_root_on_the_last_allowed_iteration(transient):
    import time
    from types import SimpleNamespace
    from scipy.sparse import eye
    from kflame.flame.diffusion import _newton
    target=np.array([0.,0.,1.,.5,0.]);start=target+np.array([1.,1.,0.,0.,1.])
    p=SimpleNamespace(n=1,nv=4,Tref=300.,
        residual=lambda value,**kwargs:value-target,
        jacobian=lambda *args:eye(len(target),format='csc'))
    options=dict(previous=start,dt=1e-4) if transient else {}
    solved,ok,reason=_newton(p,start,time.perf_counter()+5.,max_iterations=1,**options)
    assert ok,reason
    np.testing.assert_array_equal(solved,target)


@pytest.mark.parametrize('transport,soret',[('mixture-averaged',False),('mixture-averaged',True),
                                          ('multicomponent',False),('multicomponent',True)])
def test_cached_flow_columns_are_exact_and_include_a_donor_switch(transport,soret,monkeypatch):
    p=problem(transport,soret);x=p.pack(p.initial_fields())
    cached=p.properties(p.unpack(x));shifted=x.copy()
    shifted[3*p.nv]*=-1.
    shifted[2*p.nv+1]*=1.01;shifted[-1]*=1.03
    expected=p.residual(shifted)
    def no_refresh(*args,**kwargs):
        raise AssertionError('A flow-only perturbation refreshed thermochemical properties')
    monkeypatch.setattr(p,'properties',no_refresh)
    np.testing.assert_array_equal(p.residual(shifted,cached_properties=cached),expected)


@pytest.mark.parametrize('transport,soret',[('mixture-averaged',False),('mixture-averaged',True),
                                          ('multicomponent',False),('multicomponent',True)])
def test_residual_conserves_species_mass_and_honors_fixed_temperature(transport,soret):
    p=problem(transport,soret);fields=p.initial_fields();x=p.pack(fields)
    residual=p.residual(x)[:-1].reshape(p.n,p.nv)
    # Sum of species transport/reaction equations vanishes for sum(Y)=1,
    # independently of the unconverged flow/temperature seed.
    np.testing.assert_allclose(residual[1:-1,3:].sum(axis=1),0.,atol=1e-8)
    fixed=p.residual(x,previous=x*1.0001,dt=1e-4,fixed_temperature=fields['T'])
    np.testing.assert_allclose(fixed[:-1].reshape(p.n,p.nv)[1:-1,2],0.,atol=0.)
    for boundary in (0,p.n-1):
        changed=x.copy();changed[boundary*p.nv+3+p.excess[boundary]]+=.001
        row=p.residual(changed)[boundary*p.nv+3+p.excess[boundary]]
        assert row==pytest.approx(.001,abs=1e-14)


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
