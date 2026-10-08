"""Counterflow geometry, native independence and full Jacobian locality."""
import builtins
from pathlib import Path

import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism,resolve_mechanism
from kflame.flame.diffusion import DiffusionCase,DiffusionFlameProblem,upwind_gradient


def problem(transport='mixture-averaged',soret=False,*,radiation=False,mechanism='h2o2.yaml',fuel='H2',points=7):
    mech=load_mechanism(resolve_mechanism(mechanism))
    feeds=[]
    for text in (fuel,'O2:1,N2:3.76'):
        y=_composition(text,mech.species_names)*mech.molecular_weights;feeds.append(y/y.sum())
    case=DiffusionCase(resolve_mechanism(mechanism),*feeds,fuel_mass_flux=.05,oxidizer_mass_flux=.15,
                       transport_model=transport,soret_enabled=soret,radiation_enabled=radiation,
                       radiation_scale=2. if radiation else 1.,boundary_emissivities=(.4,.8) if radiation else (0.,0.))
    return DiffusionFlameProblem(case,mech,np.linspace(0,.02,points))


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


def perturbed_state(p,seed=3):
    x=p.pack(p.initial_fields());rng=np.random.default_rng(seed)
    x[:-1]*=1+1e-3*rng.standard_normal(len(x)-1)
    x[3*p.nv]*=-1.  # reverse the axial flux at node 3: the donor switches
    return x


def assert_columns_close(actual,expected,tolerance):
    scale=abs(expected).max(axis=1,keepdims=True)
    np.testing.assert_array_less(abs(actual-expected),np.broadcast_to(tolerance*scale+1e-12,expected.shape))


@pytest.mark.parametrize('transport,soret,radiation',[('mixture-averaged',False,False),('mixture-averaged',True,True),
                                                      ('multicomponent',False,False),('multicomponent',True,True)])
def test_compiled_block_jacobian_matches_full_residual_columns(transport,soret,radiation):
    # Local recomputation (rank-one faces, analytic or grouped chemistry,
    # exact Lambda column) against column differences of the full residual.
    p=problem(transport,soret,radiation=radiation);x=perturbed_state(p)
    expected=p.jacobian(x,p.residual,p.residual(x)).toarray()
    for analytic in (False,True):
        p.analytic_chemistry=analytic
        assert_columns_close(p.jacobian_blocks(x).tocsc().toarray(),expected,3e-5)
    temperature=p.unpack(x)['T']
    fixed=lambda value:p.residual(value,fixed_temperature=temperature)
    assert_columns_close(p.jacobian_blocks(x,fixed_temperature=temperature).tocsc().toarray(),
                         p.jacobian(x,fixed,fixed(x)).toarray(),3e-5)
    # The transient matrix adds mass/dt; exact at the previous state.
    transient=lambda value:p.residual(value,previous=x,dt=1e-5)
    assert_columns_close(p.jacobian_blocks(x).tocsc(1e5).toarray(),
                         p.jacobian(x,transient,transient(x)).toarray(),3e-5)


def test_compiled_jacobian_with_a_large_mechanism_and_both_multicomponent_modes():
    p=problem('multicomponent',True,mechanism='gri30.yaml',fuel='CH4',points=6);x=perturbed_state(p)
    expected=p.jacobian(x,p.residual,p.residual(x)).toarray()
    p.multicomponent_jacobian='exact'
    exact=p.jacobian_blocks(x).tocsc().toarray()
    assert_columns_close(exact,expected,3e-5)
    p.multicomponent_jacobian='frozen'
    frozen=p.jacobian_blocks(x).tocsc().toarray()
    # Only composition columns freeze face coefficients; flow and T columns stay exact.
    columns=[j*p.nv+c for j in range(p.n) for c in range(3)]+[len(x)-1]
    np.testing.assert_allclose(frozen[:,columns],exact[:,columns],rtol=1e-9,atol=1e-9*abs(exact).max())
    assert p._frozen_multicomponent()
    p.multicomponent_jacobian='auto'
    assert p._frozen_multicomponent() and not problem('multicomponent',True)._frozen_multicomponent()


@pytest.mark.parametrize('threaded',[False,True])
def test_bordered_two_ended_factorization_matches_sparse_lu(threaded,monkeypatch):
    from scipy.sparse.linalg import splu
    from kflame.flame import block_linear,diffusion
    if threaded:
        monkeypatch.setattr(block_linear,'THREAD_NODES',0)
    p=problem(points=12);x=perturbed_state(p);blocks=p.jacobian_blocks(x)
    rhs=np.random.default_rng(0).standard_normal(len(x))
    for rdt in (0.,1e4):
        factor=diffusion.BorderedFactor(blocks,rdt)
        assert factor.sparse is None
        np.testing.assert_allclose(factor.solve(rhs),splu(blocks.tocsc(rdt)).solve(rhs),rtol=1e-8,atol=1e-10)


def test_bordered_factorization_falls_back_to_sparse_lu(monkeypatch):
    from scipy.sparse.linalg import splu
    from kflame.flame import block_linear,diffusion
    def singular(*args):
        raise np.linalg.LinAlgError('singular pivot block')
    monkeypatch.setattr(block_linear,'twisted_top',singular)
    p=problem();x=perturbed_state(p);blocks=p.jacobian_blocks(x)
    factor=diffusion.BorderedFactor(blocks)
    rhs=np.arange(len(x),dtype=float)
    assert factor.sparse is not None
    np.testing.assert_allclose(factor.solve(rhs),splu(blocks.tocsc()).solve(rhs))


@pytest.mark.parametrize('mechanism,fuel',[('h2o2.yaml','H2'),('gri30.yaml','CH4')])
def test_loop_schur_thermal_system_matches_lapack(mechanism,fuel):
    # Small systems use loop kernels, large ones BLAS; both equal the
    # original LAPACK block elimination of the Dixon--Lewis system.
    from kflame.chemistry.multicomponent_kernels import (composition_workspace,temperature_terms,
                                                         thermal_schur,thermal_system,_inverse)
    from kflame.flame.diffusion_kernels import _cp_r
    p=problem('multicomponent',True,mechanism=mechanism,fuel=fuel,points=9);data=p._compiled_data()
    (log_eps,eps,crot,zrot,frot,vfit,dfit,afit,bfit,cfit,_,lo,hi,mid)=data['multi']
    mw,invW=data['mw'],data['invW'];n=mw.size;fields=p.initial_fields()
    for j in range(1,8):
        T=float(fields['T'][j]);x=np.maximum(fields['Y'][:,j]*invW/(fields['Y'][:,j]@invW),1e-20)
        terms=[np.empty(n),np.empty(n)]+[np.empty((n,n)) for _ in range(4)]
        temperature_terms(T,mw,log_eps,eps,zrot,frot,vfit,dfit,afit,bfit,cfit,*terms)
        mu,rot,bdiff,a,b,c=terms;cp=np.empty(n);_cp_r(T,lo,hi,mid,cp)
        l00=np.array([[16*T/25*x[i]*(mw[i]*sum(x[q]/bdiff[k,q] for q in range(n) if q!=k)/mw[k]+x[k]/bdiff[k,i])
                       if i!=k else 0. for i in range(n)] for k in range(n)])
        inverse=np.linalg.solve(l00,np.eye(n))
        np.testing.assert_allclose(_inverse(l00),inverse,rtol=1e-10,atol=1e-12*abs(inverse).max())
        expected,lam=thermal_system(T,x,cp,bdiff,a,b,c,rot,mw,crot,mu,l00,inverse,False)
        actual=np.empty(n)
        assert thermal_schur(T,x,cp,bdiff,a,b,c,rot,mw,crot,mu,inverse,actual,composition_workspace(n))==pytest.approx(lam,rel=1e-12)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12*abs(expected).max())


def test_radiation_is_the_optically_thin_flow1d_loss_on_interior_energy_rows():
    from kflame.flame.diffusion_kernels import STEFAN_BOLTZMANN,radiation_profile
    p=problem(radiation=True);adiabatic=problem();x=perturbed_state(p);f=p.unpack(x)
    names=list(p.mech_data.species_names);X=p.thermo.Y_to_X(f['Y'])
    inverse=1000./f['T']
    planck=sum(c*inverse**n for n,c in enumerate((-0.23093,-1.12390,9.41530,-2.99880,0.51382,-1.86840e-5)))
    kP=X[names.index('H2O')]*planck  # h2o2.yaml has no CO2; P = 1 atm
    background=STEFAN_BOLTZMANN*(.4*p.case.fuel_temperature**4+.8*p.case.oxidizer_temperature**4)
    expected=2.*2.*kP*(2.*STEFAN_BOLTZMANN*f['T']**4-background)
    np.testing.assert_allclose(radiation_profile(f['T'],np.ascontiguousarray(f['Y']),p.radiation),expected,rtol=1e-12)
    change=(p.residual(x)-adiabatic.residual(x))[:-1].reshape(p.n,p.nv)
    np.testing.assert_allclose(change[1:-1,2],expected[1:-1]/p.energy_scale,rtol=1e-9)
    change[1:-1,2]=0.
    np.testing.assert_array_equal(change,0.)


def test_radiative_flame_reports_its_loss_in_the_enthalpy_balance(tmp_path):
    from kflame import solve_diffusion_flame
    common=dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15,refine=False,max_time=60.)
    adiabatic=solve_diffusion_flame(**common,output=tmp_path/'adiabatic')
    radiative=solve_diffusion_flame(**common,radiation=True,radiation_scale=5.,output=tmp_path/'radiative')
    report=radiative['report']
    assert radiative['accepted'] and report['radiation']['enabled'] and report['radiation']['scale']==5.
    # A cold radiative start first converges the adiabatic burning branch.
    assert report['history'][0]['stage']=='adiabatic_start' and report['history'][0]['converged']
    assert radiative['T'].max()<adiabatic['T'].max()-5.
    assert report['radiation']['integrated_loss_W_m2']>0. and radiative['radiative_heat_loss'][[0,-1]].tolist()==[0.,0.]
    assert report['enthalpy_balance']['radiative_loss_between_first_and_last_faces_W_m2']==report['radiation']['integrated_loss_W_m2']
    assert not adiabatic['report']['radiation']['enabled'] and adiabatic['radiative_heat_loss'].max()==0.
    with pytest.raises(ValueError,match='radiation_scale'):
        solve_diffusion_flame(**common,radiation=True,radiation_scale=-1.,output=tmp_path/'invalid')
