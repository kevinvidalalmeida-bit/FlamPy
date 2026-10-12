"""Explicit per-Newton transport coefficients never cross states or owners."""
import gc
import time
import weakref

import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion


def make(mechanism='jetsurf2.yaml'):
    path=resolve_mechanism(mechanism)
    mech=load_mechanism(path)
    fuel='H2:1' if mechanism=='h2o2.yaml' else 'CH4:1,N2:4'
    feeds=[]
    for text in (fuel,'O2:1,N2:3.76'):
        y=_composition(text,mech.species_names)*mech.molecular_weights
        feeds.append(y/y.sum())
    # The face bank belongs to the complete Dixon--Lewis kernel.
    p=diffusion.DiffusionFlameProblem(diffusion.DiffusionCase(path,*feeds,
         transport_model='multicomponent',soret_enabled=True,multicomponent_trace_tail=None),mech,np.linspace(0.,.02,9))
    fields=p.initial_fields()
    fields['T'][:]=np.linspace(600.,2300.,9)
    y=np.arange(1.,p.ns+1.);y/=y.sum()
    fields['Y'][:]=y[:,None]
    return p,p.pack(fields)


def exact(actual,expected):
    assert actual.dtype==expected.dtype and actual.shape==expected.shape
    assert actual.strides==expected.strides
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('soret',[False,True])
def test_bank_matches_ordinary_cycle_and_rejects_stale_state(soret):
    p,x=make()
    p.soret_enabled=soret
    original=x.tobytes()
    fields=p.unpack(x)
    props,bank=p.properties_for_newton(fields)
    assert bank is not None and len(props)==7
    for actual,expected in zip(props,p.properties(fields)):
        exact(actual,expected)
    assert sum(bank[key].nbytes for key in ('multi','thermal','lambda'))==8*8*(p.ns*p.ns+p.ns+1)
    for key in ('multi','thermal','lambda','T','Y','z','dz'):
        assert not bank[key].flags.writeable
        assert not np.shares_memory(bank[key],x)
    if not soret:
        assert not np.any(bank['thermal'])
    actual=p.jacobian_blocks(x,cached_properties=props,transport_bank=bank)
    reference=p.jacobian_blocks(x,cached_properties=props)
    for key in ('lower','diag','upper','border','mass'):
        exact(getattr(actual,key),getattr(reference,key))
    changed=x.copy();changed[2]+=1e-4
    fields_b=p.unpack(changed)
    props_b=p.properties(fields_b)
    assert not diffusion._valid_face_bank(p,bank,fields_b['T'],fields_b['Y'],props_b)
    fallback=p.jacobian_blocks(changed,cached_properties=props_b,transport_bank=bank)
    reference_b=p.jacobian_blocks(changed,cached_properties=props_b)
    for key in ('lower','diag','upper','border','mass'):
        exact(getattr(fallback,key),getattr(reference_b,key))
    assert x.tobytes()==original


def test_bank_rejects_owner_configuration_grid_and_cap(monkeypatch):
    p,x=make()
    fields=p.unpack(x)
    props,bank=p.properties_for_newton(fields)
    other,_=make()
    assert not diffusion._valid_face_bank(other,bank,fields['T'],fields['Y'],props)
    old=p.P;p.P*=1.01
    assert not diffusion._valid_face_bank(p,bank,fields['T'],fields['Y'],props)
    p.P=old
    old=p.dz.copy();p.dz[0]*=1.01
    assert not diffusion._valid_face_bank(p,bank,fields['T'],fields['Y'],props)
    p.dz[:]=old
    for attr,value in (('analytic_chemistry',False),('chemistry_tile_nodes',0),
                       ('compiled_faces',False),('multicomponent_jacobian','exact')):
        old=getattr(p,attr);setattr(p,attr,value)
        assert not diffusion._valid_face_bank(p,bank,fields['T'],fields['Y'],props)
        setattr(p,attr,old)
    monkeypatch.setattr(diffusion,'_FACE_BANK_BUDGET_BYTES',0)
    ordinary,disabled=p.properties_for_newton(fields)
    assert disabled is None
    for actual,expected in zip(ordinary,props):
        exact(actual,expected)


@pytest.mark.parametrize('max_age,refresh_rate,expected_banks',[
    (1,None,[1.,-.5,.25]),(4,None,[1.]),(4,0.,[1.,None]),
])
def test_newton_bank_lifetime_rejection_aging_and_unexpected_refresh(
        monkeypatch,max_age,refresh_rate,expected_banks):
    p,x=make('h2o2.yaml');x[0]=1.
    model_banks=[];exports=[];references=[]
    def properties(fields):
        return (float(fields['m'][0]/p.mref),)
    def bank_properties(fields):
        props=properties(fields)
        witness=np.array([props[0]])
        references.append(weakref.ref(witness));exports.append(props[0])
        return props,dict(marker=props[0],witness=witness)
    def residual(value,**options):
        assert options['cached_properties'][0]==value[0]
        result=np.zeros_like(value);result[0]=value[0]
        return result
    class Factor:
        def solve(self,value):
            return 3.*value
    class Model:
        def __init__(self,problem,value,fixed,*,cached_properties=None,transport_bank=None):
            assert cached_properties[0]==value[0]
            if transport_bank is not None:
                assert transport_bank['marker']==value[0]
            model_banks.append(None if transport_bank is None else transport_bank['marker'])
            self.age=0;self.fixed=False
            self.blocks=type('Blocks',(),{'n':problem.n})()
        def factor(self,rdt):
            return Factor()
    monkeypatch.setattr(diffusion,'_face_bank_available',lambda problem:True)
    monkeypatch.setattr(p,'properties',properties)
    monkeypatch.setattr(p,'properties_for_newton',bank_properties)
    monkeypatch.setattr(p,'residual',residual)
    monkeypatch.setattr(diffusion,'_LinearModel',Model)
    _,accepted,reason=diffusion._newton(p,x,time.perf_counter()+10.,max_iterations=3,
                      max_age=max_age,refresh_rate=refresh_rate)
    assert not accepted and reason=='iteration_limit'
    assert model_banks==expected_banks
    if max_age==4:
        assert exports==[1.]  # Aged Jacobians never build trial banks.
    else:
        assert -2. in exports and -2. not in model_banks # Rejected full trial.
    gc.collect()
    assert all(ref() is None for ref in references)


def test_bank_rejects_signed_zero_and_malformed_packet():
    p,x=make()
    fields=p.unpack(x)
    fields['Y'][0,:]=0.
    x=p.pack(fields)
    fields=p.unpack(x)
    props,bank=p.properties_for_newton(fields)
    assert diffusion._valid_face_bank(p,bank,fields['T'],fields['Y'],props)
    changed=x.copy()
    state=p.unpack(changed)
    state['Y'][0,4]=-0.
    assert np.array_equal(fields['Y'],state['Y'])
    assert not diffusion._valid_face_bank(p,bank,state['T'],state['Y'],props)
    missing=bank.copy();del missing['multi']
    assert not diffusion._valid_face_bank(p,missing,fields['T'],fields['Y'],props)
    wrong_owner=bank.copy();wrong_owner['owner']=object()
    assert not diffusion._valid_face_bank(p,wrong_owner,fields['T'],fields['Y'],props)
