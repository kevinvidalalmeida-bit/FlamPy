"""A-B-A assembly against ordinary transport, with and without eligible faces."""
import hashlib
import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion, diffusion_kernels


@pytest.mark.parametrize('name', ['jetsurf2.yaml', 'aramcomech2.yaml', 'aramcomech3.yaml'])
@pytest.mark.parametrize('profile,soret,fixed', [('uniform', True, False), ('hot', False, True)])
def test_temperature_coefficients_preserve_uncached_assembly(monkeypatch,name,profile,soret,fixed):
    try:
        path=resolve_mechanism(name)
    except FileNotFoundError:
        pytest.skip('External mechanism is not installed')
    mech=load_mechanism(path)
    feeds=[]
    for text in ('CH4:1,N2:4','O2:1,N2:3.76'):
        y=_composition(text,mech.species_names)*mech.molecular_weights
        feeds.append(y/y.sum())
    case=diffusion.DiffusionCase(path,*feeds,transport_model='multicomponent',
        soret_enabled=soret,radiation_enabled=True,boundary_emissivities=(.2,.7))
    n=9
    problem=diffusion.DiffusionFlameProblem(case,mech,np.linspace(0.,1.,n)**1.3*.02)
    problem.multicomponent_jacobian='frozen'
    monkeypatch.setattr(diffusion.numba,'get_num_threads',lambda:4)
    fields=problem.initial_fields()
    fields['T'][:]=1000. if profile=='uniform' else np.linspace(1.1*problem.Tref,2300.,n)
    y=np.arange(1.,problem.ns+1.);y/=y.sum()
    for j in range(n):
        column=y*(1.+j/n*y)
        fields['Y'][:,j]=column/column.sum()
    first=problem.pack(fields)
    fields['T'][:]=np.linspace(999.99,1500.,n)
    fields['Y'][:]=fields['Y'][:,::-1]
    second=problem.pack(fields)
    screen=diffusion_kernels._temperature_cache_eligible
    cached_range=diffusion_kernels.jacobian_range_temperature_cache
    cache_calls=[]

    def observed_range(*args):
        cache_calls.append((args[0],args[1]))
        return cached_range(*args)

    def cycle(x):
        state=x.copy()
        temperature=problem.unpack(first)['T'].copy() if fixed else None
        props=problem.properties(problem.unpack(state))
        residual=problem.residual(state,cached_properties=props,fixed_temperature=temperature)
        blocks=problem.jacobian_blocks(state,cached_properties=props,fixed_temperature=temperature)
        assert state.tobytes()==x.tobytes()
        return (*props,residual,*(getattr(blocks,k) for k in
            ('lower','diag','upper','border','mass')),blocks.column())

    anchor=None
    for iteration,x in enumerate((first,second,first)):
        monkeypatch.setattr(diffusion_kernels,'_temperature_cache_eligible',lambda *args:False)
        expected=cycle(x)
        monkeypatch.setattr(diffusion_kernels,'_temperature_cache_eligible',screen)
        monkeypatch.setattr(diffusion_kernels,'jacobian_range_temperature_cache',observed_range)
        cache_calls.clear()
        actual=cycle(x)
        if iteration!=1:
            assert bool(cache_calls)==(profile=='uniform')
        hashes=[]
        for a,b in zip(expected,actual):
            assert a.shape==b.shape and a.dtype==b.dtype and a.strides==b.strides
            assert np.isfinite(a).all() and a.tobytes()==b.tobytes()
            hashes.append(hashlib.sha256(b.tobytes()).digest())
        if iteration==0:
            anchor=hashes
        elif iteration==2:
            assert hashes==anchor
