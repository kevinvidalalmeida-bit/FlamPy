"""Cross-tile ownership against the allocating legacy range, including A-B-A."""
import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion, diffusion_kernels


@pytest.mark.parametrize('n,soret,fixed', [(9, True, False), (17, False, True)])
def test_large_tiles_preserve_allocating_range_and_changed_state(monkeypatch,n,soret,fixed):
    path=resolve_mechanism('jetsurf2.yaml')
    mech=load_mechanism(path)
    feeds=[]
    for text in ('CH4:1,N2:4','O2:1,N2:3.76'):
        y=_composition(text,mech.species_names)*mech.molecular_weights
        feeds.append(y/y.sum())
    case=diffusion.DiffusionCase(path,*feeds,transport_model='multicomponent',
        soret_enabled=soret,radiation_enabled=True,boundary_emissivities=(.2,.7))
    p=diffusion.DiffusionFlameProblem(case,mech,np.linspace(0.,.02,n))
    p.multicomponent_jacobian='frozen'
    fields=p.initial_fields()
    fields['T'][:]=np.linspace(600.,1900.,n)
    # Exercise every species, including traces, rather than a sparse feed.
    y=np.arange(1.,p.ns+1.);y/=y.sum()
    fields['Y'][:]=y[:,None]
    first=p.pack(fields)
    fields['T'][:]*=.999
    fields['Y'][:]=y[::-1,None]
    second=p.pack(fields)
    carried=diffusion_kernels.jacobian_range_workspace
    temperature_carried=diffusion_kernels.jacobian_range_temperature_cache

    # Derive the argument boundary from the original signature, avoiding an
    # assumption about how many physics arguments future kernels will need.
    import inspect
    offset=list(inspect.signature(carried.py_func).parameters).index('carried_ws')
    def allocate_each_tile(*args):
        # Original range discards its private buffers at each tile.
        diffusion_kernels.jacobian_range(*args[:offset],*args[offset+2:])
        return args[offset]

    saved=[]
    for x in (first,second,first):
        temperature=p.unpack(x)['T'].copy() if fixed else None
        monkeypatch.setattr(diffusion_kernels,'jacobian_range_workspace',allocate_each_tile)
        monkeypatch.setattr(diffusion_kernels,'jacobian_range_temperature_cache',allocate_each_tile)
        expected=p.jacobian_blocks(x,fixed_temperature=temperature)
        monkeypatch.setattr(diffusion_kernels,'jacobian_range_workspace',carried)
        monkeypatch.setattr(diffusion_kernels,'jacobian_range_temperature_cache',temperature_carried)
        actual=p.jacobian_blocks(x,fixed_temperature=temperature)
        for key in ('lower','diag','upper','border','mass'):
            a,b=getattr(actual,key),getattr(expected,key)
            assert a.shape==b.shape and a.dtype==b.dtype
            assert np.isfinite(a).all() and a.tobytes()==b.tobytes()
        saved.append(actual)
    for key in ('lower','diag','upper','border','mass'):
        assert getattr(saved[0],key).tobytes()==getattr(saved[2],key).tobytes()
