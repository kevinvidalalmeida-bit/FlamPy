"""Independent transport regressions for off-manifold Newton states."""
from types import SimpleNamespace

import numpy as np
import pytest

from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.transport import NativeTransport
from kflame.chemistry import transport as transport_module


@pytest.fixture(scope='module')
def backend():
    path=resolve_mechanism('h2o2.yaml');mech=load_mechanism(path)
    problem=SimpleNamespace(P=101325.,case=SimpleNamespace(mech=path))
    return NativeSpeciesBackend(problem,mech)


def state(kind):
    Y=np.arange(1.,11.);Y/=Y.sum()
    if kind=='scaled':
        Y*=1.01
    elif kind=='species':
        Y[0]+=1e-5
    return Y


@pytest.mark.parametrize('kind',['normalized','scaled','species'])
def test_fallback_and_point_routes_preserve_the_unnormalized_state(backend,monkeypatch,kind):
    Y=state(kind);T=np.array([300.,1000.,2000.]);grid=np.tile(Y[:,None],(1,len(T)))
    fast=backend.eval_faces_native(T,grid)
    monkeypatch.setattr(backend.problem,'use_numba_transport',False,raising=False)
    fallback=backend.eval_faces_native(T,grid)
    for actual,expected in zip(fallback,fast):
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=1e-15)
    point=backend.eval_midpoint(1000.,1000.,Y,Y)
    full=backend.eval_midpoint_full_transport(1000.,1000.,Y,Y)
    np.testing.assert_allclose(point[1],fast[1][:,1],rtol=2e-12)
    np.testing.assert_allclose(full[1],fast[1][:,1],rtol=2e-12)
    _,_,diffusion,_=backend.transport.eval_all(1000.,backend.problem.P,Y,backend.thermo.cp_R(1000.),backend.invW)
    np.testing.assert_allclose(diffusion,fast[1][:,1],rtol=2e-12)


@pytest.mark.parametrize('kind',['normalized','scaled','species'])
@pytest.mark.parametrize('pressure',[101325.,202650.])
def test_soret_and_diffusion_match_cantera_without_normalizing_Y(backend,monkeypatch,kind,pressure):
    ct=pytest.importorskip('cantera')
    gas=ct.Solution(resolve_mechanism('h2o2.yaml'));Y=state(kind)
    T=np.array([300.,1000.,2000.]);grid=np.tile(Y[:,None],(1,len(T)))
    transport=NativeTransport(backend.mech)
    fast=transport.eval_faces_poly_fast(T,pressure,grid,backend.invW)[1]
    fused=transport.thermal_diff_coeffs(T,pressure,grid)
    monkeypatch.setattr(transport_module,'_mix_soret_fused',None)
    fallback=transport.thermal_diff_coeffs(T,pressure,grid)
    for j,temp in enumerate(T):
        gas.set_unnormalized_mass_fractions(Y);gas.TP=temp,pressure
        assert gas.Y.sum()==pytest.approx(Y.sum(),abs=1e-14)
        np.testing.assert_allclose(fast[:,j],gas.mix_diff_coeffs,rtol=2e-11)
        np.testing.assert_allclose(fused[:,j],gas.thermal_diff_coeffs,rtol=2e-11,atol=1e-20)
        np.testing.assert_allclose(fallback[:,j],gas.thermal_diff_coeffs,rtol=2e-11,atol=1e-20)
