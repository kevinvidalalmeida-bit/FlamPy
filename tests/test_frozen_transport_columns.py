"""Compare every tiled column against the original scalar face operations."""
import numpy as np
import pytest

from kflame.flame import diffusion_kernels as kernels
from kflame.flame.frozen_transport_columns import frozen_composition_columns


@pytest.mark.parametrize('ns',[3,23,24,53,111,581])
@pytest.mark.parametrize('soret',[False,True])
def test_ordered_columns_match_faces_with_traces_and_tail(ns,soret):
    rng=np.random.default_rng(1301+ns)
    mw=rng.uniform(2.,200.,ns);inv=1./mw
    YW=rng.uniform(.001,1.,(ns,5));YW/=YW.sum(axis=0)
    # Zero and tiny fractions exercise Wilke floors and nonzero FD increments.
    YW[0]=0.;YW[1]=1e-60
    TW=np.array([650.,800.,1200.,1900.,2200.]);dz=np.array([.001,.002,.003,.004])
    Yb=np.repeat(YW[:,2,None],ns+2,axis=1)
    for q in range(ns):Yb[q,q+2]+=1e-10+1e-5*abs(Yb[q,0])
    roots=(rng.uniform(.01,.02,ns),rng.uniform(.01,.02,ns))
    phis=(rng.uniform(.5,2.,(ns,ns)),rng.uniform(.5,2.,(ns,ns)))
    td=[(np.empty(0),np.empty((0,0)),np.empty(0),roots[s],phis[s],np.empty((0,0))) for s in range(2)]
    multi=(rng.uniform(-.01,.01,(ns,ns)),rng.uniform(-.01,.01,(ns,ns)))
    thermal=tuple(rng.uniform(-1e-5,1e-5,ns) if soret else np.zeros(ns) for _ in range(2))
    witnesses=[YW,Yb,*roots,*phis,*multi,*thermal,mw,inv]
    before=[a.tobytes() for a in witnesses]
    for a in witnesses:a.flags.writeable=False
    actual=frozen_composition_columns(ns,YW,TW,dz,101325.,inv,mw,Yb,True,True,*td,multi,thermal)
    for side in range(2):
        a=side+1;T=.5*(TW[a]+TW[a+1]);expected=np.empty((ns,ns));mu=np.empty(ns)
        for q in range(ns):
            yl=YW[:,a].copy();yr=YW[:,a+1].copy()
            (yr if side==0 else yl)[q]=Yb[q,q+2]
            yf=.5*(yl+yr);denom=0.
            for k in range(ns):denom+=yf[k]*inv[k]
            wm=1./max(denom,1e-300);rho=101325.*wm/(kernels.R_UNIV*T)
            flux=np.empty(ns);scratch=np.empty(ns)
            kernels.multicomponent_flux(yl,yr,TW[a],TW[a+1],dz[a],T,rho,wm,
                                        multi[side],thermal[side],inv,mw,flux,scratch)
            expected[:,q]=flux
            X=np.empty(ns)
            for k in range(ns):X[k]=yf[k]*wm*inv[k]
            mu[q]=kernels.face_viscosity(X,roots[side],phis[side])
        assert np.isfinite(actual[side]).all() and np.isfinite(actual[2+side]).all()
        assert actual[side].tobytes()==expected.tobytes()
        assert actual[2+side].tobytes()==mu.tobytes()
    assert [a.tobytes() for a in witnesses]==before


def test_compact_workspace_keeps_ordinary_constructor_and_private_rotation():
    ordinary=kernels._node_workspace_temperature_cache(24)
    compact=kernels._node_workspace_temperature_cache(24,True)
    assert len(ordinary)==len(compact)==41
    for slot in (17,18):
        assert ordinary[slot][0].size==24
        assert all(a.size==0 for a in compact[slot])
    assert compact[19][0].size==24
    rotated=kernels._rotate_face_workspace(compact)
    assert rotated[17][0] is compact[18][0]
    assert not np.shares_memory(compact[35][0],compact[35][1])
