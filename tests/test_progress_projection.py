"""Stability selection tested against independent eigenvalue calculations."""
import numpy as np
from kflame.fgm.projection import stable_matrix, stable_blend, progress_projection


def test_routh_matches_eigenvalues_for_nonsymmetric_matrices():
    rng=np.random.default_rng(842)
    matrices=rng.normal(size=(2000,3,3))
    margin=rng.uniform(.01,.1,2000)
    expected=np.linalg.eigvals(matrices).real.min(axis=1)>margin
    np.testing.assert_array_equal(stable_matrix(matrices,margin),expected)


def test_rank_one_blend_preserves_valid_states_and_stabilizes_bad_ones():
    rng=np.random.default_rng(413);base=[];end=[]
    for _ in range(800):
        a=rng.normal(size=(3,3))
        b=a+np.outer(rng.normal(size=3),rng.normal(size=3))*4.
        if np.linalg.eigvals(b).real.min()>.02:
            base.append(a);end.append(b)
    base,end=np.array(base),np.array(end);margin=np.full(len(base),.01)
    gamma=stable_blend(base,end,margin)
    assert len(base)>20 and np.any(gamma==0.) and np.any(gamma>0.)
    assert np.all((gamma>=0.)&(gamma<=1.))
    assert np.all(np.linalg.eigvals(base+gamma[:,None,None]*(end-base)).real.min(axis=1)>margin)
    np.testing.assert_array_equal(gamma[stable_matrix(base,margin)],0.)


def test_thermodynamic_projection_is_normalized_and_chart_invariant():
    rng=np.random.default_rng(594)
    Y=rng.uniform(.1,1.,(6,11));Y/=Y.sum(axis=0)
    original=Y.copy();dY=rng.normal(size=(11,6,3));dY-=dY.mean(axis=1)[:,None,:]
    dh=rng.normal(size=(11,3))*1e6;hk=rng.normal(size=(6,11))*1e6
    mw=np.array([2.,18.,16.,28.,32.,44.]);cp=np.full(11,1300.);T=np.linspace(300.,2000.,11)
    bilger=np.array([1.,2.,3.,-1.,.4,.1]);progress=np.array([0.,1.,0.,2.,.2,1.])
    a,b=progress_projection(Y,T,dY,dh,hk,cp,mw,bilger,progress)
    M=np.stack([np.einsum('s,nsk->nk',bilger,dY),np.einsum('s,nsk->nk',progress,dY),dh],axis=1)
    v=np.linalg.solve(M,np.tile([0.,1.,0.],(11,1))[...,None])[...,0]
    action=np.einsum('sn,nsk,nk->n',a,dY,v)+b*np.einsum('nk,nk->n',dh,v)
    np.testing.assert_allclose(action,1.,atol=2e-12)
    change=np.array([[2.,.3,0.],[0.,1.,.2],[.1,0.,3.]])
    other_a,other_b=progress_projection(Y,T,dY@change,dh@change,hk,cp,mw,bilger,progress)
    np.testing.assert_allclose(other_a,a,atol=2e-12)
    np.testing.assert_allclose(other_b,b,atol=2e-16)
    np.testing.assert_array_equal(Y,original)
