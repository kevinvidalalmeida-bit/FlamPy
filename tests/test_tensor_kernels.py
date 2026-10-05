"""Compiled interpolation checked against independent SciPy splines."""
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from kflame.fgm.tensor_kernels import evaluate_tensor, evaluate_progress


def test_tensor_values_and_derivatives_match_scipy_at_nonuniform_knots():
    rng=np.random.default_rng(823)
    axes=[np.array([0.,.02,.12,.35,.7,1.]),np.array([0.,.08,.19,.4,.91,1.]),np.linspace(0.,1.,11)]
    knots=[make_interp_spline(a,np.zeros(len(a)),k=2).t for a in axes]
    shape=np.array([len(a) for a in axes]);strides=np.array([shape[1]*shape[2],shape[2],1])
    coefficients=rng.normal(size=(np.prod(shape),7))
    points=np.r_[rng.uniform(0.,1.,(121,3)),np.zeros((1,3)),np.ones((1,3)),[[.35,.4,.5]],[[1.+1e-8,0.,.7]]]
    bases=[BSpline(k,np.eye(n),2)(points[:,i]) for i,(k,n) in enumerate(zip(knots,shape))]
    derivatives=[BSpline(k,np.eye(n),2)(points[:,i],nu=1) for i,(k,n) in enumerate(zip(knots,shape))]
    c=coefficients.reshape(*shape,7)
    expected=np.einsum('ni,nj,nk,ijkf->nf',*bases,c)
    expected_gradient=np.stack([np.einsum('ni,nj,nk,ijkf->nf',*[derivatives[k] if k==axis else bases[k] for k in range(3)],c[:,:,:,:4]) for axis in range(3)],axis=1)
    for threshold in (1,2048):
        actual,gradient=evaluate_tensor(points,knots,strides,coefficients,differentiated=4,parallel_threshold=threshold)
        np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=2e-14)
        np.testing.assert_allclose(gradient,expected_gradient,atol=5e-13,rtol=5e-13)


def test_progress_guide_matches_bilinear_scipy_derivatives():
    rng=np.random.default_rng(32);shape=np.array([4,5,9]);strides=np.array([45,9,1])
    knots=np.r_[np.zeros(3),np.arange(1,shape[2]-2),np.full(3,shape[2]-2)]/(shape[2]-2)
    points=rng.uniform(0.,1.,(39,3));base=np.floor(points[:,:2]*(shape[:2]-1)).astype(int)
    coefficients=rng.normal(size=(np.prod(shape),3));field=coefficients.reshape(*shape,3)
    values,gradient=evaluate_progress(points,base,knots,shape,strides,coefficients,differentiated=3)
    expected=[];derivative=[]
    for x,b in zip(points,base):
        uv=x[:2]*(shape[:2]-1)-b
        w=[np.array([1.-u,u]) for u in uv];dw=[np.array([-1.,1.])*(n-1) for n in shape[:2]]
        local=field[b[0]:b[0]+2,b[1]:b[1]+2]
        spline=BSpline(knots,local,2,axis=2)
        expected.append(np.einsum('i,j,ijf->f',*w,spline(x[2])))
        derivative.append([np.einsum('i,j,ijf->f',dw[0],w[1],spline(x[2])),np.einsum('i,j,ijf->f',w[0],dw[1],spline(x[2])),np.einsum('i,j,ijf->f',*w,spline(x[2],nu=1))])
    np.testing.assert_allclose(values,expected,atol=2e-15)
    np.testing.assert_allclose(gradient,derivative,atol=2e-14)
