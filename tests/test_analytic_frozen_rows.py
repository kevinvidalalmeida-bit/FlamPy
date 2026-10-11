"""Independent residual differences exercise all neighboring composition rows."""
import numpy as np
import pytest

from kflame.flame.analytic_frozen_rows import composition_row_derivative
from kflame.flame.diffusion_kernels import _window_rows


@pytest.mark.parametrize('fixed',[False,True])
@pytest.mark.parametrize('mass',[.2,-.2,0.,1e-6,-1e-6])
def test_composition_rows_against_centered_residual(fixed,mass):
    rng=np.random.default_rng(27)
    ns,n=5,7
    scalars=(.3,12.,1000.,50.,100.,1e7,.3,.3,300.,300.)
    invW=np.array([.5,.03,.06,.05,.02])
    radiation=(True,.7,101325.,3,4,200.,invW)
    excess=np.arange(n)%ns
    Tfixed=np.linspace(800.,1500.,n)
    for j in range(n):
        m=np.full(5,mass);V=rng.normal(size=5);T=np.linspace(800.,1500.,5)
        Y=rng.uniform(.1,.3,(ns,5))
        rho=1./(Y.T@invW);cp=rng.uniform(800.,1200.,5)
        omega=rng.normal(size=(ns,5));hk=rng.uniform(1e5,5e5,(ns,5))
        lam=np.full(4,.05);J=rng.normal(0,.02,(ns,4));mu=np.full(4,1e-5)
        dz=np.array([.001,.002,.003,.004])
        feeds=np.full(ns,1./ns)
        dleft=rng.normal(0,.01,(ns,ns));dright=rng.normal(0,.01,(ns,ns))
        muleft=rng.normal(0,1e-6,ns);muright=rng.normal(0,1e-6,ns)
        rates=rng.normal(size=(ns,ns))
        denom=Y[:,2]@invW
        for q in range(ns):
            drho=-rho[2]*invW[q]/denom;dcp=10.*(q+1)
            for w in range(1,4):
                if not 0<=j-2+w<n:
                    continue
                actual=np.empty(ns+3)
                composition_row_derivative(actual,w,n,j,q,m,V,T,Y,rho,hk,J,dz,
                    scalars,excess,fixed,radiation,drho,dcp,rates[:,q],dleft,dright,muleft,muright)
                h=2e-6;values=[]
                for sign in (-1.,1.):
                    step=sign*h
                    yp=Y.copy();yp[q,2]+=step
                    rp=rho.copy();rp[2]=1./(denom+step*invW[q])
                    cpp=cp.copy();cpp[2]+=step*dcp
                    op=omega.copy();op[:,2]+=step*rates[:,q]
                    jp=J.copy();jp[:,1]+=step*dleft[:,q];jp[:,2]+=step*dright[:,q]
                    mup=mu.copy();mup[1]+=step*muleft[q];mup[2]+=step*muright[q]
                    out=np.empty(ns+3)
                    _window_rows(out,w,n,j,m,V,T,yp,rp,cpp,op,hk,lam,jp,mup,dz,0.,
                        scalars,feeds,feeds,excess,fixed,Tfixed,radiation)
                    values.append(out)
                expected=(values[1]-values[0])/(2*h)
                np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-8)
