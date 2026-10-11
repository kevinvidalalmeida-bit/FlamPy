"""Ordered frozen-transport columns using eight independent composition lanes.

The caller owns one immutable base bank and one range's private buffers. Only
the valid-bank Jacobian path consumes these tables. All dot products and Wilke
reductions retain the original ascending species order, without fastmath or
BLAS reassociation. Inputs are read-only; temporary tables belong to this call.
"""
from numba import njit
import numpy as np
from kflame.chemistry.mechanism import R_UNIV

@njit(cache=True)
def _prepare_columns(YL,YR,TL,TR,spacing,P,invW,mw,Yb,left_changed,scratch,Xface,factors,qstart,count):
    ns=mw.size
    Tf=.5*(TL+TR)
    for column in range(count):
        q=qstart+column
        sL=0.;sR=0.;denom=0.
        for k in range(ns):
            yl=Yb[k,2+k] if left_changed and k==q else YL[k]
            yr=Yb[k,2+k] if not left_changed and k==q else YR[k]
            sL+=yl*invW[k]
            sR+=yr*invW[k]
            denom+=.5*(yl+yr)*invW[k]
        WL=1./max(sL,1e-300);WR=1./max(sR,1e-300)
        wm=1./max(denom,1e-300)
        rho=P*wm/(R_UNIV*Tf)
        factors[column]=rho/wm**2
        for k in range(ns):
            yl=Yb[k,2+k] if left_changed and k==q else YL[k]
            yr=Yb[k,2+k] if not left_changed and k==q else YR[k]
            scratch[k,column]=mw[k]*(yr*WR*invW[k]-yl*WL*invW[k])/spacing
            Xface[k,column]=.5*(yl+yr)*wm*invW[k]

@njit(cache=True, nogil=True, fastmath=False)
def _dots8(scratch, Xface, factors, Tf, gradT, D, DT, mw, root, phi, flux, mu):
    """Escribe flux y mu con bloques de ocho columnas y resto escalar."""
    ns, ncolumns = scratch.shape
    stop = ncolumns - ncolumns % 8
    for q in range(ncolumns):
        mu[q] = 0.0

    for k in range(ns):
        for q in range(0, stop, 8):
            dot0 = dot1 = dot2 = dot3 = 0.0
            dot4 = dot5 = dot6 = dot7 = 0.0
            wilke0 = wilke1 = wilke2 = wilke3 = 0.0
            wilke4 = wilke5 = wilke6 = wilke7 = 0.0
            for l in range(ns):
                dkl = D[k, l]
                phikl = phi[k, l]
                dot0 += dkl * scratch[l, q]
                dot1 += dkl * scratch[l, q + 1]
                dot2 += dkl * scratch[l, q + 2]
                dot3 += dkl * scratch[l, q + 3]
                dot4 += dkl * scratch[l, q + 4]
                dot5 += dkl * scratch[l, q + 5]
                dot6 += dkl * scratch[l, q + 6]
                dot7 += dkl * scratch[l, q + 7]
                wilke0 += phikl * Xface[l, q]
                wilke1 += phikl * Xface[l, q + 1]
                wilke2 += phikl * Xface[l, q + 2]
                wilke3 += phikl * Xface[l, q + 3]
                wilke4 += phikl * Xface[l, q + 4]
                wilke5 += phikl * Xface[l, q + 5]
                wilke6 += phikl * Xface[l, q + 6]
                wilke7 += phikl * Xface[l, q + 7]

            flux[k, q] = factors[q] * mw[k] * dot0 - DT[k] / Tf * gradT
            flux[k, q + 1] = factors[q + 1] * mw[k] * dot1 - DT[k] / Tf * gradT
            flux[k, q + 2] = factors[q + 2] * mw[k] * dot2 - DT[k] / Tf * gradT
            flux[k, q + 3] = factors[q + 3] * mw[k] * dot3 - DT[k] / Tf * gradT
            flux[k, q + 4] = factors[q + 4] * mw[k] * dot4 - DT[k] / Tf * gradT
            flux[k, q + 5] = factors[q + 5] * mw[k] * dot5 - DT[k] / Tf * gradT
            flux[k, q + 6] = factors[q + 6] * mw[k] * dot6 - DT[k] / Tf * gradT
            flux[k, q + 7] = factors[q + 7] * mw[k] * dot7 - DT[k] / Tf * gradT
            mu[q] += max(Xface[k, q], 1e-300) * root[k] ** 2 / max(wilke0, 1e-300)
            mu[q + 1] += max(Xface[k, q + 1], 1e-300) * root[k] ** 2 / max(wilke1, 1e-300)
            mu[q + 2] += max(Xface[k, q + 2], 1e-300) * root[k] ** 2 / max(wilke2, 1e-300)
            mu[q + 3] += max(Xface[k, q + 3], 1e-300) * root[k] ** 2 / max(wilke3, 1e-300)
            mu[q + 4] += max(Xface[k, q + 4], 1e-300) * root[k] ** 2 / max(wilke4, 1e-300)
            mu[q + 5] += max(Xface[k, q + 5], 1e-300) * root[k] ** 2 / max(wilke5, 1e-300)
            mu[q + 6] += max(Xface[k, q + 6], 1e-300) * root[k] ** 2 / max(wilke6, 1e-300)
            mu[q + 7] += max(Xface[k, q + 7], 1e-300) * root[k] ** 2 / max(wilke7, 1e-300)

        for q in range(stop, ncolumns):
            dot = 0.0
            wilkedot = 0.0
            for l in range(ns):
                dot += D[k, l] * scratch[l, q]
                wilkedot += phi[k, l] * Xface[l, q]
            flux[k, q] = factors[q] * mw[k] * dot - DT[k] / Tf * gradT
            mu[q] += max(Xface[k, q], 1e-300) * root[k] ** 2 / max(wilkedot, 1e-300)

@njit(cache=True)
def frozen_composition_columns(ns,YW,TW,dzW,P,invW,mw,Yb,has_left,has_right,tleft,tright,frozen_multi,frozen_thermal):
    flux_left=np.empty((ns,ns))
    flux_right=np.empty((ns,ns))
    mu_left=np.empty(ns)
    mu_right=np.empty(ns)
    width=8
    scratch_batch=np.empty((ns,width))
    xface_batch=np.empty((ns,width))
    factors_batch=np.empty(width)
    flux_tile=np.empty((ns,width))
    mu_tile=np.empty(width)
    for side in range(2):
        if (side==0 and not has_left) or (side==1 and not has_right):
            continue
        a=1+side
        td=tleft if side==0 else tright
        for first in range(0,ns,width):
            count=min(width,ns-first)
            _prepare_columns(YW[:,a],YW[:,a+1],TW[a],TW[a+1],dzW[a],P,invW,mw,Yb,
                            side==1,scratch_batch,xface_batch,factors_batch,first,count)
            # Fill unused tail lanes with the last valid input; outputs are discarded.
            for q in range(count,width):
                factors_batch[q]=factors_batch[count-1]
                for k in range(ns):
                    scratch_batch[k,q]=scratch_batch[k,count-1]
                    xface_batch[k,q]=xface_batch[k,count-1]
            _dots8(scratch_batch,xface_batch,factors_batch,.5*(TW[a]+TW[a+1]),
                  (TW[a+1]-TW[a])/dzW[a],frozen_multi[side],frozen_thermal[side],mw,td[3],td[4],flux_tile,mu_tile)
            result=flux_left if side==0 else flux_right
            mu_result=mu_left if side==0 else mu_right
            for q in range(count):
                mu_result[first+q]=mu_tile[q]
                for k in range(ns):
                    result[k,first+q]=flux_tile[k,q]
    return flux_left,flux_right,mu_left,mu_right
