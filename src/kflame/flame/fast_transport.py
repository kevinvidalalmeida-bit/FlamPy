"""Face-blocked, vectorized mixture-averaged transport.

Same closure as ``face_temperature_data`` + ``_face_mixture_ws`` (the
mixture-averaged diffusion, Wilke viscosity, combination-averaged
conductivity, binary diffusion fits of D_kj/T^1.5 in log T), evaluated
differently:

* no O(ns^2) temporaries (1/D_kj and the Wilke matrix) are written per face;
  the mixture sums are accumulated directly;
* a block of faces shares one pass over each mechanism row (five fit
  coefficients and two Wilke constants), so the mechanism tables are read
  from memory once per block instead of once per face;
* reductions may be reordered (``fastmath`` reassociation/contraction only),
  which lets the compiler use SIMD. Results agree with the scalar kernels up
  to rounding; no approximation of the physical model is introduced.

``coef[f, k] = rho W_k D_k / Wmix`` is the coefficient of the molar-gradient
species flux used by the residual.
"""
import math

import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV

# Reassociation and FMA contraction only: no NaN/Inf or reciprocal shortcuts.
FAST = {'reassoc', 'contract'}
FACE_BLOCK = 8


def mixture_tables(transport, mw):
    """Mechanism-constant tables in the layout of :func:`mixture_faces`."""
    diff = np.asarray(transport._diff_poly, dtype=np.float64)
    ns = diff.shape[0]
    dpoly = np.ascontiguousarray(np.moveaxis(diff, 2, 0))
    ratio = mw[None, :]/mw[:, None]
    quarter = np.ascontiguousarray(ratio**.25)
    inverse_den = np.ascontiguousarray(1./np.sqrt(8.*(1.+1./ratio)))
    ddiag = np.ascontiguousarray(diff[np.arange(ns), np.arange(ns)])
    return (dpoly, quarter, inverse_den, np.ascontiguousarray(transport._cond_poly, dtype=np.float64),
            np.ascontiguousarray(transport._visc_poly, dtype=np.float64), ddiag)


@njit(cache=True, fastmath=FAST)
def _poly4(c, x):
    return c[0]+x*(c[1]+x*(c[2]+x*(c[3]+x*c[4])))


@njit(cache=True, nogil=True, fastmath=FAST)
def mixture_faces(start, stop, T, Y, P, invW, mw, tables, tscale, lam, mu, coef):
    """lam, mu and coef of faces [start, stop) at temperature tscale*(T_L+T_R)/2.

    Face composition is (Y_L+Y_R)/2. Serial and GIL-free with private scratch;
    outputs of different ranges never overlap.
    """
    dpoly, quarter, iden, cond_poly, visc_poly, ddiag = tables
    ns = Y.shape[0]
    B = FACE_BLOCK
    L = np.empty(B)
    t15 = np.empty(B)
    wm = np.empty(B)
    X = np.empty((B, ns))
    Xc = np.empty((B, ns))
    root = np.empty((B, ns))
    iroot = np.empty((B, ns))
    SD = np.empty((B, ns))
    Smu = np.empty((B, ns))
    for f0 in range(start, stop, B):
        nb = min(B, stop-f0)
        for b in range(nb):
            f = f0+b
            Tf = tscale*(.5*(T[f]+T[f+1]))
            logT = math.log(Tf)
            L[b] = logT
            sqrtT = math.sqrt(Tf)
            t15[b] = Tf*sqrtT
            quarterT = Tf**.25
            inv = 0.
            for k in range(ns):
                inv += .5*(Y[k, f]+Y[k, f+1])*invW[k]
            w = 1./max(inv, 1e-300)
            wm[b] = w
            sum1 = 0.
            sum2 = 0.
            for k in range(ns):
                x = .5*(Y[k, f]+Y[k, f+1])*w*invW[k]
                X[b, k] = x
                xc = max(x, 1e-300)
                Xc[b, k] = xc
                cond = sqrtT*_poly4(cond_poly[k], logT)
                sum1 += xc*cond
                sum2 += xc/max(cond, 1e-300)
                r = abs(quarterT*_poly4(visc_poly[k], logT))
                root[b, k] = r
                iroot[b, k] = 1./max(r, 1e-150)
            lam[f] = .5*(sum1+1./max(sum2, 1e-300))
        d0, d1, d2, d3, d4 = dpoly[0], dpoly[1], dpoly[2], dpoly[3], dpoly[4]
        for k in range(ns):
            c0, c1, c2, c3, c4 = d0[k], d1[k], d2[k], d3[k], d4[k]
            q, wd = quarter[k], iden[k]
            for b in range(nb):
                x = L[b]
                tb = t15[b]
                xc = Xc[b]
                sd = 0.
                # One masked loop vectorizes; split ranges around k do not.
                for j in range(ns):
                    binary = tb*(c0[j]+x*(c1[j]+x*(c2[j]+x*(c3[j]+x*c4[j]))))
                    term = xc[j]/max(binary, 1e-300)
                    sd += term if j != k else 0.
                SD[b, k] = sd
                rk = root[b, k]
                xb = X[b]
                ir = iroot[b]
                smu = 0.
                for j in range(ns):
                    t = 1.+rk*ir[j]*q[j]
                    smu += xb[j]*t*t*wd[j]
                Smu[b, k] = smu
        for b in range(nb):
            f = f0+b
            w = wm[b]
            Tf = tscale*(.5*(T[f]+T[f+1]))
            rho = P*w/(R_UNIV*Tf)
            m = 0.
            for k in range(ns):
                xc = Xc[b, k]
                if SD[b, k] <= 0.:
                    D = t15[b]*_poly4(ddiag[k], L[b])/P
                else:
                    D = (w-xc*mw[k])/(P*w*max(SD[b, k], 1e-300))
                coef[f, k] = rho*mw[k]/w*D
                m += xc*root[b, k]**2/max(Smu[b, k], 1e-300)
            mu[f] = m


@njit(cache=True, nogil=True, fastmath=FAST)
def mixture_fluxes(start, stop, Y, dz, invW, coef, J):
    """Corrected molar-gradient fluxes J[k, f] of faces [start, stop) (no Soret)."""
    ns = Y.shape[0]
    for f in range(start, stop):
        sL = 0.
        sR = 0.
        for k in range(ns):
            sL += Y[k, f]*invW[k]
            sR += Y[k, f+1]*invW[k]
        WL = 1./sL
        WR = 1./sR
        h = dz[f]
        total = 0.
        for k in range(ns):
            value = -coef[f, k]*(Y[k, f+1]*WR-Y[k, f]*WL)*invW[k]/h
            J[k, f] = value
            total += value
        for k in range(ns):
            J[k, f] -= Y[k, f]*total


@njit(cache=True, nogil=True)
def face_range_fast(start, stop, T, Y, dz, P, invW, mw, tables, lam, mu, J, coef):
    """Residual face properties: lam, Wilke mu and fluxes of faces [start, stop)."""
    mixture_faces(start, stop, T, Y, P, invW, mw, tables, 1., lam, mu, coef)
    mixture_fluxes(start, stop, Y, dz, invW, coef, J)


def soret_tables(transport):
    """log(eps_kj) and the nine C*_kj fit coefficients as (9, ns, ns) planes."""
    return (np.ascontiguousarray(np.log(np.asarray(transport._eps_pair, dtype=np.float64))),
            np.ascontiguousarray(np.moveaxis(np.asarray(transport._cstar_poly, dtype=np.float64), 2, 0)))


@njit(cache=True, nogil=True, fastmath=FAST)
def mixture_faces_soret(start, stop, T, Y, P, invW, mw, tables, stables, tscale, lam, mu, coef, DT):
    """:func:`mixture_faces` plus the mixture-averaged thermal diffusion.

    Same expressions as ``_face_mixture_ws`` with ``soret``: mole fractions
    clipped at 1e-20 for the Soret diffusivities and the Wilke-type a_k,
    C*_kj fits in log(T/eps_kj) and the zero-sum correction of D^T.
    """
    dpoly, quarter, iden, cond_poly, visc_poly, ddiag = tables
    logeps, cstar = stables
    ns = Y.shape[0]
    B = FACE_BLOCK
    L = np.empty(B)
    t15 = np.empty(B)
    wm = np.empty(B)
    Yf = np.empty((B, ns))
    X = np.empty((B, ns))
    Xc = np.empty((B, ns))
    Xs = np.empty((B, ns))
    root = np.empty((B, ns))
    iroot = np.empty((B, ns))
    SD = np.empty((B, ns))
    SDs = np.empty((B, ns))
    Smu = np.empty((B, ns))
    Ss = np.empty((B, ns))
    a = np.empty((B, ns))
    Ds = np.empty((B, ns))
    value = np.empty((B, ns))
    d0, d1, d2, d3, d4 = dpoly[0], dpoly[1], dpoly[2], dpoly[3], dpoly[4]
    for f0 in range(start, stop, B):
        nb = min(B, stop-f0)
        for b in range(nb):
            f = f0+b
            Tf = tscale*(.5*(T[f]+T[f+1]))
            logT = math.log(Tf)
            L[b] = logT
            sqrtT = math.sqrt(Tf)
            t15[b] = Tf*sqrtT
            quarterT = Tf**.25
            inv = 0.
            for k in range(ns):
                y = .5*(Y[k, f]+Y[k, f+1])
                Yf[b, k] = y
                inv += y*invW[k]
            w = 1./max(inv, 1e-300)
            wm[b] = w
            sum1 = 0.
            sum2 = 0.
            for k in range(ns):
                x = Yf[b, k]*w*invW[k]
                X[b, k] = x
                xc = max(x, 1e-300)
                Xc[b, k] = xc
                Xs[b, k] = max(x, 1e-20)
                cond = sqrtT*_poly4(cond_poly[k], logT)
                sum1 += xc*cond
                sum2 += xc/max(cond, 1e-300)
                r = abs(quarterT*_poly4(visc_poly[k], logT))
                root[b, k] = r
                iroot[b, k] = 1./max(r, 1e-150)
            lam[f] = .5*(sum1+1./max(sum2, 1e-300))
        for k in range(ns):
            c0, c1, c2, c3, c4 = d0[k], d1[k], d2[k], d3[k], d4[k]
            q, wd = quarter[k], iden[k]
            for b in range(nb):
                x = L[b]
                tb = t15[b]
                xc = Xc[b]
                xs = Xs[b]
                sd = 0.
                sds = 0.
                for j in range(ns):
                    ib = 1./max(tb*(c0[j]+x*(c1[j]+x*(c2[j]+x*(c3[j]+x*c4[j])))), 1e-300)
                    keep = 1. if j != k else 0.
                    sd += keep*xc[j]*ib
                    sds += keep*xs[j]*ib
                SD[b, k] = sd
                SDs[b, k] = sds
                rk = root[b, k]
                xb = X[b]
                ir = iroot[b]
                smu = 0.
                ss = 0.
                for j in range(ns):
                    t = 1.+rk*ir[j]*q[j]
                    phi = t*t*wd[j]
                    smu += xb[j]*phi
                    ss += xs[j]*phi if j != k else 0.
                Smu[b, k] = smu
                Ss[b, k] = ss
        for b in range(nb):
            f = f0+b
            w = wm[b]
            Tf = tscale*(.5*(T[f]+T[f+1]))
            rho = P*w/(R_UNIV*Tf)
            m = 0.
            for k in range(ns):
                xc = Xc[b, k]
                diag = t15[b]*_poly4(ddiag[k], L[b])/P
                if SD[b, k] <= 0.:
                    Dk = diag
                else:
                    Dk = (w-xc*mw[k])/(P*w*max(SD[b, k], 1e-300))
                coef[f, k] = rho*mw[k]/w*Dk
                m += xc*root[b, k]**2/max(Smu[b, k], 1e-300)
                if SDs[b, k] <= 0.:
                    Ds[b, k] = diag
                else:
                    Ds[b, k] = (w-Xs[b, k]*mw[k])/(P*w*max(SDs[b, k], 1e-300))
                a[b, k] = (15./4.)*root[b, k]**2/mw[k]/(1.+1.065*Ss[b, k]/Xs[b, k]) if Yf[b, k] >= 1e-20 else 0.
            mu[f] = m
        # value_k = sum_j scale_kj (Y_k a_j - Y_j a_k); scale_kk = 0.
        for k in range(ns):
            c0, c1, c2, c3, c4 = d0[k], d1[k], d2[k], d3[k], d4[k]
            le = logeps[k]
            e0, e1, e2, e3, e4 = cstar[0, k], cstar[1, k], cstar[2, k], cstar[3, k], cstar[4, k]
            e5, e6, e7, e8 = cstar[5, k], cstar[6, k], cstar[7, k], cstar[8, k]
            mk = mw[k]
            for b in range(nb):
                x = L[b]
                tb = t15[b]
                ab = a[b]
                yb = Yf[b]
                s_a = 0.
                s_y = 0.
                for j in range(ns):
                    binary = tb*(c0[j]+x*(c1[j]+x*(c2[j]+x*(c3[j]+x*c4[j]))))
                    z = x-le[j]
                    star = e8[j]
                    star = star*z+e7[j]
                    star = star*z+e6[j]
                    star = star*z+e5[j]
                    star = star*z+e4[j]
                    star = star*z+e3[j]
                    star = star*z+e2[j]
                    star = star*z+e1[j]
                    star = star*z+e0[j]
                    sc = (1.2*star-1.)*P/(binary*(mk+mw[j])) if j != k else 0.
                    s_a += sc*ab[j]
                    s_y += sc*yb[j]
                value[b, k] = Yf[b, k]*s_a-a[b, k]*s_y
        for b in range(nb):
            f = f0+b
            w = wm[b]
            total = 0.
            for k in range(ns):
                d = value[b, k]*Ds[b, k]*mw[k]*w
                DT[f, k] = d
                total += d
            for k in range(ns):
                DT[f, k] -= Yf[b, k]*total


@njit(cache=True, nogil=True, fastmath=FAST)
def mixture_fluxes_soret(start, stop, T, Y, dz, invW, coef, DT, J):
    """Corrected molar-gradient fluxes with the thermal-diffusion term."""
    mixture_fluxes(start, stop, Y, dz, invW, coef, J)
    for f in range(start, stop):
        g = (T[f+1]-T[f])/dz[f]/(.5*(T[f]+T[f+1]))
        for k in range(Y.shape[0]):
            J[k, f] -= DT[f, k]*g


@njit(cache=True, nogil=True)
def face_range_fast_soret(start, stop, T, Y, dz, P, invW, mw, tables, stables, lam, mu, J, coef, DT):
    mixture_faces_soret(start, stop, T, Y, P, invW, mw, tables, stables, 1., lam, mu, coef, DT)
    mixture_fluxes_soret(start, stop, T, Y, dz, invW, coef, DT, J)
