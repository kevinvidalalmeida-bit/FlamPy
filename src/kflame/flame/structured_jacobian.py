"""Analytic counterflow Jacobian with frozen face transport, in structured form.

The mixture-averaged residual is differentiated analytically. Face transport
coefficients (rho W_k D_k / Wmix, thermal diffusion, conductivity and
viscosity) are held at their base composition but keep their temperature
dependence, which is obtained from one extra face evaluation at a perturbed
face temperature. Everything else -- molar gradients, mean molecular weights,
the correction velocity, ideal-gas density, cp, enthalpies, upwinding, the
stagnation blend, radiation and analytic chemistry -- is differentiated
exactly. The physical residual is never modified: this matrix only shapes
Newton directions, and every step is still accepted on the complete residual.

With frozen coefficients a face flux depends on the composition of either
node through a diagonal plus two rank-one terms. Hence each off-diagonal
block A[j, j+-1] is ``diag(d) + P Q^T`` with at most ``RANK`` columns
(two species terms, the temperature column, the energy row and the
continuity row). Diagonal blocks are dense: they carry the dense chemistry
Jacobian. Assembling a node costs O(ns**2), against O(ns**2) work per column
(O(ns**3) per node) for column-wise finite differences.
"""
import math

import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV
from kflame.flame.fast_transport import mixture_faces, mixture_faces_soret
from kflame.flame.diffusion_kernels import (STAGNATION, STEFAN_BOLTZMANN, ONE_ATMOSPHERE,
                                            _face_mixture_ws, _tdata, face_temperature_data)

RANK = 5
# Relative face-temperature step of the transport-coefficient derivative.
TEMPERATURE_STEP = 1e-6


@njit(cache=True, nogil=True)
def face_transport_range(start, stop, T, Y, dz, P, invW, mw, mixture, soret,
                         coef, DT, lam, mu, dcoef, dDT, dlam, dmu):
    """Frozen-composition face coefficients and their face-temperature slopes.

    ``coef[f, k] = rho W_k D_k / Wmix`` multiplies the molar-gradient flux,
    exactly as in the residual's ``mixture_flux``. Faces [start, stop) only;
    serial, GIL-free and with private scratch.
    """
    ns = Y.shape[0]
    cond_poly, diff_poly, visc_poly, quarter_ratio, wilke_den, eps_pair, cstar_poly, tiny = mixture
    cond, invB, diagB, root, phi, scale = _tdata(ns, False)
    Yf, X, D, DTk = np.empty(ns), np.empty(ns), np.empty(ns), np.zeros(ns)
    mix_ws = np.empty((5 if soret else 2, ns))
    for f in range(start, stop):
        Tf = .5*(T[f]+T[f+1])
        for k in range(ns):
            Yf[k] = .5*(Y[k, f]+Y[k, f+1])
        for side in range(2):
            Tq = Tf if side == 0 else Tf*(1.+TEMPERATURE_STEP)
            face_temperature_data(Tq, P, mw, cond_poly, diff_poly, visc_poly, quarter_ratio, wilke_den,
                                  soret, eps_pair, cstar_poly, cond, invB, diagB, root, phi, scale)
            rho, wm, lq, mq = _face_mixture_ws(Tq, Yf, P, invW, mw, cond, invB, diagB, root, phi,
                                               soret, scale, tiny, X, D, DTk, mix_ws)
            if side == 0:
                lam[f], mu[f] = lq, mq
                for k in range(ns):
                    coef[f, k] = rho*mw[k]/wm*D[k]
                    DT[f, k] = DTk[k] if soret else 0.
            else:
                h = Tq-Tf
                dlam[f], dmu[f] = (lq-lam[f])/h, (mq-mu[f])/h
                for k in range(ns):
                    dcoef[f, k] = (rho*mw[k]/wm*D[k]-coef[f, k])/h
                    dDT[f, k] = ((DTk[k] if soret else 0.)-DT[f, k])/h


@njit(cache=True, nogil=True)
def fast_face_transport_range(start, stop, T, Y, P, invW, mw, tables, stables, soret,
                              coef, DT, lam, mu, dcoef, dDT, dlam, dmu):
    """Vectorized variant of :func:`face_transport_range`."""
    if soret:
        mixture_faces_soret(start, stop, T, Y, P, invW, mw, tables, stables, 1., lam, mu, coef, DT)
        mixture_faces_soret(start, stop, T, Y, P, invW, mw, tables, stables, 1.+TEMPERATURE_STEP,
                            dlam, dmu, dcoef, dDT)
    else:
        mixture_faces(start, stop, T, Y, P, invW, mw, tables, 1., lam, mu, coef)
        mixture_faces(start, stop, T, Y, P, invW, mw, tables, 1.+TEMPERATURE_STEP, dlam, dmu, dcoef)
    for f in range(start, stop):
        h = TEMPERATURE_STEP*(.5*(T[f]+T[f+1]))
        dlam[f] = (dlam[f]-lam[f])/h
        dmu[f] = (dmu[f]-mu[f])/h
        for k in range(Y.shape[0]):
            dcoef[f, k] = (dcoef[f, k]-coef[f, k])/h
            if soret:
                dDT[f, k] = (dDT[f, k]-DT[f, k])/h


@njit(cache=True)
def _face_terms(f, Y, T, dz, coef, DT, dcoef, dDT, soret, invW, w):
    """Flux derivatives of face f between nodes f (L) and f+1 (R).

    dJ_k/dY_mR = dR_k delta_km + p1R_k q1R_m + p2_k q2R_m
    dJ_k/dY_mL = dL_k delta_km + p1L_k q1L_m - p2_k q2L_m
    dJ_k/dT_R = dTR_k, dJ_k/dT_L = dTL_k.
    ``w`` rows: dR, p1R, q1R, q2R, dL, p1L, q1L, q2L, p2, dTR, dTL.
    """
    ns = Y.shape[0]
    h = dz[f]
    sL = 0.
    sR = 0.
    for k in range(ns):
        sL += Y[k, f]*invW[k]
        sR += Y[k, f+1]*invW[k]
    WL = 1./sL
    WR = 1./sR
    CXL = 0.
    CXR = 0.
    Asum = 0.
    dAsum = 0.
    for k in range(ns):
        xl = Y[k, f]*(WL*invW[k])
        xr = Y[k, f+1]*(WR*invW[k])
        CXL += coef[f, k]*xl
        CXR += coef[f, k]*xr
        Asum += coef[f, k]*(xr-xl)/h
        dAsum += dcoef[f, k]*(xr-xl)/h
    Tf = .5*(T[f]+T[f+1])
    dT = T[f+1]-T[f]
    for k in range(ns):
        xl = Y[k, f]*(WL*invW[k])
        xr = Y[k, f+1]*(WR*invW[k])
        c = coef[f, k]
        w[0, k] = -c*WR*invW[k]/h
        w[1, k] = c*xr/h
        w[2, k] = WR*invW[k]
        w[3, k] = WR*invW[k]*(c-CXR)
        w[4, k] = c*WL*invW[k]/h+Asum
        w[5, k] = -c*xl/h
        w[6, k] = WL*invW[k]
        w[7, k] = WL*invW[k]*(c-CXL)
        w[8, k] = Y[k, f]/h
        dJdTf = -dcoef[f, k]*(xr-xl)/h+Y[k, f]*dAsum
        explicit = 0.
        if soret:
            dJdTf -= dDT[f, k]*dT/(Tf*h)-DT[f, k]*dT/(Tf*Tf*h)
            explicit = DT[f, k]/(Tf*h)
        w[9, k] = .5*dJdTf-explicit
        w[10, k] = .5*dJdTf+explicit


@njit(cache=True)
def _nasa_cp(T, nasa_lo, nasa_hi, nasa_tmid, invW, cpk, dcpk):
    """Mass-specific species cp_k and dcp_k/dT [J/(kg K), J/(kg K^2)]."""
    for k in range(invW.size):
        c = nasa_hi[k] if T > nasa_tmid[k] else nasa_lo[k]
        cpk[k] = R_UNIV*invW[k]*(c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4]))))
        dcpk[k] = R_UNIV*invW[k]*(c[1]+T*(2.*c[2]+T*(3.*c[3]+T*4.*c[4])))


@njit(cache=True)
def _radiation_derivatives(T, Y, j, radiation, dY):
    """dq/dT and dq/dY of the optically thin loss at node j (zero when off)."""
    enabled, scale, P, iH2O, iCO2, background, invW = radiation
    ns = Y.shape[0]
    for k in range(ns):
        dY[k] = 0.
    if not enabled:
        return 0.
    inv_wmix = 0.
    for k in range(ns):
        inv_wmix += Y[k, j]*invW[k]
    wm = 1./max(inv_wmix, 1e-300)
    x = 1000./T[j]
    num = 0.
    dnum = 0.
    a = 0.
    b = 0.
    if iH2O >= 0:
        a = -0.23093+x*(-1.12390+x*(9.41530+x*(-2.99880+x*(0.51382+x*-1.86840e-5))))
        da = -1.12390+x*(2.*9.41530+x*(-3.*2.99880+x*(4.*0.51382+x*5.*-1.86840e-5)))
        num += Y[iH2O, j]*invW[iH2O]*a
        dnum += Y[iH2O, j]*invW[iH2O]*da
    if iCO2 >= 0:
        b = 18.741+x*(-121.310+x*(273.500+x*(-194.050+x*(56.310+x*-5.8169))))
        db = -121.310+x*(2.*273.500+x*(-3.*194.050+x*(4.*56.310+x*5.*-5.8169)))
        num += Y[iCO2, j]*invW[iCO2]*b
        dnum += Y[iCO2, j]*invW[iCO2]*db
    p = P/ONE_ATMOSPHERE
    kP = p*wm*num
    emission = 2.*STEFAN_BOLTZMANN*T[j]**4-background
    factor = scale*2.*emission*p
    for k in range(ns):
        dY[k] = -factor*num*wm*wm*invW[k]
    if iH2O >= 0:
        dY[iH2O] += factor*wm*invW[iH2O]*a
    if iCO2 >= 0:
        dY[iCO2] += factor*wm*invW[iCO2]*b
    dkP = p*wm*dnum*(-1000./(T[j]*T[j]))
    return scale*2.*(dkP*emission+kP*8.*STEFAN_BOLTZMANN*T[j]**3)


@njit(cache=True)
def _rank1(Aout, scale, p, q, ns):
    for k in range(ns):
        s = scale*p[k]
        if s != 0.:
            for m in range(ns):
                Aout[3+k, 3+m] += s*q[m]


@njit(cache=True)
def _node_rows(j, m, V, T, Y, rho, cp, omega, hk, J0, lam0, mu0, dz, scalars,
               fuel_Y, oxidizer_Y, excess, fixed, nasa, invW, radiation, dy, dt, partials_start,
               coef, DT, dcoef, dDT, dlam, dmu, soret,
               diag, lowD, lowP, lowQ, upD, upP, upQ, border, ws, diag_offset=0):
    """Rows of node j against x[j-1], x[j], x[j+1]; writes diag[j-diag_offset], A[j,j-1], A[j,j+1]."""
    ns, n = Y.shape
    nv = ns+3
    mref, aref, Tref, lref, sscale, escale = scalars[0], scalars[1], scalars[2], scalars[3], scalars[4], scalars[5]
    nasa_lo, nasa_hi, nasa_tmid = nasa
    wm_, wp_, cpk, dcpk, cpkm, dcpkm, cpkp, dcpkp, drad, gH, Jbar, tmp = ws
    B = diag[j-diag_offset]
    B[:, :] = 0.
    has_low = j >= 1
    has_up = j <= n-2
    if has_low:
        lowD[j-1, :] = 0.
        lowP[j-1, :, :] = 0.
        lowQ[j-1, :, :] = 0.
    if has_up:
        upD[j, :] = 0.
        upP[j, :, :] = 0.
        upQ[j, :, :] = 0.
    # Faces: f- = j-1 (node j on its right), f+ = j (node j on its left).
    if has_low:
        _face_terms(j-1, Y, T, dz, coef, DT, dcoef, dDT, soret, invW, wm_)
    if has_up:
        _face_terms(j, Y, T, dz, coef, DT, dcoef, dDT, soret, invW, wp_)
    if j == 0 or j == n-1:
        # Boundary rows: (m-target)/mref, V/aref, (T-Tfeed)/Tref,
        # species (m(Y-feed)+J_face)/mref and the excess row sum(Y)-1.
        feed = fuel_Y if j == 0 else oxidizer_Y
        e = excess[j]
        B[0, 0] = 1.
        B[1, 1] = 1.
        B[2, 2] = 1.
        if j == 0:
            w = wp_
            # Node j is the left node of face 0; the neighbor is its right.
            for k in range(ns):
                if k == e:
                    continue
                r = 3+k
                B[r, 0] = Y[k, j]-feed[k]
                B[r, 2] = w[10, k]*Tref/mref
                B[r, r] += (m[j]+w[4, k])/mref
                s1 = w[5, k]/mref
                s2 = -w[8, k]/mref
                for q in range(ns):
                    B[r, 3+q] += s1*w[6, q]+s2*w[7, q]
                upD[j, r] = w[0, k]/mref
                upP[j, r, 0] = w[1, k]/mref
                upP[j, r, 1] = w[8, k]/mref
                upP[j, r, 2] = w[9, k]*Tref/mref
            for q in range(ns):
                upQ[j, 3+q, 0] = w[2, q]
                upQ[j, 3+q, 1] = w[3, q]
            upQ[j, 2, 2] = 1.
        else:
            w = wm_
            for k in range(ns):
                if k == e:
                    continue
                r = 3+k
                B[r, 0] = Y[k, j]-feed[k]
                B[r, 2] = w[9, k]*Tref/mref
                B[r, r] += (m[j]+w[0, k])/mref
                s1 = w[1, k]/mref
                s2 = w[8, k]/mref
                for q in range(ns):
                    B[r, 3+q] += s1*w[2, q]+s2*w[3, q]
                lowD[j-1, r] = w[4, k]/mref
                lowP[j-1, r, 0] = w[5, k]/mref
                lowP[j-1, r, 1] = -w[8, k]/mref
                lowP[j-1, r, 2] = w[10, k]*Tref/mref
            for q in range(ns):
                lowQ[j-1, 3+q, 0] = w[6, q]
                lowQ[j-1, 3+q, 1] = w[7, q]
            lowQ[j-1, 2, 2] = 1.
        for q in range(ns):
            B[3+e, 3+q] = 1.
        _border_rows(j, n, m, V, T, Y, rho, dz, mref, aref, Tref, invW, border)
        return
    hm, hp = dz[j-1], dz[j]
    vol = .5*(hm+hp)
    if m[j] >= 0.:
        donor, sp, dirn = j-1, hm, 1.
    else:
        donor, sp, dirn = j+1, hp, -1.
    conv = m[j]*dirn/sp
    # Node thermochemistry.
    inv_wmix = 0.
    for k in range(ns):
        inv_wmix += Y[k, j]*invW[k]
    wmix = 1./inv_wmix
    drho_dT = -rho[j]/T[j]
    _nasa_cp(T[j], nasa_lo, nasa_hi, nasa_tmid, invW, cpk, dcpk)
    dcp_dT = 0.
    for k in range(ns):
        dcp_dT += Y[k, j]*dcpk[k]
    # ---- Row 0: continuity (m_j - m_{j-1} + hm (rho V)_{j-1} + hm (rho V)_j)/mref.
    B[0, 0] = 1.
    B[0, 1] = hm*rho[j]*aref/mref
    B[0, 2] = hm*V[j]*drho_dT*Tref/mref
    for q in range(ns):
        B[0, 3+q] = -hm*V[j]*rho[j]*wmix*invW[q]/mref
    inv_wl = 0.
    for k in range(ns):
        inv_wl += Y[k, j-1]*invW[k]
    wl = 1./inv_wl
    lowD[j-1, 0] = -1.
    lowP[j-1, 0, 2] = -hm*V[j-1]*rho[j-1]/T[j-1]*Tref/mref
    lowP[j-1, 0, 4] = 1.
    lowQ[j-1, 1, 4] = hm*rho[j-1]*aref/mref
    for q in range(ns):
        lowQ[j-1, 3+q, 4] = -hm*V[j-1]*rho[j-1]*wl*invW[q]/mref
    _border_rows(j, n, m, V, T, Y, rho, dz, mref, aref, Tref, invW, border)
    # ---- Row 1: momentum.
    gV = dirn*(V[j]-V[donor])/sp
    dVm = V[j]-V[j-1]
    dVp = V[j+1]-V[j]
    B[1, 0] = gV*mref/lref
    B[1, 1] = (conv+2.*rho[j]*V[j]+(mu0[j]/hp+mu0[j-1]/hm)/vol)*aref/lref
    B[1, 2] = (V[j]**2*drho_dT-(.5*dmu[j]*dVp/hp-.5*dmu[j-1]*dVm/hm)/vol)*Tref/lref
    for q in range(ns):
        B[1, 3+q] = -V[j]**2*rho[j]*wmix*invW[q]/lref
    lowD[j-1, 1] = (-mu0[j-1]/(hm*vol)-(conv if donor == j-1 else 0.))*aref/lref
    lowP[j-1, 1, 2] = (.5*dmu[j-1]*dVm/hm)/vol*Tref/lref
    upD[j, 1] = (-mu0[j]/(hp*vol)-(conv if donor == j+1 else 0.))*aref/lref
    upP[j, 1, 2] = -(.5*dmu[j]*dVp/hp)/vol*Tref/lref
    # ---- Species rows: (m gradY + (J+ - J-)/vol - omega)/sscale.
    pj = j-partials_start
    for k in range(ns):
        r = 3+k
        gY = dirn*(Y[k, j]-Y[k, donor])/sp
        B[r, 0] = gY*mref/sscale
        B[r, 2] = ((wp_[10, k]-wm_[9, k])/vol-dt[k, pj])*Tref/sscale
        row = dy[k, pj]
        for q in range(ns):
            B[r, 3+q] = -row[q]/sscale
        B[r, r] += (conv+(wp_[4, k]-wm_[0, k])/vol)/sscale
        lowD[j-1, r] = (-wm_[4, k]/vol-(conv if donor == j-1 else 0.))/sscale
        lowP[j-1, r, 0] = -wm_[5, k]/(vol*sscale)
        lowP[j-1, r, 1] = wm_[8, k]/(vol*sscale)
        lowP[j-1, r, 2] = -wm_[10, k]/vol*Tref/sscale
        upD[j, r] = (wp_[0, k]/vol-(conv if donor == j+1 else 0.))/sscale
        upP[j, r, 0] = wp_[1, k]/(vol*sscale)
        upP[j, r, 1] = wp_[8, k]/(vol*sscale)
        upP[j, r, 2] = wp_[9, k]/vol*Tref/sscale
    # Face rank-one terms in the diagonal block: + (dJ+/dY_L - dJ-/dY_R)/vol.
    inv = 1./(vol*sscale)
    _rank1(B, inv, wp_[5], wp_[6], ns)
    for k in range(ns):
        tmp[k] = -wp_[8, k]
    _rank1(B, inv, tmp, wp_[7], ns)
    _rank1(B, -inv, wm_[1], wm_[2], ns)
    _rank1(B, -inv, wm_[8], wm_[3], ns)
    for q in range(ns):
        lowQ[j-1, 3+q, 0] = wm_[6, q]
        lowQ[j-1, 3+q, 1] = wm_[7, q]
        upQ[j, 3+q, 0] = wp_[2, q]
        upQ[j, 3+q, 1] = wp_[3, q]
    lowQ[j-1, 2, 2] = 1.
    upQ[j, 2, 2] = 1.
    # ---- Row 2: energy.
    if fixed:
        B[2, :] = 0.
        B[2, 2] = 1.
        lowP[j-1, 2, 2] = 0.
        upP[j, 2, 2] = 0.
        return
    _nasa_cp(T[j-1], nasa_lo, nasa_hi, nasa_tmid, invW, cpkm, dcpkm)
    _nasa_cp(T[j+1], nasa_lo, nasa_hi, nasa_tmid, invW, cpkp, dcpkp)
    gT = dirn*(T[j]-T[donor])/sp
    dTm = T[j]-T[j-1]
    dTp = T[j+1]-T[j]
    bwd = .5+m[j]/(2.*STAGNATION*mref)
    blended = 0. < bwd < 1.
    dgm = 0.
    for k in range(ns):
        Jbar[k] = .5*(J0[k, j-1]+J0[k, j])
        if blended:
            gH[k] = bwd*(hk[k, j]-hk[k, j-1])/hm+(1.-bwd)*(hk[k, j+1]-hk[k, j])/hp
            dgm += Jbar[k]*((hk[k, j]-hk[k, j-1])/hm-(hk[k, j+1]-hk[k, j])/hp)
        else:
            gH[k] = dirn*(hk[k, j]-hk[k, donor])/sp
    dgm /= 2.*STAGNATION*mref
    B[2, 0] = (cp[j]*gT+(dgm if blended else 0.))*mref/escale
    # Temperature columns.
    sumj = 0.
    suml = 0.
    sumu = 0.
    for k in range(ns):
        sumj += .5*(wm_[9, k]+wp_[10, k])*gH[k]+cpk[k]*omega[k, j]+hk[k, j]*dt[k, pj]
        suml += .5*wm_[10, k]*gH[k]
        sumu += .5*wp_[9, k]*gH[k]
        if blended:
            sumj += Jbar[k]*cpk[k]*(bwd/hm-(1.-bwd)/hp)
            suml += -Jbar[k]*bwd*cpkm[k]/hm
            sumu += Jbar[k]*(1.-bwd)*cpkp[k]/hp
        else:
            sumj += Jbar[k]*dirn*cpk[k]/sp
            if donor == j-1:
                suml += -Jbar[k]*dirn*cpkm[k]/sp
            else:
                sumu += -Jbar[k]*dirn*cpkp[k]/sp
    qT = _radiation_derivatives(T, Y, j, radiation, drad)
    B[2, 2] = (m[j]*cp[j]*dirn/sp+m[j]*dcp_dT*gT+(lam0[j]/hp+lam0[j-1]/hm)/vol
               -(.5*dlam[j]*dTp/hp-.5*dlam[j-1]*dTm/hm)/vol+sumj+qT)*Tref/escale
    lowD[j-1, 2] = (-lam0[j-1]/(hm*vol)+.5*dlam[j-1]*dTm/(hm*vol)+suml
                    -(m[j]*cp[j]*dirn/sp if donor == j-1 else 0.))*Tref/escale
    upD[j, 2] = (-lam0[j]/(hp*vol)-.5*dlam[j]*dTp/(hp*vol)+sumu
                 -(m[j]*cp[j]*dirn/sp if donor == j+1 else 0.))*Tref/escale
    # Composition columns of the energy row.
    gp1L = 0.
    gp2 = 0.
    gp1R = 0.
    gm2 = 0.
    gp1Rl = 0.
    gp2l = 0.
    gp1Ru = 0.
    gp2u = 0.
    for k in range(ns):
        gp1L += gH[k]*wp_[5, k]
        gp2 += gH[k]*wp_[8, k]
        gp1R += gH[k]*wm_[1, k]
        gm2 += gH[k]*wm_[8, k]
        gp1Rl += gH[k]*wm_[5, k]
        gp2l += gH[k]*wm_[8, k]
        gp1Ru += gH[k]*wp_[1, k]
        gp2u += gH[k]*wp_[8, k]
    for q in range(ns):
        tmp[q] = 0.
    for k in range(ns):
        h = hk[k, j]
        row = dy[k, pj]
        for q in range(ns):
            tmp[q] += h*row[q]
    for q in range(ns):
        value = (m[j]*gT*cpk[q]+tmp[q]+drad[q]
                 +.5*(gH[q]*wm_[0, q]+gp1R*wm_[2, q]+gm2*wm_[3, q])
                 +.5*(gH[q]*wp_[4, q]+gp1L*wp_[6, q]-gp2*wp_[7, q]))
        B[2, 3+q] = value/escale
        lowQ[j-1, 3+q, 3] = .5*(gH[q]*wm_[4, q]+gp1Rl*wm_[6, q]-gp2l*wm_[7, q])/escale
        upQ[j, 3+q, 3] = .5*(gH[q]*wp_[0, q]+gp1Ru*wp_[2, q]+gp2u*wp_[3, q])/escale
    lowP[j-1, 2, 3] = 1.
    upP[j, 2, 3] = 1.


@njit(cache=True)
def _border_rows(j, n, m, V, T, Y, rho, dz, mref, aref, Tref, invW, border):
    """Final continuity row against nodes n-2 (border[0]) and n-1 (border[1])."""
    if j < n-2:
        return
    side = j-(n-2)
    ns = Y.shape[0]
    h = dz[n-2]
    inv_w = 0.
    for k in range(ns):
        inv_w += Y[k, j]*invW[k]
    wmix = 1./inv_w
    border[side, 0] = 1. if side == 1 else -1.
    border[side, 1] = h*rho[j]*aref/mref
    border[side, 2] = -h*V[j]*rho[j]/T[j]*Tref/mref
    for q in range(ns):
        border[side, 3+q] = -h*V[j]*rho[j]*wmix*invW[q]/mref


@njit(cache=True)
def _row_workspace(ns):
    return (np.empty((11, ns)), np.empty((11, ns)), np.empty(ns), np.empty(ns), np.empty(ns), np.empty(ns),
            np.empty(ns), np.empty(ns), np.empty(ns), np.empty(ns), np.empty(ns), np.empty(ns))


@njit(cache=True, nogil=True)
def rows_range(start, stop, m, V, T, Y, rho, cp, omega, hk, J0, lam0, mu0, dz, scalars,
               fuel_Y, oxidizer_Y, excess, fixed, nasa, invW, radiation, dy, dt, partials_start,
               coef, DT, dcoef, dDT, dlam, dmu, soret,
               diag, lowD, lowP, lowQ, upD, upP, upQ, border, diag_offset):
    """Rows of nodes [start, stop); node j writes diag[j-diag_offset], A[j,j-1], A[j,j+1] only.

    ``diag`` may be the whole grid's blocks (offset 0) or a chunk whose first
    block belongs to node ``diag_offset``.
    """
    ws = _row_workspace(Y.shape[0])
    for j in range(start, stop):
        _node_rows(j, m, V, T, Y, rho, cp, omega, hk, J0, lam0, mu0, dz, scalars,
                   fuel_Y, oxidizer_Y, excess, fixed, nasa, invW, radiation, dy, dt, partials_start,
                   coef, DT, dcoef, dDT, dlam, dmu, soret,
                   diag, lowD, lowP, lowQ, upD, upP, upQ, border, ws, diag_offset)


# Bytes of analytic chemistry partials held by one range at a time.
_PARTIALS_TILE_BYTES = 16*1024**2


class StructuredBlocks:
    """Bordered block-tridiagonal Jacobian with ``diag + P Q^T`` coupling blocks.

    ``lower[i] = A[i+1, i]`` and ``upper[i] = A[i, i+1]`` as in
    :class:`BorderedBlocks`; ``dense()`` returns that representation.
    """

    def __init__(self, diag, lowD, lowP, lowQ, upD, upP, upQ, border, mass, shape=None):
        self.diag, self.border, self.mass = diag, border, mass
        self.lowD, self.lowP, self.lowQ = lowD, lowP, lowQ
        self.upD, self.upP, self.upQ = upD, upP, upQ
        self.n, self.nv = (diag.shape[0], diag.shape[1]) if diag is not None else shape
        self.consumed = False
        self.assemble = None

    @property
    def lazy(self):
        return self.diag is None

    @staticmethod
    def _expand(D, P, Q):
        out = np.einsum('iar,ibr->iab', P, Q)
        index = np.arange(D.shape[1])
        out[:, index, index] += D
        return out

    def dense(self):
        from kflame.flame.diffusion import BorderedBlocks
        if self.lazy:
            diag = np.empty((self.n, self.nv, self.nv))
            self.assemble(0, self.n, diag, 0)
            return BorderedBlocks(self._expand(self.lowD, self.lowP, self.lowQ), diag,
                                  self._expand(self.upD, self.upP, self.upQ), self.border, self.mass)
        return BorderedBlocks(self._expand(self.lowD, self.lowP, self.lowQ), self.diag,
                              self._expand(self.upD, self.upP, self.upQ), self.border, self.mass)


def structured_blocks(problem, x, *, fixed_temperature=None, cached_properties=None, lazy=False):
    """Analytic frozen-transport Jacobian of a mixture-averaged counterflow problem.

    With ``lazy`` only the coupling blocks, border and mass are allocated; the
    dense diagonal blocks are assembled on demand by ``blocks.assemble(start,
    stop, buffer, offset)`` (also filling the coupling blocks of those nodes),
    so a factorization can consume them chunk by chunk without ever holding
    all n*nv*nv entries. Raises ValueError when the analytic chemistry is
    unavailable; callers then keep the finite-difference Jacobian.
    """
    from kflame.chemistry.analytic import prepare_species_partials, species_partials_range_into
    from kflame.flame.buffer_pool import pool_of
    from kflame.flame.diffusion import _split, heavy_threads
    multicomponent = problem.transport_model != 'mixture-averaged' or problem.backend.uses_multicomponent_flux
    if multicomponent and getattr(problem, 'trace_tail', None) is None:
        raise ValueError('Structured Jacobian requires mixture-averaged transport')
    # Pruned multicomponent model: mixture-form face derivatives around the
    # multicomponent base fluxes, conductivity and viscosity (a preconditioner).
    data = problem._compiled_data()
    f = problem.unpack(x)
    T, Y = np.ascontiguousarray(f['T']), np.ascontiguousarray(f['Y'])
    m, V = np.ascontiguousarray(f['m']), np.ascontiguousarray(f['V'])
    if cached_properties is None:
        rho, cp, omega, hk, lam0, J0, mu0 = problem.properties(dict(f, T=T, Y=Y))
    else:
        rho, cp, omega, hk, lam0, J0, mu0 = cached_properties
    rho, cp = np.ascontiguousarray(rho), np.ascontiguousarray(cp)
    omega, hk = np.ascontiguousarray(omega), np.ascontiguousarray(hk)
    J0, lam0, mu0 = np.ascontiguousarray(J0), np.ascontiguousarray(lam0), np.ascontiguousarray(mu0)
    T, Y = prepare_species_partials(problem.backend, T, Y)
    ns, n, nv = problem.ns, problem.n, problem.nv
    soret = bool(problem.soret_enabled)
    coef, DT, dcoef, dDT = (np.empty((n-1, ns)) for _ in range(4))
    lam, mu, dlam, dmu = (np.empty(n-1) for _ in range(4))
    if 'fast_mixture' in data and getattr(problem, 'fast_transport', True):
        if not soret:
            DT[:] = 0.
            dDT[:] = 0.
        _split(fast_face_transport_range, n-1, T, Y, float(problem.P), data['invW'], data['mw'],
               data['fast_mixture'], data['fast_soret'], soret, coef, DT, lam, mu, dcoef, dDT, dlam, dmu,
               minimum=2 if ns >= 100 else 32, threads=heavy_threads() if ns >= 100 else None)
    else:
        _split(face_transport_range, n-1, T, Y, problem.dz, float(problem.P), data['invW'], data['mw'],
               data['mixture'], soret, coef, DT, lam, mu, dcoef, dDT, dlam, dmu, minimum=4)
    c = problem.case
    scalars = (float(problem.mref), float(problem.aref), float(problem.Tref), float(problem.lref),
               float(problem.species_scale), float(problem.energy_scale))
    fixed = fixed_temperature is not None
    nasa = (problem.backend._nb_nasa_lo, problem.backend._nb_nasa_hi, problem.backend._nb_nasa_tmid)
    pool = pool_of(problem)
    lowD, upD = np.zeros((n-1, nv)), np.zeros((n-1, nv))
    lowP, lowQ = np.zeros((n-1, nv, RANK)), np.zeros((n-1, nv, RANK))
    upP, upQ = np.zeros((n-1, nv, RANK)), np.zeros((n-1, nv, RANK))
    border = np.zeros((2, nv))
    fuel_Y = np.ascontiguousarray(c.fuel_Y, dtype=float)
    oxidizer_Y = np.ascontiguousarray(c.oxidizer_Y, dtype=float)
    excess = np.ascontiguousarray(problem.excess, dtype=np.int64)
    tile = max(1, min(32, _PARTIALS_TILE_BYTES//(8*ns*(ns+1))))
    backend, dz, invW, radiation = problem.backend, problem.dz, data['invW'], problem.radiation

    def assemble(start, stop, buffer, offset):
        """Rows of nodes [start, stop): diagonal blocks into buffer[j-offset]."""
        # Private, reused chemistry tile of this call (logical axes K, B, K).
        dy = pool.take((tile, ns, ns)).transpose(1, 0, 2)
        dt = np.empty((ns, tile))
        for first in range(start, stop, tile):
            last = min(first+tile, stop)
            species_partials_range_into(backend, T, Y, first, last, dy, dt)
            rows_range(first, last, m, V, T, Y, rho, cp, omega, hk, J0, lam0, mu0, dz, scalars,
                       fuel_Y, oxidizer_Y, excess, fixed, nasa, invW, radiation, dy, dt, first,
                       coef, DT, dcoef, dDT, dlam, dmu, soret,
                       buffer, lowD, lowP, lowQ, upD, upP, upQ, border, offset)
    mass = np.zeros((n, nv))
    mass[1:-1, 1] = rho[1:-1]*problem.aref/problem.lref
    if not fixed:
        mass[1:-1, 2] = rho[1:-1]*cp[1:-1]*problem.Tref/problem.energy_scale
    mass[1:-1, 3:] = (rho[1:-1]/problem.species_scale)[:, None]
    if lazy:
        blocks = StructuredBlocks(None, lowD, lowP, lowQ, upD, upP, upQ, border, mass.ravel(), shape=(n, nv))
        blocks.assemble = assemble
        return blocks
    diag = pool.take((n, nv, nv))
    _split(lambda start, stop: assemble(start, stop, diag, 0), n, minimum=2,
           threads=heavy_threads() if ns >= 100 else None)
    blocks = StructuredBlocks(diag, lowD, lowP, lowQ, upD, upP, upQ, border, mass.ravel())
    blocks.assemble = assemble
    return blocks
