"""Compiled counterflow residual rows, face transport and local Jacobian blocks.

The residual and the Jacobian share the same row functions. Perturbing one
node changes only its own thermochemistry, the two adjacent faces and three
row blocks, so the Jacobian kernel recomputes exactly those quantities with
the same finite-difference steps as a column-by-column evaluation of the full
residual. Face quantities that depend only on temperature (pure-species fits,
binary diffusion, Wilke factors, Soret collision ratios) are reused by the
composition columns of the same face.
"""
import math

import numpy as np
from numba import njit

from kflame.chemistry.jacobian import grouped_node
from kflame.chemistry.mechanism import R_UNIV
from kflame.chemistry.multicomponent_kernels import (composition_terms, composition_terms_ws,
                                                     composition_workspace, temperature_terms)


@njit(cache=True)
def boundary_rows(out, o, j, face, m, V, T, Y, J, feed, target, Tfeed, mref, aref, Tref, excess):
    ns = Y.shape[0]
    out[o] = (m[j]-target)/mref
    out[o+1] = V[j]/aref
    out[o+2] = (T[j]-Tfeed)/Tref
    total = 0.
    for k in range(ns):
        out[o+3+k] = (m[j]*(Y[k, j]-feed[k])+J[k, face])/mref
        total += Y[k, j]
    out[o+3+excess] = total-1.


STEFAN_BOLTZMANN = 5.670374419e-8
ONE_ATMOSPHERE = 101325.


@njit(cache=True)
def radiative_loss(T, Y, j, radiation):
    """Optically thin radiative loss [W/m3] at node j; zero when disabled.

    Planck-mean absorption of H2O and CO2 from the polynomial fits used by
    Cantera's Flow1D (3.2), kP = P/1 atm * sum_k X_k a_k(1000/T), and
    q = scale * 2 kP (2 sigma T^4 - background), where ``background`` holds
    sigma (eps_fuel T_fuel^4 + eps_oxidizer T_oxidizer^4) of the two inlets.
    """
    enabled, scale, P, iH2O, iCO2, background, invW = radiation
    if not enabled:
        return 0.
    inv_wmix = 0.
    for k in range(Y.shape[0]):
        inv_wmix += Y[k, j]*invW[k]
    wmix = 1./max(inv_wmix, 1e-300)
    x = 1000./T[j]
    kP = 0.
    if iH2O >= 0:
        poly = -0.23093+x*(-1.12390+x*(9.41530+x*(-2.99880+x*(0.51382+x*-1.86840e-5))))
        kP += Y[iH2O, j]*wmix*invW[iH2O]*poly
    if iCO2 >= 0:
        poly = 18.741+x*(-121.310+x*(273.500+x*(-194.050+x*(56.310+x*-5.8169))))
        kP += Y[iCO2, j]*wmix*invW[iCO2]*poly
    kP *= P/ONE_ATMOSPHERE
    return scale*2.*kP*(2.*STEFAN_BOLTZMANN*T[j]**4-background)


@njit(cache=True)
def radiation_profile(T, Y, radiation):
    q = np.zeros(T.size)
    for j in range(T.size):
        q[j] = radiative_loss(T, Y, j, radiation)
    return q


@njit(cache=True)
def interior_rows(out, o, j, m, V, T, Y, rho, cp, omega, hk, lam, J, mu, hm, hp, L,
                  mref, lref, Tref, species_scale, energy_scale, fixed, Tfixed, radiation):
    """Steady rows of node j; faces j-1 and j lie between nodes (j-1,j), (j,j+1)."""
    ns = Y.shape[0]
    volume = .5*(hm+hp)
    out[o] = (m[j]-m[j-1]+hm*(rho[j-1]*V[j-1]+rho[j]*V[j]))/mref
    if m[j] >= 0.:
        donor = j-1
        spacing = hm
        direction = 1.
    else:
        donor = j+1
        spacing = hp
        direction = -1.
    gradV = direction*(V[j]-V[donor])/spacing
    gradT = direction*(T[j]-T[donor])/spacing
    out[o+1] = (m[j]*gradV+rho[j]*V[j]**2+L
                -(mu[j]*(V[j+1]-V[j])/hp-mu[j-1]*(V[j]-V[j-1])/hm)/volume)/lref
    enthalpy_diffusion = 0.
    reaction_enthalpy = 0.
    for k in range(ns):
        gradY = direction*(Y[k, j]-Y[k, donor])/spacing
        out[o+3+k] = (m[j]*gradY+(J[k, j]-J[k, j-1])/volume-omega[k, j])/species_scale
        if not fixed:
            gradH = direction*(hk[k, j]-hk[k, donor])/spacing
            enthalpy_diffusion += .5*(J[k, j-1]+J[k, j])*gradH
            reaction_enthalpy += hk[k, j]*omega[k, j]
    if fixed:
        out[o+2] = (T[j]-Tfixed)/Tref
    else:
        out[o+2] = (m[j]*cp[j]*gradT
                    -(lam[j]*(T[j+1]-T[j])/hp-lam[j-1]*(T[j]-T[j-1])/hm)/volume
                    +enthalpy_diffusion+reaction_enthalpy
                    +radiative_loss(T, Y, j, radiation))/energy_scale


@njit(cache=True)
def assemble_residual(m, V, T, Y, L, rho, cp, omega, hk, lam, J, mu, dz,
                      fuel_Y, oxidizer_Y, mfuel, moxidizer, Tfuel, Toxidizer,
                      mref, aref, Tref, lref, species_scale, energy_scale, excess,
                      transient, oldV, oldT, oldY, dt, fixed, temperature, radiation):
    ns, n = Y.shape
    nv = ns+3
    residual = np.zeros(n*nv+1)
    for j in range(n):
        offset = j*nv
        if j == 0:
            boundary_rows(residual, offset, j, 0, m, V, T, Y, J, fuel_Y, mfuel, Tfuel,
                          mref, aref, Tref, excess[j])
            continue
        if j == n-1:
            boundary_rows(residual, offset, j, n-2, m, V, T, Y, J, oxidizer_Y, -moxidizer,
                          Toxidizer, mref, aref, Tref, excess[j])
            continue
        interior_rows(residual, offset, j, m, V, T, Y, rho, cp, omega, hk, lam, J, mu,
                      dz[j-1], dz[j], L, mref, lref, Tref, species_scale, energy_scale,
                      fixed, temperature[j], radiation)
        if transient:
            for k in range(ns):
                residual[offset+3+k] += rho[j]*(Y[k, j]-oldY[k, j])/(dt*species_scale)
            if not fixed:
                residual[offset+2] += rho[j]*cp[j]*(T[j]-oldT[j])/(dt*energy_scale)
            residual[offset+1] += rho[j]*(V[j]-oldV[j])/(dt*lref)
    residual[-1] = (m[-1]-m[-2]+dz[-1]*(rho[-2]*V[-2]+rho[-1]*V[-1]))/mref
    return residual


@njit(cache=True)
def _poly4(c, logT):
    return c[0]+logT*(c[1]+logT*(c[2]+logT*(c[3]+logT*c[4])))


@njit(cache=True)
def face_temperature_data(Tf, P, mw, cond_poly, diff_poly, visc_poly, quarter_ratio, wilke_den,
                          soret, eps_pair, cstar_poly, cond, invB, diagB, root, phi, scale):
    """Composition-independent face data, shared by all composition columns."""
    ns = mw.size
    logT = math.log(Tf)
    sqrtT = math.sqrt(Tf)
    TsqrtT = Tf*sqrtT
    quarter = Tf**.25
    for k in range(ns):
        cond[k] = sqrtT*_poly4(cond_poly[k], logT)
        root[k] = abs(quarter*_poly4(visc_poly[k], logT))
        diagB[k] = TsqrtT*_poly4(diff_poly[k, k], logT)
    for k in range(ns):
        for j in range(k+1, ns):
            binary = TsqrtT*_poly4(diff_poly[k, j], logT)
            inverse = 1./max(binary, 1e-300)
            invB[k, j] = inverse
            invB[j, k] = inverse
            if soret:
                z = math.log(Tf/eps_pair[k, j])
                star = cstar_poly[k, j, 8]
                for power in range(7, -1, -1):
                    star = star*z+cstar_poly[k, j, power]
                value = (1.2*star-1)/(binary/P)/(mw[k]+mw[j])
                scale[k, j] = value
                scale[j, k] = value
        invB[k, k] = 0.
        scale[k, k] = 0.
    for k in range(ns):
        for j in range(ns):
            ratio = root[k]/max(root[j], 1e-150)
            phi[k, j] = (1.+ratio*quarter_ratio[k, j])**2/wilke_den[k, j]


@njit(cache=True)
def face_viscosity(X, root, phi):
    ns = X.size
    value = 0.
    for k in range(ns):
        total = 0.
        for j in range(ns):
            total += X[j]*phi[k, j]
        value += max(X[k], 1e-300)*root[k]**2/max(total, 1e-300)
    return value


@njit(cache=True)
def mixture_sums(wm, Yf, invW, cond, invB, phi, X, SD, Smu):
    """Mole fractions and the composition sums of conductivity, D_km and Wilke."""
    ns = Yf.size
    sum1 = 0.
    sum2 = 0.
    for k in range(ns):
        X[k] = Yf[k]*wm*invW[k]
        xk = max(X[k], 1e-300)
        sum1 += xk*cond[k]
        sum2 += xk/max(cond[k], 1e-300)
    for k in range(ns):
        sumd = 0.
        total = 0.
        for j in range(ns):
            if j != k:
                sumd += max(X[j], 1e-300)*invB[k, j]
            total += X[j]*phi[k, j]
        SD[k] = sumd
        Smu[k] = total
    return sum1, sum2


@njit(cache=True)
def mixture_from_sums(Tf, P, wm, X, sum1, sum2, SD, Smu, mw, diagB, root, D):
    rho = P*wm/(R_UNIV*Tf)
    lam = .5*(sum1+1./max(sum2, 1e-300))
    mu = 0.
    for k in range(mw.size):
        if SD[k] <= 0.:
            D[k] = diagB[k]/P
        else:
            D[k] = (wm-max(X[k], 1e-300)*mw[k])/(P*wm*max(SD[k], 1e-300))
        mu += max(X[k], 1e-300)*root[k]**2/max(Smu[k], 1e-300)
    return rho, lam, mu


@njit(cache=True)
def mixture_rank1(l, Ynew, delta, inv, wm, X, sum1, sum2, SD, Smu, invW, cond, invB, phi, Xn, SDn, Smun):
    """Composition sums after Yf[l] += delta, in O(ns).

    Every other mole fraction scales by Wmix'/Wmix; species l changes in
    addition. Equal to recomputing the sums up to rounding (and the
    1e-300 floors of absent species, which do not reach any result).
    """
    ns = X.size
    inv2 = inv+delta*invW[l]
    wm2 = 1./max(inv2, 1e-300)
    s = wm2/wm
    xl = Ynew*wm2*invW[l]
    for j in range(ns):
        Xn[j] = s*X[j]
    Xn[l] = xl
    dp = max(xl, 1e-300)-s*max(X[l], 1e-300)
    draw = xl-s*X[l]
    for k in range(ns):
        SDn[k] = s*SD[k]+(dp*invB[k, l] if k != l else 0.)
        Smun[k] = s*Smu[k]+draw*phi[k, l]
    return wm2, s*sum1+dp*cond[l], s*sum2+dp/max(cond[l], 1e-300)


@njit(cache=True)
def face_mixture(Tf, Yf, P, invW, mw, cond, invB, diagB, root, phi, soret, scale, tiny, X, D, DT):
    """Mixture-averaged face state with the arithmetic of the grid kernels."""
    ns = Yf.size
    inv_wmix = 0.
    for k in range(ns):
        inv_wmix += Yf[k]*invW[k]
    wm = 1./max(inv_wmix, 1e-300)
    SD = np.empty(ns)
    Smu = np.empty(ns)
    sum1, sum2 = mixture_sums(wm, Yf, invW, cond, invB, phi, X, SD, Smu)
    rho, lam, mu = mixture_from_sums(Tf, P, wm, X, sum1, sum2, SD, Smu, mw, diagB, root, D)
    if soret:
        # Same clipped mole fractions and mixture diffusivities as the
        # fused mixture-averaged Soret kernel.
        denom = 0.
        for k in range(ns):
            denom += Yf[k]/mw[k]
        wmix = 1./max(denom, 1e-300)
        Xs = np.empty(ns)
        for k in range(ns):
            Xs[k] = max(Yf[k]*wmix/mw[k], 1e-20)
        a = np.zeros(ns)
        diffusion = np.empty(ns)
        for k in range(ns):
            total = 0.
            for j in range(ns):
                if j != k:
                    total += invB[j, k]*max(Xs[j], tiny)
            if total <= 0.:
                diffusion[k] = diagB[k]/P
            else:
                diffusion[k] = (wmix-max(Xs[k], tiny)*mw[k])/(P*wmix*max(total, 1e-300))
            if Yf[k] >= 1e-20:
                total = 0.
                for j in range(ns):
                    if j != k:
                        total += Xs[j]*phi[k, j]
                a[k] = (15./4.)*root[k]**2/mw[k]/(1+1.065*total/Xs[k])
        total = 0.
        for k in range(ns):
            value = 0.
            for j in range(ns):
                value += scale[k, j]*(Yf[k]*a[j]-Yf[j]*a[k])
            DT[k] = value*diffusion[k]*mw[k]*wmix
            total += DT[k]
        for k in range(ns):
            DT[k] -= Yf[k]*total
    return rho, wm, lam, mu


@njit(cache=True)
def mixture_flux(YL, YR, TL, TR, spacing, Tf, rho, wm, D, DT, soret, mw, out):
    """Molar-gradient corrected flux, matching the frozen grid expression."""
    ns = mw.size
    sL = 0.
    sR = 0.
    for k in range(ns):
        sL += YL[k]/mw[k]
        sR += YR[k]/mw[k]
    WL = 1./sL
    WR = 1./sR
    total = 0.
    for k in range(ns):
        coeff = rho*mw[k]/wm*D[k]
        value = -coeff*(YR[k]*(WR/mw[k])-YL[k]*(WL/mw[k]))/spacing
        out[k] = value
        total += value
    for k in range(ns):
        out[k] -= YL[k]*total
        if soret:
            out[k] -= DT[k]/Tf*(TR-TL)/spacing


@njit(cache=True)
def multicomponent_flux(YL, YR, TL, TR, spacing, Tf, rho, wm, multi, thermal, invW, mw, out, scratch):
    """J_k = rho W_k/Wmix^2 sum_l D_kl W_l dX_l/dz - D^T_k/T dT/dz; ``scratch`` holds W_l dX_l/dz."""
    ns = mw.size
    sL = 0.
    sR = 0.
    for k in range(ns):
        sL += YL[k]*invW[k]
        sR += YR[k]*invW[k]
    WL = 1./max(sL, 1e-300)
    WR = 1./max(sR, 1e-300)
    for k in range(ns):
        scratch[k] = mw[k]*(YR[k]*WR*invW[k]-YL[k]*WL*invW[k])/spacing
    gradT = (TR-TL)/spacing
    factor = rho/wm**2
    for k in range(ns):
        value = 0.
        for l in range(ns):
            value += multi[k, l]*scratch[l]
        out[k] = factor*mw[k]*value-thermal[k]/Tf*gradT


@njit(cache=True)
def _cp_r(T, nasa_lo, nasa_hi, nasa_tmid, out):
    for k in range(out.size):
        c = nasa_hi[k] if T > nasa_tmid[k] else nasa_lo[k]
        out[k] = c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4])))


@njit(cache=True)
def _multi_temperature(Tf, multi_data, mw, mdata):
    (log_eps_pair, epsilon, crot, zrot, frot298, viscosity_fit, diffusion_fit,
     astar_fit, bstar_fit, cstar_fit, dense, nasa_lo, nasa_hi, nasa_tmid) = multi_data
    mu, rot_relax, bdiff, a, b, c, cp_r = mdata
    _cp_r(Tf, nasa_lo, nasa_hi, nasa_tmid, cp_r)
    temperature_terms(Tf, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit, diffusion_fit,
                      astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c)


@njit(cache=True)
def _mdata(ns):
    return (np.empty(ns), np.empty(ns), np.empty((ns, ns)), np.empty((ns, ns)), np.empty((ns, ns)),
            np.empty((ns, ns)), np.empty(ns))


@njit(cache=True)
def face_state(TL, TR, YL, YR, spacing, P, invW, mw, mixture, multi_data, multicomponent, soret,
               tdata, mdata, reuse, X, D, DT, Yf, multi, thermal, flux, cws):
    """lambda, mu and the species fluxes at one face; fills ``flux``.

    With ``reuse``, temperature-only data in ``tdata``/``mdata`` belong to
    this face temperature and are not recomputed.
    """
    cond_poly, diff_poly, visc_poly, quarter_ratio, wilke_den, eps_pair, cstar_poly, tiny = mixture
    cond, invB, diagB, root, phi, scale = tdata
    Tf = .5*(TL+TR)
    for k in range(mw.size):
        Yf[k] = .5*(YL[k]+YR[k])
    if not reuse:
        face_temperature_data(Tf, P, mw, cond_poly, diff_poly, visc_poly, quarter_ratio, wilke_den,
                              soret and not multicomponent, eps_pair, cstar_poly,
                              cond, invB, diagB, root, phi, scale)
        if multicomponent:
            _multi_temperature(Tf, multi_data, mw, mdata)
    if multicomponent:
        crot, dense = multi_data[2], multi_data[10]
        mu, rot_relax, bdiff, a, b, c, cp_r = mdata
        if dense:
            rho, lam, wm = composition_terms(Tf, P, Yf, cp_r, mw, invW, crot, mu, rot_relax, bdiff,
                                             a, b, c, dense, multi, thermal)
        else:
            rho, lam, wm = composition_terms_ws(Tf, P, Yf, cp_r, mw, invW, crot, mu, rot_relax, bdiff,
                                                a, b, c, multi, thermal, cws)
        if not soret:
            thermal[:] = 0.
        multicomponent_flux(YL, YR, TL, TR, spacing, Tf, rho, wm, multi, thermal, invW, mw, flux, X)
        wsum = 0.
        for k in range(mw.size):
            wsum += Yf[k]*invW[k]
        wsum = 1./max(wsum, 1e-300)
        for k in range(mw.size):
            X[k] = Yf[k]*wsum*invW[k]
        mu = face_viscosity(X, root, phi)
    else:
        rho, wm, lam, mu = face_mixture(Tf, Yf, P, invW, mw, cond, invB, diagB, root, phi,
                                        soret, scale, tiny, X, D, DT)
        mixture_flux(YL, YR, TL, TR, spacing, Tf, rho, wm, D, DT, soret, mw, flux)
    return lam, mu


@njit(cache=True)
def _frozen_face(TL, TR, YL, YR, P, invW, mw, multi_data, soret, mdata, cws, Yf, multi, thermal, lam, side):
    """Base multicomponent coefficients of one face, kept for composition columns."""
    crot = multi_data[2]
    mu, rot_relax, bdiff, a, b, c, cp_r = mdata
    Tf = .5*(TL+TR)
    for k in range(mw.size):
        Yf[k] = .5*(YL[k]+YR[k])
    state = composition_terms_ws(Tf, P, Yf, cp_r, mw, invW, crot, mu, rot_relax, bdiff,
                                 a, b, c, multi, thermal, cws)
    lam[side] = state[1]
    if not soret:
        thermal[:] = 0.


@njit(cache=True)
def _tdata(ns):
    return (np.empty(ns), np.empty((ns, ns)), np.empty(ns), np.empty(ns),
            np.empty((ns, ns)), np.zeros((ns, ns)))


@njit(cache=True, nogil=True)
def face_range(start, stop, T, Y, dz, P, invW, mw, mixture, multi_data, multicomponent, soret, lam, mu, J):
    """Face lambda, mu and fluxes of faces [start, stop), with the Jacobian's face functions.

    Serial and GIL-free: callers split the faces among threads. Each call
    owns its scratch arrays; outputs of different ranges never overlap.
    """
    ns = Y.shape[0]
    tdata = _tdata(ns)
    mdata = _mdata(ns if multicomponent else 1)
    cws = composition_workspace(ns if multicomponent else 1)
    flux = np.empty(ns)
    X, D, DT, Yf = np.empty(ns), np.empty(ns), np.zeros(ns), np.empty(ns)
    multi, thermal = np.empty((ns, ns)), np.empty(ns)
    for f in range(start, stop):
        lam[f], mu[f] = face_state(T[f], T[f+1], Y[:, f], Y[:, f+1], dz[f], P, invW, mw, mixture,
                                   multi_data, multicomponent, soret, tdata, mdata, False,
                                   X, D, DT, Yf, multi, thermal, flux, cws)
        for k in range(ns):
            J[k, f] = flux[k]


@njit(cache=True)
def _node_workspace(ns, multicomponent):
    """Scratch arrays of one node's columns, reused by all nodes of a chunk."""
    nb, nv, nm = ns+2, ns+3, ns if multicomponent else 1
    return (np.zeros((8, 5)), np.zeros((ns, 5)), np.zeros((ns, 5)), np.zeros((ns, 5)), np.zeros((ns, 4)),
            np.empty(nb), np.empty((ns, nb)), np.empty(nv), np.empty(nb), np.empty(nb),
            np.empty((ns, nb)), np.empty((ns, nb)), np.zeros((3, nv)), np.empty(nv),
            _tdata(ns), _tdata(ns), _tdata(ns), _mdata(nm), _mdata(nm), _mdata(nm),
            np.empty(ns), np.empty(ns), np.zeros(ns), np.empty(ns), np.empty((ns, ns)), np.empty(ns),
            np.empty(ns), np.zeros((2, 3, ns)), np.zeros((2, 4)), np.zeros((2, ns)), np.empty(ns),
            np.empty(ns), np.empty(ns), np.empty((ns, 2)), composition_workspace(nm),
            np.zeros((2, nm, nm)), np.zeros((2, nm)), np.zeros(2))


@njit(cache=True)
def _window_rows(out, w, n, j, mW, VW, TW, YW, rhoW, cpW, omegaW, hkW, lamW, JW, muW, dzW, L,
                 scalars, fuel_Y, oxidizer_Y, excess, fixed, Tfixed, radiation):
    mref, aref, Tref, lref, species_scale, energy_scale, mfuel, moxidizer, Tfuel, Toxidizer = scalars
    g = j-2+w
    if g == 0:
        boundary_rows(out, 0, w, w, mW, VW, TW, YW, JW, fuel_Y, mfuel, Tfuel, mref, aref, Tref, excess[g])
    elif g == n-1:
        boundary_rows(out, 0, w, w-1, mW, VW, TW, YW, JW, oxidizer_Y, -moxidizer, Toxidizer,
                      mref, aref, Tref, excess[g])
    else:
        interior_rows(out, 0, w, mW, VW, TW, YW, rhoW, cpW, omegaW, hkW, lamW, JW, muW,
                      dzW[w-1], dzW[w], L, mref, lref, Tref, species_scale, energy_scale,
                      fixed, Tfixed[g], radiation)


@njit(cache=True)
def _analytic_node(j, Tb, Yb, P, invW, nasa_lo, nasa_hi, nasa_tmid, omega, jy, jt,
                   rhob, cpb, omegab, hkb):
    """[base, T, Y...] node states with exact ideal-gas thermo and analytic rates.

    Production rates are linear in the column step through d(omega)/dY and
    d(omega)/dT at constant pressure, so the column difference equals the
    analytic derivative. Density, cp and enthalpy are evaluated exactly.
    """
    ns = invW.size
    nb = ns+2
    cpk = np.empty(ns)
    hkk = np.empty(ns)
    for it in range(2):
        T = Tb[0] if it == 0 else Tb[1]
        for k in range(ns):
            c = nasa_hi[k] if T > nasa_tmid[k] else nasa_lo[k]
            cpk[k] = c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4])))
            hkk[k] = (c[0]+T*(c[1]/2.0+T*(c[2]/3.0+T*(c[3]/4.0+T*c[4]/5.0)))+c[5]/T)*R_UNIV*T*invW[k]
        for col in range(nb):
            if (col == 1) != (it == 1):
                continue
            inv_wmix = 0.
            cp_mix = 0.
            for k in range(ns):
                inv_wmix += Yb[k, col]*invW[k]
                cp_mix += Yb[k, col]*cpk[k]*R_UNIV*invW[k]
                hkb[k, col] = hkk[k]
            if inv_wmix < 1.0e-300:
                inv_wmix = 1.0e-300
            rhob[col] = P*(1./inv_wmix)/(R_UNIV*T)
            cpb[col] = cp_mix
    for k in range(ns):
        omegab[k, 0] = omega[k, j]
        omegab[k, 1] = omega[k, j]+jt[k, j]*(Tb[1]-Tb[0])
        for l in range(ns):
            omegab[k, 2+l] = omega[k, j]+jy[k, j, l]*(Yb[l, 2+l]-Yb[l, 0])


@njit(cache=True)
def _node_columns(j, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                  fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                  multicomponent, soret, kinetics, analytic, jy, jt, lower, diag, upper, border, ws,
                  radiation, frozen):
    ns, n = Y.shape
    nv = ns+3
    mref, aref, Tref = scalars[0], scalars[1], scalars[2]
    (nasa_lo, nasa_hi, nasa_tmid, W, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
     is_three_body, is_falloff, is_reversible, has_troe, troe_A, troe_T3, troe_T1, troe_T2,
     r_idx, r_nu, r_count, p_idx, p_nu, p_count, r_plan, p_plan,
     net_idx, net_nu, net_count, eff_idx, eff_delta, eff_count, delta_nu) = kinetics
    # Window of global nodes j-2..j+2 and faces j-2..j+1. Slots outside
    # the grid are never read: only rows of existing nodes are formed.
    (window, YW, omegaW, hkW, JW, Tb, Yb, steps, rhob, cpb, omegab, hkb, base, trial,
     tleft, tright, tpert, mleft, mright, mpert, X, D, DT, Yf, multi, thermal, flux,
     sums, face_scalars, YfB, Xn, SDn, Smun, saved_J, cws, frozen_multi, frozen_thermal, frozen_lam) = ws
    mW, VW, TW, rhoW, cpW = window[0, :5], window[1, :5], window[2, :5], window[3, :5], window[4, :5]
    lamW, muW, dzW = window[5, :4], window[6, :4], window[7, :4]
    for w in range(5):
        g = j-2+w
        if 0 <= g < n:
            mW[w], VW[w], TW[w], rhoW[w], cpW[w] = m[g], V[g], T[g], rho[g], cp[g]
            for k in range(ns):
                YW[k, w] = Y[k, g]
                omegaW[k, w] = omega[k, g]
                hkW[k, w] = hk[k, g]
    for w in range(4):
        f = j-2+w
        if 0 <= f < n-1:
            lamW[w], muW[w], dzW[w] = lam0[f], mu0[f], dz[f]
            for k in range(ns):
                JW[k, w] = J0[k, f]
    # Thermochemistry of node j for the [base, T, Y...] columns.
    o = j*nv
    nb = ns+2
    Tb[:] = T[j]
    Tb[1] = (x[o+2]+1e-7*max(abs(x[o+2]), 1.))*Tref
    for c in range(3):
        steps[c] = 1e-7*max(abs(x[o+c]), 1.)
    for k in range(ns):
        steps[3+k] = 1e-10+1e-5*abs(x[o+3+k])
        for c in range(nb):
            Yb[k, c] = Y[k, j]
        Yb[k, 2+k] += steps[3+k]
    if analytic:
        _analytic_node(j, Tb, Yb, P, invW, nasa_lo, nasa_hi, nasa_tmid, omega, jy, jt,
                       rhob, cpb, omegab, hkb)
    else:
        grouped_node(0, Tb, Yb, P, nasa_lo, nasa_hi, nasa_tmid, W, invW,
                     A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
                     is_three_body, is_falloff, is_reversible, has_troe,
                     troe_A, troe_T3, troe_T1, troe_T2,
                     r_idx, r_nu, r_count, p_idx, p_nu, p_count, r_plan, p_plan,
                     net_idx, net_nu, net_count, eff_idx, eff_delta, eff_count,
                     delta_nu, rhob, cpb, omegab, hkb)
        for k in range(ns):
            for c in range(nb):
                hkb[k, c] *= invW[k]
    # The base column of the same kernel keeps the difference consistent.
    rhoW[2], cpW[2] = rhob[0], cpb[0]
    for k in range(ns):
        omegaW[k, 2] = omegab[k, 0]
        hkW[k, 2] = hkb[k, 0]
    final_index = n+1-j
    for w in range(1, 4):
        if 0 <= j-2+w < n:
            _window_rows(base[w-1], w, n, j, mW, VW, TW, YW, rhoW, cpW, omegaW, hkW, lamW, JW, muW,
                         dzW, L, scalars, fuel_Y, oxidizer_Y, excess, fixed, Tfixed, radiation)
    final_base = 0.
    if j >= n-2:
        i = final_index
        final_base = (mW[i]-mW[i-1]+dzW[i-1]*(rhoW[i-1]*VW[i-1]+rhoW[i]*VW[i]))/mref
    # Face scratch: left face (window 1) has node j on its right side.
    has_left = j >= 1
    has_right = j <= n-2
    mixture_data = mixture
    soret_mix = soret and not multicomponent
    if has_left:
        face_temperature_data(.5*(TW[1]+TW[2]), P, mw, mixture_data[0], mixture_data[1], mixture_data[2],
                              mixture_data[3], mixture_data[4], soret_mix, mixture_data[5],
                              mixture_data[6], tleft[0], tleft[1], tleft[2], tleft[3], tleft[4], tleft[5])
        if multicomponent:
            _multi_temperature(.5*(TW[1]+TW[2]), multi_data, mw, mleft)
            if frozen:
                _frozen_face(TW[1], TW[2], YW[:, 1], YW[:, 2], P, invW, mw, multi_data, soret, mleft, cws, Yf,
                             frozen_multi[0], frozen_thermal[0], frozen_lam, 0)
    if has_right:
        face_temperature_data(.5*(TW[2]+TW[3]), P, mw, mixture_data[0], mixture_data[1], mixture_data[2],
                              mixture_data[3], mixture_data[4], soret_mix, mixture_data[5],
                              mixture_data[6], tright[0], tright[1], tright[2], tright[3], tright[4], tright[5])
        if multicomponent:
            _multi_temperature(.5*(TW[2]+TW[3]), multi_data, mw, mright)
            if frozen:
                _frozen_face(TW[2], TW[3], YW[:, 2], YW[:, 3], P, invW, mw, multi_data, soret, mright, cws, Yf,
                             frozen_multi[1], frozen_thermal[1], frozen_lam, 1)
    # Base composition sums of both faces for O(ns) composition columns.
    rank1 = not multicomponent and not soret
    if rank1:
        for side in range(2):
            if (side == 0 and not has_left) or (side == 1 and not has_right):
                continue
            a = 1+side
            td = tleft if side == 0 else tright
            inv = 0.
            for k in range(ns):
                YfB[side, k] = .5*(YW[k, a]+YW[k, a+1])
                inv += YfB[side, k]*invW[k]
            wm = 1./max(inv, 1e-300)
            s1, s2 = mixture_sums(wm, YfB[side], invW, td[0], td[1], td[4],
                                  sums[side, 0], sums[side, 1], sums[side, 2])
            face_scalars[side, 0], face_scalars[side, 1] = inv, wm
            face_scalars[side, 2], face_scalars[side, 3] = s1, s2
    saved_lam = (0., 0.)
    saved_mu = (0., 0.)
    for c in range(nv):
        step = steps[c]
        if c == 0:
            mW[2] = (x[o]+step)*mref
        elif c == 1:
            VW[2] = (x[o+1]+step)*scalars[1]
        else:
            column = c-1
            if c == 2:
                TW[2] = Tb[1]
            else:
                YW[c-3, 2] = Yb[c-3, column]
            rhoW[2], cpW[2] = rhob[column], cpb[column]
            for k in range(ns):
                omegaW[k, 2] = omegab[k, column]
                hkW[k, 2] = hkb[k, column]
            saved_lam = (lamW[1], lamW[2])
            saved_mu = (muW[1], muW[2])
            for k in range(ns):
                saved_J[k, 0] = JW[k, 1]
                saved_J[k, 1] = JW[k, 2]
            if multicomponent and frozen and c >= 3:
                for side in range(2):
                    if (side == 0 and not has_left) or (side == 1 and not has_right):
                        continue
                    a = 1+side
                    td = tleft if side == 0 else tright
                    Tf = .5*(TW[a]+TW[a+1])
                    inv = 0.
                    for k in range(ns):
                        Yf[k] = .5*(YW[k, a]+YW[k, a+1])
                        inv += Yf[k]*invW[k]
                    wm = 1./max(inv, 1e-300)
                    multicomponent_flux(YW[:, a], YW[:, a+1], TW[a], TW[a+1], dzW[a], Tf, P*wm/(R_UNIV*Tf), wm,
                                        frozen_multi[side], frozen_thermal[side], invW, mw, flux, X)
                    for k in range(ns):
                        X[k] = Yf[k]*wm*invW[k]
                        JW[k, a] = flux[k]
                    lamW[a] = frozen_lam[side]
                    muW[a] = face_viscosity(X, td[3], td[4])
            elif rank1 and c >= 3:
                l = c-3
                for side in range(2):
                    if (side == 0 and not has_left) or (side == 1 and not has_right):
                        continue
                    a = 1+side
                    td = tleft if side == 0 else tright
                    Ynew = .5*(YW[l, a]+YW[l, a+1])
                    wm2, s1, s2 = mixture_rank1(l, Ynew, Ynew-YfB[side, l], face_scalars[side, 0],
                                                face_scalars[side, 1], sums[side, 0], face_scalars[side, 2],
                                                face_scalars[side, 3], sums[side, 1], sums[side, 2], invW,
                                                td[0], td[1], td[4], Xn, SDn, Smun)
                    Tf = .5*(TW[a]+TW[a+1])
                    rho_f, lamW[a], muW[a] = mixture_from_sums(Tf, P, wm2, Xn, s1, s2, SDn, Smun, mw,
                                                               td[2], td[3], D)
                    mixture_flux(YW[:, a], YW[:, a+1], TW[a], TW[a+1], dzW[a], Tf, rho_f, wm2, D, DT,
                                 False, mw, flux)
                    for k in range(ns):
                        JW[k, a] = flux[k]
            elif has_left:
                lamW[1], muW[1] = face_state(TW[1], TW[2], YW[:, 1], YW[:, 2], dzW[1], P, invW, mw,
                                             mixture, multi_data, multicomponent, soret,
                                             tpert if c == 2 else tleft, mpert if c == 2 else mleft, c != 2,
                                             X, D, DT, Yf, multi, thermal, flux, cws)
                for k in range(ns):
                    JW[k, 1] = flux[k]
            if has_right and not ((rank1 or (multicomponent and frozen)) and c >= 3):
                lamW[2], muW[2] = face_state(TW[2], TW[3], YW[:, 2], YW[:, 3], dzW[2], P, invW, mw,
                                             mixture, multi_data, multicomponent, soret,
                                             tpert if c == 2 else tright, mpert if c == 2 else mright, c != 2,
                                             X, D, DT, Yf, multi, thermal, flux, cws)
                for k in range(ns):
                    JW[k, 2] = flux[k]
        for w in range(1, 4):
            g = j-2+w
            if g < 0 or g >= n:
                continue
            _window_rows(trial, w, n, j, mW, VW, TW, YW, rhoW, cpW, omegaW, hkW, lamW, JW, muW,
                         dzW, L, scalars, fuel_Y, oxidizer_Y, excess, fixed, Tfixed, radiation)
            for r in range(nv):
                value = (trial[r]-base[w-1, r])/step
                if w == 1:
                    upper[j-1, r, c] = value
                elif w == 2:
                    diag[j, r, c] = value
                else:
                    lower[j, r, c] = value
        if j >= n-2:
            i = final_index
            value = (mW[i]-mW[i-1]+dzW[i-1]*(rhoW[i-1]*VW[i-1]+rhoW[i]*VW[i]))/mref
            border[j-(n-2), c] = (value-final_base)/step
        # Restore the base window.
        mW[2], VW[2], TW[2], rhoW[2], cpW[2] = m[j], V[j], T[j], rhob[0], cpb[0]
        if c >= 2:
            if c >= 3:
                YW[c-3, 2] = Y[c-3, j]
            for k in range(ns):
                omegaW[k, 2] = omegab[k, 0]
                hkW[k, 2] = hkb[k, 0]
            lamW[1], lamW[2] = saved_lam
            muW[1], muW[2] = saved_mu
            for k in range(ns):
                JW[k, 1] = saved_J[k, 0]
                JW[k, 2] = saved_J[k, 1]


@njit(cache=True, nogil=True)
def jacobian_range(start, stop, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                   fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                   multicomponent, soret, kinetics, analytic, jy, jt, radiation, frozen,
                   lower, diag, upper, border):
    """Columns of nodes [start, stop) of the steady block-tridiagonal Jacobian.

    Serial and GIL-free: callers split nodes among threads. Node j writes
    only upper[j-1][:, c], diag[j][:, c], lower[j][:, c] and border, so
    different ranges never write the same entries.

    ``lower[j]`` holds d(rows j+1)/d(x_j), ``upper[j]`` d(rows j)/d(x_j+1)
    and ``border`` the final continuity row against nodes n-2 and n-1.
    With ``analytic``, production-rate columns use d(omega)/dY and d(omega)/dT.
    With ``frozen`` (multicomponent only), composition columns keep the face
    D_kl, D^T and lambda of the base state, as Cantera's Jacobian keeps its
    transport properties; gradients, density, Wmix, viscosity and every
    temperature column remain exact. The residual itself is never frozen.
    """
    ns = Y.shape[0]
    ws = _node_workspace(ns, multicomponent)
    for j in range(start, stop):
        _node_columns(j, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                      fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                      multicomponent, soret, kinetics, analytic, jy, jt, lower, diag, upper, border, ws,
                      radiation, frozen)


# Block elimination kernels are shared with the premixed solver.
from kflame.flame.block_linear import twisted_bottom, twisted_middle, twisted_solve, twisted_top  # noqa: E402,F401
