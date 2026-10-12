"""Compiled counterflow residual rows, face transport and local Jacobian blocks.

The residual and the Jacobian share the same row functions. Perturbing one
node changes only its own thermochemistry, the two adjacent faces and three
row blocks, so the Jacobian kernel recomputes exactly those quantities with
the same finite-difference steps as a column-by-column evaluation of the full
residual. Face quantities that depend only on temperature (pure-species fits,
binary diffusion, Wilke factors, Soret collision ratios) are reused by the
composition columns of the same face.

Large-species thermal blocks use contiguous rows while preserving the original
order of every column reduction in the shared multicomponent kernels.
"""
import math

import numpy as np
from numba import njit
from kflame.flame.frozen_transport_columns import frozen_composition_columns
from kflame.flame.algebraic_frozen_transport import frozen_composition_columns as algebraic_composition_columns
from kflame.flame.frozen_transport_derivatives import frozen_face_derivatives
from kflame.flame.analytic_frozen_rows import composition_row_derivative

from kflame.chemistry.jacobian import grouped_node
from kflame.chemistry.mechanism import R_UNIV
from kflame.chemistry.multicomponent_trace import trace_face
from kflame.chemistry.multicomponent_kernels import (composition_terms, composition_terms_ws,
                                                     composition_workspace, temperature_terms,
                                                     _temperature_terms_symmetric)


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
# Relative axial mass flux below which the enthalpy-flux gradient is blended.
STAGNATION = 1e-4
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
    # sum J_k dh_k/dz is not proportional to m: a pure donor switch would make
    # the residual jump at stagnation, where no discrete root may then exist.
    # Within |m| < STAGNATION*mref the two one-sided gradients are blended
    # linearly; elsewhere the upwind (Cantera) gradient is unchanged.
    backward = .5+m[j]/(2.*STAGNATION*mref)
    for k in range(ns):
        gradY = direction*(Y[k, j]-Y[k, donor])/spacing
        out[o+3+k] = (m[j]*gradY+(J[k, j]-J[k, j-1])/volume-omega[k, j])/species_scale
        if not fixed:
            if backward >= 1. or backward <= 0.:
                gradH = direction*(hk[k, j]-hk[k, donor])/spacing
            else:
                gradH = (backward*(hk[k, j]-hk[k, j-1])/hm
                         +(1.-backward)*(hk[k, j+1]-hk[k, j])/hp)
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
    # Every row, including both boundaries and the border, is overwritten.
    residual = np.empty(n*nv+1)
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


@njit(cache=True,fastmath=False)
def face_viscosity(X,root,phi):
    """Eight independent ordered Wilke sums; scalar fallback for small/strided inputs."""
    ns=X.size;value=0.;k=0
    if ns<24 or not phi.flags.c_contiguous:
        for k in range(ns):
            total=0.
            for j in range(ns):total+=X[j]*phi[k,j]
            value+=max(X[k],1e-300)*root[k]**2/max(total,1e-300)
        return value
    while k+8<=ns:
        t0,t1,t2,t3,t4,t5,t6,t7=0.,0.,0.,0.,0.,0.,0.,0.
        for j in range(ns):
            t0+=X[j]*phi[k,j];t1+=X[j]*phi[k+1,j]
            t2+=X[j]*phi[k+2,j];t3+=X[j]*phi[k+3,j]
            t4+=X[j]*phi[k+4,j];t5+=X[j]*phi[k+5,j]
            t6+=X[j]*phi[k+6,j];t7+=X[j]*phi[k+7,j]
        value+=max(X[k],1e-300)*root[k]**2/max(t0,1e-300)
        value+=max(X[k+1],1e-300)*root[k+1]**2/max(t1,1e-300)
        value+=max(X[k+2],1e-300)*root[k+2]**2/max(t2,1e-300)
        value+=max(X[k+3],1e-300)*root[k+3]**2/max(t3,1e-300)
        value+=max(X[k+4],1e-300)*root[k+4]**2/max(t4,1e-300)
        value+=max(X[k+5],1e-300)*root[k+5]**2/max(t5,1e-300)
        value+=max(X[k+6],1e-300)*root[k+6]**2/max(t6,1e-300)
        value+=max(X[k+7],1e-300)*root[k+7]**2/max(t7,1e-300)
        k+=8
    for k in range(k,ns):
        total=0.
        for j in range(ns):total+=X[j]*phi[k,j]
        value+=max(X[k],1e-300)*root[k]**2/max(total,1e-300)
    return value


@njit(cache=True)
def face_viscosity_temperature_data(Tf, visc_poly, quarter_ratio, wilke_den, root, phi):
    """Only the mixture temperature data consumed by multicomponent viscosity.

    Keep the original root/phi expressions and pair order. Conductivity,
    binary diffusion and Soret come from the full multicomponent closure.
    """
    ns = root.size
    logT = math.log(Tf)
    quarter = Tf**.25
    for k in range(ns):
        root[k] = abs(quarter*_poly4(visc_poly[k], logT))
    for k in range(ns):
        for j in range(ns):
            ratio = root[k]/max(root[j], 1e-150)
            phi[k, j] = (1.+ratio*quarter_ratio[k, j])**2/wilke_den[k, j]


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
    """Mixture face state; standalone calls own their scratch arrays."""
    return _face_mixture_ws(Tf, Yf, P, invW, mw, cond, invB, diagB, root, phi,
                            soret, scale, tiny, X, D, DT, np.empty((5 if soret else 2, Yf.size)))


@njit(cache=True)
def _face_mixture_ws(Tf, Yf, P, invW, mw, cond, invB, diagB, root, phi, soret, scale, tiny, X, D, DT, ws):
    """Mixture-averaged face state with the arithmetic of the grid kernels."""
    ns = Yf.size
    inv_wmix = 0.
    for k in range(ns):
        inv_wmix += Yf[k]*invW[k]
    wm = 1./max(inv_wmix, 1e-300)
    SD, Smu = ws[0], ws[1]
    sum1, sum2 = mixture_sums(wm, Yf, invW, cond, invB, phi, X, SD, Smu)
    rho, lam, mu = mixture_from_sums(Tf, P, wm, X, sum1, sum2, SD, Smu, mw, diagB, root, D)
    if soret:
        Xs, a, diffusion = ws[2], ws[3], ws[4]
        # Same clipped mole fractions and mixture diffusivities as the
        # fused mixture-averaged Soret kernel.
        denom = 0.
        for k in range(ns):
            denom += Yf[k]/mw[k]
        wmix = 1./max(denom, 1e-300)
        for k in range(ns):
            Xs[k] = max(Yf[k]*wmix/mw[k], 1e-20)
        # Reset the absent-species entries on every reuse.
        a[:] = 0.
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
    # Independent row accumulators expose SIMD across species while each
    # dot product retains its original ascending l order (no fastmath).
    k = 0
    while k+8 <= ns and ns > 20:
        v0, v1, v2, v3 = 0., 0., 0., 0.
        v4, v5, v6, v7 = 0., 0., 0., 0.
        for l in range(ns):
            v0 += multi[k, l]*scratch[l]
            v1 += multi[k+1, l]*scratch[l]
            v2 += multi[k+2, l]*scratch[l]
            v3 += multi[k+3, l]*scratch[l]
            v4 += multi[k+4, l]*scratch[l]
            v5 += multi[k+5, l]*scratch[l]
            v6 += multi[k+6, l]*scratch[l]
            v7 += multi[k+7, l]*scratch[l]
        out[k] = factor*mw[k]*v0-thermal[k]/Tf*gradT
        out[k+1] = factor*mw[k+1]*v1-thermal[k+1]/Tf*gradT
        out[k+2] = factor*mw[k+2]*v2-thermal[k+2]/Tf*gradT
        out[k+3] = factor*mw[k+3]*v3-thermal[k+3]/Tf*gradT
        out[k+4] = factor*mw[k+4]*v4-thermal[k+4]/Tf*gradT
        out[k+5] = factor*mw[k+5]*v5-thermal[k+5]/Tf*gradT
        out[k+6] = factor*mw[k+6]*v6-thermal[k+6]/Tf*gradT
        out[k+7] = factor*mw[k+7]*v7-thermal[k+7]/Tf*gradT
        k += 8
    for k in range(k, ns):
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
     astar_fit, bstar_fit, cstar_fit, dense, nasa_lo, nasa_hi, nasa_tmid) = multi_data[:14]
    mu, rot_relax, bdiff, a, b, c, cp_r = mdata
    _cp_r(Tf, nasa_lo, nasa_hi, nasa_tmid, cp_r)
    if len(multi_data) == 15:
        if multi_data[14]:
            _temperature_terms_symmetric(Tf, mw, log_eps_pair, epsilon, zrot, frot298,
                viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit,
                mu, rot_relax, bdiff, a, b, c)
            return
    temperature_terms(Tf, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit, diffusion_fit,
                      astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c)


@njit(cache=True)
def _mdata(ns):
    return (np.empty(ns), np.empty(ns), np.empty((ns, ns)), np.empty((ns, ns)), np.empty((ns, ns)),
            np.empty((ns, ns)), np.empty(ns))


@njit(cache=True)
def face_state(TL, TR, YL, YR, spacing, P, invW, mw, mixture, multi_data, multicomponent, soret,
               tdata, mdata, reuse, X, D, DT, Yf, multi, thermal, flux, cws):
    """Standalone face evaluation with the same caller-owned outputs."""
    mix_ws = np.empty((5 if soret else 2, 0 if multicomponent else mw.size))
    return _face_state_ws(TL, TR, YL, YR, spacing, P, invW, mw, mixture, multi_data,
                          multicomponent, soret, tdata, mdata, reuse, X, D, DT, Yf,
                          multi, thermal, flux, cws, mix_ws)


@njit(cache=True)
def _face_state_ws(TL, TR, YL, YR, spacing, P, invW, mw, mixture, multi_data, multicomponent, soret,
                   tdata, mdata, reuse, X, D, DT, Yf, multi, thermal, flux, cws, mix_ws):
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
        if multicomponent:
            face_viscosity_temperature_data(Tf, visc_poly, quarter_ratio, wilke_den, root, phi)
            if len(multi_data) == 16:
                # Pruned model: its kernel evaluates only the kept pairs.
                _cp_r(Tf, multi_data[11], multi_data[12], multi_data[13], mdata[6])
            else:
                _multi_temperature(Tf, multi_data, mw, mdata)
        else:
            face_temperature_data(Tf, P, mw, cond_poly, diff_poly, visc_poly, quarter_ratio, wilke_den,
                                  soret, eps_pair, cstar_poly,
                                  cond, invB, diagB, root, phi, scale)
    if multicomponent:
        crot, dense = multi_data[2], multi_data[10]
        mu, rot_relax, bdiff, a, b, c, cp_r = mdata
        traced = False
        if len(multi_data) == 16:
            rho, lam, wm, majors = trace_face(Tf, P, Yf, cp_r, mw, invW, crot, multi_data[0], multi_data[1],
                                              multi_data[3], multi_data[4], multi_data[5], multi_data[6],
                                              multi_data[7], multi_data[8], multi_data[9], multi_data[15],
                                              multi, thermal)
            traced = majors >= 0
            if not traced:
                _multi_temperature(Tf, multi_data, mw, mdata)
        if traced:
            pass
        elif dense:
            rho, lam, wm = composition_terms(Tf, P, Yf, cp_r, mw, invW, crot, mu, rot_relax, bdiff,
                                             a, b, c, dense, multi, thermal)
        else:
            rho, lam, wm = composition_terms_ws(Tf, P, Yf, cp_r, mw, invW, crot, mu, rot_relax, bdiff,
                                                a, b, c, multi, thermal, cws)
        if not soret:
            thermal[:] = 0.
        multicomponent_flux(YL, YR, TL, TR, spacing, Tf, rho, wm, multi, thermal, invW, mw, flux, X)
        for k in range(mw.size):
            X[k] = Yf[k]*wm*invW[k]
        mu = face_viscosity(X, root, phi)
    else:
        rho, wm, lam, mu = _face_mixture_ws(Tf, Yf, P, invW, mw, cond, invB, diagB, root, phi,
                                        soret, scale, tiny, X, D, DT, mix_ws)
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
def _tdata(ns, multicomponent=False):
    # Multicomponent faces consume only root and phi. Keep typed empty arrays
    # for the mixture-only tuple entries without reserving their O(ns**2) data.
    nmix = 0 if multicomponent else ns
    return (np.empty(nmix), np.empty((nmix, nmix)), np.empty(nmix), np.empty(ns),
            np.empty((ns, ns)), np.zeros((nmix, nmix)))


@njit(cache=True, nogil=True)
def face_range(start, stop, T, Y, dz, P, invW, mw, mixture, multi_data, multicomponent, soret, lam, mu, J):
    """Face lambda, mu and fluxes of faces [start, stop), with the Jacobian's face functions.

    Serial and GIL-free: callers split the faces among threads. Each call
    owns its scratch arrays; outputs of different ranges never overlap.
    """
    ns = Y.shape[0]
    tdata = _tdata(ns, multicomponent)
    mdata = _mdata(ns if multicomponent else 1)
    cws = composition_workspace(ns if multicomponent else 1)
    flux = np.empty(ns)
    nmix = 0 if multicomponent else ns
    X, D, DT, Yf = np.empty(ns), np.empty(nmix), np.zeros(nmix), np.empty(ns)
    multi, thermal = np.empty((ns, ns)), np.empty(ns)
    mix_ws = np.empty((5 if soret else 2, nmix))
    for f in range(start, stop):
        lam[f], mu[f] = _face_state_ws(T[f], T[f+1], Y[:, f], Y[:, f+1], dz[f], P, invW, mw, mixture,
                                   multi_data, multicomponent, soret, tdata, mdata, False,
                                   X, D, DT, Yf, multi, thermal, flux, cws, mix_ws)
        for k in range(ns):
            J[k, f] = flux[k]


@njit(cache=True)
def _node_workspace(ns, multicomponent, base_bank=False):
    """Private scratch of one range; mixture-only payloads are empty in multi mode."""
    nb, nv, nm = ns+2, ns+3, ns if multicomponent else 1
    nmix = 0 if multicomponent else ns
    # Transposed backing arrays keep species columns contiguous at each node/face.
    window = (
        np.zeros((8, 5)), np.zeros((5, ns)).T, np.zeros((5, ns)).T,
        np.zeros((5, ns)).T, np.zeros((4, ns)).T,
    )
    node = (
        np.empty(nb), np.empty((nb, ns)).T, np.empty(nv),
        np.empty(nb), np.empty(nb), np.empty((nb, ns)).T,
        np.empty((nb, ns)).T, np.zeros((3, nv)), np.empty(nv),
    )
    face_data = (
        _tdata(ns, multicomponent), _tdata(ns, multicomponent), _tdata(ns, multicomponent),
        _mdata(0 if base_bank else nm), _mdata(0 if base_bank else nm), _mdata(nm),
    )
    transport = (
        np.empty(ns), np.empty(nmix), np.zeros(nmix),
        np.empty(ns), np.empty((ns, ns)), np.empty(ns),
        np.empty(ns), np.zeros((2, 3, nmix)), np.zeros((2, 0 if multicomponent else 4)),
        np.zeros((2, nmix)), np.empty(nmix), np.empty(nmix),
        np.empty(nmix), np.empty((2, ns)).T, composition_workspace(nm),
        (np.zeros((nm, nm)), np.zeros((nm, nm))),
        (np.zeros(nm), np.zeros(nm)), (np.zeros(1), np.zeros(1)),
        np.empty((5, nmix)), np.empty((2, ns)),
    )
    return window+node+face_data+transport


@njit(cache=True)
def _rotate_face_workspace(ws):
    """Exchange complete private base-face slots; perturbation scratch stays put.

    The previous right face becomes the next node's left face. Its temperature
    data, multicomponent data and frozen coefficients must move together. Only
    array references are exchanged; neither slot aliases the other or tpert.
    """
    return (ws[:14]
            + (ws[15], ws[14], ws[16], ws[18], ws[17], ws[19])
            + ws[20:35]
            + ((ws[35][1], ws[35][0]), (ws[36][1], ws[36][0]),
               (ws[37][1], ws[37][0]))
            + ws[38:])


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
    return _analytic_node_ws(j, Tb, Yb, P, invW, nasa_lo, nasa_hi, nasa_tmid, omega, jy, jt,
                             rhob, cpb, omegab, hkb, np.empty((2, invW.size)))


@njit(cache=True)
def _analytic_node_ws(j, Tb, Yb, P, invW, nasa_lo, nasa_hi, nasa_tmid, omega, jy, jt,
                      rhob, cpb, omegab, hkb, thermo_ws, partials_start=0):
    """[base, T, Y...] node states with exact ideal-gas thermo and analytic rates.

    Production rates are linear in the column step through d(omega)/dY and
    d(omega)/dT at constant pressure, so the column difference equals the
    analytic derivative. Density, cp and enthalpy are evaluated exactly.
    """
    ns = invW.size
    nb = ns+2
    cpk, hkk = thermo_ws[0], thermo_ws[1]
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
        omegab[k, 1] = omega[k, j]+jt[k, j-partials_start]*(Tb[1]-Tb[0])
        for l in range(ns):
            omegab[k, 2+l] = omega[k, j]+jy[k, j-partials_start, l]*(Yb[l, 2+l]-Yb[l, 0])


@njit(cache=True)
def _node_workspace_temperature_cache(ns, base_bank=False):
    # valid, geometric face index, exact perturbed Tf, conductivity.
    return _node_workspace(ns, True, base_bank)+(np.zeros(4),)


@njit(cache=True)
def _temperature_coefficients_finite(multi, thermal):
    for k in range(thermal.size):
        if not math.isfinite(thermal[k]):
            return False
        for l in range(thermal.size):
            if not math.isfinite(multi[k, l]):
                return False
    return True


@njit(cache=True)
def temperature_face_flux_cached(TL, TR, YL, YR, spacing, P, invW, mw,
                                 tdata, Yf, multi, thermal, flux, X, lam):
    # Same Yf/denominator/rho/flux/viscosity arithmetic as the full face.
    Tf = .5*(TL+TR)
    for k in range(mw.size):
        Yf[k] = .5*(YL[k]+YR[k])
    denom = 0.0
    for k in range(mw.size):
        denom += Yf[k]*invW[k]
    wm = 1.0/max(denom, 1e-300)
    rho = P*wm/(R_UNIV*Tf)
    multicomponent_flux(YL, YR, TL, TR, spacing, Tf, rho, wm,
                        multi, thermal, invW, mw, flux, X)
    for k in range(mw.size):
        X[k] = Yf[k]*wm*invW[k]
    mu = face_viscosity(X, tdata[3], tdata[4])
    return lam, mu


@njit(cache=True, nogil=True)
def _temperature_cache_eligible(start, stop, x, T, Tref, nv):
    """Exact face-temperature screen; ranges with no possible hits stay legacy."""
    for j in range(start+1, stop):
        previous = x[(j-1)*nv+2]
        current = x[j*nv+2]
        Tp_previous = (previous+1e-7*max(abs(previous), 1.))*Tref
        Tp_current = (current+1e-7*max(abs(current), 1.))*Tref
        left = .5*(T[j-1]+Tp_current)
        right = .5*(Tp_previous+T[j])
        if left == right and math.isfinite(left) and left > 0.:
            return True
    return False


@njit(cache=True)
def _node_columns(j, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                  fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                  multicomponent, soret, kinetics, analytic, jy, jt, lower, diag, upper, border, ws,
                  radiation, frozen, reuse_left=False, partials_start=0, temperature_meta=None, base_bank=None,
                  analytic_transport=False, algebraic_transport=False, thermal_bank=None):
    """One node's columns; reuse_left requires an already rotated workspace."""
    if temperature_meta is not None and not reuse_left:
        temperature_meta[0] = 0.
    ns, n = Y.shape
    nv = ns+3
    mref, aref, Tref = scalars[0], scalars[1], scalars[2]
    (nasa_lo, nasa_hi, nasa_tmid, W, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
     is_three_body, is_falloff, is_reversible, has_troe, troe_A, troe_T3, troe_T1, troe_T2,
     r_idx, r_nu, r_count, p_idx, p_nu, p_count, r_plan, p_plan,
     net_idx, net_nu, net_count, eff_idx, eff_delta, eff_count, delta_nu) = kinetics[:33]
    # Window of global nodes j-2..j+2 and faces j-2..j+1. Slots outside
    # the grid are never read: only rows of existing nodes are formed.
    (window, YW, omegaW, hkW, JW, Tb, Yb, steps, rhob, cpb, omegab, hkb, base, trial,
     tleft, tright, tpert, mleft, mright, mpert, X, D, DT, Yf, multi, thermal, flux,
     sums, face_scalars, YfB, Xn, SDn, Smun, saved_J, cws, frozen_multi, frozen_thermal, frozen_lam, mix_ws, thermo_ws) = ws
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
        _analytic_node_ws(j, Tb, Yb, P, invW, nasa_lo, nasa_hi, nasa_tmid, omega, jy, jt,
                       rhob, cpb, omegab, hkb, thermo_ws, partials_start)
    else:
        if len(kinetics)>33:
            grouped_node(0, Tb, Yb, P, nasa_lo, nasa_hi, nasa_tmid, W, invW,
                         A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
                         is_three_body, is_falloff, is_reversible, has_troe,
                         troe_A, troe_T3, troe_T1, troe_T2,
                         r_idx, r_nu, r_count, p_idx, p_nu, p_count, r_plan, p_plan,
                         net_idx, net_nu, net_count, eff_idx, eff_delta, eff_count,
                         delta_nu, rhob, cpb, omegab, hkb, kinetics[33],kinetics[34])
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
        # With reuse_left the range driver has already rotated every base
        # buffer into the left slot; tpert/mpert never enter that rotation.
        if not reuse_left:
            if multicomponent:
                face_viscosity_temperature_data(.5*(TW[1]+TW[2]), mixture_data[2], mixture_data[3], mixture_data[4], tleft[3], tleft[4])
                if frozen and base_bank is not None:
                    _copy_frozen_bank(j-1,base_bank,frozen_multi[0],frozen_thermal[0],frozen_lam[0],0)
                else:
                    _multi_temperature(.5*(TW[1]+TW[2]), multi_data, mw, mleft)
                    if frozen:
                        _frozen_face(TW[1], TW[2], YW[:, 1], YW[:, 2], P, invW, mw, multi_data, soret, mleft, cws, Yf,
                                     frozen_multi[0], frozen_thermal[0], frozen_lam[0], 0)
            else:
                face_temperature_data(.5*(TW[1]+TW[2]), P, mw, mixture_data[0], mixture_data[1], mixture_data[2],
                                      mixture_data[3], mixture_data[4], soret_mix, mixture_data[5],
                                      mixture_data[6], tleft[0], tleft[1], tleft[2], tleft[3], tleft[4], tleft[5])
    if has_right:
        if multicomponent:
            face_viscosity_temperature_data(.5*(TW[2]+TW[3]), mixture_data[2], mixture_data[3], mixture_data[4], tright[3], tright[4])
            if frozen and base_bank is not None:
                _copy_frozen_bank(j,base_bank,frozen_multi[1],frozen_thermal[1],frozen_lam[1],0)
            else:
                _multi_temperature(.5*(TW[2]+TW[3]), multi_data, mw, mright)
                if frozen:
                    _frozen_face(TW[2], TW[3], YW[:, 2], YW[:, 3], P, invW, mw, multi_data, soret, mright, cws, Yf,
                                 frozen_multi[1], frozen_thermal[1], frozen_lam[1], 0)
        else:
            face_temperature_data(.5*(TW[2]+TW[3]), P, mw, mixture_data[0], mixture_data[1], mixture_data[2],
                                  mixture_data[3], mixture_data[4], soret_mix, mixture_data[5],
                                  mixture_data[6], tright[0], tright[1], tright[2], tright[3], tright[4], tright[5])
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
    analytic_columns = analytic_transport and analytic and base_bank is not None
    dleft, dright = np.empty((1,1)), np.empty((1,1))
    muleft, muright = np.empty(1), np.empty(1)
    if analytic_columns:
        if has_left:
            valid, dleft, muleft = frozen_face_derivatives(
                YW[:,1],YW[:,2],TW[1],TW[2],dzW[1],P,invW,mw,frozen_multi[0],tleft[3],tleft[4],False)
            analytic_columns = analytic_columns and valid
        if has_right:
            valid, dright, muright = frozen_face_derivatives(
                YW[:,2],YW[:,3],TW[2],TW[3],dzW[2],P,invW,mw,frozen_multi[1],tright[3],tright[4],True)
            analytic_columns = analytic_columns and valid
    if base_bank is not None and not analytic_columns:
        if algebraic_transport:
            flux_left,flux_right,mu_left,mu_right = algebraic_composition_columns(
                ns,YW,TW,dzW,P,invW,mw,Yb,has_left,has_right,tleft,tright,frozen_multi,frozen_thermal)
        else:
            flux_left,flux_right,mu_left,mu_right = frozen_composition_columns(
                ns,YW,TW,dzW,P,invW,mw,Yb,has_left,has_right,tleft,tright,frozen_multi,frozen_thermal)
    for c in range(nv):
        if analytic_columns and c >= 3:
            q = c-3
            invmix = 0.
            for k in range(ns):
                invmix += Y[k,j]*invW[k]
            drho = -rhob[0]*invW[q]/invmix
            coefficients = nasa_hi[q] if T[j] > nasa_tmid[q] else nasa_lo[q]
            temperature = T[j]
            dcp = R_UNIV*invW[q]*(coefficients[0]+temperature*(coefficients[1]+temperature*(coefficients[2]+temperature*(coefficients[3]+temperature*coefficients[4]))))
            for w in range(1,4):
                if not 0 <= j-2+w < n:
                    continue
                composition_row_derivative(trial,w,n,j,q,mW,VW,TW,YW,rhoW,hkW,JW,dzW,
                    scalars,excess,fixed,radiation,drho,dcp,jy[:,j-partials_start,q],dleft,dright,muleft,muright)
                for r in range(nv):
                    if w == 1:
                        upper[j-1,r,c] = trial[r]
                    elif w == 2:
                        diag[j,r,c] = trial[r]
                    else:
                        lower[j,r,c] = trial[r]
            if j >= n-2:
                border[j-(n-2),c] = dz[-1]*drho*V[j]/mref
            continue
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
                    if base_bank is not None:
                        result = flux_left if side == 0 else flux_right
                        for k in range(ns):
                            JW[k,a] = result[k,c-3]
                        lamW[a] = frozen_lam[side][0]
                        muW[a] = mu_left[c-3] if side == 0 else mu_right[c-3]
                    else:
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
                        lamW[a] = frozen_lam[side][0]
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
            elif c == 2 and thermal_bank is not None and ((not has_left or thermal_bank[2][j-1,2] == 1.)
                                                         and (not has_right or thermal_bank[2][j,2] == 1.)):
                delta = TW[2]-T[j]
                if has_left:
                    for k in range(ns):
                        JW[k,1] = saved_J[k,0]+delta*thermal_bank[1][j-1,k]
                    lamW[1] = saved_lam[0]+.5*delta*thermal_bank[2][j-1,0]
                    muW[1] = saved_mu[0]+.5*delta*thermal_bank[2][j-1,1]
                if has_right:
                    for k in range(ns):
                        JW[k,2] = saved_J[k,1]+delta*thermal_bank[0][j,k]
                    lamW[2] = saved_lam[1]+.5*delta*thermal_bank[2][j,0]
                    muW[2] = saved_mu[1]+.5*delta*thermal_bank[2][j,1]
            elif has_left:
                if (temperature_meta is not None and c == 2 and multicomponent and frozen and reuse_left
                    and temperature_meta[0] == 1. and temperature_meta[1] == j-1
                    and temperature_meta[2] == .5*(TW[1]+TW[2])
                    and _temperature_coefficients_finite(multi, thermal)):
                    lamW[1], muW[1] = temperature_face_flux_cached(
                        TW[1], TW[2], YW[:, 1], YW[:, 2], dzW[1], P, invW, mw,
                        tpert, Yf, multi, thermal, flux, X, temperature_meta[3])
                else:
                    lamW[1], muW[1] = _face_state_ws(TW[1], TW[2], YW[:, 1], YW[:, 2], dzW[1], P, invW, mw,
                                                 mixture, multi_data, multicomponent, soret,
                                                 tpert if c == 2 else tleft, mpert if c == 2 else mleft, c != 2,
                                                 X, D, DT, Yf, multi, thermal, flux, cws, mix_ws)
                for k in range(ns):
                    JW[k, 1] = flux[k]
            used_thermal = (c == 2 and thermal_bank is not None
                            and (not has_left or thermal_bank[2][j-1,2] == 1.)
                            and (not has_right or thermal_bank[2][j,2] == 1.))
            if has_right and not used_thermal and not ((rank1 or (multicomponent and frozen)) and c >= 3):
                lamW[2], muW[2] = _face_state_ws(TW[2], TW[3], YW[:, 2], YW[:, 3], dzW[2], P, invW, mw,
                                             mixture, multi_data, multicomponent, soret,
                                             tpert if c == 2 else tright, mpert if c == 2 else mright, c != 2,
                                             X, D, DT, Yf, multi, thermal, flux, cws, mix_ws)
                for k in range(ns):
                    JW[k, 2] = flux[k]
                if temperature_meta is not None and c == 2:
                    perturbed_Tf = .5*(TW[2]+TW[3])
                    temperature_meta[0] = 1. if (math.isfinite(perturbed_Tf)
                        and perturbed_Tf > 0. and math.isfinite(lamW[2])
                        and math.isfinite(muW[2])) else 0.
                    temperature_meta[1] = j
                    temperature_meta[2] = perturbed_Tf
                    temperature_meta[3] = lamW[2]
            elif temperature_meta is not None and c == 2:
                temperature_meta[0] = 0.
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
                   lower, diag, upper, border, reuse_faces=True, partials_start=0):
    """Columns of nodes [start, stop) of the steady block-tridiagonal Jacobian.

    Serial and GIL-free: callers split nodes among threads. Node j writes
    only upper[j-1][:, c], diag[j][:, c], lower[j][:, c] and border, so
    different ranges never write the same entries.

    ``lower[j]`` holds d(rows j+1)/d(x_j), ``upper[j]`` d(rows j)/d(x_j+1)
    and ``border`` the final continuity row against nodes n-2 and n-1.
    With ``analytic``, production-rate columns use d(omega)/dY and d(omega)/dT.
    Reused faces rotate complete private slots within this call; each range
    seeds its own first left face and owns its temperature-perturbation slot.
    With ``frozen`` (multicomponent only), composition columns keep the face
    D_kl, D^T and lambda of the base state, as Cantera's Jacobian keeps its
    transport properties; gradients, density, Wmix, viscosity and every
    temperature column remain exact. The residual itself is never frozen.
    """
    ns = Y.shape[0]
    ws = _node_workspace(ns, multicomponent)
    for j in range(start, stop):
        reuse_left = reuse_faces and j > start
        if reuse_left:
            ws = _rotate_face_workspace(ws)
        _node_columns(j, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                      fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                      multicomponent, soret, kinetics, analytic, jy, jt, lower, diag, upper, border, ws,
                      radiation, frozen, reuse_left, partials_start)


@njit(cache=True, nogil=True)
def jacobian_range_workspace(start, stop, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                   fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                   multicomponent, soret, kinetics, analytic, jy, jt, radiation, frozen,
                   lower, diag, upper, border, carried_ws, carry_left, reuse_faces=True, partials_start=0):
    """Fill a tile using the private workspace owned by its exterior range.

    Only adjacent tiles of the same state/grid/configuration may carry faces.
    The owning range seeds its first left face; concurrent ranges never share
    buffers. Kept separate from the legacy void kernel to avoid escaping its
    scratch arrays into Python on small systems.
    """
    ns = Y.shape[0]
    ws = carried_ws
    for j in range(start, stop):
        reuse_left = reuse_faces and (j > start or carry_left)
        if reuse_left:
            ws = _rotate_face_workspace(ws)
        _node_columns(j, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                      fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                      multicomponent, soret, kinetics, analytic, jy, jt, lower, diag, upper, border, ws,
                      radiation, frozen, reuse_left, partials_start)

    return ws


@njit(cache=True, nogil=True)
def jacobian_range_temperature_cache(start, stop, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                   fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                   multicomponent, soret, kinetics, analytic, jy, jt, radiation, frozen,
                   lower, diag, upper, border, carried_ws, carry_left, reuse_faces=True, partials_start=0):
    """Fill a tile using the private workspace owned by its exterior range.

    Only adjacent tiles of the same state/grid/configuration may carry faces.
    The owning range seeds its first left face; concurrent ranges never share
    buffers. Kept separate from the legacy void kernel to avoid escaping its
    scratch arrays into Python on small systems.
    """
    ns = Y.shape[0]
    ws = carried_ws
    for j in range(start, stop):
        reuse_left = reuse_faces and (j > start or carry_left)
        if reuse_left:
            ws = _rotate_face_workspace(ws)
        _node_columns(j, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                      fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                      multicomponent, soret, kinetics, analytic, jy, jt, lower, diag, upper, border, ws[:-1],
                      radiation, frozen, reuse_left, partials_start, ws[-1])

    return ws


# Block elimination kernels are shared with the premixed solver.
from kflame.flame.block_linear import twisted_bottom, twisted_middle, twisted_solve, twisted_top  # noqa: E402,F401


@njit(cache=True, nogil=True)
def face_range_bank(start, stop, T, Y, dz, P, invW, mw, mixture, multi_data, multicomponent, soret, lam, mu, J, bank_multi, bank_thermal, bank_lam):
    """Face lambda, mu and fluxes of faces [start, stop), with the Jacobian's face functions.

    Serial and GIL-free: callers split the faces among threads. Each call
    owns its scratch arrays; outputs of different ranges never overlap.
    """
    ns = Y.shape[0]
    tdata = _tdata(ns, multicomponent)
    mdata = _mdata(ns if multicomponent else 1)
    cws = composition_workspace(ns if multicomponent else 1)
    flux = np.empty(ns)
    nmix = 0 if multicomponent else ns
    X, D, DT, Yf = np.empty(ns), np.empty(nmix), np.zeros(nmix), np.empty(ns)
    multi, thermal = np.empty((ns, ns)), np.empty(ns)
    mix_ws = np.empty((5 if soret else 2, nmix))
    for f in range(start, stop):
        lam[f], mu[f] = _face_state_ws(T[f], T[f+1], Y[:, f], Y[:, f+1], dz[f], P, invW, mw, mixture,
                                   multi_data, multicomponent, soret, tdata, mdata, False,
                                   X, D, DT, Yf, multi, thermal, flux, cws, mix_ws)
        for k in range(ns):
            J[k, f] = flux[k]
            bank_thermal[f,k] = thermal[k]
            for l in range(ns):
                bank_multi[f,k,l] = multi[k,l]
        bank_lam[f] = lam[f]


@njit(cache=True)
def _copy_frozen_bank(face, bank, multi, thermal, lam, side):
    for k in range(thermal.size):
        thermal[k] = bank[1][face,k]
        for l in range(thermal.size):
            multi[k,l] = bank[0][face,k,l]
    lam[side] = bank[2][face]


from kflame.flame.thermal_face_derivatives import (temperature_branch_safe,
                                                   face_temperature_tangents)


@njit(cache=True, nogil=True)
def face_range_bank_thermal(start, stop, T, Y, dz, P, invW, mw, mixture, multi_data,
                            multicomponent, soret, lam, mu, J, bank_multi,
                            bank_thermal, bank_lam, Tref, dleft, dright, derivatives):
    """Original primal face evaluation plus local, guarded implicit tangents.

    Per-face inverse/Schur factors are consumed before the scratch is reused.
    The persistent extra bank is O(faces*species), not O(faces*species**2).
    """
    ns = Y.shape[0]
    tdata, mdata = _tdata(ns,True), _mdata(ns)
    cws = composition_workspace(ns)
    X, Yf, flux = np.empty(ns), np.empty(ns), np.empty(ns)
    D, DT = np.empty(0), np.empty(0)
    multi, thermal = np.empty((ns,ns)), np.empty(ns)
    mix_ws = np.empty((2,0))
    for f in range(start,stop):
        lam[f],mu[f] = _face_state_ws(T[f],T[f+1],Y[:,f],Y[:,f+1],dz[f],P,invW,mw,
            mixture,multi_data,True,soret,tdata,mdata,False,X,D,DT,Yf,
            multi,thermal,flux,cws,mix_ws)
        bank_lam[f] = lam[f]
        for k in range(ns):
            J[k,f] = flux[k]
            bank_thermal[f,k] = thermal[k]
            dleft[f,k],dright[f,k] = 0.,0.
            for l in range(ns):
                bank_multi[f,k,l] = multi[k,l]
        derivatives[f,:] = 0.
        eligible = (multicomponent and not multi_data[10]
                    and temperature_branch_safe(T[f],T[f+1],Tref,
                        multi_data[11],multi_data[12],multi_data[13]))
        for k in range(ns):
            eligible = eligible and Y[k,f] >= 0. and Y[k,f+1] >= 0.
        if eligible:
            valid,left,right,dlam,dmu = face_temperature_tangents(
                T[f],T[f+1],dz[f],P,invW,mw,mixture,multi_data,soret,
                mdata,tdata,cws,Y[:,f],Y[:,f+1],Yf,multi,thermal,
                .5e-7*max(abs(T[f]),abs(T[f+1]),Tref))
            if valid:
                dleft[f,:],dright[f,:] = left,right
                derivatives[f,0],derivatives[f,1],derivatives[f,2] = dlam,dmu,1.


@njit(cache=True, nogil=True)
def jacobian_range_bank(start, stop, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                   fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                   multicomponent, soret, kinetics, analytic, jy, jt, radiation, frozen,
                   lower, diag, upper, border, carried_ws, carry_left, base_bank, reuse_faces=True, partials_start=0,
                   analytic_transport=False, algebraic_transport=False, thermal_bank=None):
    """Fill a tile using the private workspace owned by its exterior range.

    Only adjacent tiles of the same state/grid/configuration may carry faces.
    The owning range seeds its first left face; concurrent ranges never share
    buffers. Kept separate from the legacy void kernel to avoid escaping its
    scratch arrays into Python on small systems.
    """
    ns = Y.shape[0]
    ws = carried_ws
    for j in range(start, stop):
        reuse_left = reuse_faces and (j > start or carry_left)
        if reuse_left:
            ws = _rotate_face_workspace(ws)
        _node_columns(j, x, m, V, T, Y, rho, cp, omega, hk, lam0, mu0, J0, dz, L, scalars,
                      fuel_Y, oxidizer_Y, excess, fixed, Tfixed, P, invW, mw, mixture, multi_data,
                      multicomponent, soret, kinetics, analytic, jy, jt, lower, diag, upper, border, ws[:-1],
                      radiation, frozen, reuse_left, partials_start, ws[-1], base_bank, analytic_transport, algebraic_transport,thermal_bank)

    return ws
