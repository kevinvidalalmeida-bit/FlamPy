"""Experimental temperature transport tangents, private to a face bank.

Only transport is linearized. Thermochemistry and residual row assembly keep
their existing temperature trial. All tangents are per kelvin of an endpoint.
"""
import math
import numpy as np
from numba import njit
from kflame.chemistry.mechanism import R_UNIV
from kflame.chemistry.transport_temperature_derivatives import (
    temperature_term_tangents, nasa_cp_r_with_tangent, _poly_with_derivative)
from kflame.chemistry.thermal_block_derivatives import (block_temperature_tangents,
                                                      block_temperature_tangents_valid,
                                                      block_trial_floor_safe)
from kflame.chemistry.multicomponent_kernels import temperature_terms
from kflame.chemistry.thermal_implicit_derivatives import implicit_temperature_tangent


@njit(cache=True, fastmath=False)
def temperature_branch_safe(TL, TR, Tref, low, high, mid):
    """Keep the complete legacy trial when a constitutive branch may change."""
    t = .5*(TL+TR)
    h = .5e-7*max(abs(TL), abs(TR), Tref)
    if not math.isfinite(t) or t <= 0. or not math.isfinite(h):
        return False
    for k in range(mid.size):
        if t == mid[k] or (t > mid[k]) != (t+h > mid[k]):
            return False
        c = high[k] if t > mid[k] else low[k]
        base = c[0]+t*(c[1]+t*(c[2]+t*(c[3]+t*c[4]))) - 2.5
        upper = c[0]+(t+h)*(c[1]+(t+h)*(c[2]+(t+h)*(c[3]+(t+h)*c[4]))) - 2.5
        if abs(base-1e-3) <= 1e-10 or (base > 1e-3) != (upper > 1e-3):
            return False
    return True


@njit(cache=True, fastmath=False)
def face_temperature_tangents(TL, TR, spacing, P, invW, mw, mixture,
                              multi_data, soret, mdata, tdata, cws,
                              YL, YR, Yf, multi, thermal, guard_step=0.):
    """Reuse the current primal inverse, Schur LU and solved energy vector.

    No primal buffers are modified. Full ordinary diffusion tangents retain
    the inverse diagonal subtraction even for trace species and roundoff.
    Only two vectors and two scalar tangents escape into the face bank.
    """
    n = mw.size
    t = .5*(TL+TR)
    mu, rot, bd, a, b, c, cp = mdata
    dmu, drot = np.empty(n), np.empty(n)
    dbd, da, db, dc = (np.empty((n,n)), np.empty((n,n)),
                       np.empty((n,n)), np.empty((n,n)))
    invmix,sL,sR = 0.,0.,0.
    for k in range(n):
        if (not math.isfinite(YL[k]) or YL[k]<0. or not math.isfinite(YR[k]) or YR[k]<0.
            or not math.isfinite(invW[k]) or invW[k]<=0.):
            return False,np.zeros(n),np.zeros(n),0.,0.
        invmix+=Yf[k]*invW[k];sL+=YL[k]*invW[k];sR+=YR[k]*invW[k]
    if (not math.isfinite(invmix) or invmix<=1e-300 or not math.isfinite(sL)
        or sL<=1e-300 or not math.isfinite(sR) or sR<=1e-300):
        return False,np.zeros(n),np.zeros(n),0.,0.
    if guard_step>0.:
        upper=t+guard_step
        temperature_terms(upper,mw,multi_data[0],multi_data[1],multi_data[3],multi_data[4],
            multi_data[5],multi_data[6],multi_data[7],multi_data[8],multi_data[9],
            dmu,drot,dbd,da,db,dc)
        if not block_trial_floor_safe(mw,bd,dbd,mu,dmu):
            return False,np.zeros(n),np.zeros(n),0.,0.
    temperature_term_tangents(t, mw, multi_data[0], multi_data[1], multi_data[3],
        multi_data[4], multi_data[5], multi_data[6], multi_data[7], multi_data[8],
        multi_data[9], mu, rot, bd, a, b, c, dmu, drot, dbd, da, db, dc)
    cp_copy, dcp = np.empty(n), np.empty(n)
    nasa_cp_r_with_tangent(t, multi_data[11], multi_data[12], multi_data[13], cp_copy, dcp)
    vectors = cws[0]
    x = vectors[0]
    if not block_temperature_tangents_valid(t,x,cp,dcp,mw,multi_data[2],mu,rot,bd,a,b,c,
                                           dmu,drot,dbd,da,db,dc):
        return False, np.zeros(n), np.zeros(n), 0., 0.
    dA, dB, dC, dD, dE = block_temperature_tangents(
        t, x, cp, dcp, mw, multi_data[2], mu, rot, bd, a, b, c,
        dmu, drot, dbd, da, db, dc)
    wm = 1./max(invmix,1e-300)
    dMulti, dThermal, dLambda = implicit_temperature_tangent(
        t, P, wm, mw, cws, dA, dB, dC, dD, dE)
    left, right = np.empty(n), np.empty(n)
    gradient = np.empty(n)
    WL, WR = 1./max(sL,1e-300), 1./max(sR,1e-300)
    for k in range(n):
        gradient[k] = mw[k]*(YR[k]*WR*invW[k]-YL[k]*WL*invW[k])/spacing
    factor = P/(R_UNIV*t*wm)
    gradT = (TR-TL)/spacing
    for k in range(n):
        primal, tangent = 0., 0.
        for j in range(n):
            primal += multi[k,j]*gradient[j]
            tangent += dMulti[k,j]*gradient[j]
        coeff = thermal[k] if soret else 0.
        dcoeff = dThermal[k] if soret else 0.
        partial = factor*mw[k]*(tangent-primal/t) - (dcoeff/t-coeff/(t*t))*gradT
        left[k] = .5*partial+coeff/(t*spacing)
        right[k] = .5*partial-coeff/(t*spacing)
    # Differentiate the exact Wilke function at fixed face composition.
    root, phi = tdata[3], tdata[4]
    quarter_ratio, denominator = mixture[3], mixture[4]
    droot = np.empty(n)
    for k in range(n):
        value, slope = _poly_with_derivative(math.log(t),mixture[2][k],4)
        if guard_step>0.:
            upper_value,_ = _poly_with_derivative(math.log(t+guard_step),mixture[2][k],4)
            if value*upper_value<=0. or abs((t+guard_step)**.25*upper_value)<=1e-140:
                return False,left,right,dLambda,0.
        raw = t**.25*value
        droot[k] = (1. if raw >= 0. else -1.)*t**.25/t*(.25*value+slope)
        if root[k] <= 1e-140:
            return False, left, right, dLambda, 0.
    dViscosity = 0.
    for k in range(n):
        total, dtotal = 0., 0.
        for j in range(n):
            Xj = Yf[j]*wm*invW[j]
            total += Xj*phi[k,j]
            ratio = root[k]/root[j]*quarter_ratio[k,j]
            dratio = quarter_ratio[k,j]*(droot[k]/root[j]-root[k]/root[j]*droot[j]/root[j])
            dtotal += Xj*2.*(1.+ratio)*dratio/denominator[k,j]
        if total <= 1e-280:
            return False, left, right, dLambda, 0.
        weight = max(Yf[k]*wm*invW[k],1e-300)
        dViscosity += weight*(2.*root[k]*droot[k]/total - root[k]**2/total*(dtotal/total))
    valid = math.isfinite(dLambda) and math.isfinite(dViscosity)
    for k in range(n):
        valid = valid and math.isfinite(left[k]) and math.isfinite(right[k])
    return valid, left, right, dLambda, dViscosity
