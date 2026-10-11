"""Analytic face composition tangents with frozen multicomponent transport.

The diffusivity matrix, thermal diffusivities, conductivity and temperature
data are held fixed, as in the existing composition Jacobian policy. Only the
mixture weights, mole-fraction gradient and Wilke mixture viscosity vary.
All species columns cost O(ns**2), with private C-contiguous outputs and no
input mutation. This module does not change the primal transport closure.
"""

import math

import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV


@njit(cache=True, nogil=True, fastmath=False)
def frozen_face_derivatives(YL, YR, TL, TR, dz, P, invW, mw, D, root, phi,
                            left_changed):
    """Return ``(valid, dflux, dmu)`` for one face's composition columns.

    ``dflux[k, q]`` and ``dmu[q]`` differentiate with respect to species q
    at YL when ``left_changed`` is true, otherwise at YR. The inputs are
    unnormalized mass fractions; face composition is their arithmetic mean.
    Temperatures, pressure, D, root and phi are constant. Consequently the
    frozen Soret term has zero composition derivative.

    Nonnegative zero species use the right physical limit of Wilke's
    numerator, ignoring its artificial 1e-300 floor at exactly zero. For
    positive subfloor mole fractions the actual clamp derivative is zero.
    Exactly-on-floor and unsafe states return valid=False and zero outputs;
    callers must fall back to their existing finite-difference path.
    """
    ns = mw.size
    dflux = np.zeros((ns, ns))
    dmu = np.zeros(ns)
    if (ns == 0 or YL.size != ns or YR.size != ns or invW.size != ns
            or root.size != ns or D.shape[0] != ns or D.shape[1] != ns
            or phi.shape[0] != ns or phi.shape[1] != ns):
        return False, dflux, dmu
    if (not math.isfinite(TL) or not math.isfinite(TR)
            or not math.isfinite(dz) or not math.isfinite(P)
            or TL <= 0.0 or TR <= 0.0 or dz <= 0.0 or P <= 0.0):
        return False, dflux, dmu

    floor = 1e-300
    Tf = 0.5 * TL + 0.5 * TR
    sL = 0.0
    sR = 0.0
    sF = 0.0
    for k in range(ns):
        if (not math.isfinite(YL[k]) or not math.isfinite(YR[k])
                or YL[k] < 0.0 or YR[k] < 0.0
                or not math.isfinite(invW[k]) or invW[k] <= 0.0
                or not math.isfinite(mw[k]) or mw[k] <= 0.0
                or not math.isfinite(root[k]) or root[k] < 0.0):
            return False, dflux, dmu
        sL += YL[k] * invW[k]
        sR += YR[k] * invW[k]
        sF += (0.5 * YL[k] + 0.5 * YR[k]) * invW[k]
    if (not math.isfinite(sL) or not math.isfinite(sR)
            or not math.isfinite(sF) or sL <= floor or sR <= floor
            or sF <= floor):
        return False, dflux, dmu

    WL = 1.0 / sL
    WR = 1.0 / sR
    wm = 1.0 / sF
    pressure_scale = P / (R_UNIV * Tf)
    side_sum = sL if left_changed else sR
    sign = -1.0 if left_changed else 1.0
    gradient_scale = sign / dz / side_sum
    if (not math.isfinite(WL) or not math.isfinite(WR)
            or not math.isfinite(wm) or not math.isfinite(pressure_scale)
            or not math.isfinite(gradient_scale)):
        return False, dflux, dmu

    gradient = np.empty(ns)
    side_projection_input = np.empty(ns)
    X = np.empty(ns)
    beta = np.empty(ns)
    for k in range(ns):
        gradient[k] = mw[k] * (YR[k] * WR * invW[k]
                              - YL[k] * WL * invW[k]) / dz
        side_y = YL[k] if left_changed else YR[k]
        side_projection_input[k] = mw[k] * invW[k] * side_y / side_sum
        X[k] = (0.5 * YL[k] + 0.5 * YR[k]) * wm * invW[k]
        beta[k] = 0.5 * wm * invW[k]
        if (not math.isfinite(gradient[k])
                or not math.isfinite(side_projection_input[k])
                or not math.isfinite(X[k]) or not math.isfinite(beta[k])
                or X[k] == floor):
            return False, dflux, dmu

    # Each D and phi row is traversed in ascending species order. These two
    # projections replace ns separate perturbed matrix-vector products.
    flux_dot = np.empty(ns)
    side_dot = np.empty(ns)
    numerator_slope = np.empty(ns)
    denominator_weight = np.empty(ns)
    common_mu = 0.0
    for k in range(ns):
        dot = 0.0
        projection = 0.0
        denominator = 0.0
        for l in range(ns):
            if (not math.isfinite(D[k, l]) or not math.isfinite(phi[k, l])
                    or phi[k, l] < 0.0):
                return False, dflux, dmu
            dot += D[k, l] * gradient[l]
            projection += D[k, l] * side_projection_input[l]
            denominator += X[l] * phi[k, l]
        root_squared = root[k] * root[k]
        if (not math.isfinite(dot) or not math.isfinite(projection)
                or not math.isfinite(denominator) or denominator <= floor
                or not math.isfinite(root_squared)):
            return False, dflux, dmu
        flux_dot[k] = dot
        side_dot[k] = projection
        if X[k] == 0.0 or X[k] > floor:
            # The physical zero limit and smooth positive branch are
            # homogeneous of degree zero; their shared normalization terms
            # cancel algebraically, rather than numerically.
            numerator_slope[k] = root_squared / denominator
            denominator_weight[k] = (root_squared * X[k] / denominator
                                     / denominator)
        else:
            # Positive mole fractions below the artificial floor have a
            # constant numerator, but their Wilke denominator still varies.
            numerator_slope[k] = 0.0
            denominator_weight[k] = (root_squared * floor / denominator
                                     / denominator)
            common_mu += root_squared * floor / denominator
        if (not math.isfinite(numerator_slope[k])
                or not math.isfinite(denominator_weight[k])
                or not math.isfinite(common_mu)):
            return False, dflux, dmu

    for q in range(ns):
        dmu[q] = beta[q] * (numerator_slope[q] + common_mu)
    valid = True
    for k in range(ns):
        flux_scale = mw[k] * pressure_scale
        for q in range(ns):
            dflux[k, q] = flux_scale * (
                0.5 * invW[q] * flux_dot[k]
                + sF * gradient_scale * (D[k, q] * mw[q] * invW[q]
                                         - invW[q] * side_dot[k]))
            dmu[q] -= beta[q] * denominator_weight[k] * phi[k, q]
            if not math.isfinite(dflux[k, q]):
                valid = False
    for q in range(ns):
        if not math.isfinite(dmu[q]):
            valid = False
    if not valid:
        for k in range(ns):
            dmu[k] = 0.0
            for q in range(ns):
                dflux[k, q] = 0.0
    return valid, dflux, dmu
