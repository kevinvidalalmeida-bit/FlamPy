"""Opt-in O(ns**2) finite composition trials of frozen transport.

This keeps the existing finite perturbations and row differences. It evaluates
the perturbed face closure using algebraic rank-one updates, not infinitesimal
derivatives. Coefficients D, DT, root and phi stay frozen. Negative increments
use the ordered scalar-compatible fallback; no physical residual is changed.
"""

import math

import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV
from kflame.flame.frozen_transport_columns import (
    frozen_composition_columns as _original_frozen_composition_columns,
)


@njit(cache=True, nogil=True, fastmath=False)
def _rational_face_trials(YL, YR, TL, TR, spacing, P, invW, mw, Yb,
                          left_changed, D, DT, root, phi, flux, mu):
    """Fill one active face's private outputs; False requests whole-call fallback."""
    ns = mw.size
    floor = 1e-300
    Tf = 0.5 * (TL + TR)
    if (not math.isfinite(TL) or not math.isfinite(TR)
            or not math.isfinite(Tf) or not math.isfinite(spacing)
            or not math.isfinite(P) or TL <= 0.0 or TR <= 0.0
            or spacing <= 0.0 or P <= 0.0):
        return False
    sL = 0.0
    sR = 0.0
    sF = 0.0
    for k in range(ns):
        if (not math.isfinite(YL[k]) or not math.isfinite(YR[k])
                or YL[k] < 0.0 or YR[k] < 0.0
                or not math.isfinite(invW[k]) or invW[k] <= 0.0
                or not math.isfinite(mw[k]) or mw[k] <= 0.0
                or not math.isfinite(root[k]) or root[k] < 0.0
                or not math.isfinite(DT[k])):
            return False
        sL += YL[k] * invW[k]
        sR += YR[k] * invW[k]
        sF += 0.5 * (YL[k] + YR[k]) * invW[k]
    if (not math.isfinite(sL) or not math.isfinite(sR)
            or not math.isfinite(sF) or sL <= floor or sR <= floor
            or sF <= floor):
        return False
    WL = 1.0 / sL
    WR = 1.0 / sR
    pressure_scale = P / (R_UNIV * Tf)
    gradT = (TR - TL) / spacing
    if (not math.isfinite(WL) or not math.isfinite(WR)
            or not math.isfinite(pressure_scale) or pressure_scale <= 0.0
            or not math.isfinite(gradT)):
        return False

    side_sum = sL if left_changed else sR
    sign = -1.0 if left_changed else 1.0
    gradient = np.empty(ns)
    side_input = np.empty(ns)
    u = np.empty(ns)
    root_squared = np.empty(ns)
    delta = np.empty(ns)
    face_sum_trial = np.empty(ns)
    scaled_floor = np.empty(ns)
    update_factor = np.empty(ns)
    for q in range(ns):
        side_y = YL[q] if left_changed else YR[q]
        trial_y = Yb[q, 2 + q]
        if not math.isfinite(trial_y) or trial_y < 0.0:
            return False
        step = trial_y - side_y
        # Production uses positive trials. Subtractive updates can lose a
        # dominant species' remaining mole sum; preserve the scalar route.
        if step < 0.0:
            return False
        weighted_step = step * invW[q]
        side_sum_trial = side_sum + weighted_step
        delta[q] = 0.5 * weighted_step
        face_sum_trial[q] = sF + delta[q]
        scaled_floor[q] = floor * face_sum_trial[q]
        if (not math.isfinite(step) or not math.isfinite(weighted_step)
                or not math.isfinite(side_sum_trial) or side_sum_trial <= floor
                or not math.isfinite(face_sum_trial[q])
                or face_sum_trial[q] <= floor
                or not math.isfinite(scaled_floor[q]) or scaled_floor[q] <= 0.0):
            return False
        update_factor[q] = sign * step / spacing / side_sum_trial
        gradient[q] = mw[q] * (YR[q] * WR * invW[q]
                               - YL[q] * WL * invW[q]) / spacing
        side_input[q] = mw[q] * invW[q] * side_y / side_sum
        u[q] = 0.5 * (YL[q] + YR[q]) * invW[q]
        root_squared[q] = root[q] * root[q]
        if (not math.isfinite(update_factor[q])
                or not math.isfinite(gradient[q])
                or not math.isfinite(side_input[q])
                or not math.isfinite(u[q]) or not math.isfinite(root_squared[q])):
            return False
        mu[q] = 0.0

    gdot = np.empty(ns)
    sidedot = np.empty(ns)
    wilke_base = np.empty(ns)
    for k in range(ns):
        g = 0.0
        s = 0.0
        h = 0.0
        for l in range(ns):
            if (not math.isfinite(D[k, l]) or not math.isfinite(phi[k, l])
                    or phi[k, l] < 0.0):
                return False
            g += D[k, l] * gradient[l]
            s += D[k, l] * side_input[l]
            h += phi[k, l] * u[l]
        if not math.isfinite(g) or not math.isfinite(s) or not math.isfinite(h):
            return False
        gdot[k] = g
        sidedot[k] = s
        wilke_base[k] = h

    # Row-major traversal of D and phi. Each mu column retains ascending-k
    # accumulation; every column is a complete finite trial, with its own
    # side/face normalization and exact (in real arithmetic) Wilke floors.
    for k in range(ns):
        thermal_flux = DT[k] / Tf * gradT
        for q in range(ns):
            dot_trial = gdot[k] + update_factor[q] * (
                D[k, q] * mw[q] * invW[q] - invW[q] * sidedot[k])
            value = pressure_scale * face_sum_trial[q] * mw[k] * dot_trial - thermal_flux
            numerator = u[k] + (delta[q] if k == q else 0.0)
            denominator = wilke_base[k] + phi[k, q] * delta[q]
            if (not math.isfinite(value) or not math.isfinite(numerator)
                    or not math.isfinite(denominator)
                    or numerator < 0.0 or denominator < 0.0):
                return False
            flux[k, q] = value
            if numerator <= scaled_floor[q] or denominator <= scaled_floor[q]:
                # Scale back before multiplying floored contributions: a
                # subnormal scaled floor must not underflow before division.
                mu[q] += (max(numerator / face_sum_trial[q], floor)
                          * root_squared[k]
                          / max(denominator / face_sum_trial[q], floor))
            else:
                mu[q] += root_squared[k] * numerator / denominator
            if not math.isfinite(mu[q]):
                return False
    return True


@njit(cache=True, nogil=True, fastmath=False)
def frozen_composition_columns(ns, YW, TW, dzW, P, invW, mw, Yb, has_left,
                               has_right, tleft, tright, frozen_multi,
                               frozen_thermal):
    """Same interface/output ownership as the original finite-trial helper.

    Inactive face buffers are allocated privately and left uninitialized,
    matching the existing contract. If either active face is unsafe, discard
    all provisional outputs and run the ORIGINAL helper for the entire call.
    """
    flux_left = np.empty((ns, ns))
    flux_right = np.empty((ns, ns))
    mu_left = np.empty(ns)
    mu_right = np.empty(ns)
    for side in range(2):
        if (side == 0 and not has_left) or (side == 1 and not has_right):
            continue
        a = 1 + side
        td = tleft if side == 0 else tright
        flux = flux_left if side == 0 else flux_right
        mu = mu_left if side == 0 else mu_right
        valid = _rational_face_trials(
            YW[:, a], YW[:, a + 1], TW[a], TW[a + 1], dzW[a], P, invW, mw,
            Yb, side == 1, frozen_multi[side], frozen_thermal[side], td[3],
            td[4], flux, mu,
        )
        if not valid:
            return _original_frozen_composition_columns(
                ns, YW, TW, dzW, P, invW, mw, Yb, has_left, has_right,
                tleft, tright, frozen_multi, frozen_thermal,
            )
    return flux_left, flux_right, mu_left, mu_right

