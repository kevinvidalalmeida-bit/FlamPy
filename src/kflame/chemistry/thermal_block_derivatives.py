"""Private, fixed-composition temperature tangents of Dixon--Lewis blocks.

Differentiate the formulas in ``multicomponent_kernels._diffusion_matrix``
and ``thermal_schur``. This module neither changes nor evaluates the primal
closure. The parent owns numerical validation and implicit tangent solves.

The input ``cp`` is cp/R, ``dcp`` is its selected-branch T derivative, and
``bd`` includes the production overwrite of its diagonal. All inputs are
read-only to these functions; arbitrary array strides are supported by the
scalar reads. No symmetry of the ordered pair coefficients is assumed.
"""

import math

import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV


_FLOOR = 1.0e-300
_MIN_C_INTERNAL = 1.0e-3


@njit(cache=True, fastmath=False, nogil=True)
def block_temperature_tangents_valid(
    t, x, cp, dcp, mw, crot, mu, rot, bd, a, b, c,
    dmu, drot, dbd, da, db, dc, cp_margin=1.0e-12, floor_rtol=1.0e-12,
):
    """Conservative local domain/branch guard, without modifying inputs.

    ``cp_margin`` is an absolute cp/R distance from the internal-mode switch.
    ``floor_rtol`` is a relative exclusion band about each T-dependent floor.
    Stable active floors are permitted: their denominator derivative is zero.

    This is a local guard, not a proof that a finite temperature step keeps
    its branch. The caller must also check NASA joins (not supplied here),
    verify the actual cp/transport active sets at any trial temperature, and
    reject nonfinite outputs or singular primal blocks before tangent solves.
    Input dimensions/ranks and float dtype are part of the API contract;
    this helper checks sizes and matrix shapes within that contract.
    """
    if not math.isfinite(t) or t <= 0.0:
        return False
    if (not math.isfinite(cp_margin) or cp_margin < 0.0
            or not math.isfinite(floor_rtol) or floor_rtol < 0.0):
        return False
    n = x.size
    if n == 0:
        return False
    if (cp.size != n or dcp.size != n or mw.size != n or crot.size != n
            or mu.size != n or rot.size != n or dmu.size != n
            or drot.size != n):
        return False
    if (bd.shape[0] != n or bd.shape[1] != n
            or a.shape[0] != n or a.shape[1] != n
            or b.shape[0] != n or b.shape[1] != n
            or c.shape[0] != n or c.shape[1] != n
            or dbd.shape[0] != n or dbd.shape[1] != n
            or da.shape[0] != n or da.shape[1] != n
            or db.shape[0] != n or db.shape[1] != n
            or dc.shape[0] != n or dc.shape[1] != n):
        return False
    for i in range(n):
        if (not math.isfinite(x[i]) or x[i] < 0.0
                or not math.isfinite(cp[i]) or not math.isfinite(dcp[i])
                or not math.isfinite(mw[i]) or mw[i] <= 0.0
                or not math.isfinite(crot[i]) or crot[i] < 0.0
                or not math.isfinite(mu[i]) or mu[i] <= 0.0
                or not math.isfinite(rot[i]) or rot[i] <= 0.0
                or not math.isfinite(dmu[i]) or not math.isfinite(drot[i])):
            return False
        if abs((cp[i] - 2.5) - _MIN_C_INTERNAL) <= cp_margin:
            return False
    for i in range(n):
        for j in range(n):
            if (not math.isfinite(bd[i, j]) or bd[i, j] <= 0.0
                    or not math.isfinite(a[i, j])
                    or not math.isfinite(b[i, j])
                    or not math.isfinite(c[i, j])
                    or not math.isfinite(dbd[i, j])
                    or not math.isfinite(da[i, j])
                    or not math.isfinite(db[i, j])
                    or not math.isfinite(dc[i, j])):
                return False
            # L01/L12 use (wi+wj)*bd[j,i]; traversing every ordered pair
            # here covers that transpose without a symmetry assumption.
            raw = (mw[i] + mw[j]) * bd[i, j]
            if not math.isfinite(raw) or raw <= 0.0:
                return False
            if abs(raw - _FLOOR) <= floor_rtol * _FLOOR:
                return False
            # L22 uses max(bd[i,j], floor) independently of the pair sum.
            if abs(bd[i, j] - _FLOOR) <= floor_rtol * _FLOOR:
                return False
    return True


@njit(cache=True, fastmath=False, nogil=True)
def block_temperature_tangents(
    t, x, cp, dcp, mw, crot, mu, rot, bd, a, b, c,
    dmu, drot, dbd, da, db, dc,
):
    """Return (dL00, dL01, dL11, dL12, dL22), all per kelvin.

    The first four arrays have shape (N,N); dL22 has shape (N,) and is the
    derivative of the diagonal internal-mode block, not a dense diagonal
    matrix. Every output is independently allocated float64 C storage;
    outputs alias neither each other nor inputs.

    Preconditions: finite positive T, fixed x/mw/crot, correctly shaped
    finite primitive arrays, and stable branches. Call the validity helper
    and the caller's NASA/finite-step guards before accepting the tangent.
    This kernel does not impose extra physical clamps or run that guard
    again. At exact switch/floor equality it returns the selected branch's
    expression; that value is not advertised as a two-sided derivative.

    L00/L11 binary denominators are NOT floored in production. L01/L12
    have a floored pair denominator; L22 floors each reciprocal binary.
    Zero/inactive internal modes stay zero rather than differentiating
    their indicator. The input dbd diagonal must already differentiate
    the overwritten production binary diagonal.
    """
    n = x.size
    dl00 = np.empty((n, n), dtype=np.float64)
    dl01 = np.empty((n, n), dtype=np.float64)
    dl11 = np.empty((n, n), dtype=np.float64)
    dl12 = np.empty((n, n), dtype=np.float64)
    dl22 = np.empty(n, dtype=np.float64)

    # L00: its diagonal is identically zero. No binary floor belongs here.
    pref00 = 16.0 * t / 25.0
    dpref00 = 16.0 / 25.0
    for k in range(n):
        summed = 0.0
        dsummed = 0.0
        for j in range(n):
            if k != j:
                fraction = x[j] / bd[k, j]
                summed += fraction
                dsummed -= fraction * (dbd[k, j] / bd[k, j])
        summed /= mw[k]
        dsummed /= mw[k]
        for j in range(n):
            if k == j:
                dl00[k, j] = 0.0
            else:
                fraction = x[k] / bd[k, j]
                bracket = mw[j] * summed + fraction
                dbracket = (mw[j] * dsummed
                            - fraction * (dbd[k, j] / bd[k, j]))
                dl00[k, j] = (dpref00 * x[j] * bracket
                              + pref00 * x[j] * dbracket)

    # L01: row-contiguous writes, with each column reduction in ascending i.
    diagonal01 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            weight = -1.6 * x[i] * x[j] * mw[i]
            star = 1.2 * c[j, i] - 1.0
            numerator = weight * t * star
            dnumerator = weight * (star + t * 1.2 * dc[j, i])
            raw = (mw[j] + mw[i]) * bd[j, i]
            denominator = max(raw, _FLOOR)
            value = numerator / denominator
            derivative = dnumerator / denominator
            if raw > _FLOOR:
                ddenominator = (mw[j] + mw[i]) * dbd[j, i]
                derivative -= value * (ddenominator / denominator)
            dl01[i, j] = derivative
            diagonal01[j] += derivative
    for j in range(n):
        dl01[j, j] -= diagonal01[j]

    # L11: no clamp on term1, and both its diagonal terms are differentiated.
    ratio = np.empty(n, dtype=np.float64)
    dratio = np.empty(n, dtype=np.float64)
    diagonal11 = np.zeros(n, dtype=np.float64)
    ddiagonal11 = np.zeros(n, dtype=np.float64)
    five_over_3pi = 5.0 / (3.0 * math.pi)
    for k in range(n):
        ratio[k] = crot[k] / rot[k]
        dratio[k] = -ratio[k] * (drot[k] / rot[k])
    for i in range(n):
        wi = mw[i]
        for j in range(n):
            wj = mw[j]
            wj2 = wj * wj
            constant = 16.0 * t * x[j] / 25.0
            dconstant = 16.0 * x[j] / 25.0
            pair_squared = (wi + wj) ** 2
            term1 = bd[i, j] * pair_squared
            dterm1 = dbd[i, j] * pair_squared
            relax = 1.0 + five_over_3pi * (ratio[j] + ratio[i])
            drelax = five_over_3pi * (dratio[j] + dratio[i])
            term2 = 4.0 * wj * a[i, j] * relax
            dterm2 = 4.0 * wj * (da[i, j] * relax + a[i, j] * drelax)

            weight = constant * x[i] * wi / (wj * term1)
            dweight = (dconstant * x[i] * wi / (wj * term1)
                       - weight * (dterm1 / term1))
            numerator = 13.75 * wj2 - 3.0 * wj2 * b[i, j] - term2 * wj
            dnumerator = -3.0 * wj2 * db[i, j] - dterm2 * wj
            dl11[i, j] = dweight * numerator + weight * dnumerator

            diag_weight = x[i] / term1
            ddiag_weight = -diag_weight * (dterm1 / term1)
            diag_numerator = (7.5 * wj2 + wi * wi * (6.25 - 3.0 * b[i, j])
                              + term2 * wi)
            ddiag_numerator = -3.0 * wi * wi * db[i, j] + dterm2 * wi
            diagonal11[j] += diag_weight * diag_numerator
            ddiagonal11[j] += (ddiag_weight * diag_numerator
                              + diag_weight * ddiag_numerator)
    for j in range(n):
        constant = 16.0 * t * x[j] / 25.0
        dconstant = 16.0 * x[j] / 25.0
        dl11[j, j] -= (dconstant * diagonal11[j]
                       + constant * ddiagonal11[j])

    # cp/R-2.5 > 1e-3 is the exact production internal-mode branch.
    internal_constant = np.zeros(n, dtype=np.float64)
    dinternal_constant = np.zeros(n, dtype=np.float64)
    constant1 = np.zeros(n, dtype=np.float64)
    dconstant1 = np.zeros(n, dtype=np.float64)
    constant2 = np.zeros(n, dtype=np.float64)
    dconstant2 = np.zeros(n, dtype=np.float64)
    for k in range(n):
        cint = cp[k] - 2.5
        if cint > _MIN_C_INTERNAL:
            thermal_log_derivative = dcp[k] / cint + drot[k] / rot[k]
            internal_constant[k] = (
                32.0 * t * mw[k] * x[k] * crot[k]
                / (5.0 * math.pi * cint * rot[k])
            )
            dinternal_constant[k] = (
                32.0 * mw[k] * x[k] * crot[k]
                / (5.0 * math.pi * cint * rot[k])
                - internal_constant[k] * thermal_log_derivative
            )
            constant1[k] = 4.0 * t * x[k] / cint
            dconstant1[k] = 4.0 * x[k] / cint - constant1[k] * (dcp[k] / cint)
            constant2[k] = 12.0 * mw[k] * crot[k] / (5.0 * math.pi * cint * rot[k])
            dconstant2[k] = -constant2[k] * thermal_log_derivative

    # L12: transpose orientations and the ADDITIVE diagonal correction matter.
    diagonal12 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if cp[j] - 2.5 <= _MIN_C_INTERNAL:
                # The entire inactive column is identically zero. Avoid
                # evaluating a needless quotient/log-derivative there.
                dl12[i, j] = 0.0
                continue
            numerator = x[i] * a[j, i] * internal_constant[j]
            dnumerator = x[i] * (da[j, i] * internal_constant[j]
                                 + a[j, i] * dinternal_constant[j])
            raw = (mw[i] + mw[j]) * bd[j, i]
            denominator = max(raw, _FLOOR)
            value = numerator / denominator
            derivative = dnumerator / denominator
            if raw > _FLOOR:
                ddenominator = (mw[i] + mw[j]) * dbd[j, i]
                derivative -= value * (ddenominator / denominator)
            dl12[i, j] = derivative
            diagonal12[j] += derivative
    for j in range(n):
        dl12[j, j] += diagonal12[j]

    # L22: inactive primal diagonal is one, so its tangent is zero.
    for i in range(n):
        cint = cp[i] - 2.5
        if cint > _MIN_C_INTERNAL:
            first = 0.0
            dfirst = 0.0
            second = 0.0
            dsecond = 0.0
            for j in range(n):
                denominator = max(bd[i, j], _FLOOR)
                inverse_b = 1.0 / denominator
                dinverse_b = 0.0
                if bd[i, j] > _FLOOR:
                    # Avoid denominator**2, which underflows near the floor.
                    dinverse_b = -inverse_b * (dbd[i, j] / denominator)
                first += inverse_b * x[j]
                dfirst += dinverse_b * x[j]
                if j != i:
                    second += a[i, j] * inverse_b * (x[j] / mw[j])
                    dsecond += (da[i, j] * inverse_b + a[i, j] * dinverse_b) * (x[j] / mw[j])
            viscous = (-8.0 / math.pi * mw[i] * x[i] ** 2 * crot[i]
                       / (cint ** 2 * R_UNIV * mu[i] * rot[i]))
            dviscous = -viscous * (
                2.0 * dcp[i] / cint + dmu[i] / mu[i] + drot[i] / rot[i]
            )
            total = first + constant2[i] * second
            dtotal = dfirst + dconstant2[i] * second + constant2[i] * dsecond
            dl22[i] = dviscous - dconstant1[i] * total - constant1[i] * dtotal
        else:
            dl22[i] = 0.0
    return dl00, dl01, dl11, dl12, dl22


_BINARY_FLOOR=1e-300
_ROOT_MIN=1e-140

@njit(cache=True, fastmath=False, nogil=True)
def block_trial_floor_safe(mw, bd, trialbd, mu, trialmu, floor_rtol=1.0e-12):
    """Return whether both actual endpoints preserve usable floor branches.

    Caller guarantees vector shapes (N,), matrix shapes (N,N), floating
    dtype and immutable, correctly populated inputs. No output/scratch
    arrays are allocated; scalar reads support C, F and strided inputs.
    Work is O(N^2) in C row order for both binary matrices plus O(N) vector
    checks. Every ordered pair, including the overwritten binary diagonal,
    is visited; transposed L01/L12 pair indices are therefore covered too.

    All mw, binary coefficients and viscosities must be finite positive at
    both endpoints. L00/L11 have no binary floor, so negative/zero binaries
    must be rejected even when a thermal block would otherwise floor them.
    The two actual max-denominator branch flags must match for:
      * max(bd[i,j], 1e-300) in L22;
      * max((mw[i]+mw[j])*bd[i,j], 1e-300) in L01/L12.
    Stable active floors remain eligible away from the exclusion band.
    Equality and values within floor_rtol*1e-300 of either floor are declined
    at either endpoint. No value, branch, denominator or input is repaired.

    sqrt(mu) and sqrt(trialmu) must both exceed 1e-140, deliberately above
    the Wilke denominator's 1e-150 root floor. The wrapper still checks its
    actual root array and signed-fit branch where applicable (module note).
    """
    if not math.isfinite(floor_rtol) or floor_rtol < 0.0:
        return False
    n = mw.size
    if n == 0:
        return False
    margin = floor_rtol * _BINARY_FLOOR
    for i in range(n):
        if not math.isfinite(mw[i]) or mw[i] <= 0.0:
            return False
        if (not math.isfinite(mu[i]) or mu[i] <= 0.0
                or not math.isfinite(trialmu[i]) or trialmu[i] <= 0.0):
            return False
        # Positive finite mu gives a finite square root. Check both actual
        # values, not a prediction mu+h*dmu or a floor-adjusted substitute.
        if math.sqrt(mu[i]) <= _ROOT_MIN or math.sqrt(trialmu[i]) <= _ROOT_MIN:
            return False
    for i in range(n):
        wi = mw[i]
        for j in range(n):
            base_binary = bd[i, j]
            trial_binary = trialbd[i, j]
            if (not math.isfinite(base_binary) or base_binary <= 0.0
                    or not math.isfinite(trial_binary) or trial_binary <= 0.0):
                return False
            if ((base_binary > _BINARY_FLOOR)
                    != (trial_binary > _BINARY_FLOOR)):
                return False
            if (abs(base_binary - _BINARY_FLOOR) <= margin
                    or abs(trial_binary - _BINARY_FLOOR) <= margin):
                return False
            pair_weight = wi + mw[j]
            base_pair = pair_weight * base_binary
            trial_pair = pair_weight * trial_binary
            if (not math.isfinite(base_pair) or base_pair <= 0.0
                    or not math.isfinite(trial_pair) or trial_pair <= 0.0):
                return False
            if ((base_pair > _BINARY_FLOOR)
                    != (trial_pair > _BINARY_FLOOR)):
                return False
            if (abs(base_pair - _BINARY_FLOOR) <= margin
                    or abs(trial_pair - _BINARY_FLOOR) <= margin):
                return False
    return True
