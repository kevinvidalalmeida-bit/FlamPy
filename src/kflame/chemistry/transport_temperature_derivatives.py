"""Temperature derivatives of native fitted transport and NASA polynomials."""
import math
import numpy as np
from numba import njit
from kflame.chemistry.mechanism import R_UNIV

@njit(cache=True, fastmath=False, nogil=True)
def _poly_with_derivative(z, coefficients, last_degree):
    """Horner value and derivative with respect to the polynomial argument."""
    value = coefficients[last_degree]
    derivative = 0.0
    for degree in range(last_degree - 1, -1, -1):
        # Use the previous value, before its Horner update.
        derivative = derivative * z + value
        value = value * z + coefficients[degree]
    return value, derivative


@njit(cache=True, fastmath=False, nogil=True)
def temperature_term_tangents(
    t, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit, diffusion_fit,
    astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c,
    dmu, drot_relax, dbdiff, da, db, dc,
):
    """Fill temperature tangents; all supplied primal buffers stay unchanged.

    The caller supplies the unchanged original primal arrays.
    Polynomial values are re-evaluated only to compute derivatives.
    Diagonal dbdiff differentiates the override, not the binary-fit diagonal.
    max(1,zrot[k]) is constant under a temperature derivative, including zrot=1.
    """
    n = mw.size
    logt = math.log(t)
    sqrt_t = math.sqrt(t)
    inv_t = 1.0 / t
    alpha = 0.5 * math.pi ** 1.5
    beta = 0.25 * math.pi ** 2 + 2.0
    gamma = math.pi ** 1.5
    for k in range(n):
        value, derivative = _poly_with_derivative(logt, viscosity_fit[k], 4)
        # Do not use a log derivative divided by the fitted value: it may be 0.
        dmu[k] = sqrt_t * inv_t * (0.5 * value * value + 2.0 * value * derivative)
        tr = epsilon[k] * inv_t
        sqrt_tr = math.sqrt(tr)
        frot = (1.0 + alpha * sqrt_tr + beta * tr + gamma * sqrt_tr * tr)
        # -T F'(T) = alpha/2 sqrt(tr) + beta tr + 3 gamma/2 tr^(3/2).
        minus_t_dfrot = (0.5 * alpha * sqrt_tr + beta * tr
                        + 1.5 * gamma * sqrt_tr * tr)
        drot_relax[k] = rot_relax[k] * inv_t * minus_t_dfrot / frot
        for j in range(n):
            z = logt - log_eps_pair[k, j]
            _, derivative = _poly_with_derivative(z, astar_fit[k, j], 8)
            da[k, j] = derivative * inv_t
            _, derivative = _poly_with_derivative(z, bstar_fit[k, j], 8)
            db[k, j] = derivative * inv_t
            _, derivative = _poly_with_derivative(z, cstar_fit[k, j], 8)
            dc[k, j] = derivative * inv_t
            value, derivative = _poly_with_derivative(logt, diffusion_fit[k, j], 4)
            dbdiff[k, j] = sqrt_t * (1.5 * value + derivative)
        dbdiff[k, k] = (1.2 * R_UNIV / mw[k]
                       * (mu[k] * a[k, k] + t * dmu[k] * a[k, k]
                          + t * mu[k] * da[k, k]))


@njit(cache=True, fastmath=False, nogil=True)
def nasa_cp_r_with_tangent(t, low, high, midpoints, cp_r, dcp_r):
    """NASA-7 cp/R and derivative of the selected branch, both in caller buffers.

    Exactly the production choice: high iff T > Tmid, low at equality.  At a
    switch the returned derivative belongs to that low polynomial; it is not
    a claim of a two-sided derivative or of continuity across the join.
    """
    for k in range(midpoints.size):
        coeff = high[k] if t > midpoints[k] else low[k]
        cp_r[k] = coeff[0] + t * (coeff[1] + t * (
            coeff[2] + t * (coeff[3] + t * coeff[4])))
        dcp_r[k] = coeff[1] + t * (2.0 * coeff[2] + t * (
            3.0 * coeff[3] + t * 4.0 * coeff[4]))

