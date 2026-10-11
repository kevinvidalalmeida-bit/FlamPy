"""Frozen operation-order reference before the multicomponent HPC campaign.

Dixon--Lewis formulation adapted from Cantera (BSD-3-Clause); see
CANTERA_TRANSPORT_LICENSE.txt. Keep this oracle independent of current helpers.
"""

import math

import numpy as np

from numba import njit

from kflame.chemistry.mechanism import R_UNIV

_MIN_C_INTERNAL = 1e-3

_BLAS_SPECIES = 24

# Original source SHA256: 5b1ceefa8d41aff2deee2bfbc19519a84cf73a4d9f5032bd217e5dcc43005c74

@njit(cache=True)
def temperature_terms(t, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit, diffusion_fit,
                      astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c):
    """Composition-independent face data: viscosities, Zrot, collision ratios, D_kj."""
    n = mw.size
    logt = math.log(t)
    sqrt_t = math.sqrt(t)
    for k in range(n):
        value = viscosity_fit[k, 4]
        for degree in range(3, -1, -1):
            value = value * logt + viscosity_fit[k, degree]
        mu[k] = sqrt_t * value * value
        tr = epsilon[k] / t
        sqrt_tr = math.sqrt(tr)
        frot = (1.0 + 0.5 * math.pi ** 1.5 * sqrt_tr
                + (0.25 * math.pi ** 2 + 2.0) * tr
                + math.pi ** 1.5 * sqrt_tr * tr)
        rot_relax[k] = max(1.0, zrot[k]) * frot298[k] / frot
        for j in range(n):
            z = logt - log_eps_pair[k, j]
            aa, bb, cc = astar_fit[k, j, 8], bstar_fit[k, j, 8], cstar_fit[k, j, 8]
            for degree in range(7, -1, -1):
                aa = aa * z + astar_fit[k, j, degree]
                bb = bb * z + bstar_fit[k, j, degree]
                cc = cc * z + cstar_fit[k, j, degree]
            a[k, j], b[k, j], c[k, j] = aa, bb, cc
            value = diffusion_fit[k, j, 4]
            for degree in range(3, -1, -1):
                value = value * logt + diffusion_fit[k, j, degree]
            bdiff[k, j] = t * sqrt_t * value
        bdiff[k, k] = 1.2 * R_UNIV * t * mu[k] * a[k, k] / mw[k]

@njit(cache=True)
def composition_workspace(n):
    """Scratch arrays of one face system, reusable across faces of one thread."""
    vectors = np.empty((12, n))
    matrices = np.empty((9, n, n))
    pivots = np.empty((2, n), dtype=np.int64)
    return vectors, matrices, pivots

@njit(cache=True)
def _diffusion_matrix(t, y, mix_weight, inv_w, mw, bdiff, x, l00):
    n = mw.size
    for k in range(n):
        x[k] = max(y[k] * mix_weight * inv_w[k], 1e-20)
    for k in range(n):
        summed = 0.0
        for j in range(n):
            if k != j:
                summed += x[j] / bdiff[k, j]
        summed /= mw[k]
        for j in range(n):
            l00[k, j] = 16.0 * t / 25.0 * x[j] * (mw[j] * summed + x[k] / bdiff[k, j]) if k != j else 0.0

@njit(cache=True)
def _multi_matrix(t, pressure, mix_weight, x, mw, inverse, dt, multi, thermal):
    n = mw.size
    prefactor = 16.0 * t * mix_weight / (25.0 * pressure)
    for k in range(n):
        thermal[k] = dt[k]
        for j in range(n):
            multi[k, j] = prefactor * x[k] / mw[j] * (inverse[k, j] - inverse[k, k])

@njit(cache=True)
def composition_terms_ws(t, pressure, y, cp_r, mw, inv_w, crot, mu, rot_relax, bdiff, a, b, c,
                         multi, thermal, workspace):
    """Exact Schur path of ``composition_terms`` without per-face allocation."""
    n = mw.size
    vectors, matrices, pivots = workspace
    x, column, dt = vectors[0], vectors[1], vectors[2]
    l00, lu, inverse = matrices[0], matrices[1], matrices[2]
    denom = 0.0
    for k in range(n):
        denom += y[k] * inv_w[k]
    mix_weight = 1.0 / max(denom, 1e-300)
    _diffusion_matrix(t, y, mix_weight, inv_w, mw, bdiff, x, l00)
    _inverse_into(l00, lu, pivots[0], column, inverse)
    lam = thermal_schur(t, x, cp_r, bdiff, a, b, c, rot_relax, mw, crot, mu, inverse, dt, workspace)
    _multi_matrix(t, pressure, mix_weight, x, mw, inverse, dt, multi, thermal)
    return pressure * mix_weight / (R_UNIV * t), lam, mix_weight

@njit(cache=True)
def _lu(a, piv):
    """In-place LU with partial pivoting for the small face systems."""
    n = a.shape[0]
    for k in range(n):
        p = k
        big = abs(a[k, k])
        for i in range(k + 1, n):
            if abs(a[i, k]) > big:
                big = abs(a[i, k])
                p = i
        piv[k] = p
        if p != k:
            for j in range(n):
                a[k, j], a[p, j] = a[p, j], a[k, j]
        pivot = a[k, k]
        if pivot == 0.0:
            raise np.linalg.LinAlgError("Singular multicomponent face matrix")
        for i in range(k + 1, n):
            factor = a[i, k] / pivot
            a[i, k] = factor
            if factor != 0.0:
                for j in range(k + 1, n):
                    a[i, j] -= factor * a[k, j]

@njit(cache=True)
def _lu_solve(a, piv, b):
    n = a.shape[0]
    for k in range(n):
        p = piv[k]
        if p != k:
            b[k], b[p] = b[p], b[k]
    for i in range(1, n):
        value = b[i]
        for j in range(i):
            value -= a[i, j] * b[j]
        b[i] = value
    for i in range(n - 1, -1, -1):
        value = b[i]
        for j in range(i + 1, n):
            value -= a[i, j] * b[j]
        b[i] = value / a[i, i]

@njit(cache=True)
def _inverse_into(matrix, lu, piv, column, inverse):
    n = matrix.shape[0]
    if n >= _BLAS_SPECIES:
        inverse[:, :] = np.linalg.inv(matrix)
        return
    lu[:, :] = matrix
    _lu(lu, piv)
    for j in range(n):
        column[:] = 0.0
        column[j] = 1.0
        _lu_solve(lu, piv, column)
        for i in range(n):
            inverse[i, j] = column[i]

@njit(cache=True)
def thermal_schur(temperature, x, cp_r, bdiff, astar, bstar, cstar, rot_relax, mw, crot,
                  viscosity, l00_inverse, dthermal, workspace):
    """Loop form of ``thermal_system`` with its exact block elimination.

    Same Dixon--Lewis blocks: L00 is inverted, the diagonal L22 eliminated,
    and the N by N energy Schur system solved; no 3N matrix is formed.
    Fills ``dthermal`` and returns the conductivity.
    """
    n = x.size
    t = float(temperature)
    vectors, matrices, pivots = workspace
    cint, internal, ratio = vectors[3], vectors[4], vectors[5]
    internal_constant, constant1, constant2 = vectors[6], vectors[7], vectors[8]
    l22, rhs = vectors[9], vectors[10]
    l01, l11, l12 = matrices[3], matrices[4], matrices[5]
    ainv_b, d_einv, schur = matrices[6], matrices[7], matrices[8]
    piv = pivots[1]
    for k in range(n):
        cint[k] = cp_r[k] - 2.5
        internal[k] = 1.0 if cint[k] > _MIN_C_INTERNAL else 0.0
        ratio[k] = crot[k] / rot_relax[k]
    # L01 with its column-sum diagonal correction.
    for j in range(n):
        column = 0.0
        for i in range(n):
            value = (-1.6 * t * x[i] * x[j] * mw[i] * (1.2 * cstar[j, i] - 1.0)
                     / max((mw[j] + mw[i]) * bdiff[j, i], 1.0e-300))
            l01[i, j] = value
            column += value
        l01[j, j] -= column
    # L11.
    five_over_3pi = 5.0 / (3.0 * math.pi)
    for j in range(n):
        wj = mw[j]
        wj2 = wj * wj
        constant = 16.0 * t * x[j] / 25.0
        diagonal = 0.0
        for i in range(n):
            wi = mw[i]
            term1 = bdiff[i, j] * (wi + wj) ** 2
            term2 = 4.0 * wj * astar[i, j] * (1.0 + five_over_3pi * (ratio[j] + ratio[i]))
            l11[i, j] = constant * x[i] * wi / (wj * term1) * (13.75 * wj2 - 3.0 * wj2 * bstar[i, j] - term2 * wj)
            diagonal += x[i] / term1 * (7.5 * wj2 + wi * wi * (6.25 - 3.0 * bstar[i, j]) + term2 * wi)
        l11[j, j] -= constant * diagonal
    # L12 and the diagonal internal-mode block L22.
    for k in range(n):
        internal_constant[k] = 0.0
        constant1[k] = 0.0
        constant2[k] = 0.0
        if internal[k] > 0.0:
            internal_constant[k] = 32.0 * t * mw[k] * x[k] * crot[k] / (5.0 * math.pi * cint[k] * rot_relax[k])
            constant1[k] = 4.0 * t * x[k] / cint[k]
            constant2[k] = 12.0 * mw[k] * crot[k] / (5.0 * math.pi * cint[k] * rot_relax[k])
    for j in range(n):
        column = 0.0
        for i in range(n):
            value = x[i] * astar[j, i] * internal_constant[j] / max((mw[i] + mw[j]) * bdiff[j, i], 1.0e-300)
            l12[i, j] = value
            column += value
        l12[j, j] += column
    for i in range(n):
        if internal[i] > 0.0:
            first = 0.0
            second = 0.0
            for j in range(n):
                inverse_b = 1.0 / max(bdiff[i, j], 1.0e-300)
                first += inverse_b * x[j]
                if j != i:
                    second += astar[i, j] * inverse_b * (x[j] / mw[j])
            l22[i] = (-8.0 / math.pi * mw[i] * x[i] ** 2 * crot[i]
                      / (cint[i] ** 2 * R_UNIV * viscosity[i] * rot_relax[i])
                      - constant1[i] * (first + constant2[i] * second))
        else:
            l22[i] = 1.0
    # Eliminate L00 and L22: schur = L11 - L01^T A^-1 L01 - L12 E^-1 L12^T.
    for i in range(n):
        for k in range(n):
            d_einv[i, k] = l12[i, k] / l22[k]
    if n >= _BLAS_SPECIES:
        # Larger mechanisms: the two N^3 products go to BLAS.
        ainv_b[:, :] = np.dot(l00_inverse, l01)
        schur[:, :] = l11 - np.dot(np.ascontiguousarray(l01.T), ainv_b) - np.dot(d_einv, np.ascontiguousarray(l12.T))
    else:
        for i in range(n):
            for j in range(n):
                value = 0.0
                for k in range(n):
                    value += l00_inverse[i, k] * l01[k, j]
                ainv_b[i, j] = value
        for i in range(n):
            for j in range(n):
                value = l11[i, j]
                for k in range(n):
                    value -= l01[k, i] * ainv_b[k, j] + d_einv[i, k] * l12[j, k]
                schur[i, j] = value
    for i in range(n):
        value = x[i]
        for k in range(n):
            value -= d_einv[i, k] * (x[k] * internal[k])
        rhs[i] = value
    _lu(schur, piv)
    _lu_solve(schur, piv, rhs)
    energy = rhs
    conductivity = 0.0
    for i in range(n):
        first = 0.0
        third = x[i] * internal[i]
        for k in range(n):
            first -= ainv_b[i, k] * energy[k]
            third -= l12[k, i] * energy[k]
        dthermal[i] = 1.6 / R_UNIV * mw[i] * x[i] * first
        conductivity += x[i] * energy[i] + x[i] * internal[i] * third / l22[i]
    return -4.0 * conductivity

# Original source SHA256: 6f8164481ddad47ebaf9779145ca101656ff1182e3fce56bc4766ad51a31f57d

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
