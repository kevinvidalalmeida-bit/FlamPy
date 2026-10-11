"""Compiled Dixon--Lewis assembly and exact block elimination.

Formulation adapted from Cantera 3.2 MultiTransport.cpp (BSD-3-Clause).
See CANTERA_TRANSPORT_LICENSE.txt.
"""
import math
import numpy as np
from kflame.chemistry.mechanism import R_UNIV

try:
    from numba import njit, prange
except ImportError:
    def njit(**kwargs):
        return lambda function: function
    prange = range

_MIN_C_INTERNAL = 1e-3
# Loop kernels win on small face systems; LAPACK/BLAS above this size.
_BLAS_SPECIES = 24


@njit(cache=True, parallel=True)
def evaluate_faces(temperature, pressure, y, cp_r, mw, inv_w, log_eps_pair,
                   epsilon, crot, zrot, frot298, viscosity_fit, diffusion_fit,
                   astar_fit, bstar_fit, cstar_fit, dense):
    """Independent face systems with one BLAS thread per Numba worker."""
    faces = temperature.size
    n = mw.size
    rho = np.empty(faces)
    conductivity = np.empty(faces)
    wmix = np.empty(faces)
    multi = np.empty((faces, n, n))
    thermal = np.empty((faces, n))
    for face in prange(faces):
        rho[face], conductivity[face], wmix[face] = evaluate_face(
            temperature[face], pressure, y[:, face], np.ascontiguousarray(cp_r[:, face]),
            mw, inv_w, log_eps_pair, epsilon, crot, zrot, frot298, viscosity_fit,
            diffusion_fit, astar_fit, bstar_fit, cstar_fit, dense, multi[face], thermal[face])
    return rho, conductivity, wmix, multi.transpose(1, 2, 0), thermal.T


@njit(cache=True)
def evaluate_face(t, pressure, y, cp_r, mw, inv_w, log_eps_pair, epsilon, crot, zrot,
                  frot298, viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit,
                  dense, multi, thermal):
    """One face: returns rho, conductivity, Wmix; fills multi[k,j] and thermal[k]."""
    n = mw.size
    mu = np.empty(n)
    rot_relax = np.empty(n)
    bdiff = np.empty((n, n))
    a = np.empty((n, n))
    b = np.empty((n, n))
    c = np.empty((n, n))
    temperature_terms(t, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit, diffusion_fit,
                      astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c)
    return composition_terms(t, pressure, y, cp_r, mw, inv_w, crot, mu, rot_relax, bdiff, a, b, c,
                             dense, multi, thermal)


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


@njit(cache=True, fastmath=False)
def _temperature_terms_symmetric(t, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit, diffusion_fit,
                      astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c):
    """Composition-independent face data: viscosities, Zrot, collision ratios, D_kj."""
    # Private entry: callers must own a certified immutable snapshot and
    # independent writable outputs. Invalid T retains the original behavior.
    if not math.isfinite(t) or t <= 0.0:
        return temperature_terms(t, mw, log_eps_pair, epsilon, zrot, frot298,
                                 viscosity_fit, diffusion_fit, astar_fit, bstar_fit,
                                 cstar_fit, mu, rot_relax, bdiff, a, b, c)
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
        for j in range(k, n):
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
            if j != k:
                a[j, k], b[j, k], c[j, k], bdiff[j, k] = aa, bb, cc, bdiff[k, j]
        bdiff[k, k] = 1.2 * R_UNIV * t * mu[k] * a[k, k] / mw[k]


@njit(cache=True, parallel=True)
def _evaluate_faces_symmetric(temperature, pressure, y, cp_r, mw, inv_w, log_eps_pair,
                   epsilon, crot, zrot, frot298, viscosity_fit, diffusion_fit,
                   astar_fit, bstar_fit, cstar_fit, dense):
    """Private certified-snapshot entry; face-local temperature buffers are disjoint."""
    faces = temperature.size
    n = mw.size
    rho = np.empty(faces)
    conductivity = np.empty(faces)
    wmix = np.empty(faces)
    multi = np.empty((faces, n, n))
    thermal = np.empty((faces, n))
    for face in prange(faces):
        rho[face], conductivity[face], wmix[face] = _evaluate_face_symmetric(
            temperature[face], pressure, y[:, face], np.ascontiguousarray(cp_r[:, face]),
            mw, inv_w, log_eps_pair, epsilon, crot, zrot, frot298, viscosity_fit,
            diffusion_fit, astar_fit, bstar_fit, cstar_fit, dense, multi[face], thermal[face])
    return rho, conductivity, wmix, multi.transpose(1, 2, 0), thermal.T


@njit(cache=True)
def _evaluate_face_symmetric(t, pressure, y, cp_r, mw, inv_w, log_eps_pair, epsilon, crot, zrot,
                  frot298, viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit,
                  dense, multi, thermal):
    """One face: returns rho, conductivity, Wmix; fills multi[k,j] and thermal[k]."""
    n = mw.size
    mu = np.empty(n)
    rot_relax = np.empty(n)
    bdiff = np.empty((n, n))
    a = np.empty((n, n))
    b = np.empty((n, n))
    c = np.empty((n, n))
    _temperature_terms_symmetric(t, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit, diffusion_fit,
                      astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c)
    return composition_terms(t, pressure, y, cp_r, mw, inv_w, crot, mu, rot_relax, bdiff, a, b, c,
                             dense, multi, thermal)


@njit(cache=True, parallel=True, fastmath=False)
def _evaluate_faces_blocked(temperature, pressure, y, cp_r, mw, inv_w, log_eps_pair,
                            epsilon, crot, zrot, frot298, viscosity_fit, diffusion_fit,
                            astar_fit, bstar_fit, cstar_fit, dense, symmetric, blocksize):
    """Block-owned scratch; every face retains its ordered scalar operations.

    The symmetric entry requires the same immutable snapshot certificate as
    _evaluate_faces_symmetric. Dense evaluation retains its original route.
    No mutable workspace is shared between blocks or concurrent invocations.
    """
    if blocksize < 1:
        raise ValueError("blocksize must be a positive integer")
    if dense:
        if symmetric:
            return _evaluate_faces_symmetric(
                temperature, pressure, y, cp_r, mw, inv_w, log_eps_pair,
                epsilon, crot, zrot, frot298, viscosity_fit, diffusion_fit,
                astar_fit, bstar_fit, cstar_fit, dense)
        return evaluate_faces(
            temperature, pressure, y, cp_r, mw, inv_w, log_eps_pair,
            epsilon, crot, zrot, frot298, viscosity_fit, diffusion_fit,
            astar_fit, bstar_fit, cstar_fit, dense)
    faces, n = temperature.size, mw.size
    rho, conductivity, wmix = np.empty(faces), np.empty(faces), np.empty(faces)
    multi, thermal = np.empty((faces, n, n)), np.empty((faces, n))
    blocks = faces // blocksize + (1 if faces % blocksize else 0)
    for block in prange(blocks):
        start = np.int64(block) * blocksize
        stop = start + min(blocksize, faces-start)
        mu, rot_relax, cp_face = np.empty(n), np.empty(n), np.empty(n)
        bdiff, a, b, c = (np.empty((n, n)), np.empty((n, n)),
                          np.empty((n, n)), np.empty((n, n)))
        workspace = composition_workspace(n)
        for face in range(start, stop):
            cp_face[:] = cp_r[:, face]
            if symmetric:
                _temperature_terms_symmetric(
                    temperature[face], mw, log_eps_pair, epsilon, zrot, frot298,
                    viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit,
                    mu, rot_relax, bdiff, a, b, c)
            else:
                temperature_terms(
                    temperature[face], mw, log_eps_pair, epsilon, zrot, frot298,
                    viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit,
                    mu, rot_relax, bdiff, a, b, c)
            rho[face], conductivity[face], wmix[face] = composition_terms_ws(
                temperature[face], pressure, y[:, face], cp_face, mw, inv_w,
                crot, mu, rot_relax, bdiff, a, b, c, multi[face], thermal[face], workspace)
    return rho, conductivity, wmix, multi.transpose(1, 2, 0), thermal.T


@njit(cache=True)
def composition_terms(t, pressure, y, cp_r, mw, inv_w, crot, mu, rot_relax, bdiff, a, b, c,
                      dense, multi, thermal):
    n = mw.size
    if not dense:
        return composition_terms_ws(t, pressure, y, cp_r, mw, inv_w, crot, mu, rot_relax, bdiff,
                                    a, b, c, multi, thermal, composition_workspace(n))
    denom = 0.0
    for k in range(n):
        denom += y[k] * inv_w[k]
    mix_weight = 1.0 / max(denom, 1e-300)
    x = np.empty(n)
    l00 = np.empty((n, n))
    _diffusion_matrix(t, y, mix_weight, inv_w, mw, bdiff, x, l00)
    inverse = np.linalg.solve(l00, np.eye(n))
    dt, lam = thermal_system(t, x, cp_r, bdiff, a, b, c, rot_relax, mw, crot, mu, l00, inverse, dense)
    _multi_matrix(t, pressure, mix_weight, x, mw, inverse, dt, multi, thermal)
    return pressure * mix_weight / (R_UNIV * t), lam, mix_weight


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
                if n >= _BLAS_SPECIES and a.flags.c_contiguous:
                    # Independent trailing-row updates are contiguous and
                    # vectorizable without reassociation or fused arithmetic.
                    row_i = a[i, k+1:]
                    row_k = a[k, k+1:]
                    for q in range(row_i.size):
                        row_i[q] -= factor * row_k[q]
                else:
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
def _inverse(matrix):
    n = matrix.shape[0]
    inverse = np.empty((n, n))
    _inverse_into(matrix, np.empty((n, n)), np.empty(n, dtype=np.int64), np.empty(n), inverse)
    return inverse


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
    if n >= 400 and l11.flags.c_contiguous:
        # Traverse C-order coefficient rows. Each column's reduction still
        # accumulates in ascending i, with the original expression/order.
        diagonal11 = np.zeros(n)
        constant11, squared_w = np.empty(n), np.empty(n)
        for j in range(n):
            squared_w[j] = mw[j] * mw[j]
            constant11[j] = 16.0 * t * x[j] / 25.0
        for i in range(n):
            wi = mw[i]
            for j in range(n):
                wj, wj2 = mw[j], squared_w[j]
                term1 = bdiff[i, j] * (wi + wj) ** 2
                term2 = 4.0 * wj * astar[i, j] * (1.0 + five_over_3pi * (ratio[j] + ratio[i]))
                l11[i, j] = constant11[j] * x[i] * wi / (wj * term1) * (13.75 * wj2 - 3.0 * wj2 * bstar[i, j] - term2 * wj)
                diagonal11[j] += x[i] / term1 * (7.5 * wj2 + wi * wi * (6.25 - 3.0 * bstar[i, j]) + term2 * wi)
        for j in range(n):
            l11[j, j] -= constant11[j] * diagonal11[j]
    else:
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
        # L00 and its LU scratch are dead after inversion; matrices[2]
        # retains the inverse for _multi_matrix. Reuse the dead slots for
        # products while keeping the original transpose copies/GEMM flags.
        product01, product12 = matrices[0], matrices[1]
        if (ainv_b.flags.c_contiguous and product01.flags.c_contiguous
                and product12.flags.c_contiguous):
            # Numba requires statically C-contiguous dot outputs as well.
            # The runtime guard makes these aliases allocation-free.
            output_b = np.ascontiguousarray(ainv_b)
            output01 = np.ascontiguousarray(product01)
            output12 = np.ascontiguousarray(product12)
            np.dot(l00_inverse, l01, output_b)
            np.dot(np.ascontiguousarray(l01.T), ainv_b, output01)
            np.dot(d_einv, np.ascontiguousarray(l12.T), output12)
            for i in range(n):
                for j in range(n):
                    schur[i, j] = (l11[i, j] - product01[i, j]) - product12[i, j]
        else:
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


@njit(cache=True)
def thermal_system(temperature, x, cp_r, bdiff, astar, bstar, cstar,
                   rot_relax, mw, crot, viscosity, l00, l00_inverse, dense):
    n = x.size
    l = np.zeros((3 * n, 3 * n), dtype=np.float64)
    l[:n, :n] = l00

    cinternal = np.asarray(cp_r, dtype=np.float64) - 2.5
    has_internal = cinternal > _MIN_C_INTERNAL
    mw_i = mw[:, None]
    mw_j = mw[None, :]
    x_i = x[:, None]
    x_j = x[None, :]

    # L00,10 and its transpose L10,00
    l01 = (
        -1.6
        * float(temperature)
        * x_i
        * x_j
        * mw_i
        * (1.2 * cstar.T - 1.0)
        / np.maximum((mw_j + mw_i) * bdiff.T, 1.0e-300)
    )
    sums01 = np.sum(l01, axis=0)
    for k in range(n):
        l01[k, k] -= sums01[k]
    l[:n, n : 2 * n] = l01
    l[n : 2 * n, :n] = l01.T

    # L10,10
    five_over_3pi = 5.0 / (3.0 * math.pi)
    # All N^2 coefficients are formed in array operations. The 3N solve
    # remains dense LAPACK work, but removing Python-level i/j loops is
    # material for the small blocks that occur at every flame face.
    wi = mw[:, None]
    wj = mw[None, :]
    xi = x[:, None]
    term1 = bdiff * (wi + wj) ** 2
    term2 = 4.0 * wj * astar * (
        1.0
        + five_over_3pi
        * (
            (crot / rot_relax)[None, :]
            + (crot / rot_relax)[:, None]
        )
    )
    constant1 = 16.0 * float(temperature) * x / 25.0
    wj2 = wj * wj
    l11 = (
        constant1[None, :]
        * xi
        * wi
        / (wj * term1)
        * (13.75 * wj2 - 3.0 * wj2 * bstar - term2 * wj)
    )
    diag11 = constant1 * np.sum(
        xi
        / term1
        * (
            7.5 * wj2
            + wi * wi * (6.25 - 3.0 * bstar)
            + term2 * wi
        ),
        axis=0,
    )
    for k in range(n):
        l11[k, k] -= diag11[k]
    l[n : 2 * n, n : 2 * n] = l11

    # L10,01, its transpose L01,10, and the internal-mode block.
    l12 = np.zeros((n, n), dtype=np.float64)
    internal_constant = np.zeros(n, dtype=np.float64)
    internal_constant[has_internal] = (
        32.0
        * float(temperature)
        * mw[has_internal]
        * x[has_internal]
        * crot[has_internal]
        / (
            5.0
            * math.pi
            * cinternal[has_internal]
            * rot_relax[has_internal]
        )
    )
    l12[:, :] = (
        xi
        * astar.T
        * internal_constant[None, :]
        / np.maximum((wi + wj) * bdiff.T, 1.0e-300)
    )
    sums12 = np.sum(l12, axis=0)
    for k in range(n):
        l12[k, k] += sums12[k]
    l[n : 2 * n, 2 * n :] = l12
    l[2 * n :, n : 2 * n] = l12.T

    l22 = np.zeros((n, n), dtype=np.float64)
    inv_bdiff = 1.0 / np.maximum(bdiff, 1.0e-300)

    constant1 = np.zeros(n, dtype=np.float64)
    constant2 = np.zeros(n, dtype=np.float64)
    constant1[has_internal] = (
        4.0 * float(temperature) * x[has_internal] / cinternal[has_internal]
    )
    constant2[has_internal] = (
        12.0
        * mw[has_internal]
        * crot[has_internal]
        / (
            5.0
            * math.pi
            * cinternal[has_internal]
            * rot_relax[has_internal]
        )
    )
    off_diagonal = np.ones((n, n), dtype=np.float64) - np.eye(n)
    total = inv_bdiff @ x + constant2 * np.sum(
        off_diagonal * astar * inv_bdiff * (x / mw)[None, :], axis=1
    )
    diagonal = np.ones(n, dtype=np.float64)
    diagonal[has_internal] = (
        -8.0
        / math.pi
        * mw[has_internal]
        * x[has_internal] ** 2
        * crot[has_internal]
        / (
            cinternal[has_internal] ** 2
            * R_UNIV
            * viscosity[has_internal]
            * rot_relax[has_internal]
        )
        - constant1[has_internal] * total[has_internal]
    )
    for k in range(n):
        l22[k, k] = diagonal[k]
    l[2 * n :, 2 * n :] = l22

    rhs = np.concatenate((np.zeros(n), x, np.where(has_internal, x, 0.0)))
    if dense:
        solution = np.linalg.solve(l, rhs)
    else:
        # Exact block elimination: E=L22 is diagonal and A=L00 has
        # already been inverted for the ordinary diffusion coefficients.
        # The remaining system is N by N, rather than 3N by 3N.
        if l00_inverse is None:
            l00_inverse = np.linalg.solve(l[:n, :n], np.eye(n))
        # ``l00_inverse`` can be an arbitrary-strided view at this boundary.
        # A contiguous copy avoids a slower generic BLAS path without changing
        # the values or the block-elimination arithmetic.
        ainv_b = np.ascontiguousarray(l00_inverse) @ l01
        d_einv = l12 / diagonal[None, :]
        schur = l11 - l01.T @ ainv_b - d_einv @ l12.T
        energy_rhs = x - d_einv @ rhs[2 * n:]
        energy = np.linalg.solve(schur, energy_rhs)
        solution = np.concatenate((
            -ainv_b @ energy,
            energy,
            (rhs[2 * n:] - l12.T @ energy) / diagonal,
        ))
    dthermal = 1.6 / R_UNIV * mw * x * solution[:n]
    # ``MultiTransport::thermalConductivity`` sums the two energetic
    # blocks (indices n through 3n), not only the translational/rotational
    # one.  Species without an internal mode have a zero third RHS entry.
    conductivity = -4.0 * float(
        np.dot(x, solution[n : 2 * n])
        + np.dot(np.where(has_internal, x, 0.0), solution[2 * n :])
    )
    return dthermal, conductivity
