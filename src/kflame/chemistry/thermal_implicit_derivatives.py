"""Private temperature tangent of a completed multicomponent Schur solve.

Call immediately after ``composition_terms_ws`` for the same face, before its
workspace is reused. All composition inputs (including the floored mole
fractions), pressure, mixture molecular weight, and the internal-mode active
set are fixed. ``dA, dB, dC, dD, dE`` are derivatives per kelvin of the exact
production L00, L01, L11, L12, and diagonal L22 blocks, respectively.

The workspace and block tangents are read-only inputs. The return values are
new C-contiguous float64 arrays for dMulti and dThermal, and scalar dLambda.
This prototype does not add a validity/fallback policy for branch crossings,
change production, run validation on import, or claim a measured speedup.
"""

import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV
from kflame.chemistry.multicomponent_kernels import _lu_solve


@njit(cache=True, nogil=True, fastmath=False)
def implicit_temperature_tangent(t, P, wm, mw, workspace, dA, dB, dC, dD, dE):
    """Differentiate the retained face system without refactorizing it.

    Required retained slots after ``composition_terms_ws`` are:
      matrices[2]: Ainv; [3]: B; [5]: D; [6]: Q=Ainv@B;
      matrices[7]: H=D/E; [8]: factored Schur matrix;
      pivots[1]: Schur pivots;
      vectors[0]: x; [4]: internal mask; [9]: E; [10]: solved v.
    C (matrices[4]) is not needed: the supplied dC enters the differentiated
    right-hand side. The L00 LU scratch is not retained on large mechanisms;
    all L00 solves here therefore multiply by the retained Ainv.

    Validity is the caller's responsibility: t and P must be positive, the
    workspace must describe this same state, E must be nonzero, and all block
    derivatives must use the same stable internal/NASA branches as the primal.
    The full inverse tangent uses two cubic matrix products. It deliberately
    keeps the diffusion diagonal subtraction rather than replacing it with a
    projected action that would assume an exactly zero composition-gradient
    sum. No retained array, pivot, or caller-supplied derivative is modified.
    """
    vectors, matrices, pivots = workspace
    n = mw.size
    x = vectors[0]
    internal = vectors[4]
    E = vectors[9]
    v = vectors[10]
    ainv = matrices[2]
    B = matrices[3]
    D = matrices[5]
    Q = matrices[6]
    H = matrices[7]
    schur_lu = matrices[8]

    # Recover the other two primal solution blocks from the retained v:
    # A u + B v = 0, B.T u + C v + D w = x,
    # D.T v + E w = x*internal.
    u = np.empty(n, dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        first = 0.0
        third = x[i] * internal[i]
        for k in range(n):
            first -= Q[i, k] * v[k]
            third -= D[k, i] * v[k]
        u[i] = first
        w[i] = third / E[i]

    # Coefficient changes move to the right-hand side. x and x*internal
    # have zero tangent at fixed composition and a stable internal mask.
    r0 = np.empty(n, dtype=np.float64)
    r1 = np.empty(n, dtype=np.float64)
    r2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        value0 = 0.0
        value1 = 0.0
        value2 = -dE[i] * w[i]
        for k in range(n):
            value0 -= dA[i, k] * u[k] + dB[i, k] * v[k]
            value1 -= (dB[k, i] * u[k] + dC[i, k] * v[k]
                       + dD[i, k] * w[k])
            value2 -= dD[k, i] * v[k]
        r0[i] = value0
        r1[i] = value1
        r2[i] = value2

    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        value = 0.0
        for k in range(n):
            value += ainv[i, k] * r0[k]
        z[i] = value

    # Reuse S = C - B.T@Ainv@B - D@diag(1/E)@D.T. _lu_solve
    # mutates only its last argument, so solve on a private copy of the RHS.
    rhs = np.empty(n, dtype=np.float64)
    for i in range(n):
        value = r1[i]
        for k in range(n):
            value -= B[k, i] * z[k] + H[i, k] * r2[k]
        rhs[i] = value
    dv = rhs.copy()
    _lu_solve(schur_lu, pivots[1], dv)

    dthermal = np.empty(n, dtype=np.float64)
    conductivity_tangent = 0.0
    for i in range(n):
        du = z[i]
        third = r2[i]
        for k in range(n):
            du -= Q[i, k] * dv[k]
            third -= D[k, i] * dv[k]
        dw = third / E[i]
        dthermal[i] = 1.6 / R_UNIV * mw[i] * x[i] * du
        conductivity_tangent += x[i] * dv[i] + x[i] * internal[i] * dw
    dlambda = -4.0 * conductivity_tangent

    # d(Ainv)/dT = -Ainv @ dA @ Ainv. Keep a C-order product/output for
    # Numba's three-argument dot. ascontiguousarray copies only if needed;
    # neither dot output aliases the retained Ainv or an input derivative.
    inverse_da = np.dot(np.ascontiguousarray(ainv), np.ascontiguousarray(dA))
    dmulti = np.empty((n, n), dtype=np.float64)
    np.dot(inverse_da, np.ascontiguousarray(ainv), dmulti)
    prefactor = 16.0 * t * wm / (25.0 * P)
    dprefactor = prefactor / t
    for i in range(n):
        inverse_diagonal = ainv[i, i]
        # dmulti temporarily holds minus the inverse tangent. Capture its
        # diagonal before converting this row to diffusion coefficients.
        dinverse_diagonal = -dmulti[i, i]
        for j in range(n):
            dinverse = -dmulti[i, j]
            dmulti[i, j] = x[i] / mw[j] * (
                dprefactor * (ainv[i, j] - inverse_diagonal)
                + prefactor * (dinverse - dinverse_diagonal)
            )
    return dmulti, dthermal, dlambda
