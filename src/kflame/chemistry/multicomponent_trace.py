"""Multicomponent transport of large mechanisms: Dixon--Lewis without trace--trace collisions.

Every coupling of the Dixon--Lewis systems between two species i != j comes
from i-j collisions and is proportional to x_i x_j. The species are split per
face into *majors* M (the most abundant ones) and *traces* T, the smallest
species whose mole fractions add up to at most ``tail``. The model keeps every
collision term that involves a major species (and the self terms of every
species) and omits collisions between two distinct trace species, including
their share of the diagonal sums. The omitted terms are O(x_traza) relative;
on flames of GRI-Mech 3.0, USC Mech II and AramcoMech 3.0 with tail 1e-8 the
species diffusion fluxes change by at most 2e-6 relative to the complete model
(mass weighted 2e-9), and D_T and lambda by 1e-14.

Two exact algebraic facts make the model cheap. The ordinary diffusion
matrix D (with D_kk = 0) and the thermal solution do not depend on the
rank-one regularization u v^T of the singular Stefan--Maxwell block L00
(any u with sum(u) != 0), so u sits on the most abundant species only; and
without trace--trace collisions every trace block of the 3N system is
diagonal (a 3 x 3 block per trace species). Cantera's choice (u on every
species) leaves the rows of all small species parallel up to O(x), with a
condition number of order 1/x_min; here every other row of L00 is a
Stefan--Maxwell row proportional to its own x, and rows are equilibrated
before each LU. The
m x m Schur complement of L00 is then formed with BLAS, and the thermal
system is solved by iterative refinement on the major block (its trace
corrections are O(tail) per entry, so two or three sweeps reach round-off).
Only the collision pairs of the model are evaluated: O(m n) instead of
O(n^2) polynomial fits. D, D_T and lambda are those of the pruned model to
round-off, and D keeps the exact mass conservation of the complete model.
"""
import math

import numpy as np
from numba import njit, prange, types
from numba.np.linalg import _LAPACK

from kflame.chemistry.mechanism import R_UNIV
from kflame.chemistry.multicomponent_kernels import (
    _temperature_terms_symmetric, composition_terms_ws, composition_workspace)

# Default tail: mole fraction of all trace species together at a face.
TRACE_TAIL = 1e-8
# 'auto' applies the reduction from this mechanism size on (JetSurF 2.0 and
# AramcoMech); smaller mechanisms keep the complete Dixon--Lewis kernel.
TRACE_MIN_SPECIES = 200
_MIN_C_INTERNAL = 1e-3
_REFINEMENT_SWEEPS = 12

_getrf = _LAPACK().numba_xxgetrf(types.float64)
_KIND = ord('d')


def resolve_trace_tail(setting, n_species):
    """None (complete model) or the tail of the reduced model for this mechanism."""
    if setting is None or setting is False:
        return None
    if isinstance(setting, str):
        if setting != 'auto':
            raise ValueError("multicomponent_trace_tail must be 'auto', True, False/None or a tail in (0, 1e-2)")
        return TRACE_TAIL if n_species >= TRACE_MIN_SPECIES else None
    if setting is True:
        return TRACE_TAIL
    tail = float(setting)
    if not math.isfinite(tail) or not 0. < tail < 1e-2:
        raise ValueError("multicomponent_trace_tail must be 'auto', True, False/None or a tail in (0, 1e-2)")
    return tail


@njit(cache=True)
def _lu_factor(K, buffer, ipiv):
    """LAPACK LU of K (row pivoting, as numpy): factors are kept in ``buffer``."""
    N = K.shape[0]
    for i in range(N):
        for j in range(N):
            buffer[j, i] = K[i, j]
    return _getrf(_KIND, N, N, buffer.ctypes, N, ipiv.ctypes) == 0


@njit(cache=True)
def _lu_solve(buffer, ipiv, r):
    """Solve K d = r in place with the factors of :func:`_lu_factor`.

    LAPACK saw the Fortran view of ``buffer``, i.e. K itself: K = P L U with
    F(i, j) = buffer[j, i].
    """
    N = r.size
    for i in range(N):
        p = ipiv[i]-1
        if p != i:
            value = r[i]
            r[i] = r[p]
            r[p] = value
    for i in range(N):
        value = r[i]
        for j in range(i):
            value -= buffer[j, i]*r[j]
        r[i] = value
    for i in range(N-1, -1, -1):
        value = r[i]
        for j in range(i+1, N):
            value -= buffer[j, i]*r[j]
        r[i] = value/buffer[i, i]


@njit(cache=True)
def _equilibrate_rows(K):
    """Scale each row of K to unit max-norm in place; returns the scales."""
    N = K.shape[0]
    scales = np.empty(N)
    for i in range(N):
        big = 0.
        for j in range(K.shape[1]):
            big = max(big, abs(K[i, j]))
        scales[i] = 1./big if big > 0. else 1.
        for j in range(K.shape[1]):
            K[i, j] *= scales[i]
    return scales


@njit(cache=True)
def _trace_solve(Binv, rhs, out):
    """out = blockdiag(B_r)^-1 rhs for the 3 x 3 trace blocks (groups x traces)."""
    for r in range(Binv.shape[0]):
        for g in range(3):
            out[g, r] = Binv[r, g, 0]*rhs[0, r]+Binv[r, g, 1]*rhs[1, r]+Binv[r, g, 2]*rhs[2, r]


@njit(cache=True)
def split_majors(x, tail, major):
    """Mark the majors: all species but the smallest ones with sum(x) <= tail."""
    n = x.size
    order = np.argsort(-x, kind='mergesort')
    m = 0
    acc = 0.
    for q in range(n-1, -1, -1):
        acc += x[order[q]]
        if acc > tail:
            m = q+1
            break
    m = max(m, min(n, 2))
    major[:] = False
    for q in range(m):
        major[order[q]] = True
    return m


@njit(cache=True)
def _mole_fractions(y, inv_w):
    n = y.size
    denom = 0.
    for k in range(n):
        denom += y[k]*inv_w[k]
    wmix = 1./max(denom, 1e-300)
    x = np.empty(n)
    for k in range(n):
        x[k] = max(y[k]*wmix*inv_w[k], 1e-20)
    return wmix, x


@njit(cache=True)
def _partition(x, tail):
    """Majors M and traces T (ascending species order) and each species' position."""
    n = x.size
    major = np.empty(n, np.bool_)
    m = split_majors(x, tail, major)
    M = np.empty(m, np.int64)
    T = np.empty(n-m, np.int64)
    position = np.empty(n, np.int64)
    pm, pt = 0, 0
    for k in range(n):
        if major[k]:
            M[pm] = k
            position[k] = pm
            pm += 1
        else:
            T[pt] = k
            position[k] = pt
            pt += 1
    return major, M, T, position


@njit(cache=True)
def _species_temperature_terms(t, mw, epsilon, zrot, frot298, viscosity_fit, mu, rot_relax):
    """Viscosities and rotational relaxation numbers (as ``temperature_terms``)."""
    logt = math.log(t)
    sqrt_t = math.sqrt(t)
    for k in range(mw.size):
        value = viscosity_fit[k, 4]
        for degree in range(3, -1, -1):
            value = value*logt+viscosity_fit[k, degree]
        mu[k] = sqrt_t*value*value
        tr = epsilon[k]/t
        sqrt_tr = math.sqrt(tr)
        frot = (1.0+0.5*math.pi**1.5*sqrt_tr+(0.25*math.pi**2+2.0)*tr+math.pi**1.5*sqrt_tr*tr)
        rot_relax[k] = max(1.0, zrot[k])*frot298[k]/frot


@njit(cache=True)
def _pair(t, logt, i, j, log_eps_pair, diffusion_fit, astar_fit, bstar_fit, cstar_fit):
    """A*, B*, C* and the binary diffusion coefficient of pair (i, j), as ``temperature_terms``."""
    z = logt-log_eps_pair[i, j]
    aa, bb, cc = astar_fit[i, j, 8], bstar_fit[i, j, 8], cstar_fit[i, j, 8]
    for degree in range(7, -1, -1):
        aa = aa*z+astar_fit[i, j, degree]
        bb = bb*z+bstar_fit[i, j, degree]
        cc = cc*z+cstar_fit[i, j, degree]
    value = diffusion_fit[i, j, 4]
    for degree in range(3, -1, -1):
        value = value*logt+diffusion_fit[i, j, degree]
    return aa, bb, cc, t*math.sqrt(t)*value


@njit(cache=True)
def _compact_pairs(t, M, T, mw, mu, log_eps_pair, diffusion_fit, astar_fit, bstar_fit, cstar_fit):
    """Pair data of the kept collisions only: rows of the majors and trace self pairs."""
    m, nt, n = M.size, T.size, mw.size
    logt = math.log(t)
    bM, aM, bsM, cM = np.empty((m, n)), np.empty((m, n)), np.empty((m, n)), np.empty((m, n))
    for p in range(m):
        i = M[p]
        for j in range(n):
            aM[p, j], bsM[p, j], cM[p, j], bM[p, j] = _pair(
                t, logt, i, j, log_eps_pair, diffusion_fit, astar_fit, bstar_fit, cstar_fit)
        bM[p, i] = 1.2*R_UNIV*t*mu[i]*aM[p, i]/mw[i]
    bT, aT, bsT = np.empty(nt), np.empty(nt), np.empty(nt)
    for r in range(nt):
        k = T[r]
        aT[r], bsT[r], _, _ = _pair(t, logt, k, k, log_eps_pair, diffusion_fit, astar_fit, bstar_fit, cstar_fit)
        bT[r] = 1.2*R_UNIV*t*mu[k]*aT[r]/mw[k]
    return bM, aM, bsM, cM, bT, aT, bsT


@njit(cache=True)
def _compact_from_full(M, T, bdiff, a, b, c):
    m, nt, n = M.size, T.size, bdiff.shape[0]
    bM, aM, bsM, cM = np.empty((m, n)), np.empty((m, n)), np.empty((m, n)), np.empty((m, n))
    for p in range(m):
        i = M[p]
        for j in range(n):
            bM[p, j], aM[p, j], bsM[p, j], cM[p, j] = bdiff[i, j], a[i, j], b[i, j], c[i, j]
    bT, aT, bsT = np.empty(nt), np.empty(nt), np.empty(nt)
    for r in range(nt):
        k = T[r]
        bT[r], aT[r], bsT[r] = bdiff[k, k], a[k, k], b[k, k]
    return bM, aM, bsM, cM, bT, aT, bsT


@njit(cache=True)
def composition_terms_trace(t, pressure, y, cp_r, mw, inv_w, crot, mu, rot_relax, bdiff, a, b, c,
                            multi, thermal, tail):
    """Pruned Dixon--Lewis face from complete pair arrays; returns rho, lambda, Wmix, m.

    Same inputs and outputs as ``composition_terms_ws`` (symmetric pair data
    required), plus the trace tail. m is the number of majors, or -1 when the
    pruned systems could not be solved (callers then use the complete kernel).
    """
    wmix, x = _mole_fractions(y, inv_w)
    major, M, T, position = _partition(x, tail)
    bM, aM, bsM, cM, bT, aT, bsT = _compact_from_full(M, T, bdiff, a, b, c)
    lam, ok = _trace_core(t, pressure, x, wmix, cp_r, mw, crot, mu, rot_relax, major, M, T, position,
                          bM, aM, bsM, cM, bT, aT, bsT, multi, thermal)
    return pressure*wmix/(R_UNIV*t), lam, wmix, M.size if ok else -1


@njit(cache=True)
def trace_face(t, pressure, y, cp_r, mw, inv_w, crot, log_eps_pair, epsilon, zrot, frot298,
               viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit, tail, multi, thermal):
    """Pruned face with its own temperature terms: only the kept collision pairs are evaluated."""
    n = mw.size
    wmix, x = _mole_fractions(y, inv_w)
    major, M, T, position = _partition(x, tail)
    mu, rot_relax = np.empty(n), np.empty(n)
    _species_temperature_terms(t, mw, epsilon, zrot, frot298, viscosity_fit, mu, rot_relax)
    bM, aM, bsM, cM, bT, aT, bsT = _compact_pairs(t, M, T, mw, mu, log_eps_pair, diffusion_fit,
                                                  astar_fit, bstar_fit, cstar_fit)
    lam, ok = _trace_core(t, pressure, x, wmix, cp_r, mw, crot, mu, rot_relax, major, M, T, position,
                          bM, aM, bsM, cM, bT, aT, bsT, multi, thermal)
    return pressure*wmix/(R_UNIV*t), lam, wmix, M.size if ok else -1


@njit(cache=True)
def _trace_core(t, pressure, x, wmix, cp_r, mw, crot, mu, rot_relax, major, M, T, position,
                bM, aM, bsM, cM, bT, aT, bsT, multi, thermal):
    """D (multi), D_T (thermal) and lambda of the pruned model; returns (lambda, ok).

    Pair data are compact: row p of ``bM``/``aM``/``bsM``/``cM`` holds the
    pairs of major M[p] with every species (column = species index); trace
    self pairs are in ``bT``/``aT``/``bsT``.
    """
    n = mw.size
    m, nt = M.size, T.size
    cst = 16.*t/25.
    inv_mw = np.empty(n)
    for k in range(n):
        inv_mw[k] = 1./mw[k]
    ib = np.empty((m, n))
    G = np.zeros(n)
    for p in range(m):
        i = M[p]
        acc = 0.
        for j in range(n):
            value = 1./bM[p, j]
            ib[p, j] = value
            if j != i:
                acc += x[j]*value
        G[i] = acc
    for r in range(nt):
        k = T[r]
        acc = 0.
        for p in range(m):
            acc += x[M[p]]*ib[p, k]
        G[k] = acc
    v = np.empty(n)
    iG = np.empty(n)
    u = np.zeros(n)
    k0 = M[0]
    for k in range(n):
        v[k] = x[k]*mw[k]
        iG[k] = 1./G[k]
        if x[k] > x[k0]:
            k0 = k
    u[k0] = G[k0]*inv_mw[k0]
    # --- Ordinary diffusion: L00 = c (u v^T - X G), u on the most abundant species ---
    A = np.empty((m, m))
    for p in range(m):
        i = M[p]
        for q in range(m):
            j = M[q]
            if p == q:
                A[p, q] = cst*(u[i]*v[i]-x[i]*G[i])
            else:
                A[p, q] = cst*(u[i]*v[j]+x[i]*x[j]*ib[p, j])
    Bh = np.empty((m, nt))      # L00[M, T] / E  (O(1))
    B00 = np.empty((m, nt))     # L00[M, T]
    for p in range(m):
        i = M[p]
        for r in range(nt):
            k = T[r]
            value = u[i]*mw[k]+x[i]*ib[p, k]
            Bh[p, r] = -value*iG[k]
            B00[p, r] = cst*x[k]*value
    Cp = np.empty((nt, m))      # L00[T, M]
    Ch = np.empty((nt, m))      # E^-1 L00[T, M]  (O(1))
    for r in range(nt):
        k = T[r]
        for q in range(m):
            j = M[q]
            Cp[r, q] = cst*x[k]*x[j]*ib[q, k]
            Ch[r, q] = -x[j]*ib[q, k]*iG[k]
    S = A.copy()
    if nt > 0:
        S -= np.dot(Bh, Cp)
    # inv(S) = inv(diag(r) S) diag(r), with equilibrated rows.
    row_scale = _equilibrate_rows(S)
    W = np.linalg.inv(S)
    for q in range(m):
        for p in range(m):
            W[q, p] *= row_scale[p]
    pref = 16.*t*wmix/(25.*pressure)
    scale = np.empty(n)
    for j in range(n):
        scale[j] = pref*inv_mw[j]
    for p in range(m):
        i = M[p]
        wii = W[p, p]
        row = multi[i]
        for q in range(m):
            row[M[q]] = x[i]*scale[M[q]]*(W[p, q]-wii)
    if nt > 0:
        invMT = -np.dot(W, Bh)
        invTM = -np.dot(Ch, W)
        R = np.dot(invTM, Bh)   # inv(L00)[T, T] = diag(1/E) - R
        for p in range(m):
            i = M[p]
            wii = W[p, p]
            row = multi[i]
            for r in range(nt):
                j = T[r]
                row[j] = x[i]*scale[j]*(invMT[p, r]-wii)
        for r in range(nt):
            k = T[r]
            row = multi[k]
            xk = x[k]
            # x_k inv_kk, with x_k/E_k = -1/(c G_k) formed without 1/x_k.
            xkk = -iG[k]/cst-xk*R[r, r]
            for q in range(m):
                j = M[q]
                row[j] = scale[j]*(xk*invTM[r, q]-xkk)
            Rr = R[r]
            for s2 in range(nt):
                j = T[s2]
                row[j] = scale[j]*(-xk*Rr[s2]-xkk)
            row[k] = 0.
    # --- Thermal 3N system ---
    cint = np.empty(n)
    internal = np.empty(n)
    ratio = np.empty(n)
    ic = np.zeros(n)
    c1 = np.zeros(n)
    c2 = np.zeros(n)
    for k in range(n):
        cint[k] = cp_r[k]-2.5
        internal[k] = 1. if cint[k] > _MIN_C_INTERNAL else 0.
        ratio[k] = crot[k]/rot_relax[k]
        if internal[k] > 0.:
            ic[k] = 32.*t*mw[k]*x[k]*crot[k]/(5.*math.pi*cint[k]*rot_relax[k])
            c1[k] = 4.*t*x[k]/cint[k]
            c2[k] = 12.*mw[k]*crot[k]/(5.*math.pi*cint[k]*rot_relax[k])
    five_over_3pi = 5./(3.*math.pi)
    d01 = np.zeros(n)
    sum11 = np.zeros(n)
    d12 = np.zeros(n)
    first = np.zeros(n)
    second = np.zeros(n)
    l11self = np.empty(n)
    # Self collisions of every species.
    for k in range(n):
        if major[k]:
            p = position[k]
            bkk, akk, bskk = bM[p, k], aM[p, k], bsM[p, k]
        else:
            r = position[k]
            bkk, akk, bskk = bT[r], aT[r], bsT[r]
        wk = mw[k]
        term1 = bkk*(2.*wk)**2
        term2 = 4.*wk*akk*(1.+five_over_3pi*(2.*ratio[k]))
        l11self[k] = cst*x[k]*x[k]/term1*(13.75*wk*wk-3.*wk*wk*bskk-term2*wk)
        sum11[k] += x[k]/term1*(7.5*wk*wk+wk*wk*(6.25-3.*bskk)+term2*wk)
        d12[k] += 2.*x[k]*akk*ic[k]/((2.*wk)*bkk)
        first[k] += x[k]/bkk
    L01mm = np.empty((m, m))
    L11mm = np.empty((m, m))
    L12mm = np.empty((m, m))
    L01mt = np.empty((m, nt))
    L11mt = np.empty((m, nt))
    L12mt = np.empty((m, nt))
    L01tm = np.empty((nt, m))
    L11tm = np.empty((nt, m))
    L12tm = np.empty((nt, m))
    # Ordered kept pairs (row i, column j), i != j: i major and j any, then
    # i trace and j major. Each is visited once; pair data are symmetric.
    for half in range(2):
        rows = M if half == 0 else T
        for i in rows:
            wi = mw[i]
            xi = x[i]
            ri = ratio[i]
            pi = position[i]
            for q in range(n if half == 0 else m):
                j = q if half == 0 else M[q]
                if j == i:
                    continue
                if half == 0:
                    pp, cc = pi, j
                else:
                    pp, cc = q, i
                ibij = ib[pp, cc]
                aij = aM[pp, cc]
                bsij = bsM[pp, cc]
                wj = mw[j]
                xj = x[j]
                iw = 1./(wi+wj)
                l01 = -1.6*t*xi*xj*wi*(1.2*cM[pp, cc]-1.)*iw*ibij
                iterm1 = ibij*iw*iw
                term2 = 4.*wj*aij*(1.+five_over_3pi*(ratio[j]+ri))
                l11 = cst*xj*xi*wi*inv_mw[j]*iterm1*(13.75*wj*wj-3.*wj*wj*bsij-term2*wj)
                l12 = xi*aij*ic[j]*iw*ibij
                d01[j] -= l01
                sum11[j] += xi*iterm1*(7.5*wj*wj+wi*wi*(6.25-3.*bsij)+term2*wi)
                d12[j] += l12
                first[i] += xj*ibij
                second[i] += aij*xj*ibij*inv_mw[j]
                pj = position[j]
                if half == 0:
                    if major[j]:
                        L01mm[pi, pj] = l01
                        L11mm[pi, pj] = l11
                        L12mm[pi, pj] = l12
                    else:
                        L01mt[pi, pj] = l01
                        L11mt[pi, pj] = l11
                        L12mt[pi, pj] = l12
                else:
                    L01tm[pi, pj] = l01
                    L11tm[pi, pj] = l11
                    L12tm[pi, pj] = l12
    d11 = np.empty(n)
    l22 = np.empty(n)
    for k in range(n):
        d11[k] = l11self[k]-cst*x[k]*sum11[k]
        if internal[k] > 0.:
            l22[k] = (-8./math.pi*mw[k]*x[k]**2*crot[k]/(cint[k]**2*R_UNIV*mu[k]*rot_relax[k])
                      - c1[k]*(first[k]+c2[k]*second[k]))
        else:
            l22[k] = 1.
    for p in range(m):
        i = M[p]
        L01mm[p, p] = d01[i]
        L11mm[p, p] = d11[i]
        L12mm[p, p] = d12[i]
    return _thermal_solve(x, internal, mw, M, T, G, cst, A, B00, Cp, L01mm, L11mm, L12mm,
                          L01mt, L11mt, L12mt, L01tm, L11tm, L12tm, d01, d11, d12, l22, thermal)


@njit(cache=True)
def _thermal_solve(x, internal, mw, M, T, G, cst, A, B00, Cp, L01mm, L11mm, L12mm,
                   L01mt, L11mt, L12mt, L01tm, L11tm, L12tm, d01, d11, d12, l22, thermal):
    """Pruned 3N system: iterative refinement on the major block; returns (lambda, ok)."""
    m, nt = M.size, T.size
    # Major block with the diagonal internal-mode group eliminated:
    # K = [[A, L01], [L01^T, L11 - L12 diag(1/l22) L12^T]].
    N2 = 2*m
    l22M = np.empty(m)
    for p in range(m):
        l22M[p] = l22[M[p]]
    scaled = np.empty((m, m))
    for p in range(m):
        for q in range(m):
            scaled[p, q] = L12mm[p, q]/l22M[q]
    reduced = L11mm-np.dot(scaled, np.ascontiguousarray(L12mm.T))
    K = np.empty((N2, N2))
    for p in range(m):
        for q in range(m):
            K[p, q] = A[p, q]
            K[p, m+q] = L01mm[p, q]
            K[m+p, q] = L01mm[q, p]
            K[m+p, m+q] = reduced[p, q]
    factors = np.empty((N2, N2))
    ipiv = np.empty(N2, np.int32)
    k_scale = _equilibrate_rows(K)
    if not _lu_factor(K, factors, ipiv):
        return 0., False
    # Trace 3x3 blocks [[E, d01, 0], [d01, d11, d12], [0, d12, l22]] inverted once.
    Binv = np.empty((nt, 3, 3))
    for r in range(nt):
        k = T[r]
        e00 = -cst*x[k]*G[k]
        b01, b11, b12, b22 = d01[k], d11[k], d12[k], l22[k]
        c00 = b11*b22-b12*b12
        c01 = -b01*b22
        c02 = b01*b12
        c11 = e00*b22
        c12 = -e00*b12
        c22 = e00*b11-b01*b01
        det = e00*c00+b01*c01
        if not (det != 0. and math.isfinite(det)):
            return 0., False
        Binv[r, 0, 0], Binv[r, 0, 1], Binv[r, 0, 2] = c00/det, c01/det, c02/det
        Binv[r, 1, 0], Binv[r, 1, 1], Binv[r, 1, 2] = c01/det, c11/det, c12/det
        Binv[r, 2, 0], Binv[r, 2, 1], Binv[r, 2, 2] = c02/det, c12/det, c22/det
    # Right-hand sides: (0, x, x*internal).
    rT = np.zeros((3, nt))
    for r in range(nt):
        k = T[r]
        rT[1, r] = x[k]
        rT[2, r] = x[k]*internal[k]
    target = np.zeros((3, m))
    for p in range(m):
        i = M[p]
        target[1, p] = x[i]
        target[2, p] = x[i]*internal[i]
    wT = np.empty((3, nt))
    yT = np.empty((3, nt))
    if nt > 0:
        _trace_solve(Binv, rT, wT)
        target[0] -= np.dot(B00, wT[0])+np.dot(L01mt, wT[1])
        target[1] -= np.dot(L01tm.T, wT[0])+np.dot(L11mt, wT[1])+np.dot(L12mt, wT[2])
        target[2] -= np.dot(L12tm.T, wT[1])
    uM = np.zeros((3, m))
    res = np.empty((3, m))
    rhs2 = np.empty(N2)
    converged = False
    previous = np.inf
    for sweep in range(_REFINEMENT_SWEEPS):
        # res = target - S3 uM, with S3 = L_MM - L_MT Bd^-1 L_TM.
        res[0] = target[0]-(np.dot(A, uM[0])+np.dot(L01mm, uM[1]))
        res[1] = target[1]-(np.dot(L01mm.T, uM[0])+np.dot(L11mm, uM[1])+np.dot(L12mm, uM[2]))
        res[2] = target[2]-np.dot(L12mm.T, uM[1])
        for p in range(m):
            res[2, p] -= l22M[p]*uM[2, p]
        if nt > 0:
            yT[0] = np.dot(Cp, uM[0])+np.dot(L01tm, uM[1])
            yT[1] = np.dot(L01mt.T, uM[0])+np.dot(L11tm, uM[1])+np.dot(L12tm, uM[2])
            yT[2] = np.dot(L12mt.T, uM[1])
            _trace_solve(Binv, yT, wT)
            res[0] += np.dot(B00, wT[0])+np.dot(L01mt, wT[1])
            res[1] += np.dot(L01tm.T, wT[0])+np.dot(L11mt, wT[1])+np.dot(L12mt, wT[2])
            res[2] += np.dot(L12tm.T, wT[1])
        # Solve L_MM delta = res through the reduced 2m system.
        for p in range(m):
            rhs2[p] = res[0, p]
            rhs2[m+p] = res[1, p]
        for p in range(m):
            value = res[2, p]/l22M[p]
            for q in range(m):
                rhs2[m+q] -= L12mm[q, p]*value
        for i in range(N2):
            rhs2[i] *= k_scale[i]
        _lu_solve(factors, ipiv, rhs2)
        change, size = 0., 0.
        for p in range(m):
            uM[0, p] += rhs2[p]
            uM[1, p] += rhs2[m+p]
        for p in range(m):
            value = res[2, p]
            for q in range(m):
                value -= L12mm[q, p]*rhs2[m+q]
            d2 = value/l22M[p]
            uM[2, p] += d2
            change = max(change, abs(rhs2[p]), abs(rhs2[m+p]), abs(d2))
            size = max(size, abs(uM[0, p]), abs(uM[1, p]), abs(uM[2, p]))
        if not math.isfinite(change):
            return 0., False
        # Converged, or stagnating at the round-off level of the major block.
        if change <= 1e-13*size or (sweep >= 2 and change > .5*previous and change <= 1e-6*size):
            converged = True
            break
        previous = change
    if not converged:
        return 0., False
    conductivity = 0.
    for p in range(m):
        i = M[p]
        thermal[i] = 1.6/R_UNIV*mw[i]*x[i]*uM[0, p]
        conductivity += x[i]*uM[1, p]+x[i]*internal[i]*uM[2, p]
    if nt > 0:
        yT[0] = np.dot(Cp, uM[0])+np.dot(L01tm, uM[1])
        yT[1] = np.dot(L01mt.T, uM[0])+np.dot(L11tm, uM[1])+np.dot(L12tm, uM[2])
        yT[2] = np.dot(L12mt.T, uM[1])
        for g in range(3):
            for r in range(nt):
                yT[g, r] = rT[g, r]-yT[g, r]
        _trace_solve(Binv, yT, wT)
        for r in range(nt):
            k = T[r]
            thermal[k] = 1.6/R_UNIV*mw[k]*x[k]*wT[0, r]
            conductivity += x[k]*wT[1, r]+x[k]*internal[k]*wT[2, r]
    return -4.*conductivity, True


@njit(cache=True)
def _complete_face(t, pressure, y, cp_r, mw, inv_w, log_eps_pair, epsilon, crot, zrot, frot298,
                   viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit, multi, thermal):
    n = mw.size
    mu, rot_relax = np.empty(n), np.empty(n)
    bdiff, a, b, c = np.empty((n, n)), np.empty((n, n)), np.empty((n, n)), np.empty((n, n))
    _temperature_terms_symmetric(t, mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit,
                                 diffusion_fit, astar_fit, bstar_fit, cstar_fit, mu, rot_relax, bdiff, a, b, c)
    return composition_terms_ws(t, pressure, y, cp_r, mw, inv_w, crot, mu, rot_relax, bdiff, a, b, c,
                                multi, thermal, composition_workspace(n))


@njit(cache=True, parallel=True)
def evaluate_faces_trace(temperature, pressure, y, cp_r, mw, inv_w, log_eps_pair, epsilon, crot, zrot,
                         frot298, viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit, tail):
    """Independent pruned face systems (one BLAS thread per Numba worker).

    Returns rho, lambda, Wmix, multi (n, n, faces), thermal (n, faces) and
    the majors per face; a face whose pruned systems fail is evaluated with
    the complete kernel (majors = -1).
    """
    faces = temperature.size
    n = mw.size
    rho, conductivity, wmix = np.empty(faces), np.empty(faces), np.empty(faces)
    multi, thermal = np.empty((faces, n, n)), np.empty((faces, n))
    majors = np.empty(faces, np.int64)
    for face in prange(faces):
        yf = np.ascontiguousarray(y[:, face])
        cpf = np.ascontiguousarray(cp_r[:, face])
        rho[face], conductivity[face], wmix[face], majors[face] = trace_face(
            temperature[face], pressure, yf, cpf, mw, inv_w, crot, log_eps_pair, epsilon, zrot,
            frot298, viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit, tail,
            multi[face], thermal[face])
        if majors[face] < 0:
            rho[face], conductivity[face], wmix[face] = _complete_face(
                temperature[face], pressure, yf, cpf, mw, inv_w, log_eps_pair, epsilon, crot, zrot,
                frot298, viscosity_fit, diffusion_fit, astar_fit, bstar_fit, cstar_fit,
                multi[face], thermal[face])
    return rho, conductivity, wmix, multi.transpose(1, 2, 0), thermal.T, majors
