"""Two-ended block elimination for ``diag + P Q^T`` coupling blocks.

The block-tridiagonal matrix has dense diagonal blocks D_i and coupling blocks
``A[i+1,i] = diag(dl_i) + Pl_i Ql_i^T`` and ``A[i,i+1] = diag(du_i) + Pu_i Qu_i^T``
with a few columns. The Schur complement of a pivot needs only its inverse
G and the rank-r products G P and Q^T G, so a node costs one dense inversion
plus O(r nv^2) work; dense coupling products and their storage disappear.
As in :mod:`block_linear` the top and bottom sweeps are independent and run
concurrently. Pivot inversion uses LAPACK; a singular or non-finite pivot
raises LinAlgError and callers fall back to a pivoted sparse LU.
"""
from concurrent.futures import wait

import numba
import numpy as np
from numba import njit, types
from numba.np.linalg import _LAPACK, F_INT_nptype

from kflame.flame import block_linear

# Numba's own LAPACK entry points (resolved by symbol name, so cacheable).
_getrf = _LAPACK().numba_xxgetrf(types.float64)
_getri = _LAPACK().numba_ez_xxgetri(types.float64)
_KIND = ord('d')


@njit(cache=True, fastmath={'reassoc', 'contract'})
def _schur_subtract(S, G, GP, dl, Pl, Ql, du, Pu, Qu):
    """S -= (dl + Pl Ql^T) G (du + Pu Qu^T), with GP = G Pu precomputed.

    Expansion: dl G du + dl (G Pu) Qu^T + Pl (Ql^T G) du + Pl (Ql^T G Pu) Qu^T,
    applied in one fused pass over S and G.
    """
    nv = S.shape[0]
    r = GP.shape[1]
    QG = np.dot(Ql.T.copy(), G)
    QGP = np.dot(Ql.T.copy(), GP)
    QuT = Qu.T.copy()
    right = np.dot(QGP, QuT)
    for t in range(r):
        for b in range(nv):
            right[t, b] += QG[t, b]*du[b]
    left = np.empty(r)
    pa = np.empty(r)
    for a in range(nv):
        da = dl[a]
        for t in range(r):
            left[t] = da*GP[a, t]
            pa[t] = Pl[a, t]
        Ga = G[a]
        Sa = S[a]
        for b in range(nv):
            acc = da*Ga[b]*du[b]
            for t in range(r):
                acc += left[t]*QuT[t, b]+pa[t]*right[t, b]
            Sa[b] -= acc


@njit(cache=True, nogil=True)
def _transpose_inplace(A):
    n = A.shape[0]
    for a in range(n):
        for b in range(a+1, n):
            value = A[a, b]
            A[a, b] = A[b, a]
            A[b, a] = value


@njit(cache=True, nogil=True)
def _invert_pivot(Gi, ipiv, GPi, P):
    """Gi <- Gi^-1 in place and GPi = Gi P; 0 if singular or non-finite.

    Otherwise returns the sign (+1 or -1) of det(Gi) read from the LU factors
    (pivot swaps and the diagonal of U); the product over all pivots is the
    sign of the determinant of the block-tridiagonal matrix.

    LAPACK reads the C-ordered buffer as its Fortran transpose, so the block
    is transposed first: the LU then pivots over the rows of A itself, as
    numpy.linalg.inv does. Pivoting the transpose (columns of A) is O(nv^2)
    cheaper but, on chemistry blocks with condition numbers of 1e18-1e21,
    gives far less accurate inverses and more Jacobian rebuilds. The
    result inv(A) in Fortran order is transposed back to C order.
    """
    n = Gi.shape[0]
    _transpose_inplace(Gi)
    if _getrf(_KIND, n, n, Gi.ctypes, n, ipiv.ctypes) != 0:
        return 0
    sign = 1
    for a in range(n):
        if Gi[a, a] < 0.:
            sign = -sign
        if ipiv[a] != a+1:
            sign = -sign
    if _getri(_KIND, n, Gi.ctypes, n, ipiv.ctypes) != 0:
        return 0
    _transpose_inplace(Gi)
    for a in range(n):
        if not np.isfinite(Gi[a, a]):
            return 0
    if P.shape[0] == n:
        GPi[:, :] = np.dot(Gi, P)
        for a in range(n):
            for t in range(GPi.shape[1]):
                if not np.isfinite(GPi[a, t]):
                    return 0
    return sign


@njit(cache=True, nogil=True)
def structured_top(diag, shift, lowD, lowP, lowQ, upD, upP, upQ, k, G, GP, copy, signs):
    """Top sweep; ``copy`` False means G is diag itself (in-place pivots)."""
    nv = diag.shape[1]
    ipiv = np.empty(nv, dtype=F_INT_nptype)
    for i in range(k):
        S = G[i]
        if copy:
            S[:, :] = diag[i]
        for a in range(nv):
            S[a, a] += shift[i, a]
        if i > 0:
            _schur_subtract(S, G[i-1], GP[i-1], lowD[i-1], lowP[i-1], lowQ[i-1],
                            upD[i-1], upP[i-1], upQ[i-1])
        signs[i] = _invert_pivot(S, ipiv, GP[i], upP[i])
        if signs[i] == 0:
            return False
    return True


@njit(cache=True, nogil=True)
def structured_bottom(diag, shift, lowD, lowP, lowQ, upD, upP, upQ, k, G, GP, copy, signs):
    n, nv = diag.shape[0], diag.shape[1]
    ipiv = np.empty(nv, dtype=F_INT_nptype)
    for i in range(n-1, k, -1):
        S = G[i]
        if copy:
            S[:, :] = diag[i]
        for a in range(nv):
            S[a, a] += shift[i, a]
        if i < n-1:
            # U_i G_{i+1} L_i with U_i = A[i,i+1] and L_i = A[i+1,i].
            _schur_subtract(S, G[i+1], GP[i+1], upD[i], upP[i], upQ[i],
                            lowD[i], lowP[i], lowQ[i])
        signs[i] = _invert_pivot(S, ipiv, GP[i], lowP[i-1])
        if signs[i] == 0:
            return False
    return True


@njit(cache=True, nogil=True)
def structured_middle(diag, shift, lowD, lowP, lowQ, upD, upP, upQ, k, G, GP, copy, signs):
    n, nv = diag.shape[0], diag.shape[1]
    ipiv = np.empty(nv, dtype=F_INT_nptype)
    S = G[k]
    if copy:
        S[:, :] = diag[k]
    for a in range(nv):
        S[a, a] += shift[k, a]
    if k > 0:
        _schur_subtract(S, G[k-1], GP[k-1], lowD[k-1], lowP[k-1], lowQ[k-1],
                        upD[k-1], upP[k-1], upQ[k-1])
    if k < n-1:
        _schur_subtract(S, G[k+1], GP[k+1], upD[k], upP[k], upQ[k],
                        lowD[k], lowP[k], lowQ[k])
    signs[k] = _invert_pivot(S, ipiv, GP[k], np.empty((0, 0)))
    return signs[k] != 0


@njit(cache=True)
def _coupling_apply(D, P, Q, v, out):
    """out = (diag(D) + P Q^T) v."""
    nv = v.size
    r = P.shape[1]
    for a in range(nv):
        out[a] = D[a]*v[a]
    for t in range(r):
        s = 0.
        for b in range(nv):
            s += Q[b, t]*v[b]
        if s != 0.:
            for a in range(nv):
                out[a] += P[a, t]*s


@njit(cache=True, fastmath={'reassoc', 'contract'})
def _mv(A, v, out):
    """out = A v accumulated in float64 (A may also be float32)."""
    for a in range(A.shape[0]):
        row = A[a]
        acc = 0.
        for b in range(v.size):
            acc += row[b]*v[b]
        out[a] = acc


@njit(cache=True, nogil=True)
def solve_forward(lowD, lowP, lowQ, upD, upP, upQ, G, k, f, y, part):
    """Forward sweep of the top (part 0, i < k) or bottom (part 1, i > k) half."""
    n, nv = G.shape[0], G.shape[1]
    r = np.empty(nv)
    t = np.empty(nv)
    if part == 0:
        for i in range(k):
            r[:] = f[i]
            if i > 0:
                _coupling_apply(lowD[i-1], lowP[i-1], lowQ[i-1], y[i-1], t)
                r -= t
            _mv(G[i], r, y[i])
    else:
        for i in range(n-1, k, -1):
            r[:] = f[i]
            if i < n-1:
                _coupling_apply(upD[i], upP[i], upQ[i], y[i+1], t)
                r -= t
            _mv(G[i], r, y[i])


@njit(cache=True, nogil=True)
def solve_middle(lowD, lowP, lowQ, upD, upP, upQ, G, k, f, y, x):
    n, nv = G.shape[0], G.shape[1]
    r = f[k].copy()
    t = np.empty(nv)
    if k > 0:
        _coupling_apply(lowD[k-1], lowP[k-1], lowQ[k-1], y[k-1], t)
        r -= t
    if k < n-1:
        _coupling_apply(upD[k], upP[k], upQ[k], y[k+1], t)
        r -= t
    _mv(G[k], r, x[k])


@njit(cache=True, nogil=True)
def solve_back(lowD, lowQ, upD, upQ, G, GP, k, y, x, part):
    """Back substitution of the top (part 0) or bottom (part 1) half."""
    n, nv = G.shape[0], G.shape[1]
    t = np.empty(nv)
    w = np.empty(nv)
    q = np.empty(GP.shape[2])
    if part == 0:
        for i in range(k-1, -1, -1):
            for a in range(nv):
                t[a] = upD[i, a]*x[i+1, a]
            _mv(G[i], t, w)
            x[i] = y[i]-w
            for s in range(q.size):
                acc = 0.
                for b in range(nv):
                    acc += upQ[i, b, s]*x[i+1, b]
                q[s] = acc
            x[i] -= np.dot(GP[i], q)
    else:
        for i in range(k+1, n):
            for a in range(nv):
                t[a] = lowD[i-1, a]*x[i-1, a]
            _mv(G[i], t, w)
            x[i] = y[i]-w
            for s in range(q.size):
                acc = 0.
                for b in range(nv):
                    acc += lowQ[i-1, b, s]*x[i-1, b]
                q[s] = acc
            x[i] -= np.dot(GP[i], q)


@njit(cache=True)
def _store(S, out):
    for a in range(S.shape[0]):
        for b in range(S.shape[1]):
            out[a, b] = S[a, b]


@njit(cache=True, nogil=True)
def chunk_top(chunk, a, b, shift, lowD, lowP, lowQ, upD, upP, upQ, G, GP, signs, prev):
    """Top sweep over nodes [a, b); chunk[i-a] holds the assembled block of node i.

    Pivots are inverted in the chunk and copied to G (``STORAGE`` dtype).
    ``prev`` holds G_{a-1} on entry (float64) and G_{b-1} on exit.
    """
    nv = chunk.shape[1]
    ipiv = np.empty(nv, dtype=F_INT_nptype)
    for i in range(a, b):
        S = chunk[i-a]
        for c in range(nv):
            S[c, c] += shift[i, c]
        if i > 0:
            Gp = prev if i == a else chunk[i-1-a]
            _schur_subtract(S, Gp, GP[i-1], lowD[i-1], lowP[i-1], lowQ[i-1], upD[i-1], upP[i-1], upQ[i-1])
        signs[i] = _invert_pivot(S, ipiv, GP[i], upP[i])
        if signs[i] == 0:
            return False
        _store(S, G[i])
    prev[:, :] = chunk[b-1-a]
    return True


@njit(cache=True, nogil=True)
def chunk_bottom(chunk, a, b, shift, lowD, lowP, lowQ, upD, upP, upQ, G, GP, signs, prev):
    """Bottom sweep over nodes b-1 down to a; ``prev`` holds G_b on entry, G_a on exit."""
    n, nv = G.shape[0], chunk.shape[1]
    ipiv = np.empty(nv, dtype=F_INT_nptype)
    for i in range(b-1, a-1, -1):
        S = chunk[i-a]
        for c in range(nv):
            S[c, c] += shift[i, c]
        if i < n-1:
            Gn = prev if i == b-1 else chunk[i+1-a]
            _schur_subtract(S, Gn, GP[i+1], upD[i], upP[i], upQ[i], lowD[i], lowP[i], lowQ[i])
        signs[i] = _invert_pivot(S, ipiv, GP[i], lowP[i-1])
        if signs[i] == 0:
            return False
        _store(S, G[i])
    prev[:, :] = chunk[0]
    return True


@njit(cache=True, nogil=True)
def chunk_middle(chunk, k, shift, lowD, lowP, lowQ, upD, upP, upQ, G, GP, signs, top, bottom):
    n, nv = G.shape[0], chunk.shape[1]
    ipiv = np.empty(nv, dtype=F_INT_nptype)
    S = chunk[0]
    for c in range(nv):
        S[c, c] += shift[k, c]
    if k > 0:
        _schur_subtract(S, top, GP[k-1], lowD[k-1], lowP[k-1], lowQ[k-1], upD[k-1], upP[k-1], upQ[k-1])
    if k < n-1:
        _schur_subtract(S, bottom, GP[k+1], upD[k], upP[k], upQ[k], lowD[k], lowP[k], lowQ[k])
    signs[k] = _invert_pivot(S, ipiv, GP[k], np.empty((0, 0)))
    if signs[k] == 0:
        return False
    _store(S, G[k])
    return True


# Bytes of one chunk of assembled diagonal blocks (two per sweep in flight).
CHUNK_BYTES = 48*1024**2
# Storage of the pivot inverses. float32 halves the memory and the solve
# traffic, but the rounded inverses perturb the damping norm enough that the
# Aramco 3.0 initial grid needed 55 instead of 39 Jacobians (47 s vs 39 s).
STORAGE = np.float64


def factor_lazy(blocks, shift, pool, signs=None):
    """Fused assembly and two-ended factorization.

    Each sweep factors one chunk of freshly assembled diagonal blocks while a
    helper thread assembles the next chunk, so the diagonal blocks are never
    all held besides the inverses: memory is n*nv*nv*itemsize(STORAGE) plus
    four chunks. Pivots are inverted in float64 and the next Schur
    complement uses that float64 inverse; solves accumulate in float64.
    """
    n, nv = blocks.n, blocks.nv
    k = n//2
    shift = np.ascontiguousarray(shift, dtype=np.float64)
    G = pool.take((n, nv, nv), STORAGE) if pool is not None else np.empty((n, nv, nv), STORAGE)
    GP = np.empty((n, nv, blocks.lowP.shape[2]))
    signs = np.zeros(n, np.int8) if signs is None else signs
    size = max(1, min(CHUNK_BYTES//(8*nv*nv), max(k, 1)))
    common = (shift, blocks.lowD, blocks.lowP, blocks.lowQ, blocks.upD, blocks.upP, blocks.upQ, G, GP, signs)
    workers = block_linear.pool()

    def take(count):
        return pool.take((count, nv, nv)) if pool is not None else np.empty((count, nv, nv))

    def sweep(ranges, kernel):
        buffers = [take(size), take(size)]
        prev = np.empty((nv, nv))
        pending = workers.submit(blocks.assemble, ranges[0][0], ranges[0][1], buffers[0], ranges[0][0])
        for index, (a, b) in enumerate(ranges):
            pending.result()
            if index+1 < len(ranges):
                na, nb = ranges[index+1]
                pending = workers.submit(blocks.assemble, na, nb, buffers[(index+1) % 2], na)
            if not kernel(buffers[index % 2], a, b, *common, prev):
                if index+1 < len(ranges):
                    pending.result()
                raise np.linalg.LinAlgError('singular or non-finite structured pivot')
        return prev

    top = [(a, min(a+size, k)) for a in range(0, k, size)]
    bottom = [(max(k+1, b-size), b) for b in range(n, k+1, -size)]
    last_top = None
    if top:
        future = workers.submit(sweep, top, chunk_top)
    try:
        last_bottom = sweep(bottom, chunk_bottom) if bottom else None
    except BaseException:
        # Never leave the top sweep running on buffers of a failed factor.
        if top:
            wait([future])
        raise
    if top:
        last_top = future.result()
    middle = take(1)
    blocks.assemble(k, k+1, middle, k)
    empty = np.empty((nv, nv))
    if not chunk_middle(middle, k, *common, last_top if last_top is not None else empty,
                        last_bottom if last_bottom is not None else empty):
        raise np.linalg.LinAlgError('singular or non-finite structured pivot')
    return G, GP, k


def parallel_solve(blocks, G, GP, k, rhs):
    """Two-thread solve: both halves sweep concurrently (GIL released)."""
    n, nv = G.shape[0], G.shape[1]
    f = np.ascontiguousarray(rhs, dtype=np.float64).reshape(n, nv)
    y = np.empty((n, nv))
    x = np.empty((n, nv))
    b = blocks
    forward = (b.lowD, b.lowP, b.lowQ, b.upD, b.upP, b.upQ, G, k, f, y)
    if n >= 16:
        top = block_linear.pool().submit(solve_forward, *forward, 0)
        solve_forward(*forward, 1)
        top.result()
    else:
        solve_forward(*forward, 0)
        solve_forward(*forward, 1)
    solve_middle(*forward, x)
    back = (b.lowD, b.lowQ, b.upD, b.upQ, G, GP, k, y, x)
    if n >= 16:
        top = block_linear.pool().submit(solve_back, *back, 0)
        solve_back(*back, 1)
        top.result()
    else:
        solve_back(*back, 0)
        solve_back(*back, 1)
    return x.reshape(n*nv)


def factor_structured(blocks, shift, threads=None, inplace=False, pool=None, signs=None):
    """Return (G, GP, k) for StructuredBlocks ``blocks`` with diagonal ``shift`` (n, nv).

    ``signs`` (int8, n), when given, receives the determinant sign of each
    pivot (see :func:`_invert_pivot`).

    ``inplace`` stores each pivot inverse over its own diagonal block (each
    block is copied into the Schur workspace before it is overwritten), so
    the factorization needs no second n*nv*nv array; the blocks are then
    marked consumed.
    """
    n, nv = blocks.n, blocks.nv
    k = n//2
    if inplace:
        G = blocks.diag
        blocks.consumed = True
    else:
        G = pool.take((n, nv, nv)) if pool is not None else np.empty((n, nv, nv))
    GP = np.empty((n, nv, blocks.lowP.shape[2]))
    signs = np.zeros(n, np.int8) if signs is None else signs
    arguments = (blocks.diag, np.ascontiguousarray(shift, dtype=np.float64), blocks.lowD, blocks.lowP,
                 blocks.lowQ, blocks.upD, blocks.upP, blocks.upQ, k, G, GP, not inplace, signs)
    threads = numba.get_num_threads() if threads is None else threads
    if n >= 8 and threads > 1:
        top = block_linear.pool().submit(structured_top, *arguments)
        ok = structured_bottom(*arguments)
        ok = top.result() and ok
    else:
        ok = structured_top(*arguments) and structured_bottom(*arguments)
    if not (ok and structured_middle(*arguments)):
        raise np.linalg.LinAlgError('singular or non-finite structured pivot')
    return G, GP, k


class StructuredFactor:
    """Bordered solve [[A, c], [r, 0]] with A in structured block form.

    Mirrors :class:`BorderedFactor`: Lambda is eliminated with one extra
    solve and a scalar Schur complement. Falls back to sparse LU of the
    assembled matrix when a pivot is singular.
    """

    def __init__(self, blocks, rdt=0., inplace=False, pool=None):
        from kflame.flame.diffusion import BorderedFactor
        n, nv = blocks.n, blocks.nv
        self.n, self.nv, self.rdt = n, nv, rdt
        self.blocks = blocks
        self.fallback = None
        self.signs = np.zeros(n, np.int8)
        if getattr(blocks, 'consumed', False):
            raise ValueError('structured blocks were consumed by an in-place factorization')
        try:
            if blocks.lazy:
                self.G, self.GP, self.k = factor_lazy(blocks, rdt*blocks.mass.reshape(n, nv), pool, self.signs)
            else:
                self.G, self.GP, self.k = factor_structured(blocks, rdt*blocks.mass.reshape(n, nv),
                                                            inplace=inplace, pool=pool, signs=self.signs)
            column = np.zeros(n*nv)
            column[nv+1:(n-1)*nv:nv] = 1.
            self.v = self._solve(column)
            self.r = blocks.border.ravel()
            with np.errstate(all='raise'):
                self.schur = -float(self.r@self.v[(n-2)*nv:])
            if not np.isfinite(self.v).all() or not abs(self.schur) > 1e-300:
                raise FloatingPointError('Lambda Schur complement')
            # Block elimination pivots only inside blocks: the first solve is
            # verified when A is still held (chunked and in-place factors no
            # longer hold it; their steps are accepted on the full residual).
            self.verify = not (blocks.lazy or inplace)
        except (FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
            self.G = self.GP = None
            if getattr(blocks, 'consumed', False) or blocks.lazy:
                # The caller rebuilds full blocks and retries out of place.
                raise np.linalg.LinAlgError('singular structured factorization')
            self.fallback = BorderedFactor(blocks.dense(), rdt)

    def _solve(self, rhs):
        return parallel_solve(self.blocks, self.G, self.GP, self.k, rhs)

    def determinant_sign(self):
        """Sign of det [[A, c], [r, 0]]: pivot signs times the Lambda Schur complement."""
        if self.fallback is not None:
            return self.fallback.determinant_sign()
        return int(np.prod(self.signs, dtype=np.int64))*int(np.sign(self.schur))

    def solve(self, rhs):
        if self.fallback is not None:
            return self.fallback.solve(rhs)
        out = self._bordered(rhs)
        if getattr(self, 'verify', False):
            from kflame.flame.diffusion import BACKWARD_ERROR_LIMIT, BorderedFactor, backward_error
            self.verify = False
            self.backward_error = backward_error(self.blocks, self.rdt, out, rhs)
            if not self.backward_error <= BACKWARD_ERROR_LIMIT:
                self.G = self.GP = None
                self.fallback = BorderedFactor(self.blocks.dense(), self.rdt)
                return self.fallback.solve(rhs)
        return out

    def _bordered(self, rhs):
        n, nv = self.n, self.nv
        u = self._solve(rhs[:-1])
        delta = (rhs[-1]-self.r@u[(n-2)*nv:])/self.schur
        out = np.empty(n*nv+1)
        np.subtract(u, self.v*delta, out=out[:-1])
        out[-1] = delta
        return out

