"""Right-preconditioned GMRES and a Newton factor that runs it.

Shared by the counterflow and premixed Newton solvers. The operator is the
matrix-free Jacobian of the complete residual at the current Newton iterate,
(F(x + eps v) - F(x)) / eps; the preconditioner is a factorization of an
approximate Jacobian (an aged one, or a mixture-averaged form of a
multicomponent problem). Steps are still accepted on the complete residual.
"""
import numpy as np


def gmres(apply_A, apply_M, b, z0, rtol, maxit):
    """Right-preconditioned GMRES for A x = b, started from z0 = M^-1 b.

    Returns (x, converged). x minimizes ||b - A x|| over M^-1 K_m(A M^-1, b);
    its first direction is the preconditioned step itself, so the result is
    never worse in residual norm than the optimally scaled z0.
    """
    beta = float(np.linalg.norm(b))
    if not beta > 0.:
        return np.zeros_like(b), True
    V = [b/beta]
    Z = []
    H = np.zeros((maxit+1, maxit))
    y = np.zeros(0)
    converged = False
    for j in range(maxit):
        z = z0/beta if j == 0 else apply_M(V[j])
        if not np.isfinite(z).all():
            break
        Z.append(z)
        w = apply_A(z)
        if not np.isfinite(w).all():
            Z.pop()
            break
        size = float(np.linalg.norm(w))
        for i in range(j+1):
            H[i, j] = float(w@V[i])
            w -= H[i, j]*V[i]
        H[j+1, j] = float(np.linalg.norm(w))
        e = np.zeros(j+2)
        e[0] = beta
        y, *_ = np.linalg.lstsq(H[:j+2, :j+1], e, rcond=None)
        residual = float(np.linalg.norm(e-H[:j+2, :j+1]@y))
        if residual <= rtol*beta:
            converged = True
            break
        if not H[j+1, j] > 1e-14*size:
            # Happy breakdown: the Krylov space is invariant; y is exact.
            converged = residual <= rtol*beta
            break
        V.append(w/H[j+1, j])
    if not Z:
        return z0, False
    return np.column_stack(Z)@y[:len(Z)], converged


class KrylovFactor:
    """Newton solves with GMRES on the exact operator, preconditioned by ``inner``.

    ``set_anchor(fun, x, f, scales)`` fixes the operator to the Jacobian of
    ``fun`` at x (F(x) = f); ``scales`` are the Newton norm weights that set
    the finite-difference step. Without an anchor ``solve`` returns the
    preconditioned step itself.
    """

    def __init__(self, inner, rtol=1e-2, iterations=20):
        self.inner = inner
        self.rtol = float(rtol)
        self.iterations = int(iterations)
        self.anchor = None
        self.statistics = dict(solves=0, iterations=0, unconverged=0)

    def set_anchor(self, fun, x, f, scales):
        self.anchor = (fun, np.array(x, dtype=float), np.array(f, dtype=float), np.asarray(scales, dtype=float))

    def solve(self, b):
        b = np.asarray(b, dtype=float)
        z0 = self.inner.solve(b)
        if self.anchor is None or b.ndim != 1:
            return z0
        fun, x, f, scales = self.anchor
        count = [0]

        def apply(v):
            peak = float(np.max(np.abs(v)/scales))
            if not peak > 0.:
                return np.zeros_like(v)
            count[0] += 1
            eps = 1e-3/peak
            return (fun(x+eps*v)-f)/eps
        step, converged = gmres(apply, self.inner.solve, b, z0, self.rtol, self.iterations)
        self.statistics['solves'] += 1
        self.statistics['iterations'] += count[0]
        self.statistics['unconverged'] += 0 if converged else 1
        return step if np.isfinite(step).all() else z0
