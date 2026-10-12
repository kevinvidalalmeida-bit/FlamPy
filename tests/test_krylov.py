"""GMRES around approximate factorizations (counterflow and premixed Newton)."""
import numpy as np

from kflame.flame.krylov import KrylovFactor, gmres


class _Inverse:
    def __init__(self, matrix):
        self.matrix = matrix
        self.calls = 0

    def solve(self, b):
        self.calls += 1
        return np.linalg.solve(self.matrix, b)


def _system(seed=0, n=40):
    rng = np.random.default_rng(seed)
    A = np.eye(n)*4.+rng.standard_normal((n, n))*.3
    approximate = A+np.diag(rng.uniform(-.4, .4, n))     # a nearby, cheaper operator
    return A, approximate, rng.standard_normal(n)


def test_exact_preconditioner_converges_at_once():
    A, _, b = _system()
    M = _Inverse(A)
    x, converged = gmres(lambda v: A@v, M.solve, b, M.solve(b), 1e-12, 5)
    assert converged
    np.testing.assert_allclose(A@x, b, atol=1e-10)


def test_approximate_preconditioner_reaches_the_tolerance():
    A, approximate, b = _system(1)
    M = _Inverse(approximate)
    x, converged = gmres(lambda v: A@v, M.solve, b, M.solve(b), 1e-10, 40)
    assert converged
    assert np.linalg.norm(A@x-b) <= 1e-10*np.linalg.norm(b)
    # The first Krylov direction is the preconditioned step: never worse than it.
    z0 = M.solve(b)
    assert np.linalg.norm(A@x-b) <= np.linalg.norm(A@z0-b)


def test_krylov_factor_without_anchor_is_the_inner_solve():
    A, approximate, b = _system(2)
    factor = KrylovFactor(_Inverse(approximate))
    np.testing.assert_array_equal(factor.solve(b), np.linalg.solve(approximate, b))


def test_krylov_factor_solves_the_anchored_jacobian():
    A, approximate, b = _system(3)
    c = np.random.default_rng(4).standard_normal(A.shape[0])
    fun = lambda x: A@x+.01*x**3-c                       # Jacobian at x: A + 0.03 diag(x^2)
    x0 = np.linspace(-1., 1., A.shape[0])
    factor = KrylovFactor(_Inverse(approximate), rtol=1e-9, iterations=40)
    factor.set_anchor(fun, x0, fun(x0), np.full_like(x0, 1e-4))   # Newton-norm weights
    step = factor.solve(b)
    J = A+np.diag(.03*x0**2)
    # Finite-difference operator: agreement to its O(eps) error.
    assert np.linalg.norm(J@step-b) <= 1e-6*np.linalg.norm(b)
    assert factor.statistics['solves'] == 1 and factor.statistics['iterations'] > 1
