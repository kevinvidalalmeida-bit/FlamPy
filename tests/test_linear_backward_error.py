"""Block eliminations verify their backward error and fall back to pivoted LU.

Review case (docs/reviews/repository-hpc-2026-10-08.md, P1): a tiny pivot
inside one block of a well-conditioned system. Elimination that pivots only
inside blocks returns [0,1,2] where the solution is [-1,1,2].
"""
import numpy as np
from scipy.sparse.linalg import splu

from kflame.flame.diffusion import BorderedBlocks, BorderedFactor, bordered_matvec
from kflame.flame.equations import BlockTridiagJacobian, factorize, solve_linear

NV = 4


def _blocks():
    n = 3
    diag = np.repeat(np.eye(NV)[None], n, axis=0)
    lower = np.zeros((n-1, NV, NV))
    upper = np.zeros((n-1, NV, NV))
    diag[0, 0, 0] = 1e-20               # variable 0 is tridiag [[1e-20,1,0],[1,1,1],[0,1,1]]
    lower[:, 0, 0] = 1.
    upper[:, 0, 0] = 1.
    return diag, lower, upper


def test_bordered_factor_detects_a_failed_block_elimination():
    diag, lower, upper = _blocks()
    border = np.zeros((2, NV))
    border[0, 1] = 1.                   # selects the interior momentum carrying Lambda
    blocks = BorderedBlocks(lower, diag, upper, border, np.zeros(3*NV))
    rhs = np.zeros(3*NV+1)
    rhs[0::NV][:3] = [1., 2., 3.]
    rhs[NV+1] = 4.
    rhs[-1] = 1.
    factor = BorderedFactor(blocks)
    solution = factor.solve(rhs)        # the first solve is verified...
    assert factor.sparse is not None    # ...and the elimination was rejected
    expected = splu(blocks.tocsc()).solve(rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(expected[0:3*NV:NV], [-1., 1., 2.], atol=1e-12)
    np.testing.assert_allclose(bordered_matvec(blocks, 0., factor.solve(rhs)), rhs, atol=1e-12)


def test_well_conditioned_blocks_keep_the_fast_elimination():
    diag, lower, upper = _blocks()
    diag[0, 0, 0] = 3.
    border = np.zeros((2, NV))
    border[0, 1] = 1.
    factor = BorderedFactor(BorderedBlocks(lower, diag, upper, border, np.zeros(3*NV)))
    factor.solve(np.arange(3*NV+1, dtype=float))
    assert factor.sparse is None and factor.backward_error < 1e-14


def test_premixed_factorization_falls_back_to_pivoted_lu():
    diag, lower, upper = _blocks()
    jmat = BlockTridiagJacobian(lower, diag, upper)
    rhs = np.zeros(3*NV)
    rhs[0::NV] = [1., 2., 3.]
    state = factorize(jmat)
    assert state['method'] == 'block_twisted'
    x = solve_linear(state, rhs)        # first solve verified: both eliminations rejected
    assert state['method'] == 'direct'
    np.testing.assert_allclose(solve_linear(state, rhs), x, atol=1e-14)
    np.testing.assert_allclose(x[0::NV], [-1., 1., 2.], atol=1e-12)
    np.testing.assert_allclose(jmat.matvec(x), rhs, atol=1e-12)


def test_a_zero_right_hand_side_keeps_the_block_elimination():
    diag, lower, upper = _blocks()
    diag[0, 0, 0] = 3.
    border = np.zeros((2, NV))
    border[0, 1] = 1.
    factor = BorderedFactor(BorderedBlocks(lower, diag, upper, border, np.zeros(3*NV)))
    np.testing.assert_array_equal(factor.solve(np.zeros(3*NV+1)), 0.)
    assert factor.sparse is None and factor.backward_error == 0.
