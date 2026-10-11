"""Independent ordered Wilke oracle: tails, exceptional values and layouts."""
import numpy as np
import pytest
from numba import njit
from kflame.flame.diffusion_kernels import face_viscosity


@njit(cache=True, fastmath=False)
def original_wilke(X, root, phi):
    # Original scalar expression, retained independently of the production
    # row grouping and its layout/size dispatch.
    value = 0.
    for k in range(X.size):
        total = 0.
        for j in range(X.size):
            total += X[j]*phi[k, j]
        value += max(X[k], 1e-300)*root[k]**2/max(total, 1e-300)
    return value


@pytest.mark.parametrize('n', (10, 21, 24, 25, 38, 53, 111, 348, 493, 581))
@pytest.mark.parametrize('layout', ('C', 'F', 'strided'))
def test_wilke_ordered_rows_preserve_scalar_bits(n, layout):
    rng = np.random.default_rng(234+n)
    phi = rng.uniform(.3, 3., (n, n))
    if layout == 'F':
        phi = np.asfortranarray(phi)
    elif layout == 'strided':
        phi = np.repeat(phi, 2, axis=1)[:, ::2]
    root = rng.uniform(.001, .01, n)
    dense = rng.uniform(0., 1., n);dense /= dense.sum()
    sparse = dense.copy();sparse[::2] = 0.
    trace = np.full(n, 1e-310);trace[-1] = 1.
    signed = np.zeros(n);signed[::2] = -0.
    phi.flags.writeable = False
    for X in (dense, sparse, trace, signed):
        actual = np.float64(face_viscosity(X, root, phi))
        expected = np.float64(original_wilke(X, root, phi))
        assert actual.tobytes() == expected.tobytes()
        assert np.isfinite(actual)


@pytest.mark.parametrize('n', (24, 53))
def test_wilke_exceptional_inputs_keep_original_behavior(n):
    X = np.full(n, 1./n);root = np.full(n, .005)
    for exceptional in (float('nan'), float('inf'), -float('inf')):
        phi = np.ones((n, n));phi[2, 3] = exceptional
        actual, expected = face_viscosity(X, root, phi), original_wilke(X, root, phi)
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert np.float64(actual).tobytes() == np.float64(expected).tobytes()
