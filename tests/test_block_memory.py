"""Exact regression checks against the allocating two-ended BLAS sweeps."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident

import numpy as np
import pytest
from numba import njit

from kflame.flame import block_linear


@njit(cache=True)
def _allocating_factor(lower, diag, upper, shift):
    # Frozen arithmetic/product order from the pre-scratch implementation.
    n, nv = diag.shape[:2]
    k = n // 2
    inverse = np.empty((n, nv, nv))
    coupling = np.empty((n, nv, nv))
    for i in range(k):
        block = diag[i].copy()
        for a in range(nv):
            block[a, a] += shift[i, a]
        if i > 0:
            block -= np.dot(lower[i-1], coupling[i-1])
        inverse[i] = np.linalg.inv(block)
        coupling[i] = np.dot(inverse[i], upper[i])
    for i in range(n-1, k, -1):
        block = diag[i].copy()
        for a in range(nv):
            block[a, a] += shift[i, a]
        if i < n-1:
            block -= np.dot(upper[i], coupling[i+1])
        inverse[i] = np.linalg.inv(block)
        coupling[i] = np.dot(inverse[i], lower[i-1])
    block = diag[k].copy()
    for a in range(nv):
        block[a, a] += shift[k, a]
    if k > 0:
        block -= np.dot(lower[k-1], coupling[k-1])
    if k < n-1:
        block -= np.dot(upper[k], coupling[k+1])
    inverse[k] = np.linalg.inv(block)
    return inverse, coupling, k


@njit(cache=True)
def _allocating_solve(lower, upper, inverse, coupling, k, rhs):
    n, nv = inverse.shape[:2]
    f = rhs.reshape((n, nv))
    y = np.empty((n, nv))
    for i in range(k):
        r = f[i].copy()
        if i > 0:
            r -= np.dot(lower[i-1], y[i-1])
        y[i] = np.dot(inverse[i], r)
    for i in range(n-1, k, -1):
        r = f[i].copy()
        if i < n-1:
            r -= np.dot(upper[i], y[i+1])
        y[i] = np.dot(inverse[i], r)
    r = f[k].copy()
    if k > 0:
        r -= np.dot(lower[k-1], y[k-1])
    if k < n-1:
        r -= np.dot(upper[k], y[k+1])
    x = np.empty((n, nv))
    x[k] = np.dot(inverse[k], r)
    for i in range(k-1, -1, -1):
        x[i] = y[i]-np.dot(coupling[i], x[i+1])
    for i in range(k+1, n):
        x[i] = y[i]-np.dot(coupling[i], x[i-1])
    return x.reshape(n*nv)


@njit(cache=True, nogil=True)
def _baseline_current_solve(lower, upper, inverse, coupling, k, rhs):
    # Frozen baseline-current, before extracting the caller-buffer API.
    n, nv = inverse.shape[0], inverse.shape[1]
    f = rhs.reshape((n, nv))
    y = np.empty((n, nv))
    r = np.empty(nv, dtype=rhs.dtype)
    product = np.empty(nv, dtype=inverse.dtype)
    for i in range(k):
        r[:] = f[i]
        if i > 0:
            np.dot(lower[i-1], y[i-1], out=product)
            r -= product
        np.dot(inverse[i], r, out=y[i])
    for i in range(n-1, k, -1):
        r[:] = f[i]
        if i < n-1:
            np.dot(upper[i], y[i+1], out=product)
            r -= product
        np.dot(inverse[i], r, out=y[i])
    r[:] = f[k]
    if k > 0:
        np.dot(lower[k-1], y[k-1], out=product)
        r -= product
    if k < n-1:
        np.dot(upper[k], y[k+1], out=product)
        r -= product
    x = np.empty((n, nv))
    np.dot(inverse[k], r, out=x[k])
    for i in range(k-1, -1, -1):
        np.dot(coupling[i], x[i+1], out=product)
        for a in range(nv):
            x[i, a] = y[i, a]-product[a]
    for i in range(k+1, n):
        np.dot(coupling[i], x[i-1], out=product)
        for a in range(nv):
            x[i, a] = y[i, a]-product[a]
    return x.reshape(n*nv)


def _layout(array, layout):
    if layout == 'fortran':
        return np.asfortranarray(array)
    if layout == 'strided':
        storage = np.empty(tuple(2*s for s in array.shape))
        view = storage[tuple(slice(None, None, 2) for _ in array.shape)]
        view[...] = array
        return view
    return array


def _system(n, nv, layout='c'):
    rng = np.random.default_rng(918 + n + nv)
    lower = rng.normal(0., .1, (n-1, nv, nv))
    upper = rng.normal(0., .1, (n-1, nv, nv))
    diag = rng.normal(0., .1, (n, nv, nv)) + 3.*np.eye(nv)
    shift = rng.uniform(-.1, .1, (n, nv))
    return tuple(_layout(a, layout) for a in (lower, diag, upper, shift))


def _assert_bits(actual, expected):
    # Includes signed zero; no tolerance can conceal a changed operation.
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint64),
        np.ascontiguousarray(expected).view(np.uint64))


@pytest.mark.parametrize('n,nv', [(1, 1), (1, 5), (2, 1), (2, 5), (3, 5), (8, 5), (33, 5)])
@pytest.mark.parametrize('layout', ['c', 'fortran', 'strided'])
def test_scratch_matches_allocating_reference_exactly(n, nv, layout):
    lower, diag, upper, shift = _system(n, nv, layout)
    originals = [a.copy() for a in (lower, diag, upper, shift)]
    expected_inverse, expected_coupling, k = _allocating_factor(lower, diag, upper, shift)
    inverse, coupling, actual_k = block_linear.factor_two_ended(
        lower, diag, upper, shift, threads=1)
    assert actual_k == k
    _assert_bits(inverse, expected_inverse)
    # coupling[k] is deliberately uninitialized and never consumed.
    _assert_bits(coupling[:k], expected_coupling[:k])
    _assert_bits(coupling[k+1:], expected_coupling[k+1:])
    rhs = np.random.default_rng(71).standard_normal(n*nv)
    original_rhs = rhs.copy()
    expected = _allocating_solve(lower, upper, expected_inverse, expected_coupling, k, rhs)
    solve_inverse = _layout(inverse, layout)
    solve_coupling = _layout(coupling, layout)
    x = block_linear.twisted_solve(lower, upper, solve_inverse, solve_coupling, k, rhs)
    _assert_bits(x, expected)
    assert x.shape == (n*nv,) and x.flags.c_contiguous
    for actual, original in zip((lower, diag, upper, shift), originals):
        _assert_bits(actual, original)
    _assert_bits(rhs, original_rhs)
    for output in (inverse, coupling, x):
        for source in (lower, diag, upper, shift, rhs):
            assert not np.shares_memory(output, source)
    assert not np.shares_memory(inverse, coupling)
    assert not np.shares_memory(x, inverse) and not np.shares_memory(x, coupling)


def test_scratch_is_private_between_threaded_factors_and_repeated_solves():
    lower, diag, upper, shift = _system(33, 5)
    serial = block_linear.factor_two_ended(lower, diag, upper, threads=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(block_linear.factor_two_ended,
                                 lower, diag, upper, threads=2) for _ in range(2)]
        factors = [task.result() for task in tasks]
    k = serial[2]
    for inverse, coupling, actual_k in factors:
        assert actual_k == k
        _assert_bits(inverse, serial[0])
        _assert_bits(coupling[:k], serial[1][:k])
        _assert_bits(coupling[k+1:], serial[1][k+1:])
        assert not np.shares_memory(inverse, serial[0])
        assert not np.shares_memory(coupling, serial[1])
    rhs = np.arange(diag.shape[0]*diag.shape[1], dtype=float)
    first = block_linear.twisted_solve(lower, upper, *serial, rhs)
    saved = first.copy()
    second = block_linear.twisted_solve(lower, upper, *serial, -rhs)
    _assert_bits(first, saved)
    assert not np.shares_memory(first, second)
    factors[0][0].fill(0.)
    _assert_bits(factors[1][0], serial[0])


@pytest.mark.parametrize('n,nv', [(1, 1), (1, 5), (2, 5), (3, 5), (8, 5),
                                  (33, 5), (31, 13), (31, 56)])
@pytest.mark.parametrize('layout', ['c', 'fortran', 'strided'])
@pytest.mark.parametrize('rhs_layout', ['c', 'strided'])
def test_solve_into_matches_baseline_current_bytes(n, nv, layout, rhs_layout):
    lower, diag, upper, shift = _system(n, nv, layout)
    inverse, coupling, k = block_linear.factor_two_ended(
        lower, diag, upper, shift, threads=1)
    inverse, coupling = (_layout(a, layout) for a in (inverse, coupling))
    rhs_source = _layout(np.random.default_rng(71).standard_normal(n*nv), rhs_layout)
    # BorderedFactor already packs strided RHS at this boundary.
    rhs = np.ascontiguousarray(rhs_source)
    inputs = (lower, upper, inverse, coupling, rhs_source, rhs)
    originals = [a.copy() for a in inputs]
    expected = _baseline_current_solve(lower, upper, inverse, coupling, k, rhs)
    y, x = (np.full((n, nv), np.nan) for _ in range(2))
    r, product = (np.full(nv, np.nan) for _ in range(2))
    actual = block_linear.twisted_solve_into(
        lower, upper, inverse, coupling, k, rhs, y, x, r, product)
    _assert_bits(actual, expected)
    _assert_bits(block_linear.twisted_solve(lower, upper, inverse, coupling, k, rhs),
                 expected)
    assert actual.shape == (n*nv,) and actual.flags.c_contiguous
    assert np.shares_memory(actual, x)
    for scratch in (y, r, product):
        assert not np.shares_memory(actual, scratch)
    assert np.isnan(y[k]).all()  # The middle scratch row is unused.
    for source, original in zip(inputs, originals):
        _assert_bits(source, original)
        assert not np.shares_memory(actual, source)


def test_solve_into_reuses_scratch_with_independent_output_views():
    n, nv = 8, 5
    lower, diag, upper, shift = _system(n, nv)
    inverse, coupling, k = block_linear.factor_two_ended(
        lower, diag, upper, shift, threads=1)
    rhs = np.random.default_rng(9).standard_normal(n*nv)
    for source in (lower, upper, inverse, coupling, rhs):
        source.setflags(write=False)
    y = np.empty((n, nv))
    r, product = np.empty(nv), np.empty(nv)
    outputs = []
    snapshots = []
    for scale in (1., -1., 2.):
        current_rhs = scale*rhs
        # Permit an independently owned bordered output backing array.
        storage = np.full(n*nv+1, 123.)
        x = storage[:-1].reshape(n, nv)
        result = block_linear.twisted_solve_into(
            lower, upper, inverse, coupling, k, current_rhs, y, x, r, product)
        expected = _baseline_current_solve(lower, upper, inverse, coupling, k, current_rhs)
        _assert_bits(result, expected)
        assert np.shares_memory(result, storage)
        assert storage[-1] == 123.
        outputs.append(result)
        snapshots.append(result.copy())
        for scratch in (y, r, product):
            scratch.fill(np.nan)
    for index, (result, saved) in enumerate(zip(outputs, snapshots)):
        _assert_bits(result, saved)
        for other in outputs[index+1:]:
            assert not np.shares_memory(result, other)


def test_solve_into_concurrent_calls_use_private_reusable_scratch():
    n, nv = 33, 5
    lower, diag, upper, shift = _system(n, nv)
    inverse, coupling, k = block_linear.factor_two_ended(
        lower, diag, upper, shift, threads=1)
    rhs = np.random.default_rng(17).standard_normal(n*nv)
    for source in (lower, upper, inverse, coupling, rhs):
        source.setflags(write=False)
    # Compile before starting workers; no shared scratch even during warmup.
    block_linear.twisted_solve_into(
        lower, upper, inverse, coupling, k, rhs.copy(),
        np.empty((n, nv)), np.empty((n, nv)), np.empty(nv), np.empty(nv))
    scales = (1., -1., 2., -2.)
    expected = [_baseline_current_solve(lower, upper, inverse, coupling, k, scale*rhs)
                for scale in scales]
    barrier = Barrier(4, timeout=30.)

    def worker():
        y = np.empty((n, nv))
        r, product = np.empty(nv), np.empty(nv)
        outputs = []
        for scale in scales:
            current_rhs = scale*rhs
            x = np.empty((n, nv))
            barrier.wait()
            outputs.append(block_linear.twisted_solve_into(
                lower, upper, inverse, coupling, k, current_rhs, y, x, r, product))
        return outputs

    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(worker) for _ in range(4)]
        results = [task.result() for task in tasks]
    all_outputs = []
    for batch in results:
        for result, reference in zip(batch, expected):
            _assert_bits(result, reference)
            all_outputs.append(result)
    for index, result in enumerate(all_outputs):
        for other in all_outputs[index+1:]:
            assert not np.shares_memory(result, other)
    saved = all_outputs[1].copy()
    all_outputs[0].fill(0.)
    _assert_bits(all_outputs[1], saved)


@pytest.mark.parametrize('value', [0., np.nan, np.inf])
@pytest.mark.parametrize('pivot', [0, 2, 4])
def test_invalid_pivots_preserve_linalg_failure(value, pivot):
    lower, diag, upper, shift = _system(5, 3)
    lower.fill(0.)
    upper.fill(0.)
    shift.fill(0.)
    diag[pivot].fill(value)
    with pytest.raises(np.linalg.LinAlgError):
        _allocating_factor(lower, diag, upper, shift)
    with pytest.raises(np.linalg.LinAlgError):
        block_linear.factor_two_ended(lower, diag, upper, shift, threads=1)


def _bordered_system(n=33, nv=5, layout='c'):
    from kflame.flame.diffusion import BorderedBlocks

    lower, diag, upper, _ = _system(n, nv, layout)
    border = np.zeros((2, nv))
    border[0, 1] = 1.
    mass = np.full(n*nv, .1)
    return BorderedBlocks(lower, diag, upper, border, mass)


def _allocating_bordered_solve(factor, rhs):
    # Frozen pre-integration BorderedFactor.solve: u and out are independent.
    n, nv = factor.n, factor.nv
    u = _baseline_current_solve(
        factor.lower, factor.upper, factor.inverse, factor.coupling, factor.k,
        np.ascontiguousarray(rhs[:-1]))
    delta = (rhs[-1]-factor.r@u[(n-2)*nv:])/factor.schur
    out = np.empty(n*nv+1)
    np.subtract(u, factor.v*delta, out=out[:-1])
    out[-1] = delta
    return out


@pytest.mark.parametrize('n,nv', [(3, 5), (31, 13), (31, 56), (33, 5)])
@pytest.mark.parametrize('layout', ['c', 'fortran', 'strided'])
@pytest.mark.parametrize('rdt', [0., 1e4])
def test_bordered_many_rhs_match_allocating_oracle_bytes(n, nv, layout, rdt):
    from kflame.flame.diffusion import BorderedFactor

    factor = BorderedFactor(_bordered_system(n, nv, layout), rdt)
    assert factor.sparse is None
    inputs = (factor.lower, factor.upper, factor.inverse, factor.coupling,
              factor.v, factor.r)
    originals = [a.copy() for a in inputs]
    original_schur = np.float64(factor.schur).tobytes()
    rng = np.random.default_rng(182)
    rhs_values = [np.zeros(n*nv+1)] + [rng.standard_normal(n*nv+1) for _ in range(6)]
    outputs, snapshots = [], []
    for index, values in enumerate(rhs_values):
        rhs = _layout(values, 'strided' if index % 2 else 'c')
        rhs.setflags(write=False)
        original_rhs = rhs.copy()
        expected = _allocating_bordered_solve(factor, rhs)
        result = factor.solve(rhs)
        _assert_bits(result, expected)
        _assert_bits(rhs, original_rhs)
        assert result.shape == (n*nv+1,)
        assert result.flags.owndata and result.flags.c_contiguous
        for source in inputs + (rhs,):
            assert not np.shares_memory(result, source)
        outputs.append(result)
        snapshots.append(result.copy())
    for actual, saved in zip(inputs, originals):
        _assert_bits(actual, saved)
    assert np.float64(factor.schur).tobytes() == original_schur
    for index, (result, saved) in enumerate(zip(outputs, snapshots)):
        _assert_bits(result, saved)
        for other in outputs[index+1:]:
            assert not np.shares_memory(result, other)
    outputs[0].fill(123.)
    for result, saved in zip(outputs[1:], snapshots[1:]):
        _assert_bits(result, saved)


@pytest.mark.parametrize('rdt', [0., 1e4])
def test_bordered_same_factor_concurrency_preserves_outputs(rdt):
    from kflame.flame.diffusion import BorderedFactor

    factor = BorderedFactor(_bordered_system(), rdt)
    assert factor.sparse is None
    rng = np.random.default_rng(93)
    rhs_values = [_layout(rng.standard_normal(factor.n*factor.nv+1),
                          'strided' if i % 2 else 'c') for i in range(4)]
    for rhs in rhs_values:
        rhs.setflags(write=False)
    originals = [rhs.copy() for rhs in rhs_values]
    expected = [_allocating_bordered_solve(factor, rhs) for rhs in rhs_values]
    # Warm the exact signatures before entering concurrent solves.
    main_outputs = []
    for rhs, reference in zip(rhs_values, expected):
        result = factor.solve(rhs)
        _assert_bits(result, reference)
        main_outputs.append(result)
    barrier = Barrier(4, timeout=30.)

    def worker():
        outputs, snapshots = [], []
        for rhs, reference in zip(rhs_values, expected):
            barrier.wait()
            result = factor.solve(rhs)
            _assert_bits(result, reference)
            outputs.append(result)
            snapshots.append(result.copy())
        for result, saved in zip(outputs, snapshots):
            _assert_bits(result, saved)
        return get_ident(), outputs

    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(worker) for _ in range(4)]
        batches = [task.result() for task in tasks]
    assert len({ident for ident, _ in batches}) == 4
    all_outputs = main_outputs + [result for _, outputs in batches for result in outputs]
    for index, result in enumerate(all_outputs):
        assert result.flags.owndata and result.flags.c_contiguous
        _assert_bits(result, expected[index % len(expected)])
        for other in all_outputs[index+1:]:
            assert not np.shares_memory(result, other)
        for source in rhs_values:
            assert not np.shares_memory(result, source)
    all_outputs[0].fill(123.)
    for index, result in enumerate(all_outputs[1:], start=1):
        _assert_bits(result, expected[index % len(expected)])
    for rhs, saved in zip(rhs_values, originals):
        _assert_bits(rhs, saved)


@pytest.mark.parametrize('threaded', [False, True])
def test_synthetic_bordered_factor_matches_sparse_lu(threaded, monkeypatch):
    from scipy.sparse.linalg import splu
    from kflame.flame.diffusion import BorderedFactor

    monkeypatch.setattr(block_linear, 'THREAD_NODES', 0 if threaded else 34)
    monkeypatch.setattr(block_linear.numba, 'get_num_threads', lambda: 2)
    blocks = _bordered_system()
    rhs = np.random.default_rng(12).standard_normal(33*5+1)
    for rdt in (0., 1e4):
        factor = BorderedFactor(blocks, rdt)
        assert factor.sparse is None
        np.testing.assert_allclose(factor.solve(rhs), splu(blocks.tocsc(rdt)).solve(rhs),
                                   rtol=1e-8, atol=1e-10)


def test_synthetic_bordered_factor_preserves_sparse_fallback(monkeypatch):
    from scipy.sparse.linalg import splu
    from kflame.flame.diffusion import BorderedFactor

    def singular(*args):
        raise np.linalg.LinAlgError('singular pivot block')

    monkeypatch.setattr(block_linear, 'twisted_top', singular)
    blocks = _bordered_system()
    factor = BorderedFactor(blocks)
    assert factor.sparse is not None
    rhs = np.random.default_rng(12).standard_normal(33*5+1)
    _assert_bits(factor.solve(rhs), splu(blocks.tocsc()).solve(rhs))
