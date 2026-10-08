"""Compiled two-ended elimination of block-tridiagonal Jacobians.

The top sweep eliminates downward to the middle block and the bottom sweep
upward; both are independent and release the GIL, so they can run in two
threads. Pivot blocks are inverted with LAPACK (getrf/getri) and coupled with
BLAS products. A singular or non-finite block raises LinAlgError; callers
fall back to pivoted LU. Shared by the premixed and counterflow solvers.
"""
from concurrent.futures import ThreadPoolExecutor
import os

import numba
import numpy as np
from numba import njit

_POOL = None
THREAD_NODES = 32


def pool():
    """Process-wide worker threads, spawned on demand."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 1, thread_name_prefix='kflame-blocks')
    return _POOL


def factor_two_ended(lower, diag, upper, shift=None, threads=None):
    """Return (inverse, coupling, k) of tridiag(lower, diag+shift, upper)."""
    n, nv = diag.shape[0], diag.shape[1]
    shift = np.zeros((n, nv)) if shift is None else np.ascontiguousarray(shift, dtype=np.float64)
    k = n//2
    inverse, coupling = np.empty((n, nv, nv)), np.empty((n, nv, nv))
    arguments = (lower, diag, upper, shift, k, inverse, coupling)
    threads = numba.get_num_threads() if threads is None else threads
    if n >= THREAD_NODES and threads > 1:
        top = pool().submit(twisted_top, *arguments)
        twisted_bottom(*arguments)
        top.result()
    else:
        twisted_top(*arguments)
        twisted_bottom(*arguments)
    twisted_middle(*arguments)
    return inverse, coupling, k


@njit(cache=True, nogil=True)
def twisted_top(lower, diag, upper, shift, k, inverse, coupling):
    nv = diag.shape[1]
    for i in range(k):
        block = diag[i].copy()
        for a in range(nv):
            block[a, a] += shift[i, a]
        if i > 0:
            block -= np.dot(lower[i-1], coupling[i-1])
        inverse[i] = np.linalg.inv(block)
        coupling[i] = np.dot(inverse[i], upper[i])


@njit(cache=True, nogil=True)
def twisted_bottom(lower, diag, upper, shift, k, inverse, coupling):
    n, nv = diag.shape[0], diag.shape[1]
    for i in range(n-1, k, -1):
        block = diag[i].copy()
        for a in range(nv):
            block[a, a] += shift[i, a]
        if i < n-1:
            block -= np.dot(upper[i], coupling[i+1])
        inverse[i] = np.linalg.inv(block)
        coupling[i] = np.dot(inverse[i], lower[i-1])


@njit(cache=True, nogil=True)
def twisted_middle(lower, diag, upper, shift, k, inverse, coupling):
    n, nv = diag.shape[0], diag.shape[1]
    block = diag[k].copy()
    for a in range(nv):
        block[a, a] += shift[k, a]
    if k > 0:
        block -= np.dot(lower[k-1], coupling[k-1])
    if k < n-1:
        block -= np.dot(upper[k], coupling[k+1])
    inverse[k] = np.linalg.inv(block)


@njit(cache=True, nogil=True)
def twisted_solve(lower, upper, inverse, coupling, k, rhs):
    n, nv = inverse.shape[0], inverse.shape[1]
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
