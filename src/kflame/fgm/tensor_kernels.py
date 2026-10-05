"""Fused quadratic tensor queries with compact analytic derivatives.

Each point reads 27 adjacent coefficients once. No (points, 27, species)
temporary or dense derivative basis is constructed. Parallel execution is
reserved for large batches; Newton's small batches use the serial kernel.
IEEE arithmetic is retained: fastmath is deliberately disabled.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def _basis(knots, coordinate):
    count = len(knots)-3
    span = min(max(np.searchsorted(knots, coordinate, side='right')-1, 2), count-1)
    interval = knots[span+1]-knots[span]
    left = (knots[span+1]-coordinate)/interval
    right = (coordinate-knots[span])/interval
    before = knots[span+1]-knots[span-1]
    after = knots[span+2]-knots[span]
    weights = np.empty(3)
    derivatives = np.empty(3)
    weights[0] = (knots[span+1]-coordinate)*left/before
    weights[1] = (coordinate-knots[span-1])*left/before+(knots[span+2]-coordinate)*right/after
    weights[2] = (coordinate-knots[span])*right/after
    derivatives[0] = -2.*left/before
    derivatives[1] = 2.*left/before-2.*right/after
    derivatives[2] = 2.*right/after
    return span-2, weights, derivatives


@njit(cache=True, nogil=True)
def _point(x, index, knots0, knots1, knots2, strides, fields, values, gradients):
    a, u, du = _basis(knots0, x[index, 0])
    b, v, dv = _basis(knots1, x[index, 1])
    c, w, dw = _basis(knots2, x[index, 2])
    differentiated = gradients.shape[2]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                node = (a+i)*strides[0]+(b+j)*strides[1]+c+k
                weight = u[i]*v[j]*w[k]
                d0, d1, d2 = du[i]*v[j]*w[k], u[i]*dv[j]*w[k], u[i]*v[j]*dw[k]
                for field in range(fields.shape[1]):
                    coefficient = fields[node, field]
                    values[index, field] += weight*coefficient
                    if field < differentiated:
                        gradients[index, 0, field] += d0*coefficient
                        gradients[index, 1, field] += d1*coefficient
                        gradients[index, 2, field] += d2*coefficient


@njit(cache=True, nogil=True)
def _serial(x, knots0, knots1, knots2, strides, fields, differentiated):
    values = np.zeros((len(x), fields.shape[1]))
    gradients = np.zeros((len(x), 3, differentiated))
    for i in range(len(x)):
        _point(x, i, knots0, knots1, knots2, strides, fields, values, gradients)
    return values, gradients


@njit(cache=True, nogil=True, parallel=True)
def _parallel(x, knots0, knots1, knots2, strides, fields, differentiated):
    values = np.zeros((len(x), fields.shape[1]))
    gradients = np.zeros((len(x), 3, differentiated))
    for i in prange(len(x)):
        _point(x, i, knots0, knots1, knots2, strides, fields, values, gradients)
    return values, gradients


def evaluate_tensor(x, knots, strides, fields, *, differentiated=0, parallel_threshold=2048):
    """Evaluate local values and optionally leading-field derivatives.

    Callers validate the chart bounds; tiny extrapolations are supported for
    numerical Jacobian probes. The knots must be clamped quadratic knots.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    fields = np.ascontiguousarray(fields, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or fields.ndim != 2:
        raise ValueError('Expected (points, 3) coordinates and a coefficient matrix')
    if not 0 <= differentiated <= fields.shape[1]:
        raise ValueError('Invalid number of differentiated fields')
    kernel = _parallel if len(x) >= parallel_threshold else _serial
    return kernel(x, *knots, np.asarray(strides, dtype=np.int64), fields, differentiated)


@njit(cache=True, nogil=True)
def _progress(x, bases, knots, shape, strides, fields, differentiated):
    values = np.zeros((len(x), fields.shape[1]))
    gradients = np.zeros((len(x), 3, differentiated))
    for n in range(len(x)):
        a, b = bases[n, 0], bases[n, 1]
        u, v = x[n, 0]*(shape[0]-1)-a, x[n, 1]*(shape[1]-1)-b
        c, w, dw = _basis(knots, x[n, 2])
        for i in range(2):
            wi, di = (u, shape[0]-1) if i else (1.-u, 1-shape[0])
            for j in range(2):
                wj, dj = (v, shape[1]-1) if j else (1.-v, 1-shape[1])
                for k in range(3):
                    node = (a+i)*strides[0]+(b+j)*strides[1]+c+k
                    for field in range(fields.shape[1]):
                        coefficient = fields[node, field]
                        values[n, field] += wi*wj*w[k]*coefficient
                        if field < differentiated:
                            gradients[n, 0, field] += di*wj*w[k]*coefficient
                            gradients[n, 1, field] += wi*dj*w[k]*coefficient
                            gradients[n, 2, field] += wi*wj*dw[k]*coefficient
    return values, gradients


def evaluate_progress(x, bases, knots, shape, strides, fields, *, differentiated=0):
    """Fused bilinear composition/loss and quadratic progress guide."""
    return _progress(np.ascontiguousarray(x, dtype=np.float64),
                     np.ascontiguousarray(bases, dtype=np.int64), knots,
                     np.asarray(shape, dtype=np.int64), np.asarray(strides, dtype=np.int64),
                     fields, differentiated)
