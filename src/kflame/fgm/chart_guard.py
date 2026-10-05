"""Offline shared-coefficient limiter for an orientation-preserving chart.

The finite audit uses the centre and eight Gauss points of each logical cell.
It is not a proof between those points. Convex blending uses the same scalar
for every field, preserving mass, elemental/source invariants and C1 splines.
"""
import itertools
import numpy as np
from scipy.interpolate import BSpline
from kflame.fgm.tensor_kernels import evaluate_tensor


def guard_chart(fields, raw, shape, knots, strides, bilger, progress):
    shape = tuple(shape)
    base = {q: raw[q].reshape((*shape, -1)) for q in fields}
    high = {q: v.reshape((*shape, -1)) for q, v in fields.items()}
    delta = {q: high[q]-base[q] for q in fields}
    base_C, delta_C = base['Y']@progress, delta['Y']@progress
    growth = np.diff(base_C, axis=2)
    if np.any(growth <= 0.):
        raise ValueError('Orientation guard requires a monotone reference polygon')
    coefficient = np.ones(shape)
    cells = np.stack(np.meshgrid(*[np.arange(n-1) for n in shape], indexing='ij'), axis=-1).reshape(-1,3)
    offsets = [(0.5,)*3]+list(itertools.product([.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)], repeat=3))
    scales = np.maximum(np.ptp(np.column_stack([raw['Y']@bilger, raw['Y']@progress, raw['h']]),axis=0),[1e-3,1e-3,1e5])
    history = []
    for iteration in range(32):
        # Preserve strict progress growth after neighbouring coefficients differ.
        for _ in range(24):
            current = np.diff(base_C+coefficient*delta_C,axis=2)
            margin = np.minimum(.01*growth,1e-12)
            bad = current <= margin
            if not bad.any():
                break
            factor = np.ones_like(current)
            factor[bad] = .99*(growth[bad]-margin[bad])/(growth[bad]-current[bad])
            reduction = np.ones(shape)
            reduction[:,:,:-1] = np.minimum(reduction[:,:,:-1],factor)
            reduction[:,:,1:] = np.minimum(reduction[:,:,1:],factor)
            coefficient *= reduction
        Y = base['Y']+coefficient[...,None]*delta['Y']
        h = base['h']+coefficient[...,None]*delta['h']
        controls = np.column_stack([(Y@bilger).ravel(),(Y@progress).ravel(),h.ravel()])
        bad_points, minimum = [], np.inf
        for offset in offsets:
            points = (cells+offset)/(np.asarray(shape)-1)
            _, derivative = evaluate_tensor(points,knots,strides,controls,differentiated=3)
            determinant = np.linalg.det(derivative.transpose(0,2,1)/scales[None,:,None])
            minimum = min(minimum,float(determinant.min()))
            if np.any(determinant <= 1e-12):
                bad_points.append(points[determinant <= 1e-12])
        history.append(dict(iteration=iteration,bad_samples=sum(len(v) for v in bad_points),minimum_scaled_determinant=minimum))
        if not bad_points:
            break
        points = np.concatenate(bad_points)
        indices = [BSpline.design_matrix(points[:,k],knots[k],2).indices.reshape(-1,3) for k in range(3)]
        support = (indices[0][:,:,None,None]*strides[0]+indices[1][:,None,:,None]*strides[1]+indices[2][:,None,None,:]).ravel()
        affected = np.unique(support)
        coefficient.ravel()[affected] *= .5 if iteration < 24 else 0.
    else:
        raise ValueError('Chart orientation guard could not produce a valid sampled closure')
    if np.any(np.diff(Y@progress,axis=2) <= 0.):
        raise ValueError('Chart guard lost strict progress growth')
    return {q:(base[q]+coefficient[...,None]*delta[q]).reshape(-1,high[q].shape[-1]) for q in high}, dict(
        samples_per_cell=9, sample_definition='centre_and_eight_Gauss_points',
        limited_coefficients=int(np.count_nonzero(coefficient < 1.)),
        minimum_coefficient=float(coefficient.min()),history=history,
        limitation='Finite sample check; runtime nodes and faces must also be checked.')
