"""Select a linear reactive coordinate from training trajectories only."""
import numpy as np
from scipy.optimize import linprog


def select_progress_weights(Y, mass_sources, shape, *, normalization=1., reference_weights=None):
    """Maximize monotonicity margin along flamelets and their initial chemistry.

    Nonnegative weights with a fixed sum remove arbitrary coordinate scaling.
    Normalize each physical direction separately, so a fast flame front does
    not hide a slow post-flame reversal. This is a necessary sampled check;
    connected-cell orientation and the actual reactor branch still need checks.
    If reference_weights are supplied, retain the existing coordinate with
    the smallest L1 change inside the strict cone, instead of maximizing the
    margin alone. Both steps use training data only. Species identities and
    withheld validation profiles are not inputs.
    """
    Y, source = np.asarray(Y, dtype=float), np.asarray(mass_sources, dtype=float)
    shape_array = np.asarray(shape)
    if (shape_array.shape != (3,) or not np.issubdtype(shape_array.dtype, np.integer)
            or np.any(shape_array < 2) or Y.ndim != 2 or source.shape != Y.shape
            or len(Y) != int(np.prod(shape_array)) or not Y.shape[1]
            or not np.isfinite(Y).all() or not np.isfinite(source).all()):
        raise ValueError('Require finite species profiles/sources on a structured three-axis table')
    if not np.isfinite(normalization) or normalization <= 0.:
        raise ValueError('Progress normalization must be finite and positive')
    reference = None if reference_weights is None else np.asarray(reference_weights,dtype=float)
    if reference is not None and (reference.shape != (Y.shape[1],) or not np.isfinite(reference).all()):
        raise ValueError('Reference progress weights must match the finite species vector')
    shape, species = tuple(shape_array), Y.shape[1]
    body = np.diff(Y.reshape(*shape, species), axis=2).reshape(-1, species)
    endpoints = source.reshape(*shape, species)[:, :, -1].reshape(-1, species)
    directions = np.concatenate([body, endpoints], axis=0)
    norms = np.sum(abs(directions), axis=1)
    if np.any(norms == 0.):
        raise ValueError('Degenerate flamelet or chemical endpoint cannot define a strict reactive coordinate')
    directions /= norms[:, None]
    constraints = np.column_stack([-directions, np.ones(len(directions))])
    objective = np.r_[np.zeros(species), -1.]
    fit = linprog(objective, A_ub=constraints, b_ub=np.zeros(len(directions)),
                  A_eq=np.r_[np.ones(species), 0.][None, :], b_eq=[1.],
                  bounds=[(0., 1.)]*species+[(0., None)], method='highs',
                  options={'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
    if not fit.success:
        raise ValueError(f'No common linear reactive coordinate found: {fit.message}')
    weights = fit.x[:-1]
    actual_margin = float(np.min(directions@weights))
    if actual_margin <= 1e-9:
        raise ValueError('No strictly increasing common progress coordinate on the sampled physical directions')
    maximum_margin = actual_margin
    target_margin = None
    if reference is not None:
        # Keep a strict guard well above the LP feasibility tolerance. The
        # maximum available margin caps the guard on narrower physical cones.
        target_margin = min(1e-6,.5*maximum_margin)
        reference = reference/normalization
        inequality = np.block([[np.eye(species),-np.eye(species)],
                               [-np.eye(species),-np.eye(species)]])
        nearest = linprog(np.r_[np.zeros(species),np.ones(species)],
            A_ub=np.concatenate([np.column_stack([-directions,np.zeros_like(directions)]),inequality]),
            b_ub=np.r_[np.full(len(directions),-target_margin),reference,-reference],
            A_eq=np.r_[np.ones(species),np.zeros(species)][None,:],b_eq=[1.],
            bounds=[(0.,1.)]*species+[(0.,None)]*species,method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not nearest.success:
            raise ValueError(f'Cannot retain a nearby strictly monotone coordinate: {nearest.message}')
        weights = nearest.x[:species]
        actual_margin = float(np.min(directions@weights))
        if actual_margin <= 1e-9:
            raise ValueError('Nearest coordinate does not retain a strict numerical monotonicity margin')
    audit = dict(method='nonnegative_linear_progress_maximum_normalized_direction_margin',
                 body_directions=len(body), chemical_endpoints=len(endpoints),
                 minimum_normalized_margin=actual_margin, optimal_margin=float(fit.x[-1]),
                 normalization=float(normalization),
                 scope='Training flamelet differences and initial homogeneous chemical directions; '
                       'check connected cells, full reactor branches and independent physical validation separately.')
    if reference is not None:
        audit.update(method='nonnegative_linear_progress_nearest_reference_L1',
            maximum_available_margin=maximum_margin,target_normalized_margin=target_margin,
            normalized_reference_L1_change=float(np.sum(abs(weights-reference))))
    return weights*normalization, audit
