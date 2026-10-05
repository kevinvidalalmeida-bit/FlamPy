"""A constrained thermodynamic tangent projection for reacting progress.

The Z and total-h balances remain conservative. The third interior equation
projects the full species and energy residuals along the chemical tangent at
constant Z/h. Applying the projection AFTER flux divergence retains the
variable-projection derivative terms. It is not a clipped diffusivity.

Gupta et al. (2021, doi:10.1080/13647830.2021.1926544) motivates testing
projected source AND transport terms. Their adiabatic Le=1 study uses a
Euclidean projection; this thermodynamic metric and constrained extension
are implementation choices and require independent validation.
"""
import numpy as np
from kflame.chemistry.mechanism import R_UNIV


def progress_projection(Y, T, dY, dh, hk, cp, mw, bilger, progress, *, metric_floor=1e-30):
    """Return the normalized species/enthalpy multipliers, by node.

    Y/hk are (species,nodes); dY is (nodes,species,3), dh (nodes,3).
    The ideal-mixture entropy Hessian is Q.T diag(1/(Wk Yk)) Q,
    Q=I-Y (1/Wk).T / sum(Y/Wk). The floor regularizes the METRIC only;
    it never changes Y, temperature, kinetics or transport properties.
    """
    if not np.isfinite(metric_floor) or metric_floor <= 0.:
        raise ValueError('Metric floor must be finite and positive')
    controls = np.stack([np.einsum('j,njk->nk', bilger, dY),
                         np.einsum('j,njk->nk', progress, dY), dh], axis=1)
    direction = np.linalg.solve(controls, np.tile([0.,1.,0.], (len(T),1))[...,None])[...,0]
    tangent = np.einsum('njk,nk->nj', dY, direction)
    dT = (dh-np.einsum('nj,njk->nk', hk.T, dY))/cp[:,None]
    tangent_T = np.einsum('nk,nk->n', dT, direction)
    W = 1./np.sum(Y/mw[:,None], axis=0)
    q = tangent-Y.T*W[:,None]*np.sum(tangent/mw[None,:], axis=1)[:,None]
    metric_tangent = q/(mw[None,:]*np.maximum(Y.T, metric_floor))
    metric_tangent -= W[:,None]/mw[None,:]*np.sum(Y.T*metric_tangent, axis=1)[:,None]
    beta = tangent_T/T**2
    alpha = R_UNIV*metric_tangent-beta[:,None]*hk.T
    denominator = np.sum(alpha*tangent, axis=1)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.):
        raise ValueError('Degenerate thermodynamic progress tangent')
    return alpha.T/denominator[None,:], beta/denominator


def _routh(matrix, shift):
    shifted = matrix.copy()
    shifted[:,np.arange(3),np.arange(3)] -= shift[:,None]
    first = np.trace(shifted,axis1=1,axis2=2)
    second = .5*(first**2-np.einsum('nij,nji->n',shifted,shifted))
    third = np.linalg.det(shifted)
    return np.column_stack([first,second,third])


def stable_matrix(matrix, margin):
    coefficients = _routh(matrix, margin)
    return np.all(coefficients>0.,axis=1)&(coefficients[:,0]*coefficients[:,1]>coefficients[:,2])


def stable_blend(conservative, projected, margin):
    """Smallest guarded rank-one blend passing the cubic Routh conditions.

    The normalized projected progress row has unit chemical-tangent action.
    Hence M(gamma)^-1 K(gamma) is affine in gamma with a rank-one update.
    All three characteristic coefficients are affine; the final Routh test
    is quadratic. This avoids iterative eigenvalue searches in Newton.
    The entire residual (source and transport) uses the selected blend.
    """
    first = _routh(conservative,margin)
    last = _routh(projected,margin)
    valid0 = np.all(first>0.,axis=1)&(first[:,0]*first[:,1]>first[:,2])
    valid1 = np.all(last>0.,axis=1)&(last[:,0]*last[:,1]>last[:,2])
    if not np.all(valid1|valid0):
        raise ValueError('Neither progress closure satisfies the diffusion margin')
    gamma = np.zeros(len(first))
    active = ~valid0
    if not active.any():
        return gamma
    start,end = first[active],last[active]
    delta = end-start
    roots = np.where(start <= 0.,-start/np.maximum(delta,1e-300),0.)
    lower = roots.max(axis=1)
    a = delta[:,0]*delta[:,1]
    b = start[:,0]*delta[:,1]+start[:,1]*delta[:,0]-delta[:,2]
    c = start[:,0]*start[:,1]-start[:,2]
    need = (a*lower+b)*lower+c <= 0.
    discriminant = np.maximum(b*b-4.*a*c,0.)
    radical = np.sqrt(discriminant)
    linear = abs(a) <= 64.*np.finfo(float).eps*np.maximum.reduce([abs(a),abs(b),abs(c)])
    with np.errstate(divide='ignore',invalid='ignore'):
        root0 = (-b-radical)/(2.*a)
        root1 = (-b+radical)/(2.*a)
        root_linear = -c/b
    root0 = np.where((root0>=lower)&(root0<=1.),root0,-np.inf)
    root1 = np.where((root1>=lower)&(root1<=1.),root1,-np.inf)
    quadratic_root = np.where(linear,root_linear,np.maximum(root0,root1))
    lower = np.where(need,np.maximum(lower,quadratic_root),lower)
    chosen = np.minimum(1.,np.maximum(0.,lower)*1.02+1e-10)
    trial = conservative[active]+chosen[:,None,None]*(projected[active]-conservative[active])
    coefficients = _routh(trial,margin[active])
    passed = np.all(coefficients>0.,axis=1)&(coefficients[:,0]*coefficients[:,1]>coefficients[:,2])
    gamma[active] = np.where(passed&np.isfinite(chosen),chosen,1.)
    return gamma
