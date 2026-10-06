"""Convective derivative weights for the native flame residual.

``upwind`` is the first-order Cantera-style scheme. ``exponential`` is the
Il'in-Allen-Southwell blend of the nonuniform three-point central derivative
with upwinding, by the cell Peclet number. Where diffusion resolves the
profile (flame front, burner boundary layer) its upwind fraction is O(dz),
so the derivative is second order; at large Peclet it tends to upwind and
does not oscillate. It is the non-conservative counterpart of the
exponentially fitted fluxes of the reduced burner closure.
"""
import numpy as np

SCHEMES = {'upwind': 0, 'exponential': 1}


def scheme_code(name):
    key = str(name).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"convection scheme must be one of {sorted(SCHEMES)}")
    return SCHEMES[key]


def upwind_fraction(peclet):
    """coth(Pe/2) - 2/Pe for |Pe|; series below 1e-3 avoids cancellation."""
    pe = np.abs(np.asarray(peclet, dtype=float))
    small = pe < 1e-3
    safe = np.where(small, 1.0, pe)
    return np.where(small, pe/6.0 - pe**3/360.0, 1.0/np.tanh(0.5*safe) - 2.0/safe)


def exponential_weights(u, rho, cp, lam_face, z):
    """Per-node weights, shape (nodes, 4): (wm, wp, hm, hp).

    dphi/dz at interior node j is wp*(phi[j+1]-phi[j]) + wm*(phi[j]-phi[j-1])
    for convected T and Y. (hm, hp) are the plain central weights used for
    the non-convective species-enthalpy gradient. Boundary rows are unused.
    The thermal cell Peclet number is used for every convected variable.
    """
    z = np.asarray(z, dtype=float)
    weights = np.zeros((len(z), 4))
    if len(z) < 3:
        return weights
    dzm, dzp, dz2 = z[1:-1]-z[:-2], z[2:]-z[1:-1], z[2:]-z[:-2]
    hm, hp = dzp/(dzm*dz2), dzm/(dzp*dz2)
    u_j = np.asarray(u, dtype=float)[1:-1]
    lam = 0.5*(np.asarray(lam_face, dtype=float)[:-1]+np.asarray(lam_face, dtype=float)[1:])
    peclet = np.abs(np.asarray(rho, dtype=float)[1:-1]*u_j)*np.asarray(cp, dtype=float)[1:-1]*0.5*dz2/np.maximum(lam, 1e-300)
    theta = upwind_fraction(peclet)
    forward = u_j > 0.0
    weights[1:-1, 0] = (1.0-theta)*hm + theta*np.where(forward, 1.0/dzm, 0.0)
    weights[1:-1, 1] = (1.0-theta)*hp + theta*np.where(forward, 0.0, 1.0/dzp)
    weights[1:-1, 2] = hm
    weights[1:-1, 3] = hp
    return weights
