"""Convective derivative weights for the native flame residual.

``upwind`` is the first-order Cantera-style scheme. ``exponential`` is the
Il'in-Allen-Southwell blend of the nonuniform three-point central derivative
with upwinding, by the cell Peclet number of each transported variable:
thermal for T, and rho*D_k for each species. Where diffusion resolves the
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


def _blend(theta, forward, hm, hp, dzm, dzp):
    wm = (1.0-theta)*hm + theta*np.where(forward, 1.0/dzm, 0.0)
    wp = (1.0-theta)*hp + theta*np.where(forward, 0.0, 1.0/dzp)
    return wm, wp


def exponential_weights(u, rho, cp, lam_face, z, rho_diffusivity_face=None):
    """Per-node weights.

    Returns (thermal, species): thermal has shape (nodes, 4) with columns
    (wm, wp, hm, hp); species has shape (species, nodes, 2) with (wm, wp).
    dphi/dz at interior node j is wp*(phi[j+1]-phi[j]) + wm*(phi[j]-phi[j-1]).
    (hm, hp) are the plain central weights for the non-convective species
    enthalpy gradient. ``rho_diffusivity_face`` is rho*D_k on the faces,
    shape (species, faces); without it species use the thermal Peclet number.
    Boundary rows are unused.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    thermal = np.zeros((n, 4))
    n_species = 0 if rho_diffusivity_face is None else np.shape(rho_diffusivity_face)[0]
    species = np.zeros((n_species, n, 2))
    if n < 3:
        return thermal, species
    dzm, dzp, dz2 = z[1:-1]-z[:-2], z[2:]-z[1:-1], z[2:]-z[:-2]
    hm, hp = dzp/(dzm*dz2), dzm/(dzp*dz2)
    u_j = np.asarray(u, dtype=float)[1:-1]
    mass_flux = np.abs(np.asarray(rho, dtype=float)[1:-1]*u_j)
    forward = u_j > 0.0
    # Discrete maximum principle on a nonuniform grid: the downstream
    # coefficient keeps its sign iff (1-theta)*|rho u|*dz_up <= 2*Gamma_down,
    # with Gamma the face diffusivity downstream of the node. Il'in's
    # theta(Pe) >= 1-2/Pe, so Pe uses the upwind cell and that face. On a
    # uniform grid this is the usual cell Peclet number; theta stays O(dz).
    upwind_cell = np.where(forward, dzm, dzp)
    lam_face = np.asarray(lam_face, dtype=float)
    lam = np.where(forward, lam_face[1:], lam_face[:-1])
    peclet = mass_flux*np.asarray(cp, dtype=float)[1:-1]*upwind_cell/np.maximum(lam, 1e-300)
    thermal[1:-1, 0], thermal[1:-1, 1] = _blend(upwind_fraction(peclet), forward, hm, hp, dzm, dzp)
    thermal[1:-1, 2] = hm
    thermal[1:-1, 3] = hp
    if n_species:
        # One weight for all species: different weights would break the
        # discrete identity sum_k Y_k = 1. The thermal weight is used: the
        # least diffusive trace species would make all species upwind and
        # first order again (burner q_outlet 4684 vs ~4805 W/m2).
        species[:, 1:-1, :] = thermal[None, 1:-1, :2]
    # Next to the outflow node, whose value is a zero-gradient copy rather than
    # a physical state, keep the one-sided upwind derivative (the same
    # boundary closure order as the outlet condition itself).
    last = n - 2
    if forward[-1]:
        thermal[last, :2] = (1.0/dzm[-1], 0.0)
        species[:, last] = (1.0/dzm[-1], 0.0)
    return thermal, species
