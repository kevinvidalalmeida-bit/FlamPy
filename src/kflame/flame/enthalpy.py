"""Total enthalpy and boundary heat-loss diagnostics, in SI units."""
import numpy as np

from kflame.flame.equations import (
    build_local_jacobian_cache, _corrected_flux_frozen, _multicomponent_flux_frozen,
)
from kflame.flame.state import unpack_state


def enthalpy_diagnostics(problem, state):
    """Use the solver's face transport and thermodynamic reference convention.

    Positive boundary heat loss denotes conduction out of the gas towards
    the isothermal burner. Local h_feed-h is a departure from the feed,
    not a local wall-loss coordinate when preferential diffusion is active.
    The independent outlet deficit is meaningful when outlet gradients vanish.
    Differences between the two estimates measure finite-grid energy closure.
    """
    u, T, Y = unpack_state(state, problem.n_points, problem.n_species)
    thermo = problem._thermo
    h = np.asarray(thermo.enthalpy_mass(T, Y))
    h_feed = float(thermo.enthalpy_mass(problem.T_in, problem.Y_in))
    cache = build_local_jacobian_cache(state, problem)
    dz = np.diff(problem.z)
    if cache['flux_model'] == 'multicomponent':
        thermal = cache['soret_face_coeff']
        if thermal is None:
            thermal = np.zeros_like(Y[:, :-1])
        J = _multicomponent_flux_frozen(
            Y[:, :-1], Y[:, 1:], T[:-1], T[1:], cache['multi_face_coeff'],
            thermal, dz, cache['W'])
    else:
        J = _corrected_flux_frozen(
            Y[:, :-1], Y[:, 1:], cache['face_coeff'], dz, cache['W'], cache['basis'])
        if cache['soret_face_coeff'] is not None:
            J -= cache['soret_face_coeff'] * (np.diff(T) / dz)[None, :]
    q_conduction = -cache['lam_face'] * np.diff(T) / dz
    h_species_face = thermo.partial_molar_enthalpies(.5 * (T[:-1] + T[1:])) * cache['invW'][:, None]
    q_diffusion = np.sum(h_species_face * J, axis=0)
    mdot = cache['rho'] * u
    enthalpy_flux = .5 * (mdot[:-1] * h[:-1] + mdot[1:] * h[1:]) + q_diffusion + q_conduction
    fields = dict(h_mass=h, enthalpy_departure_from_feed=h_feed - h,
                  z_face=.5 * (problem.z[:-1] + problem.z[1:]),
                  conductive_heat_flux=q_conduction, diffusive_enthalpy_flux=q_diffusion,
                  total_enthalpy_flux=enthalpy_flux)
    wall_flux = float(-q_conduction[0])
    deficit = float(h_feed - h[-1])
    flux_outlet = float(mdot[-1] * deficit)
    closure = abs(wall_flux - flux_outlet) / max(abs(wall_flux), abs(flux_outlet), 1.0)
    diagnostics = dict(
        enthalpy_definition='total_sensible_plus_formation', h_feed_J_kg=h_feed,
        burned_enthalpy_deficit_J_kg=deficit, burner_heat_loss_W_m2=wall_flux,
        outlet_enthalpy_loss_W_m2=flux_outlet, relative_energy_closure_error=closure,
        mass_flux_relative_spread=float(np.ptp(mdot) / max(abs(mdot[0]), 1e-300)),
        outlet_gradient_T_K_m=float((T[-1] - T[-2]) / dz[-1]),
        outlet_diffusive_enthalpy_flux_W_m2=float(q_diffusion[-1]),
        note='Boundary fluxes are finite-grid estimates; refine before interpreting small losses.',
    )
    return fields, diagnostics
