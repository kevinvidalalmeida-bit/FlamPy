"""Independent full-species checks for a projected finite-volume solution."""
import numpy as np


def species_conservation_diagnostics(species_total, mass_sources, volumes, feed_Y,
                                     mass_flux, h_species, mechanism, *,
                                     bilger_weights, progress_weights, energy_scale):
    """Audit balances that projected Z/C equations do not enforce individually.

    Fluxes are (species,faces); sources and h_species are (species,nodes).
    Use the solver's interior control volumes. The caloric residual is the
    enthalpy projection of the omitted species residual, not an added source
    for the conserved total-h equation or a predicted wall-heat correction.
    Element fluxes use kmol of atoms, with no fuel-specific assumptions.
    """
    flux, sources = np.asarray(species_total), np.asarray(mass_sources)
    volumes, feed = np.asarray(volumes), np.asarray(feed_Y)
    hk = np.asarray(h_species)
    ns = mechanism.n_species
    if (flux.ndim != 2 or sources.shape != (ns, flux.shape[1]+1)
            or flux.shape[0] != ns or hk.shape != sources.shape
            or volumes.shape != (flux.shape[1]-1,) or feed.shape != (ns,)
            or not all(np.isfinite(a).all() for a in (flux,sources,volumes,feed,hk))
            or np.any(volumes <= 0.) or not np.isfinite(mass_flux) or mass_flux <= 0.
            or not np.isfinite(energy_scale) or energy_scale <= 0.):
        raise ValueError('Require finite aligned species fluxes, nodal sources and positive control volumes/flow/scales')
    production = sources[:,1:-1]*volumes[None,:]
    residual = np.diff(flux,axis=1)-production
    integral = flux[:,-1]-mass_flux*feed-production.sum(axis=1)
    inlet = flux[:,0]-mass_flux*feed
    elements = mechanism.atom_matrix*mechanism.inv_molecular_weights[None,:]
    element_feed = elements@feed
    element_defect = elements@flux/mass_flux-element_feed[:,None]
    maximum_element_defect = np.max(abs(element_defect),axis=1)
    relative_elements = {name:float(error/value) if value > 0. else None
        for name,error,value in zip(mechanism.element_names,maximum_element_defect,element_feed)}
    caloric = np.sum(hk[:,1:-1]*residual,axis=0)
    return dict(
        maximum_species_residual_L1_over_mdot=float(np.max(np.sum(abs(residual),axis=1))/mass_flux),
        species_integral_defect_over_mdot={name:float(value/mass_flux)
            for name,value in zip(mechanism.species_names,integral)},
        maximum_species_integral_defect_over_mdot=float(np.max(abs(integral))/mass_flux),
        projected_integral_defects_over_mdot=dict(
            Z=float(np.asarray(bilger_weights)@integral/mass_flux),
            C=float(np.asarray(progress_weights)@integral/mass_flux)),
        inlet_species_face_flux_error_over_mdot=float(np.max(abs(inlet))/mass_flux),
        inlet_species_face_enthalpy_defect_W_m2=float(hk[:,0]@inlet),
        maximum_element_flux_defect_kmol_kg={name:float(value)
            for name,value in zip(mechanism.element_names,maximum_element_defect)},
        relative_element_flux_defects=relative_elements,
        maximum_relative_element_flux_defect=max((value for value in relative_elements.values()
            if value is not None),default=0.),
        maximum_element_source_defect_kmol_m3_s=float(np.max(abs(elements@sources))),
        omitted_species_caloric_residual_W_m2=float(caloric.sum()),
        omitted_species_caloric_residual_L1_W_m2=float(np.sum(abs(caloric))),
        relative_omitted_species_caloric_residual_L1=float(np.sum(abs(caloric))/energy_scale),
        scope='Same conservative finite-volume species flux law for every profile. '
              'These are diagnostics, not an acceptance tolerance or a heat-flux correction.')
