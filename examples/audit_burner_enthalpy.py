"""Audit saved burner profiles without solving flames or changing a table.

Global energy closure, nodal inlet species/enthalpy defects and the reduced
face-flux correction are separate quantities. Saved validation outcomes are
historical comparisons, not new certifications produced by this audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.chemistry.initialization import fresh_mixture
from kflame.fgm.reduced_burner import ReducedBurnerProblem, _SOURCE_SHA256
from kflame.fgm.conservation import species_conservation_diagnostics
from kflame.flame.enthalpy import inlet_enthalpy_diagnostics
from kflame.flame.equations import _corrected_flux_frozen
from examples.audit_nonadiabatic_sources import detailed_sources


def read_profile(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def steady_thermal_rates(profile, mass_flux, thermo, rho_faces, conductivity_faces):
    """Face diagnostics of a steady profile, not extra transported controls.

    Material temperature/enthalpy rates and enthalpy scalar dissipation are
    different quantities. A negative material temperature rate describes gas
    cooling; positive wall heat loss alone does not imply that sign.
    """
    T, Y, z = profile['T'], profile['Y'], profile['z']
    dz = np.diff(z)
    Tf, Yf = .5*(T[:-1]+T[1:]), .5*(Y[:,:-1]+Y[:,1:])
    _, cp_faces = thermo.enthalpy_mass_cp_mass(Tf,Yf)
    h = thermo.enthalpy_mass(T,Y)
    u = mass_flux/rho_faces
    alpha = conductivity_faces/(rho_faces*cp_faces)
    gradient_T, gradient_h = np.diff(T)/dz, np.diff(h)/dz
    material_T, material_h = u*gradient_T, u*gradient_h
    chi_h = alpha*gradient_h**2
    result = dict(
        material_temperature_rate_K_s=dict(minimum=float(material_T.min()),maximum=float(material_T.max())),
        material_total_enthalpy_rate_J_kg_s=dict(minimum=float(material_h.min()),maximum=float(material_h.max())),
        maximum_enthalpy_scalar_dissipation_J2_kg2_s=float(chi_h.max()),
        maximum_temperature_drop_from_preceding_peak_K=float(np.max(np.maximum.accumulate(T)-T)),
        peak_to_outlet_temperature_drop_K=float(T.max()-T[-1]),
        scope='Steady face estimates: DT/Dt=u*dT/dz; Dh/Dt=u*dh/dz; '
              'chi_h=lambda/(rho*cp)*(dh/dz)^2, with no factor 2. '
              'These diagnostics neither impose a cooling law nor prove an extra independent table dimension.')
    if 'qdot' in profile:
        positive_q = np.maximum(.5*(profile['qdot'][:-1]+profile['qdot'][1:]),0.)
        weights = positive_q*dz
        total = float(weights.sum())
        if total > 0.:
            result['positive_heat_release_fraction_at_negative_material_temperature_rate'] = float(
                weights[material_T < 0.].sum()/total)
            result['material_temperature_rate_at_peak_heat_release_K_s'] = float(material_T[np.argmax(positive_q)])
    return result


def profile_diagnostics(problem, profile, *, thermal_rates=False):
    """Use the same physical flux law for detailed and reduced profiles."""
    T, Y, z = profile['T'], profile['Y'], profile['z']
    mech, thermo = problem.mech, problem.model.thermo
    Tf, Yf = .5*(T[:-1]+T[1:]), .5*(Y[:,:-1]+Y[:,1:])
    rho, D, lam, W = problem.transport.eval_faces_poly_fast(
        Tf,problem.pressure,Yf,mech.inv_molecular_weights)
    coeff = rho[None,:]*mech.molecular_weights[:,None]/W[None,:]*D
    J = _corrected_flux_frozen(Y[:,:-1],Y[:,1:],coeff,np.diff(z),mech.molecular_weights,'molar')
    wall = float(lam[0]*(T[1]-T[0])/(z[1]-z[0]))
    deficit = float(problem.mass_flux*(problem.h_feed-thermo.enthalpy_mass(T[-1],Y[:,-1])))
    scale = max(abs(wall),abs(deficit),1.)
    result = dict(burner_heat_loss_W_m2=wall,outlet_enthalpy_loss_W_m2=deficit,
        relative_energy_closure_error=abs(wall-deficit)/scale,
        **inlet_enthalpy_diagnostics(thermo,T[0],Y[:,0],problem.inlet_Y,J[:,0],
            problem.mass_flux,energy_scale=scale,feed_temperature=problem.T_burner))
    if thermal_rates:
        result['steady_thermal_rates'] = steady_thermal_rates(profile,problem.mass_flux,thermo,rho,lam)
    return result


def full_species_diagnostics(problem, profile, sources):
    """Compare full balances using each profile's own grid and identical fluxes."""
    local = ReducedBurnerProblem(problem.model,profile['z'],problem.mass_flux,problem.inlet_Y,
                                interpolation='quadratic_progress')
    T, Y = profile['T'],profile['Y']
    thermo,table = problem.model.thermo,problem.model.table
    state = dict(T=T,Y=Y,h=thermo.enthalpy_mass(T,Y),
                 Z=table['bilger_weights']@Y+float(table['bilger_offset']),
                 C=table['progress_weights']@Y)
    flux = local.fluxes(state)
    scale = max(abs(float(flux['conduction'][0])),
                abs(local.mass_flux*(local.h_feed-state['h'][-1])),1.)
    return species_conservation_diagnostics(flux['species_total'],sources,local.volumes,
        local.inlet_Y,local.mass_flux,thermo.partial_molar_enthalpies(T)*local.mech.inv_molecular_weights[:,None],
        local.mech,bilger_weights=table['bilger_weights'],progress_weights=table['progress_weights'],energy_scale=scale)


def audit(table, cases, output, *, reference_cases=None, species_balances=False, thermal_rates=False):
    model = NonAdiabaticFGM(table)
    rows = []
    folders = sorted(Path(cases).glob('case_*'))
    for folder in folders:
        if not folder.is_dir():
            continue
        report_path = folder/'reduced_report.json'
        if report_path.exists():
            settings = json.loads(report_path.read_text(encoding='utf-8'))
        else:
            settings = json.loads((folder/'result.json').read_text(encoding='utf-8'))['report']
        reduced = read_profile(folder/'reduced.npz')
        if str(reduced['table_sha256']) != model.table_sha256:
            raise ValueError(f'{folder.name}: saved profile uses a different frozen table')
        reference_folder = folder if reference_cases is None else Path(reference_cases)/folder.name
        comparison = read_profile(reference_folder/'comparison.npz')
        problem = ReducedBurnerProblem(model,reduced['z'],float(comparison['mass_flux']),
            comparison['feed_Y'],interpolation=settings['interpolation'],
            curvature_limit=settings['curvature_limit'],
            progress_equation=settings.get('progress_equation','conservative'),
            metric_floor=settings.get('metric_floor',1e-30))
        expected_feed = fresh_mixture(problem.mech,settings['phi'],model.metadata['fuel'],model.metadata['oxidizer'])
        if (not np.isclose(problem.mass_flux,settings['mass_flux_kg_m2_s'],rtol=1e-12,atol=0.)
                or not np.allclose(problem.inlet_Y,expected_feed,rtol=1e-12,atol=1e-15)):
            raise ValueError(f'{folder.name}: saved comparison uses different feed/flow conditions')
        current = problem.state(reduced['logical_coordinates'])
        row = dict(case=folder.name,saved_solver_source_sha256=settings['solver_source_sha256'],
            reduced=profile_diagnostics(problem,reduced,thermal_rates=thermal_rates),
            recomputed=problem.diagnostics(reduced['logical_coordinates']),
            maximum_temperature_recovery_change_K=float(np.max(abs(current['T']-reduced['T']))),
            maximum_species_reconstruction_change=float(np.max(abs(current['Y']-reduced['Y']))))
        if species_balances:
            if 'omega_Y' not in model.table:
                raise ValueError('Full frozen species sources are required for a reduced species-balance audit')
            row['species_balances'] = dict(reduced=full_species_diagnostics(problem,reduced,
                problem.species_sources(reduced['logical_coordinates'])))
        for name,path in [('native',reference_folder/'native/flame.npz'),('cantera',reference_folder/'reference.npz')]:
            if path.exists():
                detailed = read_profile(path)
                row[name] = profile_diagnostics(problem,detailed,thermal_rates=thermal_rates)
                if species_balances:
                    sources,_ = detailed_sources(problem.mech,detailed['T'],detailed['Y'],problem.pressure)
                    row['species_balances'][name] = full_species_diagnostics(problem,detailed,sources)
        validation = folder/'validation.json'
        if validation.exists():
            saved = json.loads(validation.read_text(encoding='utf-8'))
            row.update(saved_validation_passed=saved['passed'],saved_wall_heat_error=saved['metrics']['wall_heat_flux_relative'])
        elif (folder/'result.json').exists():
            saved = json.loads((folder/'result.json').read_text(encoding='utf-8')).get('validation')
            if saved is not None:
                row.update(saved_validation_passed=saved['passed'],saved_wall_heat_error=saved['metrics']['wall_heat_flux_relative'])
        rows.append(row)
    if not rows:
        raise ValueError('No saved burner cases found')
    result = dict(table_sha256=model.table_sha256,solver_source_sha256=_SOURCE_SHA256,
        mechanism=model.metadata['mechanism'],pressure_Pa=model.metadata['pressure_Pa'],
        training_enthalpy_quality=model.training_enthalpy_quality(),
        case_count=len(rows),cases=rows,
        maximum_temperature_recovery_change_K=max(r['maximum_temperature_recovery_change_K'] for r in rows),
        maximum_recomputed_scaled_residual=max(r['recomputed']['maximum_scaled_residual'] for r in rows),
        maximum_relative_inlet_enthalpy_flux_error=max(r['reduced']['relative_inlet_enthalpy_flux_error'] for r in rows),
        note='Audit of saved profiles; no new flames, calibration, or physical validation. The nodal inlet defect and the numerical face correction use different enthalpy locations.')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-cases',type=Path,help='Independent saved reference bank for cases from a rebuilt table')
    parser.add_argument('--species-balances',action='store_true',help='Audit all species/element fluxes and the omitted caloric residual')
    parser.add_argument('--thermal-rates',action='store_true',help='Distinguish steady material cooling rates from enthalpy scalar dissipation')
    args = parser.parse_args()
    result = audit(args.table,args.cases,args.output,reference_cases=args.reference_cases,species_balances=args.species_balances,thermal_rates=args.thermal_rates)
    print({key:value for key,value in result.items() if key != 'cases'})
