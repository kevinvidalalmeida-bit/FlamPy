"""A posteriori reduced burner comparisons; no detailed state supplies controls.

Every reduced warm start is a fingerprinted training profile. Independent
native/Cantera solutions serve only as comparison data. Reports retain failed
cases; the settings and table fingerprint are saved before the first solve.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from kflame import solve_flame, solve_burner_flame
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.transport import NativeTransport
from kflame.flame.equations import _corrected_flux_frozen
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import solve_reduced_burner_fgm, ReducedConvergenceError
from examples.validate_nonadiabatic_3d import read_profile, cantera_reference, reference_input_hash
from examples.audit_nonadiabatic_sources import detailed_sources


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')


def distance(z, source):
    """Quadratic peak estimate inside the three nodes around the maximum."""
    i = int(np.argmax(abs(source)))
    if i == 0 or i == len(z)-1:
        return float(z[i])
    local = z[i-1:i+2]-z[i]
    polynomial = np.polyfit(local, abs(source[i-1:i+2]), 2)
    if polynomial[0] >= 0.:
        return float(z[i])
    peak = -polynomial[1]/(2*polynomial[0])
    return float(z[i]+peak) if local[0] <= peak <= local[-1] else float(z[i])


def source_errors(z, predicted, reference):
    denominator = float(np.trapezoid(abs(reference), z))
    return dict(peak_relative=float(np.max(abs(predicted-reference))/np.max(abs(reference))),
                L1_relative=float(np.trapezoid(abs(predicted-reference), z)/denominator),
                integral_relative=float(abs(np.trapezoid(predicted-reference, z))/denominator))


def flux_diagnostics(model, profile, mdot, feed):
    T, Y, z = profile['T'], profile['Y'], profile['z']
    mech = load_mechanism(model.metadata['mechanism'])
    transport = NativeTransport(mech)
    Tf, Yf = .5*(T[:-1]+T[1:]), .5*(Y[:, :-1]+Y[:, 1:])
    rho, D, lam, W = transport.eval_faces_poly_fast(Tf, model.metadata['pressure_Pa'], Yf, mech.inv_molecular_weights)
    coeff = rho[None, :]*mech.molecular_weights[:, None]/W[None, :]*D
    J = _corrected_flux_frozen(Y[:, :-1], Y[:, 1:], coeff, np.diff(z), mech.molecular_weights, 'molar')
    conduction = -lam*np.diff(T)/np.diff(z)
    diffusion = np.sum(model.thermo.partial_molar_enthalpies(Tf)*mech.inv_molecular_weights[:, None]*J, axis=0)
    h = model.thermo.enthalpy_mass(T, Y)
    hfeed = float(model.thermo.enthalpy_mass(model.metadata['temperature_K'], feed))
    qwall = float(-conduction[0])
    deficit = float(mdot*(hfeed-h[-1]))
    return dict(h=h, Z=model.table['bilger_weights'] @ Y+float(model.table['bilger_offset']),
                C=model.table['progress_weights'] @ Y,
                total_enthalpy_flux=mdot*.5*(h[:-1]+h[1:])+conduction+diffusion,
                conductive_heat_flux=conduction, diffusive_enthalpy_flux=diffusion,
                z_face=.5*(z[:-1]+z[1:]),
                wall_heat_flux_W_m2=qwall, outlet_heat_flux_W_m2=deficit,
                energy_closure_relative=abs(qwall-deficit)/max(abs(qwall), abs(deficit), 1.))


def comparisons(model, reduced, detailed, reference, mdot, feed, limits):
    z = reduced['z']
    dT = np.interp(z, detailed['z'], detailed['T'])
    dY = np.array([np.interp(z, detailed['z'], y) for y in detailed['Y']])
    metrics = dict(temperature_K=float(np.max(abs(reduced['T']-dT))),
                   species_absolute=float(np.max(abs(reduced['Y']-dY))))
    sources = {}
    for name in ('omega_C', 'qdot'):
        truth = np.interp(z, detailed['z'], detailed[name])
        sources[name] = source_errors(z, reduced[name], truth)
    reduced_flux, native_flux, reference_flux = [flux_diagnostics(model, p, mdot, feed) for p in (reduced, detailed, reference)]
    metrics.update(wall_heat_flux_relative=abs(reduced_flux['wall_heat_flux_W_m2']-native_flux['wall_heat_flux_W_m2'])/abs(native_flux['wall_heat_flux_W_m2']),
                   stand_off_distance_m=abs(distance(z, reduced['omega_C'])-distance(detailed['z'], detailed['omega_C'])))
    reference_errors = dict(temperature_K=float(np.max(abs(detailed['T']-np.interp(detailed['z'], reference['z'], reference['T'])))),
                            wall_heat_flux_relative=abs(native_flux['wall_heat_flux_W_m2']-reference_flux['wall_heat_flux_W_m2'])/abs(reference_flux['wall_heat_flux_W_m2']))
    checks = {k: value <= limits[k] for k, value in metrics.items()}
    for name, source in sources.items():
        for norm, value in source.items():
            checks[name+'_'+norm] = value <= limits['source_'+norm]
    return dict(metrics=metrics, sources=sources, checks=checks,
                passed=all(checks.values()), native_cantera=reference_errors,
                q_native_W_m2=native_flux['wall_heat_flux_W_m2'],
                q_reference_W_m2=reference_flux['wall_heat_flux_W_m2'])


def run(table, profiles, output, settings, *, only_phi=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = NonAdiabaticFGM(table)
    digest = hashlib.sha256((Path(table)/'nonadiabatic_fgm.npz').read_bytes()).hexdigest()
    plan = dict(settings=settings, table_sha256=digest, table_flames=len(model.metadata['rows']),
                status=settings.get('role', 'development_cases'), comparison='a_posteriori_unshifted_spatial_profiles',
                seed='fingerprinted_training_profiles_only',
                solver_source_sha256=hashlib.sha256(Path('src/kflame/fgm/reduced_burner.py').read_bytes()).hexdigest())
    plan_path = output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Output plan differs from the frozen settings or table')
    else:
        write_json(plan_path, plan)
    mech = load_mechanism(model.metadata['mechanism'])
    reports = []
    for phi in settings['phis']:
        if only_phi is not None and abs(phi-only_phi) > 1e-12:
            continue
        phi_folder = output/f'phi_{phi:.6f}'
        phi_folder.mkdir(exist_ok=True)
        native_settings = dict(mechanism=model.metadata['mechanism'], fuel=model.metadata['fuel'], oxidizer=model.metadata['oxidizer'],
                               temperature=model.metadata['temperature_K'], pressure=model.metadata['pressure_Pa'],
                               transport=model.metadata['transport'], soret=model.metadata['soret'],
                               width=settings['width_m'], initial_points=24, slope=settings['native_slope'],
                               curve=settings['native_curve'], ratio=3., prune=.005, max_points=1800, max_time=240.)
        ad_path = phi_folder/'adiabatic'
        if not ad_path.exists():
            solve_flame(phi=phi, output=ad_path, **native_settings)
        ad_meta = json.loads((ad_path/'metadata.json').read_text(encoding='utf-8'))
        if not ad_meta['accepted']:
            raise RuntimeError('Independent adiabatic mass-flux reference was not accepted')
        feed = np.asarray(ad_meta['inlet_Y'])
        ad_flux = float(model.thermo.density(model.metadata['temperature_K'], model.metadata['pressure_Pa'], feed))*ad_meta['Su']
        warm = phi_folder/'warm_065'
        if not warm.exists():
            solve_burner_flame(phi=phi, mass_flux=.65*ad_flux, output=warm, **native_settings)
        previous = warm
        for fraction in settings['mass_flux_fractions']:
            label = f'case_{phi:.6f}_{fraction:.6f}'
            case_folder = output/label
            case_folder.mkdir(exist_ok=True)
            print(f'{label}: independent detailed and reduced solves', flush=True)
            mdot = fraction*ad_flux
            native_path = case_folder/'native'
            if not native_path.exists():
                solve_burner_flame(phi=phi, mass_flux=mdot, initial_solution=previous, output=native_path, **native_settings)
            previous = native_path
            native_meta = json.loads((native_path/'metadata.json').read_text(encoding='utf-8'))
            if not native_meta['accepted']:
                raise RuntimeError('Independent native burner did not pass acceptance')
            if (abs(native_meta['mass_flux']-mdot) > 1e-12
                    or abs(native_meta['temperature']-model.metadata['temperature_K']) > 1e-10
                    or abs(native_meta['pressure']-model.metadata['pressure_Pa']) > 1e-8
                    or native_meta['transport'] != model.metadata['transport']
                    or native_meta['soret'] != model.metadata['soret']
                    or not np.allclose(native_meta['inlet_Y'], feed, rtol=0., atol=1e-14)
                    or abs(native_meta['final_width']-settings['width_m']) > 1e-10):
                raise ValueError('Cached detailed profile has different physical inputs')
            native = read_profile(native_path/'flame.npz')
            mass_sources, qdot = detailed_sources(mech, native['T'], native['Y'], model.metadata['pressure_Pa'])
            native.update(omega_C=model.table['progress_weights'] @ mass_sources, qdot=qdot)
            reference_path = case_folder/'reference.npz'
            reference = read_profile(reference_path) if reference_path.exists() else {}
            if str(reference.get('input_digest', '')) != reference_input_hash(native_meta, mdot):
                np.savez_compressed(reference_path, **cantera_reference(native_meta, mdot, model.table['progress_weights']))
            reference = read_profile(reference_path)
            phis = np.asarray(model.metadata['phis'])
            i = int(np.argmin(abs(phis-phi)))
            j = 1+int(np.argmin(abs(np.asarray(model.metadata['mass_flux_fractions'])-fraction)))
            row = next(r for r in model.metadata['rows'] if r['composition_index']==i and r['loss_index']==j)
            seed_path = Path(profiles)/row['output']/'flame.npz'
            reduced_path, report_path = case_folder/'reduced.npz', case_folder/'reduced_report.json'
            if reduced_path.exists():
                reduced, report = read_profile(reduced_path), json.loads(report_path.read_text(encoding='utf-8'))
                if (report['solver_source_sha256'] != plan['solver_source_sha256']
                        or report['residual_tolerance'] != settings['residual_tolerance']
                        or abs(report['mass_flux_kg_m2_s']-mdot) > 1e-12
                        or abs(report['phi']-phi) > 1e-12):
                    raise ValueError('Cached reduced result has different solver settings or source')
            else:
                try:
                    reduced, report = solve_reduced_burner_fgm(model, phi=phi, mass_flux=mdot,
                        seed_profile=seed_path, seed_row=row['output'], width=settings['width_m'],
                        max_spacing=settings['max_spacing_m'], residual_tolerance=settings['residual_tolerance'],
                        max_iterations=settings['max_iterations'], max_energy_error=settings['limits']['energy_closure_relative'], verbose=True)
                except ReducedConvergenceError as error:
                    reduced, report = error.profile, error.report
                np.savez_compressed(reduced_path, **reduced)
                write_json(report_path, report)
            audit = comparisons(model, reduced, native, reference, mdot, feed, settings['limits'])
            audit.update(phi=phi, fraction=fraction, mass_flux_kg_m2_s=mdot, solver=report,
                         case_label=label, accepted=report['accepted'], passed=audit['passed'] and report['accepted'])
            write_json(case_folder/'validation.json', audit)
            payload = {**{'reduced_'+k:v for k,v in reduced.items()},
                       **{'native_'+k:v for k,v in native.items()},
                       **{'reference_'+k:v for k,v in reference.items()},
                       'feed_Y':feed, 'mass_flux':mdot, 'phi':phi, 'fraction':fraction}
            for prefix, profile in [('native', native), ('reference', reference)]:
                payload.update({prefix+'_'+k:v for k,v in flux_diagnostics(model, profile, mdot, feed).items()})
            np.savez_compressed(case_folder/'comparison.npz', **payload)
            seed_destination = output/'seeds'/row['output']/'flame.npz'
            seed_destination.parent.mkdir(parents=True, exist_ok=True)
            if not seed_destination.exists():
                shutil.copyfile(seed_path, seed_destination)
            reports.append(audit)
            print(label, 'PASS' if audit['passed'] else 'FAIL', audit['metrics'], flush=True)
    write_json(output/('summary.json' if only_phi is None else f'summary_phi_{only_phi:.6f}.json'),
               dict(cases=reports, all_passed=all(r['passed'] for r in reports), table_sha256=digest))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path, default=Path('examples/reduced_burner_settings.json'))
    parser.add_argument('--only-phi', type=float)
    args = parser.parse_args()
    run(args.table, args.profiles, args.output, json.loads(args.settings.read_text(encoding='utf-8')), only_phi=args.only_phi)


if __name__ == '__main__':
    main()
