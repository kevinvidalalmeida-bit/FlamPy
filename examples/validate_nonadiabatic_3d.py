"""Withheld composition/heat-loss validation, independent of table construction.

python examples/validate_nonadiabatic_3d.py runs/nonadiabatic --output runs/validation3d
Add --recheck to reuse accepted native/reference profiles after rebuilding a table.
Cantera is used only for the independent reference, never for production lookup.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kflame import solve_flame, solve_burner_flame
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM, OutsideManifoldError
from kflame.fgm.accuracy import FGMTolerances, assess_profile
if __package__:
    from .audit_nonadiabatic_sources import detailed_sources, cantera_sources, source_metrics
else:
    from audit_nonadiabatic_sources import detailed_sources, cantera_sources, source_metrics


SOURCE_FILES = [
    'src/kflame/api.py', 'src/kflame/flame/config.py', 'src/kflame/flame/problem.py',
    'src/kflame/flame/equations.py', 'src/kflame/flame/analytic_jacobian.py',
    'src/kflame/flame/solver.py', 'src/kflame/flame/enthalpy.py',
    'src/kflame/chemistry/thermo.py', 'src/kflame/chemistry/kinetics.py',
    'src/kflame/chemistry/mechanism.py', 'src/kflame/fgm/nonadiabatic3d.py',
    'examples/validate_nonadiabatic_3d.py', 'examples/audit_nonadiabatic_sources.py',
    'src/kflame/fgm/accuracy.py', 'src/kflame/fgm/adaptive_nonadiabatic.py', 'src/kflame/fgm/search.py',
]
WITHHELD_PHIS = np.sqrt(np.array([.7, .85, 1., 1.15]) * np.array([.85, 1., 1.15, 1.3]))
WITHHELD_FRACTIONS = [.55, .35, .18, .085]


def read_profile(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def reference_input_hash(meta, flux):
    from kflame.chemistry.mechanism import resolve_mechanism
    settings = {k: meta[k] for k in ('temperature', 'pressure', 'inlet_Y', 'final_width', 'transport', 'soret', 'refinement')}
    settings['mass_flux'] = flux
    settings['mechanism_sha256'] = hashlib.sha256(Path(resolve_mechanism(meta['mechanism'])).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def cantera_reference(meta, flux, weights):
    import cantera as ct
    from kflame.chemistry.mechanism import resolve_mechanism
    gas = ct.Solution(resolve_mechanism(meta['mechanism']))
    gas.TPY = meta['temperature'], meta['pressure'], meta['inlet_Y']
    h_feed = gas.enthalpy_mass
    flame = ct.BurnerFlame(gas, width=meta['final_width'])
    flame.burner.mdot = flux
    flame.transport_model = meta['transport']
    flame.soret_enabled = meta['soret']
    flame.set_refine_criteria(**{k: meta['refinement'][k] for k in ('ratio', 'slope', 'curve', 'prune')})
    flame.solve(loglevel=0, auto=True)
    gas.TPY = flame.T[-1], meta['pressure'], flame.Y[:, -1]
    return dict(z=flame.grid, T=flame.T, Y=flame.Y, u=flame.velocity,
                omega_C=weights @ (flame.net_production_rates * gas.molecular_weights[:, None]),
                qdot=flame.heat_release_rate,
                input_digest=np.array(reference_input_hash(meta, flux)),
                burned_deficit_J_kg=np.array(h_feed - gas.enthalpy_mass))


def evaluate_case(model, native, reference, meta, phi, fraction, flux, tolerances=None):
    tolerances = tolerances or FGMTolerances()
    t = model.table
    Z = t['bilger_weights'] @ native['Y'] + float(t['bilger_offset'])
    C = t['progress_weights'] @ native['Y']
    h = model.thermo.enthalpy_mass(native['T'], native['Y'])
    mech = load_mechanism(model.metadata['mechanism'])
    mass_sources, qdot = detailed_sources(mech, native['T'], native['Y'], meta['pressure'])
    omega = t['progress_weights'] @ mass_sources
    n = len(Z)
    batch = model.lookup_batch(Z=Z, C=C, h=h, outside='mask')
    covered = batch['covered']
    prediction = {k: batch[k] for k in ('T', 'omega_C', 'qdot')}
    prediction['Y'] = batch['Y'].T
    y, temperature = batch['Y'][covered].T, batch['T'][covered]
    control_errors = np.column_stack([
        abs(t['bilger_weights'] @ y + float(t['bilger_offset']) - Z[covered]),
        abs(t['progress_weights'] @ y - C[covered]),
        abs(model.thermo.enthalpy_mass(temperature, y) - h[covered]),
        abs(y.sum(axis=0) - 1.),
    ])
    if not covered.any():
        raise RuntimeError('No withheld states inside the resolved manifold')
    source_peak = float(np.max(abs(omega)))
    qdot_peak = float(np.max(abs(qdot)))
    sources = dict(omega_C=source_metrics(native['z'], omega, prediction['omega_C'], covered),
                   qdot=source_metrics(native['z'], qdot, prediction['qdot'], covered))
    metrics = dict(
        coverage_fraction=float(covered.mean()),
        temperature_Linf_K=float(np.max(abs(prediction['T'][covered] - native['T'][covered]))),
        all_Y_Linf=float(np.max(abs(prediction['Y'][:, covered] - native['Y'][:, covered]))),
        omega_C_Linf_over_native_peak=float(np.max(abs(prediction['omega_C'][covered] - omega[covered])) / source_peak),
        qdot_Linf_over_native_peak=float(np.max(abs(prediction['qdot'][covered] - qdot[covered])) / qdot_peak),
    )
    limits = dict(coverage_fraction_min=tolerances.node_coverage_min,
                  temperature_Linf_K_max=tolerances.temperature_K, all_Y_Linf_max=tolerances.species_absolute,
                  omega_C_Linf_over_native_peak_max=tolerances.source_peak_relative['omega_C'],
                  qdot_Linf_over_native_peak_max=tolerances.source_peak_relative['qdot'])
    source_limits = {name: dict(Linf_over_truth_peak_max=tolerances.source_peak_relative[name],
                               L1_relative_max=tolerances.source_L1_relative[name],
                               integral_error_over_abs_integral_max=tolerances.source_integral_relative[name],
                               absolute_source_coverage_min=tolerances.source_coverage_min[name])
                     for name in sources}
    source_pass = all(s[key[:-4]] <= value if key.endswith('_max') else s[key[:-4]] >= value
                      for name, s in sources.items() for key, value in source_limits[name].items())
    assessment = assess_profile(dict(native, omega_C=omega, qdot=qdot), batch, tolerances)
    interpolation_pass = assessment['passed']
    reference_T = np.interp(native['z'], reference['z'], reference['T'])
    reference_Y = np.array([np.interp(native['z'], reference['z'], row) for row in reference['Y']])
    ref_metrics = dict(
        temperature_Linf_K=float(np.max(abs(native['T'] - reference_T))),
        all_Y_Linf=float(np.max(abs(native['Y'] - reference_Y))),
        burned_deficit_relative_error=float(abs(meta['heat_loss']['burned_enthalpy_deficit_J_kg']
                                              - reference['burned_deficit_J_kg']) / abs(reference['burned_deficit_J_kg'])),
        energy_closure_error=float(meta['heat_loss']['relative_energy_closure_error']),
        mass_flux_relative_spread=float(meta['heat_loss']['mass_flux_relative_spread']),
        omega_C_Linf_over_reference_peak=source_metrics(native['z'], np.interp(native['z'], reference['z'], reference['omega_C']), omega)['Linf_over_truth_peak'],
        qdot_Linf_over_reference_peak=source_metrics(native['z'], np.interp(native['z'], reference['z'], reference['qdot']), qdot)['Linf_over_truth_peak'])
    ref_limits = dict(temperature_Linf_K=5., all_Y_Linf=.002, burned_deficit_relative_error=.01,
                      energy_closure_error=.02, mass_flux_relative_spread=1e-5,
                      omega_C_Linf_over_reference_peak=.02, qdot_Linf_over_reference_peak=.02)
    ref_pass = all(ref_metrics[k] <= v for k, v in ref_limits.items())
    ct_mass, ct_q = cantera_sources(model.metadata['mechanism'], native['T'], native['Y'], meta['pressure'])
    ct_C = t['progress_weights'] @ ct_mass
    expected = flux * float(t['progress_weights'] @ (native['Y'][:,-1] - np.asarray(meta['inlet_Y'])))
    chemistry = dict(omega_C_same_state_relative_error=float(np.max(abs(omega-ct_C))/np.max(abs(ct_C))),
                     qdot_same_state_relative_error=float(np.max(abs(qdot-ct_q))/np.max(abs(ct_q))),
                     mass_source_sum_over_peak=float(np.max(abs(mass_sources.sum(axis=0)))/np.max(abs(mass_sources))),
                     Z_source_over_mass_peak=float(np.max(abs(t['bilger_weights']@mass_sources))/np.max(abs(mass_sources))),
                     integrated_progress_balance_relative_error=abs(float(np.trapezoid(omega,native['z']))-expected)/abs(expected))
    chemistry_limits = dict(omega_C_same_state_relative_error=1e-9, qdot_same_state_relative_error=1e-9,
                            mass_source_sum_over_peak=1e-10, Z_source_over_mass_peak=1e-10,
                            integrated_progress_balance_relative_error=.005)
    chemistry_pass = all(chemistry[k] < v for k,v in chemistry_limits.items())
    conservation = dict(zip(('Z_Linf', 'C_Linf', 'h_Linf_J_kg', 'sum_Y_error'),
                            np.max(control_errors, axis=0).tolist()))
    conservation_pass = (conservation['Z_Linf'] < 1e-10 and conservation['C_Linf'] < 1e-10
                         and conservation['h_Linf_J_kg'] < 1e-4 and conservation['sum_Y_error'] < 1e-10)
    # An uncovered inlet is counted in coverage, never replaced by a clipped state.
    record = dict(phi=float(phi), fraction=float(fraction), mass_flux_kg_m2_s=float(flux),
                  native_refinement=meta['refinement'],
                  native_nodes=n, reference_nodes=len(reference['z']), resolved_nodes=int(covered.sum()),
                  interpolation=metrics, interpolation_limits=limits, interpolation_passed=interpolation_pass,
                  sources=sources, source_limits=source_limits, sources_passed=source_pass,
                  interpolation_tolerances=tolerances.to_dict(), interpolation_error_ratios=assessment['ratios'],
                  chemistry=chemistry, chemistry_limits=chemistry_limits, chemistry_passed=chemistry_pass,
                  reference=ref_metrics, reference_limits=ref_limits, reference_passed=ref_pass,
                  conservation=conservation, conservation_passed=conservation_pass,
                  heat_loss=meta['heat_loss'], passed=interpolation_pass and ref_pass and conservation_pass and chemistry_pass)
    arrays = dict(z=native['z'], T=native['T'], Y=native['Y'], Z=Z, C=C, h=h,
                  omega_C=omega, qdot=qdot, covered=covered,
                  **{'fgm_' + k: v for k, v in prediction.items()},
                  **{'reference_' + k: v for k, v in reference.items()})
    return record, arrays


def spatial_refinement(output):
    path = output / 'case_01_native'
    coarse = read_profile(path / 'flame.npz')
    meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    fine_path = output / 'mesh_fine'
    if not fine_path.exists():
        settings = meta['refinement']
        solve_burner_flame(mechanism=meta['mechanism'], mass_flux=meta['mass_flux'],
                           Y=dict(zip(meta['species_names'], meta['inlet_Y'])),
                           temperature=meta['temperature'], pressure=meta['pressure'],
                           transport=meta['transport'], soret=meta['soret'],
                           width=meta['initial_width'], initial_solution=path,
                           ratio=settings['ratio'], slope=.5*settings['slope'],
                           curve=.5*settings['curve'], prune=.5*settings['prune'],
                           max_points=settings['max_points'], max_time=240., output=fine_path)
    fine = read_profile(fine_path / 'flame.npz')
    fine_meta = json.loads((fine_path / 'metadata.json').read_text(encoding='utf-8'))
    if not fine_meta['accepted']:
        raise RuntimeError('Refined mesh solution failed acceptance')
    coarse_loss, fine_loss = meta['heat_loss'], fine_meta['heat_loss']
    Tmax_change = abs(float(np.max(fine['T'])) - float(np.max(coarse['T'])))
    flux_change = abs(fine_loss['burner_heat_loss_W_m2'] - coarse_loss['burner_heat_loss_W_m2']) / abs(coarse_loss['burner_heat_loss_W_m2'])
    ultra_path = output / 'mesh_ultra'
    if not ultra_path.exists():
        settings = fine_meta['refinement']
        solve_burner_flame(mechanism=meta['mechanism'], mass_flux=meta['mass_flux'],
                           Y=dict(zip(meta['species_names'], meta['inlet_Y'])),
                           temperature=meta['temperature'], pressure=meta['pressure'],
                           transport=meta['transport'], soret=meta['soret'], width=meta['initial_width'],
                           initial_solution=fine_path, ratio=settings['ratio'], slope=.5*settings['slope'],
                           curve=.5*settings['curve'], prune=.5*settings['prune'],
                           max_points=settings['max_points'], max_time=240., output=ultra_path)
    ultra = read_profile(ultra_path / 'flame.npz')
    ultra_meta = json.loads((ultra_path / 'metadata.json').read_text(encoding='utf-8'))
    if not ultra_meta['accepted']:
        raise RuntimeError('Third spatial grid failed acceptance')
    ultra_loss = ultra_meta['heat_loss']
    fine_ultra_T = abs(float(np.max(ultra['T'])) - float(np.max(fine['T'])))
    fine_ultra_flux = abs(ultra_loss['burner_heat_loss_W_m2'] - fine_loss['burner_heat_loss_W_m2']) / abs(fine_loss['burner_heat_loss_W_m2'])
    return dict(case_index=1, coarse_nodes=len(coarse['z']), fine_nodes=len(fine['z']),
                ultra_nodes=len(ultra['z']),
                Tmax_change_K=Tmax_change, relative_heat_flux_change=flux_change,
                fine_to_ultra_Tmax_change_K=fine_ultra_T, fine_to_ultra_relative_heat_flux_change=fine_ultra_flux,
                coarse_energy_closure=coarse_loss['relative_energy_closure_error'],
                fine_energy_closure=fine_loss['relative_energy_closure_error'],
                ultra_energy_closure=ultra_loss['relative_energy_closure_error'],
                limits=dict(Tmax_change_K_max=5., relative_heat_flux_change_max=.01, energy_closure_max=.02),
                note='Check changes in Tmax and heat flux on three adaptive grids and the absolute closure tolerance. '
                     'The boundary closure estimate need not decrease monotonically when points are redistributed.',
                passed=(Tmax_change < 5. and flux_change < .01 and fine_ultra_T < 5. and fine_ultra_flux < .01
                        and max(coarse_loss['relative_energy_closure_error'], fine_loss['relative_energy_closure_error'],
                                ultra_loss['relative_energy_closure_error']) < .02))


def validate(folder, output, *, recheck=False, mesh_check=False, fine_native=False,
             withheld_phis=WITHHELD_PHIS, withheld_fractions=WITHHELD_FRACTIONS, tolerances=None):
    import cantera as ct
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = NonAdiabaticFGM(folder)
    family = model.metadata
    withheld, fractions = np.asarray(withheld_phis), np.asarray(withheld_fractions)
    if (withheld.ndim != 1 or fractions.shape != withheld.shape or not withheld.size
            or not np.isfinite(withheld).all() or not np.isfinite(fractions).all()
            or np.any(withheld <= 0.) or np.any(fractions <= 0.) or np.any(fractions >= 1.)):
        raise ValueError('Withheld compositions/fractions must be matched finite positive arrays')
    if any(np.any(np.isclose(phi, family['phis'])) for phi in withheld):
        raise ValueError('Validation compositions must not appear in training')
    if any(np.any(np.isclose(r, family['mass_flux_fractions'])) for r in fractions):
        raise ValueError('Validation mass-flux fractions must not appear in training')
    mech = load_mechanism(family['mechanism'])
    thermo = NativeThermo(mech)
    reference_meta = json.loads((folder / family['rows'][0]['output'] / 'metadata.json').read_text(encoding='utf-8'))
    settings = dict(mechanism=family['mechanism'], fuel=family['fuel'], oxidizer=family['oxidizer'],
                    temperature=family['temperature_K'], pressure=family['pressure_Pa'],
                    transport=family['transport'], soret=family['soret'],
                    width=reference_meta['initial_width'], initial_points=24, max_time=240.,
                    **reference_meta['refinement'])
    # max_points is accepted by solve_flame as a keyword.
    cases = []
    for i, (phi, fraction) in enumerate(zip(withheld, fractions)):
        print(f'Withheld phi={phi:.6g}, mass-flux fraction={fraction:g}', flush=True)
        native_path = output / f'case_{i:02d}_native'
        if not recheck and not native_path.exists():
            ad_path = output / f'case_{i:02d}_adiabatic'
            if not ad_path.exists():
                solve_flame(phi=float(phi), output=ad_path, **settings)
            ad_meta = json.loads((ad_path / 'metadata.json').read_text(encoding='utf-8'))
            if not ad_meta['accepted']:
                raise RuntimeError('Withheld adiabatic reference was not accepted')
            ad_flux = float(thermo.density(family['temperature_K'], family['pressure_Pa'],
                                         np.asarray(ad_meta['inlet_Y']))) * ad_meta['Su']
            seed_path = output / f'case_{i:02d}_warm'
            if not seed_path.exists():
                solve_burner_flame(phi=float(phi), mass_flux=.65 * ad_flux,
                                  output=seed_path, **settings)
            for r in family['mass_flux_fractions']:
                if not fraction < r < .65:
                    continue
                step = output / f'case_{i:02d}_continuation_{r:g}'
                if not step.exists():
                    solve_burner_flame(phi=float(phi), mass_flux=float(r * ad_flux),
                                       initial_solution=seed_path, output=step, **settings)
                seed_path = step
            solve_burner_flame(phi=float(phi), mass_flux=float(fraction * ad_flux),
                               initial_solution=seed_path, output=native_path, **settings)
        meta = json.loads((native_path / 'metadata.json').read_text(encoding='utf-8'))
        if not meta['accepted']:
            raise RuntimeError('Withheld native flame was not accepted')
        if not np.allclose(meta['inlet_Y'], fresh_mixture(mech, float(phi), family['fuel'], family['oxidizer']),
                           rtol=1e-12, atol=1e-15):
            raise ValueError('Cached withheld flame has a different composition; choose a new output directory')
        if fine_native:
            fine_path = output / f'case_{i:02d}_native_fine'
            if not fine_path.exists():
                refine = meta['refinement']
                solve_burner_flame(phi=float(phi), mass_flux=meta['mass_flux'], initial_solution=native_path,
                                   output=fine_path, **dict(settings, slope=.5*refine['slope'],
                                                           curve=.5*refine['curve'], prune=.5*refine['prune']))
            native_path = fine_path
            meta = json.loads((native_path / 'metadata.json').read_text(encoding='utf-8'))
            if not meta['accepted']:
                raise RuntimeError('Refined withheld native flame was not accepted')
        native = read_profile(native_path / 'flame.npz')
        suffix = '_fine' if fine_native else ''
        ref_path = output / f'case_{i:02d}_reference{suffix}.npz'
        reference = read_profile(ref_path) if ref_path.exists() else {}
        if str(reference.get('input_digest', '')) != reference_input_hash(meta, meta['mass_flux']):
            reference = cantera_reference(meta, meta['mass_flux'], model.table['progress_weights'])
            np.savez_compressed(ref_path, **reference)
        reference = read_profile(ref_path)
        result, arrays = evaluate_case(model, native, reference, meta, phi, fraction, meta['mass_flux'], tolerances)
        np.savez_compressed(output / f'case_{i:02d}_comparison.npz', **arrays)
        columns = ['z', 'T', 'fgm_T', 'Z', 'C', 'h', 'omega_C', 'fgm_omega_C', 'qdot', 'fgm_qdot', 'covered']
        with (output / f'case_{i:02d}_comparison.csv').open('w', encoding='utf-8', newline='\n') as stream:
            np.savetxt(stream, np.column_stack([arrays[k] for k in columns]),
                       delimiter=',', header=','.join(columns), comments='')
        cases.append(result)
        print(json.dumps(result), flush=True)
    root = Path(__file__).resolve().parents[1]
    mesh_result = spatial_refinement(output) if mesh_check else None
    report = dict(
        reference='Independent Cantera BurnerFlame', reference_version=ct.__version__,
        training_phis=family['phis'], training_mass_flux_fractions=family['mass_flux_fractions'],
        training_flames=len(family['rows']), mesh=family['mesh'],
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        table_sha256=hashlib.sha256((folder / 'nonadiabatic_fgm.npz').read_bytes()).hexdigest(),
        mechanism_sha256=hashlib.sha256(Path(reference_meta['mechanism']).read_bytes()).hexdigest(),
        cases=cases, spatial_refinement=mesh_result,
        passed=all(r['passed'] for r in cases) and (mesh_result is None or mesh_result['passed']),
        scope='A priori interpolation against withheld native detailed profiles, with an independent numerical '
              'Cantera comparison of those profiles. No reduced transport, CFD or experimental validation.',
        coverage_definition='Number of native spatial nodes inside adjacent valid tetrahedra divided by all '
                            'native spatial nodes. This is not a volume coverage measure.',
        source_units=dict(omega_C='kg/(m^3 s)', qdot='W/m^3'))
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recheck', action='store_true')
    parser.add_argument('--mesh-check', action='store_true')
    parser.add_argument('--fine-native', action='store_true')
    parser.add_argument('--tolerances', type=Path, help='JSON with interpolation tolerances per quantity')
    parser.add_argument('--phis', type=float, nargs='+', default=WITHHELD_PHIS.tolist())
    parser.add_argument('--fractions', type=float, nargs='+', default=WITHHELD_FRACTIONS)
    args = parser.parse_args()
    report = validate(args.folder, args.output, recheck=args.recheck, mesh_check=args.mesh_check,
                      fine_native=args.fine_native, withheld_phis=args.phis, withheld_fractions=args.fractions,
                      tolerances=FGMTolerances.from_file(args.tolerances) if args.tolerances else None)
    print('Passed:', report['passed'])
    if not report['passed']:
        raise SystemExit('Nonadiabatic 3D validation failed; inspect validation.json')


if __name__ == '__main__':
    main()
