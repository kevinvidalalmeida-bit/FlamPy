"""Compare native burner flames against independent Cantera solutions.

Run after example_heat_loss.py:
python examples/validate_burner.py runs/burner_example --h2
Cantera is imported only by this reference script, never by production.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare_case(profile, metadata, mass_flux):
    import cantera as ct
    gas = ct.Solution(metadata['mechanism'])
    gas.TPY = metadata['temperature'], metadata['pressure'], metadata['inlet_Y']
    h_feed = gas.enthalpy_mass
    flame = ct.BurnerFlame(gas, width=metadata['final_width'])
    flame.burner.mdot = mass_flux
    flame.transport_model = metadata['transport']
    flame.soret_enabled = metadata['soret']
    settings = metadata['refinement']
    flame.set_refine_criteria(**{name: settings[name] for name in ('ratio', 'slope', 'curve', 'prune')})
    flame.solve(loglevel=0, auto=True)
    gas.TPY = flame.T[-1], metadata['pressure'], flame.Y[:, -1]
    deficit_reference = h_feed - gas.enthalpy_mass
    major = [i for i in range(gas.n_species) if max(np.max(flame.Y[i]), np.max(profile['Y'][i])) > .001]
    temperature_error = float(np.max(abs(profile['T'] - np.interp(profile['z'], flame.grid, flame.T))))
    species_error = float(max(np.max(abs(profile['Y'][i] - np.interp(profile['z'], flame.grid, flame.Y[i]))) for i in major))
    native = metadata['heat_loss']
    errors = dict(temperature_Linf_K=temperature_error, major_Y_Linf=species_error,
                  burned_deficit_relative_error=abs(native['burned_enthalpy_deficit_J_kg'] - deficit_reference) / abs(deficit_reference),
                  relative_energy_closure_error=native['relative_energy_closure_error'],
                  mass_flux_relative_spread=native['mass_flux_relative_spread'])
    limits = dict(temperature_Linf_K=5., major_Y_Linf=.002,
                  burned_deficit_relative_error=.01, relative_energy_closure_error=.02,
                  mass_flux_relative_spread=1e-5)
    return dict(mass_flux_kg_m2_s=mass_flux, transport=metadata['transport'], soret=metadata['soret'],
                native_nodes=len(profile['z']), reference_nodes=len(flame.grid),
                native_Tmax_K=float(np.max(profile['T'])), reference_Tmax_K=float(np.max(flame.T)),
                native_burned_deficit_J_kg=native['burned_enthalpy_deficit_J_kg'],
                reference_burned_deficit_J_kg=deficit_reference,
                native_burner_heat_loss_W_m2=native['burner_heat_loss_W_m2'],
                errors=errors, limits=limits, passed=all(errors[k] <= limits[k] for k in limits))


def solve_matching_burner(metadata, mass_flux, *, finer=False):
    from kflame import solve_burner_flame
    refinement = metadata['refinement']
    factor = .5 if finer else 1.
    names = metadata['species_names']
    return solve_burner_flame(
        mechanism=metadata['mechanism'], Y=dict(zip(names, metadata['inlet_Y'])),
        mass_flux=mass_flux, temperature=metadata['temperature'], pressure=metadata['pressure'],
        transport=metadata['transport'], soret=metadata['soret'],
        width=metadata['initial_width'], initial_points=24, max_time=240.,
        ratio=refinement['ratio'], slope=factor * refinement['slope'], curve=factor * refinement['curve'],
        prune=factor * refinement['prune'], max_points=refinement['max_points'],
        species=tuple(name for name in ('CH4', 'H2', 'O2', 'H2O', 'CO2', 'OH') if name in names))


def main():
    from kflame import solve_burner_flame
    from kflame.fgm.nonadiabatic import BurnerFGM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--h2', action='store_true', help='Also solve H2-air with multicomponent transport and Soret')
    parser.add_argument('--withheld-mass-flux', type=float, default=None,
                        help='Validate lookup against an additional burner not in the table')
    parser.add_argument('--mesh-check', action='store_true', help='Repeat the middle burner on a finer mesh')
    args = parser.parse_args()
    family = json.loads((args.folder / 'metadata.json').read_text(encoding='utf-8'))
    cases = []
    for row in family['rows'][1:]:
        print(f"Comparing burner mass flux {row['mass_flux_kg_m2_s']:g} kg/(m2 s)", flush=True)
        path = args.folder / row['output']
        with np.load(path / 'flame.npz', allow_pickle=False) as profile:
            meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
            result = compare_case(profile, meta, row['mass_flux_kg_m2_s'])
            result['fuel_case'] = 'fixed_inlet_family'
            cases.append(result)
    if args.h2:
        print('Comparing H2-air with multicomponent transport and Soret', flush=True)
        h2 = solve_burner_flame(mechanism='h2o2.yaml', fuel='H2', phi=.7, mass_flux=.08,
                               transport='multicomponent', soret=True, initial_points=24,
                               max_time=240., species=('H2', 'O2', 'H2O', 'OH'))
        meta = json.loads((h2['output'] / 'metadata.json').read_text(encoding='utf-8'))
        result = compare_case(h2, meta, .08)
        result['fuel_case'] = 'H2_air_phi_0.7'
        cases.append(result)
    lookup_check = None
    if args.withheld_mass_flux is not None:
        flux = args.withheld_mass_flux
        if (not np.isfinite(flux) or flux <= 0
                or any(np.isclose(flux, row['mass_flux_kg_m2_s']) for row in family['rows'][1:])):
            raise SystemExit('Withheld mass flux must be positive and absent from the family')
        model = BurnerFGM(args.folder)
        reference_meta = json.loads((args.folder / 'adiabatic' / 'metadata.json').read_text(encoding='utf-8'))
        print(f'Checking withheld mass flux {flux:g} kg/(m2 s)', flush=True)
        native = solve_matching_burner(reference_meta, flux)
        meta = json.loads((native['output'] / 'metadata.json').read_text(encoding='utf-8'))
        reference_check = compare_case(native, meta, flux)
        reference_check['fuel_case'] = 'withheld_fixed_inlet_family'
        cases.append(reference_check)
        table = model.table
        progress = (table['progress_weights'] @ native['Y'] - table['beta_unburned']) / table['beta_span']
        differences = []
        for j, (c, h) in enumerate(zip(progress, native['h_mass'])):
            try:
                prediction = model.lookup(c=c, h=h)
            except ValueError:
                continue
            differences.append([abs(prediction['T'] - native['T'][j]),
                                np.max(abs(prediction['Y'] - native['Y'][:, j])),
                                abs(prediction['qdot'] - native['qdot'][j])])
        values = np.asarray(differences)
        if not len(values):
            raise SystemExit('No withheld states fall inside the resolved table')
        metrics = dict(coverage_fraction=len(values) / len(progress),
                       temperature_Linf_K=float(np.max(values[:, 0])),
                       all_Y_Linf=float(np.max(values[:, 1])),
                       qdot_Linf_over_native_peak=float(np.max(values[:, 2]) / np.max(native['qdot'])))
        limits = dict(coverage_fraction_min=.8, temperature_Linf_K_max=5.,
                      all_Y_Linf_max=.002, qdot_Linf_over_native_peak_max=.08)
        passed = (metrics['coverage_fraction'] >= .8 and metrics['temperature_Linf_K'] <= 5.
                  and metrics['all_Y_Linf'] <= .002 and metrics['qdot_Linf_over_native_peak'] <= .08)
        lookup_check = dict(mass_flux_kg_m2_s=flux, resolved_states=len(values), native_states=len(progress),
                            metrics=metrics, limits=limits, passed=passed,
                            note='Errors apply only inside the resolved manifold; unsupported inlet states are counted as uncovered.')
    mesh_check = None
    if args.mesh_check:
        row = family['rows'][1:][len(family['rows'][1:]) // 2]
        path = args.folder / row['output']
        metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        print('Checking finer spatial mesh', flush=True)
        fine = solve_matching_burner(metadata, row['mass_flux_kg_m2_s'], finer=True)
        with np.load(path / 'flame.npz') as coarse:
            delta_T = abs(float(np.max(fine['T'])) - float(np.max(coarse['T'])))
        coarse_loss, fine_loss = metadata['heat_loss'], fine['heat_loss']
        relative_heat_flux_change = abs(fine_loss['burner_heat_loss_W_m2'] - coarse_loss['burner_heat_loss_W_m2']) / abs(coarse_loss['burner_heat_loss_W_m2'])
        passed = (fine_loss['relative_energy_closure_error'] < coarse_loss['relative_energy_closure_error']
                  and delta_T < 5. and relative_heat_flux_change < .01)
        mesh_check = dict(mass_flux_kg_m2_s=row['mass_flux_kg_m2_s'],
                          coarse_nodes=metadata['nodes'], fine_nodes=len(fine['z']),
                          coarse_energy_closure=coarse_loss['relative_energy_closure_error'],
                          fine_energy_closure=fine_loss['relative_energy_closure_error'],
                          Tmax_change_K=delta_T, relative_heat_flux_change=relative_heat_flux_change,
                          passed=passed)
    # Absolute paths and user-specific directories are omitted from the report.
    root = Path(__file__).resolve().parents[1]
    sources = ['src/kflame/api.py', 'src/kflame/flame/config.py', 'src/kflame/flame/problem.py',
               'src/kflame/flame/equations.py', 'src/kflame/flame/analytic_jacobian.py',
               'src/kflame/flame/solver.py', 'src/kflame/flame/enthalpy.py',
               'src/kflame/chemistry/thermo.py', 'src/kflame/fgm/nonadiabatic.py']
    import cantera as ct
    report = dict(reference='Cantera BurnerFlame', reference_version=ct.__version__,
                  native_source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
                  cases=cases, held_out_lookup=lookup_check, mesh_refinement_check=mesh_check,
                  passed=(all(case['passed'] for case in cases)
                          and (lookup_check is None or lookup_check['passed'])
                          and (mesh_check is None or mesh_check['passed'])),
                  scope='Independent numerical comparison using the same chemistry and transport; no experimental validation.')
    (args.folder / 'reference_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], cases=len(cases), held_out_lookup=lookup_check,
                          mesh_refinement_check=mesh_check), indent=2))
    if not report['passed']:
        raise SystemExit('Burner reference validation failed')


if __name__ == '__main__':
    main()
