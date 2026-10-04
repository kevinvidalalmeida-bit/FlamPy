"""User-facing CPU flame and FGM entry points; solver tuning stays internal."""
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from kflame.chemistry.initialization import _composition, fresh_mixture
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame.config import FlameCase


def _stream(value):
    return ', '.join(f'{name}:{amount}' for name, amount in value.items()) if isinstance(value, dict) else value


def _settings(mechanism, temperature, pressure, width, transport, soret,
              initial_points, ratio, slope, curve, prune, max_points, max_time, verbose):
    if transport not in ('mixture-averaged', 'multicomponent'):
        raise ValueError("transport must be 'mixture-averaged' or 'multicomponent'")
    for name, value in dict(temperature=temperature, pressure=pressure, width=width, max_time=max_time).items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive (SI units)')
    if not isinstance(initial_points, int) or not isinstance(max_points, int) or not 3 <= initial_points <= max_points:
        raise ValueError('Require integer 3 <= initial_points <= max_points')
    if not (np.isfinite(ratio) and ratio >= 2 and 0 < slope <= 1 and 0 < curve <= 1
            and 0 <= prune < min(slope, curve)):
        raise ValueError('Require ratio >= 2, 0 < slope/curve <= 1, 0 <= prune < min(slope, curve)')
    return [
        '--mech', resolve_mechanism(str(mechanism)), '--T-in', str(temperature),
        '--P', str(pressure), '--width', str(width), '--transport-model', transport,
        '--initial-grid-points', str(initial_points), '--ratio', str(ratio),
        '--slope', str(slope), '--curve', str(curve), '--prune', str(prune),
        '--max-grid-points', str(max_points), '--max-flame-time-s', str(max_time),
        *(['--loglevel', '1'] if verbose else []),
        *(['--soret-enabled'] if soret else []),
        *(['--multicomponent-bootstrap'] if transport == 'multicomponent' or soret else []),
    ]


def _output(path, label):
    folder = Path(path) if path is not None else Path('runs') / label / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    return folder.resolve()


def solve_flame(*, mechanism='gri30.yaml', temperature=300.0, pressure=101325.0,
                phi=None, fuel='CH4', oxidizer='O2:1, N2:3.76', X=None, Y=None,
                diluent=None, dilution=0.0, width=0.03, grid=None, initial_points=8,
                transport='mixture-averaged', soret=False, ratio=2.5, slope=0.04,
                curve=0.08, prune=0.003, max_points=1600, rtol=1e-4, atol=1e-9,
                max_time=180.0, output=None, plots=False, verbose=False,
                species=('CH4', 'O2', 'CO2', 'H2O', 'OH'), mass_flux=None,
                initial_solution=None):
    """Solve an adiabatic premixed free flame on the native CPU backend.

    All arguments are keyword-only. Temperature, pressure and width use SI
    units (K, Pa and m). Specify at most one of phi with fuel/oxidizer, X
    (mole amounts), or Y (mass amounts); the default mixture has phi=1.
    Compositions accept strings or dictionaries and are normalized. Dilution
    is the final mole fraction of the diluent in the fresh mixture.

    Grid optionally supplies increasing nodes from zero to width; spatial
    adaptation remains active. Both transport models support soret=True.
    Internal Jacobian and pseudotransient settings use production defaults.

    Return a dictionary with the profiles, Su in m/s, accepted, report,
    runtime_s and output (a Path). Y has shape (species, nodes). The complete
    mechanism is retained in flame.npz; species selects CSV and plot curves.
    Always save NPZ, CSV and metadata; plots=True adds PNG and PDF figures
    and requires Matplotlib. Output must not already exist; None creates a
    dated directory in runs/flame. A rejected solve saves diagnostics before
    raising RuntimeError. See docs/api.md for all parameters and defaults.

    A positive mass_flux [kg/(m^2 s)] selects an isothermal burner instead;
    solve_burner_flame provides the explicit entry point for that mode.
    initial_solution may reference an accepted native burner output directory
    with the same feed, mechanism, pressure and surface temperature. The state
    is a continuation guess; the new flame is always solved and certified.
    """
    from kflame.chemistry.backend import NativeSpeciesBackend
    from kflame.fgm.generate import build_argparser, make_solve_options, tabulated_properties
    from kflame.flame.problem import FreeFlameProblem
    from kflame.flame.solver import solve_free_flame
    from kflame.flame.state import unpack_state
    from kflame.serialization import json_safe

    if mass_flux is not None and (not np.isfinite(mass_flux) or mass_flux <= 0):
        raise ValueError('mass_flux must be finite and positive [kg/(m^2 s)]')
    argv = _settings(mechanism, temperature, pressure, width, transport, soret,
                     initial_points, ratio, slope, curve, prune, max_points, max_time, verbose)
    args = build_argparser().parse_args(argv)
    mech = load_mechanism(args.mech)
    if sum(value is not None for value in (phi, X, Y)) > 1:
        raise ValueError('Specify only one composition: phi, X, or Y')
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0 or atol <= 0:
        raise ValueError('rtol and atol must be finite and positive')
    if not 0 <= dilution < 1 or (diluent is None and dilution != 0):
        raise ValueError('Specify a diluent and 0 <= dilution < 1 (final mole fraction)')
    inlet = None
    if X is not None or Y is not None:
        inlet = _composition(_stream(X if X is not None else Y), mech.species_names)
        if X is not None:
            inlet *= mech.molecular_weights
        inlet /= inlet.sum()
    if dilution:
        if inlet is None:
            inlet = fresh_mixture(mech, 1.0 if phi is None else phi, _stream(fuel), _stream(oxidizer))
        mole = inlet / mech.molecular_weights
        mole /= mole.sum()
        extra = _composition(_stream(diluent), mech.species_names)
        mole = (1 - dilution) * mole + dilution * extra / extra.sum()
        inlet = mole * mech.molecular_weights
        inlet /= inlet.sum()
    seed = None
    if initial_solution is not None:
        if mass_flux is None:
            raise ValueError('initial_solution continuation is supported for burner flames only')
        seed_path = Path(initial_solution)
        seed_meta = json.loads((seed_path / 'metadata.json').read_text(encoding='utf-8'))
        if (not seed_meta.get('accepted') or seed_meta.get('backend') != 'native_cpu'
                or seed_meta.get('flow_type') != 'isothermal_burner'):
            raise ValueError('initial_solution must be an accepted native burner flame')
        if Path(resolve_mechanism(seed_meta['mechanism'])).read_bytes() != Path(args.mech).read_bytes():
            raise ValueError('Continuation requires the same mechanism')
        current_inlet = inlet if inlet is not None else fresh_mixture(mech, 1. if phi is None else phi, _stream(fuel), _stream(oxidizer))
        if (not np.isclose(seed_meta['temperature'], temperature, rtol=1e-12)
                or not np.isclose(seed_meta['pressure'], pressure, rtol=1e-12)
                or not np.allclose(seed_meta['inlet_Y'], current_inlet, rtol=1e-12, atol=1e-15)):
            raise ValueError('Continuation requires the same feed, pressure and burner temperature')
        with np.load(seed_path / 'flame.npz', allow_pickle=False) as saved:
            seed = {key: saved[key].copy() for key in ('z', 'T', 'Y', 'species_names')}
        if list(seed['species_names']) != list(mech.species_names):
            raise ValueError('Continuation mechanism species order differs')
        if (seed['z'].ndim != 1 or len(seed['z']) < 3 or seed['z'][0] != 0.
                or np.any(np.diff(seed['z']) <= 0.) or seed['T'].shape != seed['z'].shape
                or seed['Y'].shape != (mech.n_species, len(seed['z']))
                or any(not np.isfinite(seed[k]).all() for k in ('z', 'T', 'Y'))):
            raise ValueError('Continuation profile must contain consistent finite T,Y and increasing z')
        if grid is None and np.isclose(seed['z'][-1], width, rtol=1e-12):
            grid = seed['z']
    if grid is not None:
        grid = np.asarray(grid, dtype=float)
        if (grid.ndim != 1 or not 3 <= grid.size <= max_points or not np.isfinite(grid).all()
                or grid[0] != 0 or not np.isclose(grid[-1], width, rtol=1e-12, atol=0)
                or not np.all(np.diff(grid) > 0)):
            raise ValueError('grid must have 3..max_points increasing finite nodes, from 0 to width')
    species = tuple(species)
    missing = set(species) - set(mech.species_names)
    if missing:
        raise ValueError(f'Output species absent from mechanism: {sorted(missing)}')
    if plots:
        import matplotlib.pyplot as plt
    case = FlameCase(
        mech=args.mech, T_in=temperature, P=pressure, width=width,
        phi=1.0 if phi is None else phi, fuel=_stream(fuel), oxidizer=_stream(oxidizer),
        inlet_mass_fractions=None if inlet is None else tuple(inlet),
        initial_grid=None if grid is None else tuple(grid), steady_rtol=rtol, steady_atol=atol,
        inlet_mass_flux=mass_flux, cantera_seed_grid=mass_flux is None,
        transport_model=transport, soret_enabled=soret, ratio=ratio, slope=slope, curve=curve, prune=prune,
    )
    folder = _output(output, 'burner' if mass_flux is not None else 'flame')
    started = time.perf_counter()
    problem = FreeFlameProblem(case, n_points=initial_points, mech_data=mech)
    problem.assume_finite_y = True
    problem.backend_factory = NativeSpeciesBackend
    options = replace(make_solve_options(args), refine_ratio=ratio, refine_slope=slope,
                      refine_curve=curve, refine_prune=prune, refine_max_points=max_points)
    initial_state = None
    if seed is not None:
        from kflame.flame.state import pack_state
        seed_T = np.interp(problem.z, seed['z'], seed['T'])
        seed_Y = problem._sanitize_Y(np.array([np.interp(problem.z, seed['z'], row) for row in seed['Y']]))
        seed_T[0] = temperature
        seed_u = float(mass_flux) / problem._thermo.density(seed_T, pressure, seed_Y)
        initial_state = pack_state(seed_u, seed_T, seed_Y)
    state, ok, report = solve_free_flame(problem, x0=initial_state, options=options)
    elapsed = time.perf_counter() - started
    u, T, y = unpack_state(state, problem.n_points, problem.n_species)
    rho, cp, conductivity, qdot, _ = tabulated_properties(problem, T, y, {})
    fields = dict(z=problem.z, T=T, u=u, Y=y, rho=rho, cp_mass=cp,
                  conductivity=conductivity, qdot=qdot, species_names=np.asarray(problem.species_names))
    heat_loss = None
    if problem.is_burner:
        from kflame.flame.enthalpy import enthalpy_diagnostics
        enthalpy_fields, heat_loss = enthalpy_diagnostics(problem, state)
        fields.update(enthalpy_fields)
    np.savez_compressed(folder / 'flame.npz', **fields)
    indices = [problem.species_names.index(name) for name in species]
    extra_columns = [fields['h_mass'], fields['enthalpy_departure_from_feed']] if problem.is_burner else []
    extra_headers = ['h_total_J_kg', 'h_feed_minus_h_J_kg'] if problem.is_burner else []
    np.savetxt(folder / 'profiles.csv', np.column_stack([problem.z, T, u, rho, cp, conductivity, qdot, *extra_columns, *y[indices]]),
               delimiter=',', header=','.join(['z_m', 'T_K', 'u_m_s', 'rho_kg_m3', 'cp_J_kg_K',
                                             'conductivity_W_m_K', 'qdot_W_m3', *extra_headers, *['Y_' + name for name in species]]), comments='')
    accepted = bool(ok and report.get('final_accepted') and report.get('grid_converged'))
    if problem.is_burner:
        # The cold, nonreacting branch also solves the steady equations.
        # This is a branch diagnostic, not a physical extinction threshold.
        reacting = bool(np.max(T) - temperature > 50.0 and np.max(qdot) > 1.0)
        accepted = accepted and reacting
        report['reacting_branch'] = reacting
    velocity = {'inlet_velocity': float(u[0]), 'mass_flux': float(mass_flux)} if problem.is_burner else {'Su': float(u[0])}
    metadata = dict(software='FlamPy', backend='native_cpu', mechanism=args.mech,
                    species_names=problem.species_names, temperature=temperature,
                    pressure=pressure, transport=transport, soret=soret, inlet_Y=problem.Y_in,
                    initial_grid=case.initial_grid, initial_points=initial_points,
                    refinement=dict(ratio=ratio, slope=slope, curve=curve, prune=prune, max_points=max_points),
                    tolerances=dict(rtol=rtol, atol=atol),
                    initial_width=width, final_width=problem.width, nodes=problem.n_points,
                    **velocity, flow_type='isothermal_burner' if problem.is_burner else 'free_flame',
                    heat_loss=heat_loss, runtime_s=elapsed, accepted=accepted, report=report)
    if initial_solution is not None:
        metadata['continuation_from'] = str(Path(initial_solution).resolve())
    (folder / 'metadata.json').write_text(json.dumps(json_safe(metadata), indent=2), encoding='utf-8')
    if not accepted:
        raise RuntimeError(f'Flame failed acceptance or mesh convergence; diagnostics: {folder}')
    if plots:
        with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'cm', 'xtick.direction': 'in', 'ytick.direction': 'in'}):
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            mm = 1000 * problem.z
            axes[0, 0].plot(mm, T)
            axes[0, 0].set_ylabel('Temperature [K]')
            axes[0, 1].plot(mm, qdot)
            axes[0, 1].set_ylabel('Heat release [W/m³]')
            for name, index in zip(species, indices):
                axes[1, 0].plot(mm, y[index], label=name)
            axes[1, 0].set_ylabel('Mass fraction')
            if species:
                axes[1, 0].legend(fontsize=8)
            axes[1, 1].semilogy(.5 * (mm[:-1] + mm[1:]), np.diff(mm), '.-')
            axes[1, 1].set_ylabel('Adaptive cell spacing [mm]')
            for ax in axes.flat:
                ax.set_xlabel('z [mm]')
                ax.grid(alpha=.2)
            for extension in ('png', 'pdf'):
                fig.savefig(folder / f'profiles.{extension}', dpi=200)
            plt.close(fig)
    return dict(**fields, **velocity, heat_loss=heat_loss, accepted=accepted, output=folder, report=report, runtime_s=elapsed)


def solve_burner_flame(*, mass_flux, **kwargs):
    """Solve a premixed flame conducting heat towards an isothermal burner.

    mass_flux is the imposed feed flux in kg/(m^2 s). Other keyword arguments
    match solve_flame; temperature is the burner surface and feed temperature.
    The model has T(0)=temperature, a prescribed inlet mass flux, diffusive
    species inlet conditions, and zero temperature/species outlet gradients.
    No free-flame phase condition or artificial chemical heat scaling is used.

    Return inlet_velocity (not Su), total enthalpy profiles and heat_loss
    diagnostics, in addition to the usual fields. This planar gas model does
    not solve conduction inside the solid or multidimensional wall quenching.
    """
    if mass_flux is None:
        raise ValueError('mass_flux is required for a burner flame')
    return solve_flame(mass_flux=mass_flux, **kwargs)


def generate_fgm(*, phis=(0.7, 0.9, 1.0, 1.1, 1.4), mechanism='gri30.yaml',
                 temperature=300.0, pressure=101325.0, fuel='CH4', oxidizer='O2:1, N2:3.76',
                 width=0.03, initial_points=8, transport='mixture-averaged', soret=False,
                 ratio=2.5, slope=0.04, curve=0.08, prune=0.003, max_points=1600,
                 max_time=180.0, output=None, plots=False, verbose=False, export=True,
                 species=('CO2', 'H2O'), progress_species='CO2:1.0,H2O:1.0,CO:1.0,H2:0.5',
                 progress_points=241, adaptive_phi=True, target_defect=0.01,
                 max_bridges_per_round=10, max_adaptive_rounds=64,
                 max_flamelets=256):
    """Build an accepted native FGM table on inlet composition Z and progress c.

    All arguments are keyword-only; physical inputs use SI units. Fuel and
    oxidizer specify mole-basis streams. Phis must be positive and strictly
    increasing. With adaptive_phi=True, at least three initial phis are
    required and new flames are solved at logarithmic midpoints until the
    leave-one-out defect reaches target_defect (0.01 means 1%). Execution
    limits raise an error if that target is not reached. A fixed family
    (adaptive_phi=False) needs at least two phis. Both modes adapt each flame
    spatially and redistribute the progress grid.

    Return the output Path after checking flame acceptance and table structure.
    Save the full NPZ table, raw profiles, metadata and continuation traces.
    Export=True adds all-species .fla and CSV files. Plots=True requires
    Matplotlib and exactly two valid species. For H2 with h2o2.yaml, change
    fuel, progress_species and plot species together; see docs/api.md.

    Output must not already exist; None creates a dated directory in runs/fgm.
    This API uses one process and disables persistent seed caching. The
    progress weights and refinement defect require physical review for each
    new family; interpolation consistency is not experimental validation.
    """
    from kflame.fgm.generate import main
    argv = _settings(mechanism, temperature, pressure, width, transport, soret,
                     initial_points, ratio, slope, curve, prune, max_points, max_time, verbose)
    if not isinstance(progress_species, str) or not progress_species.strip():
        raise ValueError('progress_species must be a non-empty weighted species string')
    if not isinstance(progress_points, int) or progress_points < 2:
        raise ValueError('progress_points must be an integer greater than one')
    phis = np.asarray(phis, dtype=float)
    if phis.ndim != 1 or phis.size < 2 or not np.isfinite(phis).all() or np.any(phis <= 0) or np.any(np.diff(phis) <= 0):
        raise ValueError('phis must contain at least two finite, positive, strictly increasing values')
    if adaptive_phi:
        if phis.size < 3:
            raise ValueError('adaptive_phi requires at least three initial phis')
        if not np.isfinite(target_defect) or target_defect <= 0:
            raise ValueError('target_defect must be finite and positive')
        if not all(isinstance(value, int) and value >= minimum for value, minimum in (
            (max_bridges_per_round, 1), (max_adaptive_rounds, 0), (max_flamelets, phis.size)
        )):
            raise ValueError('Invalid adaptive FGM refinement limits')
    mech = load_mechanism(resolve_mechanism(str(mechanism)))
    fresh_mixture(mech, float(phis[0]), _stream(fuel), _stream(oxidizer))
    if plots:
        if len(species) != 2 or set(species) - set(mech.species_names):
            raise ValueError('FGM plots require two species present in the mechanism')
        from kflame.fgm.plot import main as plot
    folder = _output(output, 'fgm')
    command = [*argv, '--fuel', _stream(fuel), '--oxidizer', _stream(oxidizer),
               '--phi-values', ','.join(map(str, phis)), '--save-raw-profiles',
               '--disable-seed-cache', '--parallel-workers', '1',
               '--progress-species', progress_species, '--n-c', str(progress_points),
               '--output-root', str(folder.parent), '--run-name', folder.name]
    if adaptive_phi:
        from kflame.fgm.adaptive import build_adaptive_fgm
        from kflame.fgm.generate import build_argparser
        build_adaptive_fgm(
            args=build_argparser().parse_args(command),
            initial_phis=phis,
            output_dir=folder,
            target_defect=target_defect,
            max_bridges_per_round=max_bridges_per_round,
            max_rounds=max_adaptive_rounds,
            max_flames=max_flamelets,
        )
    else:
        main(command)
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if not meta['all_final_accepted'] or not meta['table_validation']['valid']:
        raise RuntimeError(f'FGM failed acceptance or table validation; diagnostics: {folder}')
    if export:
        from kflame.fgm.export import main as export_table
        export_table(['--npz', str(folder / 'fgm_table.npz'), '--all-species', '--single-flamelets',
                      '--mech', str(mechanism), '--fuel', _stream(fuel), '--oxidizer', _stream(oxidizer),
                      '--pressure', str(pressure), '--T-in', str(temperature)])
    if plots:
        plot_args = ['--run-dir', str(folder), '--sp1', species[0], '--sp2', species[1]]
        plot(plot_args)
    return folder


def generate_burner_fgm(*, mass_fluxes=(.12, .08, .04),
                        progress_species='CO2:1,H2O:1,CO:1,H2:0.5',
                        progress_points=241, max_energy_error=.02, output=None,
                        **flame_settings):
    """Build a nonadiabatic (c,h) FGM for one fixed inlet composition.

    Solve an adiabatic reference and burner flames with imposed mass fluxes
    [kg/(m^2 s)]. flame_settings uses solve_flame keywords. All flamelets must
    pass solver/mesh/branch checks and the specified boundary energy closure.
    c uses one adiabatic normalization throughout the family; h includes
    formation enthalpies. delta_h=h_adiabatic(c)-h is stored as a diagnostic.

    Preserve unreachable progress ranges with a validity mask; no endpoint
    stretching, extrapolation, or free-flame Su labels for burner results.
    Return the output Path. Load it with kflame.fgm.nonadiabatic.BurnerFGM.
    This table has fixed feed composition; it is not a three-control (Z,c,h)
    manifold or a solver for conjugate heat transfer inside the burner.
    """
    from kflame.fgm.common import parse_progress_weights
    from kflame.fgm.nonadiabatic import tabulate_burner_family
    from kflame.serialization import json_safe

    fluxes = np.asarray(mass_fluxes, dtype=float)
    if (fluxes.ndim != 1 or fluxes.size < 1 or not np.isfinite(fluxes).all()
            or np.any(fluxes <= 0) or np.unique(fluxes).size != fluxes.size):
        raise ValueError('mass_fluxes must contain distinct finite positive fluxes')
    if not isinstance(progress_points, int) or progress_points < 3:
        raise ValueError('progress_points must be an integer >= 3')
    if not np.isfinite(max_energy_error) or not 0 < max_energy_error < 1:
        raise ValueError('max_energy_error must lie between zero and one')
    if 'mass_flux' in flame_settings:
        raise ValueError('Use mass_fluxes to specify the burner family')
    mech_name = str(flame_settings.get('mechanism', 'gri30.yaml'))
    mech = load_mechanism(resolve_mechanism(mech_name))
    parsed = parse_progress_weights(progress_species)
    if not parsed or set(parsed) - set(mech.species_names) or not np.isfinite(list(parsed.values())).all():
        raise ValueError('Progress weights must be finite and refer to mechanism species')
    weights = np.array([parsed.get(name, 0.) for name in mech.species_names])
    if not np.any(weights):
        raise ValueError('Progress weights cannot all be zero')
    folder = _output(output, 'burner_fgm')
    pressure = float(flame_settings.get('pressure', 101325.))
    reference = solve_flame(output=folder / 'adiabatic', **flame_settings)
    reference['pressure'] = pressure
    reference_meta = json.loads((reference['output'] / 'metadata.json').read_text(encoding='utf-8'))
    inlet_Y = np.array(reference_meta['inlet_Y'])
    burners = []
    for index, flux in enumerate(np.sort(fluxes)[::-1]):
        result = solve_burner_flame(mass_flux=float(flux), output=folder / f'burner_{index:03d}', **flame_settings)
        if result['heat_loss']['relative_energy_closure_error'] > max_energy_error:
            raise RuntimeError(f'Burner energy closure exceeds {max_energy_error:g}; refine the mesh: {result["output"]}')
        result['pressure'] = pressure
        burners.append(result)
    # Order by actual enthalpy; input mass flux alone is not an enthalpy coordinate.
    burners.sort(key=lambda result: result['heat_loss']['burned_enthalpy_deficit_J_kg'])
    results = [reference, *burners]
    table = tabulate_burner_family(results, mech, inlet_Y, weights, progress_points)
    table['mass_flux'] = np.array([np.nan, *[r['mass_flux'] for r in burners]])
    np.savez_compressed(folder / 'burner_fgm.npz', **table)
    metadata = dict(
        format='FlamPy_burner_fgm_v1', backend='native_cpu', controls=['c', 'h'],
        fixed_inlet_composition=True, inlet_Y=inlet_Y, mechanism=mech_name,
        pressure_Pa=pressure, progress_species=parsed,
        c_definition='(weighted_species-beta_unburned)/adiabatic_beta_span',
        h_definition='total_sensible_plus_formation_J_kg',
        source_units=dict(omega_c='kg/(m^3 s)', qdot='W/m^3'),
        delta_h_definition='h_adiabatic_at_same_c_minus_h',
        interpolation='adjacent_flamelets_only_with_thermodynamic_temperature_recovery',
        progress_points=progress_points, max_energy_error=max_energy_error,
        all_final_accepted=True,
        rows=[dict(kind='adiabatic_reference', output='adiabatic', Su=reference['Su']),
              *[dict(kind='isothermal_burner', output=r['output'].name,
                     mass_flux_kg_m2_s=r['mass_flux'], inlet_velocity_m_s=r['inlet_velocity'],
                     heat_loss=r['heat_loss']) for r in burners]],
        limitations='Fixed-composition planar gas flamelets; no solid conduction, radiation, multidimensional wall quenching, or independent Z control.',
    )
    (folder / 'metadata.json').write_text(json.dumps(json_safe(metadata), indent=2), encoding='utf-8')
    return folder


def generate_nonadiabatic_fgm(*, phis=(.7, .85, 1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3),
                             mass_flux_fractions=(.65, .45, .25, .20, .16, .12, .10, .08, .06),
                             progress_species='CO2:1,H2O:1,CO:1,H2:0.5',
                             progress_points=181, max_energy_error=.02,
                             output=None, raw_only=False, reuse_from=None, **flame_settings):
    """Generate physical (Z,C,h) flamelets over inlet composition and heat loss.

    Fractions multiply the adiabatic rho_feed*Su at each phi. Solve an adiabatic
    reference plus every burner, using the same physical species progress C
    throughout the family. Z is evaluated locally from Y, with mole-basis fuel
    and oxidizer streams. h is dimensional total mass enthalpy [J/kg].

    Output must be new. Accepted profiles are checkpointed in generation.json;
    failures retain diagnostics and are not called physical extinction.
    raw_only=True saves profiles for a later build_nonadiabatic_table call.
    reuse_from optionally copies matching accepted native profiles from an
    earlier family; chemistry, feed, transport and refinement must agree.
    The default also builds a connected tetrahedral table; import the loader
    from kflame.fgm.nonadiabatic3d.NonAdiabaticFGM.
    """
    from kflame.fgm.common import parse_progress_weights
    from kflame.fgm.nonadiabatic3d import build_nonadiabatic_table
    from kflame.serialization import json_safe
    import inspect
    import shutil

    phis = np.asarray(phis, dtype=float)
    fractions = np.asarray(mass_flux_fractions, dtype=float)
    if (phis.ndim != 1 or phis.size < 2 or not np.isfinite(phis).all()
            or np.any(phis <= 0.) or np.any(np.diff(phis) <= 0.)):
        raise ValueError('phis must be positive, finite and strictly increasing, with at least two entries')
    if (fractions.ndim != 1 or not fractions.size or not np.isfinite(fractions).all()
            or np.any(fractions <= 0.) or np.any(fractions >= 1.) or np.unique(fractions).size != fractions.size):
        raise ValueError('mass_flux_fractions must be distinct and strictly between zero and one')
    if not isinstance(progress_points, int) or progress_points < 3:
        raise ValueError('progress_points must be an integer >= 3')
    if not np.isfinite(max_energy_error) or not 0. < max_energy_error < 1.:
        raise ValueError('max_energy_error must lie between zero and one')
    if any(key in flame_settings for key in ('phi', 'X', 'Y', 'mass_flux', 'diluent', 'dilution')):
        raise ValueError('Use phis and explicit fuel/oxidizer streams for variable-composition generation')
    mechanism = str(flame_settings.get('mechanism', 'gri30.yaml'))
    mech = load_mechanism(resolve_mechanism(mechanism))
    weights = parse_progress_weights(progress_species)
    if not weights or set(weights) - set(mech.species_names) or not np.isfinite(list(weights.values())).all() or not any(weights.values()):
        raise ValueError('Progress weights must be finite, nonzero and refer to mechanism species')
    fuel, oxidizer = _stream(flame_settings.get('fuel', 'CH4')), _stream(flame_settings.get('oxidizer', 'O2:1,N2:3.76'))
    for phi in phis:
        fresh_mixture(mech, float(phi), fuel, oxidizer)
    fractions = np.sort(fractions)[::-1]
    reusable = None
    if reuse_from is not None:
        reusable = Path(reuse_from).resolve()
        saved_generation = json.loads((reusable / 'generation.json').read_text(encoding='utf-8'))
        if not saved_generation.get('all_final_accepted'):
            raise ValueError('reuse_from must be a complete accepted native family')
    defaults = {name: parameter.default for name, parameter in inspect.signature(solve_flame).parameters.items()}
    def reuse(kind, phi, fraction, destination):
        if reusable is None or flame_settings.get('grid') is not None:
            return None
        matches = [r for r in saved_generation['rows'] if r['kind'] == kind and r['phi'] == phi
                   and (kind == 'adiabatic_reference' or r['fraction'] == fraction)]
        if len(matches) != 1:
            return None
        source = reusable / matches[0]['output']
        meta = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
        target = lambda k: flame_settings.get(k, defaults[k])
        matching = (meta.get('accepted') and meta.get('backend') == 'native_cpu'
                    and meta.get('flow_type') == ('free_flame' if kind == 'adiabatic_reference' else 'isothermal_burner')
                    and meta['report'].get('grid_converged')
                    and Path(meta['mechanism']).read_bytes() == Path(resolve_mechanism(mechanism)).read_bytes()
                    and list(meta['species_names']) == list(mech.species_names)
                    and np.allclose(meta['inlet_Y'], fresh_mixture(mech, phi, fuel, oxidizer), rtol=1e-12, atol=1e-15)
                    and meta['temperature'] == target('temperature') and meta['pressure'] == target('pressure')
                    and meta['initial_width'] == target('width') and meta['transport'] == target('transport')
                    and meta['soret'] == target('soret')
                    and all(v == target(k) for k, v in meta['refinement'].items())
                    and all(v == target(k) for k, v in meta['tolerances'].items()))
        if not matching:
            return None
        shutil.copytree(source, destination)
        with np.load(destination / 'flame.npz', allow_pickle=False) as saved:
            result = {k: saved[k] for k in saved.files}
        return dict(result, output=destination, **{k: meta[k] for k in
                    (('Su',) if kind == 'adiabatic_reference' else ('mass_flux', 'inlet_velocity', 'heat_loss'))})
    folder = _output(output, 'nonadiabatic_fgm')
    generation = dict(
        mechanism=mechanism, fuel=fuel, oxidizer=oxidizer, phis=phis,
        mass_flux_fractions=fractions, progress_species=weights,
        temperature_K=float(flame_settings.get('temperature', 300.)),
        pressure_Pa=float(flame_settings.get('pressure', 101325.)),
        transport=flame_settings.get('transport', 'mixture-averaged'),
        soret=flame_settings.get('soret', False), max_energy_error=max_energy_error,
        flame_settings=flame_settings,
        all_final_accepted=False, rows=[], failures=[], backend='native_cpu', reused_profiles=0)
    def checkpoint():
        (folder / 'generation.json').write_text(json.dumps(json_safe(generation), indent=2) + '\n', encoding='utf-8')
    checkpoint()
    for i, phi in enumerate(phis):
        print(f'phi={phi:g}: adiabatic reference', flush=True)
        destination = folder / f'phi_{i:03d}_adiabatic'
        reference = reuse('adiabatic_reference', float(phi), None, destination)
        if reference is None:
            reference = solve_flame(phi=float(phi), output=destination, **flame_settings)
        else:
            generation['reused_profiles'] += 1
        meta = json.loads((reference['output'] / 'metadata.json').read_text(encoding='utf-8'))
        inlet_Y = np.array(meta['inlet_Y'])
        from kflame.chemistry.thermo import NativeThermo
        mdot_ad = float(NativeThermo(mech).density(generation['temperature_K'], generation['pressure_Pa'], inlet_Y)) * reference['Su']
        generation['rows'].append(dict(composition_index=i, loss_index=0, phi=float(phi),
                                       kind='adiabatic_reference', output=reference['output'].name,
                                       inlet_Y=inlet_Y, adiabatic_mass_flux=mdot_ad, Su=reference['Su']))
        checkpoint()
        previous_burner = None
        for j, fraction in enumerate(fractions, start=1):
            flux = float(fraction * mdot_ad)
            print(f'phi={phi:g}: burner fraction={fraction:g}, mdot={flux:.6g}', flush=True)
            try:
                destination = folder / f'phi_{i:03d}_loss_{j:03d}'
                result = reuse('isothermal_burner', float(phi), float(fraction), destination)
                if result is not None and not np.isclose(result['mass_flux'], flux, rtol=1e-12):
                    raise ValueError('Reused burner mass flux differs from the new family')
                if result is None:
                    result = solve_burner_flame(phi=float(phi), mass_flux=flux,
                        output=destination, initial_solution=previous_burner, **flame_settings)
                else:
                    generation['reused_profiles'] += 1
                if result['heat_loss']['relative_energy_closure_error'] > max_energy_error:
                    raise RuntimeError('Boundary energy closure exceeds the requested tolerance; refine the spatial mesh')
            except RuntimeError as error:
                generation['failures'].append(dict(phi=float(phi), fraction=float(fraction),
                    classification='numerical_failure_not_physical_extinction', reason=str(error)))
                checkpoint()
                raise
            generation['rows'].append(dict(composition_index=i, loss_index=j, phi=float(phi),
                kind='isothermal_burner', output=result['output'].name, inlet_Y=inlet_Y,
                fraction=float(fraction), mass_flux=flux, adiabatic_mass_flux=mdot_ad,
                inlet_velocity=result['inlet_velocity'], heat_loss=result['heat_loss']))
            previous_burner = result['output']
            checkpoint()
    generation['all_final_accepted'] = True
    checkpoint()
    return folder if raw_only else build_nonadiabatic_table(folder, progress_points=progress_points)
