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
                species=('CH4', 'O2', 'CO2', 'H2O', 'OH')):
    """Solve an adiabatic premixed free flame and save NPZ, CSV and metadata.

    Use one of phi with fuel/oxidizer, X (mole amounts), or Y (mass amounts).
    Compositions accept strings or dictionaries and are normalized. Dilution
    is the final mole fraction of the diluent in the fresh mixture. Grid is
    optional, in metres, strictly increasing from zero to width. The outlet
    has zero species gradients; pressure and fresh inlet state are prescribed.
    Returned arrays use Y[species, node]. Internal Jacobian/PTC settings are
    the production defaults. A rejected solve raises after saving diagnostics.
    """
    from kflame.chemistry.backend import NativeSpeciesBackend
    from kflame.fgm.generate import build_argparser, make_solve_options, tabulated_properties
    from kflame.flame.problem import FreeFlameProblem
    from kflame.flame.solver import solve_free_flame
    from kflame.flame.state import unpack_state
    from kflame.serialization import json_safe

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
        transport_model=transport, soret_enabled=soret, ratio=ratio, slope=slope, curve=curve, prune=prune,
    )
    folder = _output(output, 'flame')
    started = time.perf_counter()
    problem = FreeFlameProblem(case, n_points=initial_points, mech_data=mech)
    problem.assume_finite_y = True
    problem.backend_factory = NativeSpeciesBackend
    options = replace(make_solve_options(args), refine_ratio=ratio, refine_slope=slope,
                      refine_curve=curve, refine_prune=prune, refine_max_points=max_points)
    state, ok, report = solve_free_flame(problem, options=options)
    elapsed = time.perf_counter() - started
    u, T, y = unpack_state(state, problem.n_points, problem.n_species)
    rho, cp, conductivity, qdot, _ = tabulated_properties(problem, T, y, {})
    fields = dict(z=problem.z, T=T, u=u, Y=y, rho=rho, cp_mass=cp,
                  conductivity=conductivity, qdot=qdot, species_names=np.asarray(problem.species_names))
    np.savez_compressed(folder / 'flame.npz', **fields)
    indices = [problem.species_names.index(name) for name in species]
    np.savetxt(folder / 'profiles.csv', np.column_stack([problem.z, T, u, rho, cp, conductivity, qdot, *y[indices]]),
               delimiter=',', header=','.join(['z_m', 'T_K', 'u_m_s', 'rho_kg_m3', 'cp_J_kg_K',
                                             'conductivity_W_m_K', 'qdot_W_m3', *['Y_' + name for name in species]]), comments='')
    accepted = bool(ok and report.get('final_accepted') and report.get('grid_converged'))
    metadata = dict(software='KFLAME', backend='native_cpu', mechanism=args.mech,
                    species_names=problem.species_names, temperature=temperature,
                    pressure=pressure, transport=transport, soret=soret, inlet_Y=problem.Y_in,
                    initial_grid=case.initial_grid, initial_points=initial_points,
                    refinement=dict(ratio=ratio, slope=slope, curve=curve, prune=prune, max_points=max_points),
                    tolerances=dict(rtol=rtol, atol=atol),
                    initial_width=width, final_width=problem.width, nodes=problem.n_points,
                    Su=float(u[0]), runtime_s=elapsed, accepted=accepted, report=report)
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
    return dict(**fields, Su=float(u[0]), accepted=accepted, output=folder, report=report, runtime_s=elapsed)


def generate_fgm(*, phis=(0.7, 0.9, 1.0, 1.1, 1.4), mechanism='gri30.yaml',
                 temperature=300.0, pressure=101325.0, fuel='CH4', oxidizer='O2:1, N2:3.76',
                 width=0.03, initial_points=8, transport='mixture-averaged', soret=False,
                 ratio=2.5, slope=0.04, curve=0.08, prune=0.003, max_points=1600,
                 max_time=180.0, output=None, plots=False, verbose=False, export=True,
                 species=('CO2', 'H2O'), progress_species='CO2:1.0,H2O:1.0,CO:1.0,H2:0.5',
                 progress_points=241, adaptive_phi=True, target_defect=0.01,
                 max_bridges_per_round=10, max_adaptive_rounds=64,
                 max_flamelets=256):
    """Generate a native FGM with adaptive c coordinates and certified flames.

    Output contains the full NPZ table, raw profiles, metadata, optional
    FlameMaster/CSV tables and the established FGM figures. Composition is
    parameterized by phis and mole-basis fuel/oxidizer streams. With
    By default, phis are the initial certified flamelets and logarithmic bridge
    flamelets are solved until the leave-one-out defect reaches target_defect.
    Set adaptive_phi=False only to build a fixed composition grid. Returns Path.
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
