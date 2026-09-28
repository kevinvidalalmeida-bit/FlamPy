"""Cantera FreeFlame adapter for the shared adaptive FGM orchestration.

Only the flame solver and its within-family continuation differ. Table fields,
progress definition, adaptive c-grid and row indicator are shared with KFLAME.
"""
from __future__ import annotations
import time
import numpy as np


class CanteraEngine:
    @staticmethod
    def configure_numba_kinetics_threads(count):
        # Shared table utilities may use Numba; Cantera itself uses OMP/BLAS=1.
        from kflame.fgm.generate import configure_numba_kinetics_threads
        return configure_numba_kinetics_threads(count)

    @staticmethod
    def parse_progress_weights(text):
        from kflame.fgm.common import parse_progress_weights
        return parse_progress_weights(text)

    @staticmethod
    def load_mechanism(path):
        import cantera as ct
        return ct.Solution(path)

    @staticmethod
    def make_solve_options(args):
        # Native FreeFlameProblem's stationary tolerances, recorded explicitly.
        return dict(steady=(1.e-4, 1.e-9), transient=(1.e-4, 1.e-11))

    @staticmethod
    def solve_flame_native(phi, args, mech, opts, weights, prev_solution=None, prev_prev_solution=None):
        import cantera as ct
        from kflame.fgm.common import compute_bilger_Z, compute_progress_variable
        from kflame.fgm.generate import FlameRecord
        start = time.perf_counter()
        gas = mech
        Z = compute_bilger_Z(phi, args, gas)
        fresh_Y = gas.Y.copy()
        # Copying a SolutionArray for the guess is supported by Cantera's API.
        # Each solver object is fresh; only profiles from this FGM are continued.
        grid = np.linspace(0, args.width, args.initial_grid_points) if prev_solution is None else prev_solution['z']
        flame = ct.FreeFlame(gas, grid=grid)
        flame.transport_model = args.transport_model
        if args.transport_model == 'mixture-averaged':
            flame.flux_gradient_basis = args.flux_gradient_basis
        flame.soret_enabled = args.soret_enabled
        flame.flame.set_steady_tolerances(default=opts['steady'])
        flame.flame.set_transient_tolerances(default=opts['transient'])
        flame.set_refine_criteria(ratio=args.ratio, slope=args.slope, curve=args.curve, prune=args.prune)
        flame.set_max_grid_points(flame.flame, args.max_grid_points)
        if args.grid_min > 0: flame.set_grid_min(args.grid_min)
        def interrupt(_):
            if time.perf_counter()-start > args.max_flame_time_s:
                raise TimeoutError('Cantera FGM flame time limit reached')
            return 0.
        flame.set_interrupt(interrupt)
        if prev_solution is not None:
            flame.set_initial_guess(data=prev_solution['solution'])
        flame.solve(loglevel=args.loglevel, auto=True, refine_grid=True)
        solve_s = time.perf_counter()-start
        if flame.transport_model != args.transport_model or flame.soret_enabled != args.soret_enabled:
            raise RuntimeError('Cantera changed the requested final transport.')
        if not np.isclose(flame.inlet.T, args.T_in) or not np.isclose(flame.P, args.P):
            raise RuntimeError('Cantera inlet condition mismatch.')
        if not np.allclose(flame.inlet.Y, fresh_Y, atol=1e-12, rtol=1e-10):
            raise RuntimeError('Cantera inlet composition changed during continuation.')
        post_start = time.perf_counter()
        z, T, Y, u = flame.grid.copy(), flame.T.copy(), flame.Y.copy(), flame.velocity.copy()
        c, beta, used = compute_progress_variable(gas.species_names, Y, T, weights)
        span = float(beta[-1]-beta[0])
        if not used or abs(span) <= 1e-14: raise RuntimeError('Degenerate Cantera progress coordinate.')
        molecular_weights = gas.molecular_weights
        omega_mass = flame.net_production_rates * molecular_weights[:, None]
        omega_beta = sum(weight*omega_mass[gas.species_index(name)] for name, weight in weights.items())
        rho, cp, conductivity = flame.density.copy(), flame.cp_mass.copy(), flame.thermal_conductivity.copy()
        qdot = flame.heat_release_rate.copy()
        finite = all(np.isfinite(v).all() for v in (z, T, Y, u, rho, cp, conductivity, qdot, omega_beta))
        accepted = bool(finite and u[0] > 0 and np.diff(z).min() > 0 and T[-1] > T[0]+100
                        and np.max(abs(Y.sum(axis=0)-1)) < 1e-6 and Y.min() > -1e-8)
        kind = 'cold' if prev_solution is None else 'cantera_previous_profile'
        rec = FlameRecord(phi=float(phi), Z=Z, solve_ok=True, residual_inf=float('nan'),
                          weighted_step_norm=float('nan'), final_accepted=accepted,
                          acceptance_criterion='cantera solve returned + finite physical profile checks',
                          solve_time_s=solve_s, n_points=len(z), width_m=float(z[-1]-z[0]),
                          Su_m_per_s=float(u[0]), z=z, u=u, T=T, rho=rho, cp_mass=cp,
                          conductivity=conductivity, qdot=qdot, omega_c=omega_beta/span,
                          Y=Y, c=c, beta=beta, predictor_kind=kind)
        state = dict(phi=np.array([phi]), z=z.copy(), solution=flame.to_array())
        stats = {key: getattr(flame, key) for key in ('grid_size_stats', 'jacobian_count_stats',
                 'jacobian_time_stats', 'eval_count_stats', 'eval_time_stats', 'time_step_stats')}
        trace = dict(predictor_kind=kind, seeded=prev_solution is not None, solve_time_s=solve_s,
                     postprocess_time_s=time.perf_counter()-post_start, final_accepted=accepted,
                     solver_profile=stats, effective_options=dict(opts, auto=True,
                     transport_model=flame.transport_model, soret_enabled=flame.soret_enabled,
                     T_in_K=flame.inlet.T, P_Pa=flame.P, refinement=flame.get_refine_criteria()),
                     initial_grid_points=len(grid), final_grid_points=len(z), final_width_m=rec.width_m)
        return rec, state, trace
