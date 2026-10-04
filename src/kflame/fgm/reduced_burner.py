"""Steady planar burner transport on a nonadiabatic (Z, C, h) manifold.

Only three transported quantities are solved. Species and chemical sources
are interpolated from the frozen table; detailed kinetics are never evaluated.
Corrected mixture-averaged species fluxes retain preferential diffusion.
The total enthalpy equation includes conduction and species enthalpy flux,
and has no additional chemical heat-release source.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csc_matrix, lil_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings

from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.transport import NativeTransport
from kflame.flame.equations import _corrected_flux_frozen

_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ReducedConvergenceError(RuntimeError):
    """A bounded manifold solve failed its residual or physical checks."""
    def __init__(self, report, profile):
        super().__init__(report['reason'])
        self.report, self.profile = report, profile


class ReducedBurnerProblem:
    """Conservative finite volumes on a strictly increasing nonuniform grid.

    Logical coordinates are numerical variables, not new physical controls.
    The default convex quadratic approximation is C1 in progression, with
    bilinear weights in the other two directions. All fields share weights;
    temperature is recovered from physical total h. Barycentric interpolation
    of the original connected mesh remains available as a comparison option.
    """
    def __init__(self, model, z, mass_flux, inlet_Y, *, interpolation='quadratic_progress'):
        meta, table = model.metadata, model.table
        if meta.get('transport') != 'mixture-averaged' or meta.get('soret', False):
            raise ValueError('Reduced burner currently supports mixture-averaged transport without Soret')
        self.model, self.z = model, np.asarray(z, dtype=float)
        if (self.z.ndim != 1 or len(self.z) < 5 or self.z[0] != 0.
                or not np.isfinite(self.z).all() or np.any(np.diff(self.z) <= 0.)):
            raise ValueError('Grid must start at zero and contain >=5 increasing finite nodes')
        self.mass_flux = float(mass_flux)
        if not np.isfinite(self.mass_flux) or self.mass_flux <= 0.:
            raise ValueError('Mass flux must be positive and finite')
        self.shape = np.asarray(table['structured_shape'], dtype=int)
        if self.shape.shape != (3,) or np.any(self.shape < 2):
            raise ValueError('A structured three-control manifold is required')
        expected = 6 * int(np.prod(self.shape - 1))
        if len(table['cells']) != expected:
            raise ValueError('Logical transport requires a complete unfolded connected mesh')
        self.strides = np.array([self.shape[1] * self.shape[2], self.shape[2], 1])
        if interpolation not in ('barycentric', 'quadratic_progress'):
            raise ValueError('Unknown manifold approximation')
        if interpolation == 'quadratic_progress' and self.shape[2] < 3:
            raise ValueError('Quadratic approximation needs at least three progress samples')
        self.interpolation = interpolation
        self.corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.knots = np.r_[np.zeros(3), np.arange(1, self.shape[2]-2),
                           np.full(3, self.shape[2]-2)]
        self.residual_evaluations = 0
        self.mech = load_mechanism(meta['mechanism'])
        self.transport = NativeTransport(self.mech)
        self.pressure, self.T_burner = meta['pressure_Pa'], meta['temperature_K']
        self.inlet_Y = np.asarray(inlet_Y, dtype=float)
        if (self.inlet_Y.shape != (self.mech.n_species,) or np.any(self.inlet_Y < 0.)
                or not np.isfinite(self.inlet_Y).all() or abs(self.inlet_Y.sum()-1.) > 1e-12):
            raise ValueError('Inlet fractions must be nonnegative and sum to one')
        self.h_feed = float(model.thermo.enthalpy_mass(self.T_burner, self.inlet_Y))
        self.Z_feed = float(table['bilger_weights'] @ self.inlet_Y + table['bilger_offset'])
        self.C_feed = float(table['progress_weights'] @ self.inlet_Y)
        self.dz = np.diff(self.z)
        self.volumes = .5 * (self.dz[:-1] + self.dz[1:])
        self.scales = self.mass_flux * np.maximum(
            np.ptp(table['controls'], axis=0), [1e-3, 1e-3, 1e5])
        # At the surface all stored flamelets have the imposed burner T.
        surface = table['T'].reshape(tuple(self.shape))[:, :, 0]
        if np.max(abs(surface-self.T_burner)) > 1e-6:
            raise ValueError('All surface samples must have the imposed burner temperature')

    def state(self, x, *, geometry=None, temperature_cache=None, changed_nodes=None):
        x = np.asarray(x).reshape(-1, 3)
        if len(x) != len(self.z) or not np.isfinite(x).all():
            raise ValueError('Three finite logical coordinates per spatial node are required')
        if geometry is None and (np.any(x < 0.) or np.any(x > 1.)):
            raise ValueError('Trial state outside the resolved logical mesh')
        logical = x * (self.shape-1)
        if self.interpolation == 'quadratic_progress':
            base = (np.minimum(np.floor(logical[:, :2]).astype(int), self.shape[:2]-2)
                    if geometry is None else geometry)
            f = logical[:, :2]-base
            bilinear = np.prod(np.where(self.corners[None, :, :],
                               f[:, None, :], 1.-f[:, None, :]), axis=2)
            # Extrapolation is permitted only for derivative probes above.
            # Accepted states always use bounded partition-of-unity weights.
            basis = BSpline.design_matrix(x[:, 2]*(self.shape[2]-2), self.knots,
                                           2, extrapolate=geometry is not None).tocsr()
            weights, indices = basis.data.reshape(-1, 3), basis.indices.reshape(-1, 3)
            nodes = ((base@self.strides[:2])[:, None]
                     +(self.corners@self.strides[:2])[None, :])[:, :, None]+indices[:, None, :]
            nodes = nodes.reshape(-1, 12)
            bary = (bilinear[:, :, None]*weights[:, None, :]).reshape(-1, 12)
            active_geometry = base
        else:
            if geometry is None:
                base = np.minimum(np.floor(logical).astype(int), self.shape-2)
                order = np.argsort(-(logical-base), axis=1)
            else:
                base, order = geometry
            f = np.take_along_axis(logical-base, order, axis=1)
            bary = np.column_stack([1-f[:, 0], f[:, 0]-f[:, 1], f[:, 1]-f[:, 2], f[:, 2]])
            offsets = np.column_stack([np.zeros(len(x), dtype=int), np.cumsum(self.strides[order], axis=1)])
            nodes = (base @ self.strides)[:, None] + offsets
            active_geometry = (base, order)
        t = self.model.table
        Y = np.einsum('ni,nij->nj', bary, t['Y'][nodes])
        h = np.einsum('ni,ni->n', bary, t['h'][nodes])
        guess = np.einsum('ni,ni->n', bary, t['T'][nodes])
        if temperature_cache is None:
            T = self.model._recover_temperature(h, Y.T, guess)
        else:
            # Colored finite differences change only one third of the nodes.
            # Keep the exactly unchanged temperatures instead of repeatedly
            # solving their thermodynamic inversions.
            T = np.asarray(temperature_cache).copy()
            T[changed_nodes] = self.model._recover_temperature(
                h[changed_nodes], Y[changed_nodes].T, guess[changed_nodes])
        return dict(Y=Y.T, T=T, h=h,
                    Z=t['bilger_weights'] @ Y.T + float(t['bilger_offset']),
                    C=t['progress_weights'] @ Y.T,
                    omega_C=np.einsum('ni,ni->n', bary, t['omega_C'][nodes]),
                    qdot=np.einsum('ni,ni->n', bary, t['qdot'][nodes]),
                    geometry=active_geometry)

    def initial_progress(self, target, row):
        """Invert the strictly monotone row C for the chosen approximation."""
        if self.interpolation == 'barycentric':
            return np.interp(target, row, np.linspace(0., 1., len(row)))
        spline = BSpline(self.knots, row, 2)
        lo, hi = np.zeros_like(target), np.ones_like(target)
        for _ in range(35):
            mid = .5*(lo+hi)
            below = spline(mid*(len(row)-2)) < target
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return .5*(lo+hi)

    def fluxes(self, state):
        T, Y, mw = state['T'], state['Y'], self.mech.molecular_weights
        Tf, Yf = .5*(T[:-1]+T[1:]), .5*(Y[:, :-1]+Y[:, 1:])
        rho, D, conductivity, W = self.transport.eval_faces_poly_fast(Tf, self.pressure, Yf, 1./mw)
        coeff = rho[None, :] * (mw[:, None]/W[None, :]) * D
        J = _corrected_flux_frozen(Y[:, :-1], Y[:, 1:], coeff, self.dz, mw, 'molar')
        conduction = -conductivity * np.diff(T)/self.dz
        hk = self.model.thermo.partial_molar_enthalpies(Tf)/mw[:, None]
        enthalpy_diffusion = np.sum(hk*J, axis=0)
        # Exponential fitting is central at low Peclet number and approaches
        # upstream advection in the convection-dominated post-flame tail.
        # Preferential species/enthalpy diffusion remains in the physical J.
        cp = self.model.thermo.cp_mass(Tf, Yf)
        peclet = self.mass_flux*self.dz*cp/conductivity
        weight = np.empty_like(peclet)
        small = peclet < 1e-3
        weight[small] = .5-peclet[small]/12.+peclet[small]**3/720.
        weight[~small] = 1./peclet[~small]-1./np.expm1(np.minimum(peclet[~small], 700.))
        def advected(name):
            return self.mass_flux*((1.-weight)*state[name][:-1]+weight*state[name][1:])
        F = np.column_stack([
            advected('Z') + self.model.table['bilger_weights'] @ J,
            advected('C') + self.model.table['progress_weights'] @ J,
            advected('h') + conduction + enthalpy_diffusion])
        # Danckwerts inflow fixes the incoming *total* enthalpy to the feed.
        # Imposing only projected Z/C fluxes does not enforce that identity
        # for every species in a reduced manifold. Use the physical inlet
        # energy flux explicitly, retaining the computed conductive loss.
        F[0, 2] = self.mass_flux*self.h_feed+conduction[0]
        return dict(total=F, species=J, conduction=conduction,
                    enthalpy_diffusion=enthalpy_diffusion, conductivity=conductivity)

    def residual(self, x, *, geometry=None, temperature_cache=None, changed_nodes=None):
        self.residual_evaluations += 1
        state = self.state(x, geometry=geometry, temperature_cache=temperature_cache, changed_nodes=changed_nodes)
        flux = self.fluxes(state)
        out = np.empty((len(self.z), 3))
        out[1:-1] = np.diff(flux['total'], axis=0)
        out[1:-1, 1] -= state['omega_C'][1:-1]*self.volumes
        out[0, 0] = flux['total'][0, 0]-self.mass_flux*self.Z_feed
        out[0, 1] = flux['total'][0, 1]-self.mass_flux*self.C_feed
        out /= self.scales
        logical = np.asarray(x).reshape(-1, 3)
        out[0, 2] = logical[0, 2]  # exactly T(0)=T_burner
        out[-1] = logical[-1]-logical[-2]  # zero gradients of T/Y/Z/C/h
        return out.ravel()

    def diagnostics(self, x):
        state, flux = self.state(x), self.fluxes(self.state(x))
        q_wall = float(-flux['conduction'][0])
        q_deficit = self.mass_flux*(self.h_feed-state['h'][-1])
        energy_scale = max(abs(q_wall), abs(q_deficit), 1.)
        inlet_species = self.mass_flux*(state['Y'][:, 0]-self.inlet_Y)+flux['species'][:, 0]
        inlet_energy = self.mass_flux*state['h'][0] + np.sum(
            self.model.thermo.partial_molar_enthalpies(self.T_burner).ravel()
            / self.mech.molecular_weights * flux['species'][:, 0])
        progress_integral = float(np.sum(state['omega_C'][1:-1]*self.volumes))
        progress_out = self.mass_flux*(state['C'][-1]-self.C_feed)
        return dict(burner_heat_loss_W_m2=q_wall, outlet_enthalpy_loss_W_m2=float(q_deficit),
                    relative_energy_closure_error=abs(q_wall-q_deficit)/energy_scale,
                    relative_total_enthalpy_flux_spread=float(np.ptp(flux['total'][:, 2])/energy_scale),
                    projected_progress_balance_error=abs(progress_integral-progress_out)/max(abs(progress_out), 1e-12),
                    inlet_species_flux_error_over_mdot=float(np.max(abs(inlet_species))/self.mass_flux),
                    inlet_enthalpy_flux_error_W_m2=float(inlet_energy-self.mass_flux*self.h_feed),
                    sum_Y_error=float(np.max(abs(state['Y'].sum(axis=0)-1.))),
                    burner_temperature_error_K=float(abs(state['T'][0]-self.T_burner)),
                    maximum_scaled_residual=float(np.max(abs(self.residual(x)))))


def _newton(problem, initial, *, residual_tolerance, max_iterations, verbose):
    x = np.clip(np.asarray(initial, dtype=float).ravel(), 1e-7, 1.-1e-5)
    x[2] = 0.
    n = len(problem.z)
    pattern = lil_matrix((3*n, 3*n), dtype=int)
    for i in range(n):
        pattern[3*i:3*i+3, max(0, 3*(i-1)):min(3*n, 3*(i+2))] = 1
    pattern = pattern.tocsc()
    history, reason = [], 'iteration_budget_exhausted'
    for iteration in range(max_iterations+1):
        f = problem.residual(x)
        maximum, norm = float(np.max(abs(f))), float(np.linalg.norm(f))
        history.append(dict(iteration=iteration, maximum_scaled_residual=maximum, norm=norm))
        if verbose and (iteration % 10 == 0 or maximum <= residual_tolerance):
            print(f'FGM iteration {iteration}: residual={maximum:.3g}', flush=True)
        if maximum <= residual_tolerance:
            return x, history, 'converged'
        if (len(history) >= 16 and norm >= .999*history[-16]['norm']
                and max(h['norm'] for h in history[-16:]) <= 1.01*norm):
            reason = 'stagnated_reduced_residual'
            break
        if iteration == max_iterations:
            break
        base_state = problem.state(x)
        geometry = base_state['geometry']
        data = np.empty_like(pattern.data, dtype=float)
        for color in range(9):
            selected = np.arange(color, len(x), 9)
            steps = np.full(len(selected), 1e-7)
            plus = x.copy()
            plus[selected] += steps
            difference = problem.residual(plus, geometry=geometry,
                temperature_cache=base_state['T'], changed_nodes=selected//3)-f
            for column, step in zip(selected, steps):
                a, b = pattern.indptr[column:column+2]
                data[a:b] = difference[pattern.indices[a:b]]/step
        jacobian = csc_matrix((data, pattern.indices, pattern.indptr), shape=pattern.shape)
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            try:
                step = spsolve(jacobian, -f)
            except (MatrixRankWarning, RuntimeError):
                reason = 'singular_reduced_jacobian'
                break
        step[2] = 0.
        if not np.isfinite(step).all():
            reason = 'nonfinite_newton_step'
            break
        positive, negative = step > 0., step < 0.
        alpha = 1.
        if positive.any():
            alpha = min(alpha, .995*float(np.min((1.-x[positive])/step[positive])))
        if negative.any():
            alpha = min(alpha, .995*float(np.min(-x[negative]/step[negative])))
        accepted = False
        for _ in range(32):
            if alpha < 1e-11:
                break
            trial = x+alpha*step
            trial[2] = 0.
            trial_norm = np.linalg.norm(problem.residual(trial))
            if trial_norm <= norm*(1.-1e-4*alpha):
                accepted = True
                break
            alpha *= .5
        if not accepted:
            # A simplex facet can change the directional derivative. Rebuild
            # one-sided physical trial derivatives and regularize the sparse
            # least-squares model instead of spending hundreds of iterations
            # on an ill-conditioned Newton direction.
            for color in range(9):
                selected = np.arange(color, len(x), 9)
                steps = np.where(x[selected] > 1.-1e-7, -1e-7, 1e-7)
                perturbed = x.copy()
                perturbed[selected] += steps
                difference = problem.residual(perturbed, temperature_cache=base_state['T'], changed_nodes=selected//3)-f
                for column, delta in zip(selected, steps):
                    a, b = pattern.indptr[column:column+2]
                    data[a:b] = difference[pattern.indices[a:b]]/delta
            actual = csc_matrix((data, pattern.indices, pattern.indptr), shape=pattern.shape)
            normal = actual.T @ actual
            gradient = actual.T @ f
            diagonal = np.maximum(normal.diagonal(), 1e-12)
            for damping in (1e-8, 1e-6, 1e-4, 1e-2, 1., 100.):
                step = spsolve(normal+diags(damping*diagonal), -gradient)
                step[2] = 0.
                # Project only an outward numerical search direction at an
                # active bound; production physical states are never clipped.
                step[(x < 1e-10) & (step < 0.)] = 0.
                step[(x > 1.-1e-10) & (step > 0.)] = 0.
                alpha = 1.
                positive, negative = step > 0., step < 0.
                if positive.any():
                    alpha = min(alpha, .995*float(np.min((1.-x[positive])/step[positive])))
                if negative.any():
                    alpha = min(alpha, .995*float(np.min(-x[negative]/step[negative])))
                for _ in range(24):
                    if alpha < 1e-11:
                        break
                    trial = x+alpha*step
                    trial[2] = 0.
                    if np.linalg.norm(problem.residual(trial)) <= norm*(1.-1e-4*alpha):
                        accepted = True
                        history[-1]['regularization'] = damping
                        break
                    alpha *= .5
                if accepted:
                    break
            if not accepted:
                reason = 'bounded_newton_step_failed'
                break
        x = trial
        history[-1]['step_length'] = float(alpha)
    return x, history, reason


def solve_reduced_burner_fgm(model, *, phi, mass_flux, seed_profile, seed_row,
                             width=None, grid=None, max_spacing=1.25e-4,
                             initial_solution=None, residual_tolerance=1e-7,
                             max_iterations=180, max_energy_error=.02,
                             interpolation='quadratic_progress', verbose=False):
    """Solve a burner using a frozen FGM and a verified training warm start.

    ``seed_row`` is an output name from the table metadata. A detailed withheld
    solution is not a seed. The seed fixes only the initial guess and grid;
    all final controls are obtained by solving the reduced balances.
    A previous reduced iterate from the same frozen table may supply a start.
    This initial implementation uses the table's fixed pressure/temperature.
    """
    if not np.isfinite(phi) or phi <= 0.:
        raise ValueError('Equivalence ratio must be positive and finite')
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.:
        raise ValueError('Residual tolerance must be positive and finite')
    if not np.isfinite(max_spacing) or max_spacing <= 0.:
        raise ValueError('Maximum spacing must be positive and finite')
    if not np.isfinite(max_energy_error) or max_energy_error <= 0.:
        raise ValueError('Energy closure tolerance must be positive and finite')
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('Iteration budget must be a positive integer')
    rows = [r for r in model.metadata['rows'] if r['output'] == seed_row]
    if len(rows) != 1 or rows[0]['loss_index'] == 0:
        raise ValueError('Seed must identify one burner training row')
    record = rows[0]
    seed_profile = Path(seed_profile)
    digest = hashlib.sha256(seed_profile.read_bytes()).hexdigest()
    if model.metadata['training_profile_sha256'][seed_row] != digest:
        raise ValueError('Seed profile does not match the accepted training fingerprint')
    with np.load(seed_profile, allow_pickle=False) as saved:
        seed = {k: saved[k] for k in ('z', 'T', 'Y')}
    width = float(seed['z'][-1] if width is None else width)
    if not np.isfinite(width) or width <= 0.:
        raise ValueError('Domain width must be positive and finite')
    if grid is None:
        anchors = np.unique(np.r_[seed['z'][seed['z'] < width], width])
        # Split seed intervals, avoiding nearly coincident nodes from a union
        # with an unrelated uniform grid. Keep the resolved flame-front mesh.
        grid = np.r_[np.concatenate([np.linspace(a, b, int(np.ceil((b-a)/max_spacing))+1)[:-1]
                                     for a, b in zip(anchors[:-1], anchors[1:])]), width]
    elif abs(np.asarray(grid)[-1]-width) > 1e-12:
        raise ValueError('Grid endpoint must match the domain width')
    mech = load_mechanism(model.metadata['mechanism'])
    feed = fresh_mixture(mech, phi, model.metadata['fuel'], model.metadata['oxidizer'])
    problem = ReducedBurnerProblem(model, grid, mass_flux, feed, interpolation=interpolation)
    i, j = record['composition_index'], record['loss_index']
    C_seed = model.table['progress_weights'] @ seed['Y']
    C_row = model.table['C'].reshape(tuple(problem.shape))[i, j]
    if np.any(np.diff(C_row) <= 0.):
        raise ValueError('Training progress samples must increase strictly')
    progress = np.minimum(.985, problem.initial_progress(np.interp(problem.z, seed['z'], C_seed), C_row))
    phis = np.asarray(model.metadata['phis'])
    if phi < phis[0] or phi > phis[-1]:
        raise ValueError('Feed equivalence ratio is outside the tabulated family')
    adiabatic = sorted([r for r in model.metadata['rows'] if r['loss_index'] == 0], key=lambda r: r['phi'])
    reference_flux = np.interp(phi, [r['phi'] for r in adiabatic], [r['adiabatic_mass_flux'] for r in adiabatic])
    fractions = np.r_[1., model.metadata['mass_flux_fractions']]
    target_fraction = mass_flux/reference_flux
    if target_fraction < fractions[-1] or target_fraction > fractions[0]:
        raise ValueError('Mass flux is outside the tabulated family')
    initial_phi = np.interp(phi, phis, np.linspace(0., 1., len(phis)))
    initial_loss = np.interp(target_fraction, fractions[::-1], np.linspace(0., 1., len(fractions))[::-1])
    initial = np.column_stack([np.full(len(problem.z), initial_phi),
                               np.full(len(problem.z), initial_loss), progress])
    if initial_solution is not None:
        if str(initial_solution.get('table_sha256', '')) != model.table_sha256:
            raise ValueError('Previous iterate must identify the same frozen FGM table')
        previous_z, previous_x = initial_solution['z'], initial_solution['logical_coordinates']
        previous_z, previous_x = np.asarray(previous_z), np.asarray(previous_x)
        if (previous_z.ndim != 1 or len(previous_z) < 2 or previous_z[0] != 0.
                or not np.isfinite(previous_z).all() or np.any(np.diff(previous_z) <= 0.)
                or previous_x.shape != (len(previous_z), 3)
                or not np.isfinite(previous_x).all()
                or np.any(previous_x < 0.) or np.any(previous_x > 1.)):
            raise ValueError('Previous reduced iterate has invalid coordinates or grid')
        initial = np.column_stack([np.interp(problem.z, previous_z, previous_x[:, k]) for k in range(3)])
        beyond = problem.z > previous_z[-1]
        initial[beyond, 2] = previous_x[-1, 2]+(1.-previous_x[-1, 2])*(
            1.-np.exp(-(problem.z[beyond]-previous_z[-1])/.008))
    started = time.perf_counter()
    x, history, reason = _newton(problem, initial, residual_tolerance=residual_tolerance,
                                 max_iterations=max_iterations, verbose=verbose)
    state, flux = problem.state(x), problem.fluxes(problem.state(x))
    diagnostics = problem.diagnostics(x)
    accepted = (reason == 'converged' and diagnostics['relative_energy_closure_error'] <= max_energy_error
                and np.max(state['T'])-problem.T_burner > 50.
                and diagnostics['sum_Y_error'] < 1e-10)
    if reason == 'converged' and not accepted:
        reason = 'physical_acceptance_failed'
    profile = {k: v for k, v in state.items() if k != 'geometry'}
    profile.update(z=problem.z, logical_coordinates=x.reshape(-1, 3),
                   table_sha256=np.array(model.table_sha256),
                   z_face=.5*(problem.z[:-1]+problem.z[1:]),
                   total_enthalpy_flux=flux['total'][:, 2],
                   conductive_heat_flux=flux['conduction'], diffusive_enthalpy_flux=flux['enthalpy_diffusion'],
                   u=mass_flux/model.thermo.density(state['T'], problem.pressure, state['Y']))
    report = dict(accepted=bool(accepted), reason=reason, phi=float(phi), mass_flux_kg_m2_s=float(mass_flux),
                  width_m=width, nodes=len(problem.z), elapsed_seconds=time.perf_counter()-started,
                  iterations=len(history)-1, residual_tolerance=residual_tolerance,
                  max_energy_error=max_energy_error, seed_row=seed_row, seed_sha256=digest,
                  backend='reduced_FGM_native_transport', controls=['Z', 'C', 'h'],
                  source='frozen_table', interpolation=interpolation,
                  detailed_source_evaluations=0, residual_evaluations=problem.residual_evaluations,
                  jacobian_colors=9, advection='Peclet_fitted_finite_volume',
                  solver_source_sha256=_SOURCE_SHA256,
                  table_sha256=model.table_sha256,
                  diagnostics=diagnostics, history=history)
    if not accepted:
        raise ReducedConvergenceError(report, profile)
    return profile, report
