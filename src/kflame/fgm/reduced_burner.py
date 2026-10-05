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
from scipy.interpolate import BSpline, make_interp_spline
from scipy.sparse import csc_matrix, lil_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings

from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.transport import NativeTransport
from kflame.flame.equations import _corrected_flux_frozen
from kflame.fgm.tensor_kernels import evaluate_tensor, evaluate_progress
from kflame.fgm.projection import progress_projection, stable_blend, stable_matrix
from kflame.fgm.chart_guard import guard_chart

_SOURCE_SHA256 = hashlib.sha256(b''.join(Path(__file__).with_name(q).read_bytes()
    for q in ('reduced_burner.py','tensor_kernels.py','projection.py','chart_guard.py','nonadiabatic3d.py'))).hexdigest()


class ReducedConvergenceError(RuntimeError):
    """A bounded manifold solve failed its residual or physical checks."""
    def __init__(self, report, profile):
        super().__init__(report['reason'])
        self.report, self.profile = report, profile


class ReducedBurnerProblem:
    """Conservative finite volumes on a strictly increasing nonuniform grid.

    Logical coordinates are numerical variables, not new physical controls.
    The default limited tensor approximation is C1 in all three directions.
    The positive control polygons and shared weights retain species mass;
    temperature is recovered from physical total h. Barycentric interpolation
    of the original connected mesh remains available as a comparison option.
    """
    def __init__(self, model, z, mass_flux, inlet_Y, *, interpolation='bounded_tensor', curvature_limit=.75,
                 kernel_backend='compiled', progress_equation='conservative', metric_floor=1e-30):
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
        if interpolation not in ('barycentric', 'quadratic_progress', 'bounded_tensor'):
            raise ValueError('Unknown manifold approximation')
        if interpolation == 'quadratic_progress' and self.shape[2] < 3:
            raise ValueError('Quadratic approximation needs at least three progress samples')
        self.interpolation = interpolation
        if progress_equation not in ('conservative', 'entropy_projection', 'adaptive_projection'):
            raise ValueError('Unknown progress equation')
        if progress_equation != 'conservative' and (interpolation == 'barycentric' or 'omega_Y' not in table):
            raise ValueError('Entropy projection requires a differentiable closure and frozen full species sources')
        if not np.isfinite(metric_floor) or metric_floor <= 0.:
            raise ValueError('Metric floor must be finite and positive')
        self.progress_equation, self.metric_floor = progress_equation, float(metric_floor)
        if kernel_backend not in ('compiled', 'numpy'):
            raise ValueError('Kernel backend must be compiled or numpy')
        self.kernel_backend = kernel_backend
        self.corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.knots = np.r_[np.zeros(3), np.arange(1, self.shape[2]-2),
                           np.full(3, self.shape[2]-2)]
        self.fields = None
        self.curvature_limit = float(curvature_limit)
        if not np.isfinite(self.curvature_limit) or not 0. <= self.curvature_limit <= 1.:
            raise ValueError('Tensor curvature limit must be between zero and one')
        if interpolation == 'bounded_tensor':
            if np.any(self.shape < 3):
                raise ValueError('Bounded tensor approximation needs >=3 samples per axis')
            caches = getattr(model, '_reduced_tensor_caches', {})
            cache = caches.get(self.curvature_limit)
            if cache is None:
                shape = tuple(self.shape)
                names = ['Y', 'h', 'T', 'omega_C', 'qdot']+(['omega_Y'] if 'omega_Y' in table else [])
                fields = {q: table[q].reshape((*shape, -1)).copy() for q in names}
                knots, statistics = [], []
                for axis in (0, 1):
                    u = np.linspace(0., 1., shape[axis])
                    controls = {}
                    for name, values in fields.items():
                        spline = make_interp_spline(u, values, k=2, axis=axis)
                        controls[name] = np.moveaxis(spline.c, 0, axis)
                    knots.append(spline.t)
                    # Limit the control polygon once, offline. A single scalar
                    # for every species, h and source retains linear invariants.
                    # No physical state is clipped during production lookup.
                    delta = controls['Y'] - fields['Y']
                    alpha = np.minimum(1., np.min(np.where(delta < 0.,
                        np.maximum(fields['Y'], 0.) / np.maximum(-delta, 1e-300),
                        np.inf), axis=-1))
                    alpha *= self.curvature_limit
                    # Species positivity alone does not make C monotone.
                    # Limit neighbouring control coefficients together; use
                    # the same scalar for h and sources, retaining invariants.
                    original_C = fields['Y']@table['progress_weights']
                    delta_C = delta@table['progress_weights']
                    growth = np.diff(original_C, axis=2)
                    if np.any(growth <= 0.):
                        raise ValueError('Compiled progress control polygon must increase strictly')
                    margin = np.minimum(growth*.01, 1e-12)
                    for _ in range(12):
                        current = np.diff(original_C+alpha*delta_C, axis=2)
                        bad = current <= margin
                        if not bad.any():
                            break
                        factor = np.ones_like(current)
                        factor[bad] = .99*(growth[bad]-margin[bad])/(growth[bad]-current[bad])
                        reduction = np.ones_like(alpha)
                        reduction[:,:,:-1] = np.minimum(reduction[:,:,:-1], factor)
                        reduction[:,:,1:] = np.minimum(reduction[:,:,1:], factor)
                        alpha *= reduction
                    current = np.diff(original_C+alpha*delta_C, axis=2)
                    factor = np.where(current <= margin,
                        .99*(growth-margin)/np.maximum(growth-current, 1e-300), 1.)
                    alpha *= np.min(factor, axis=2)[:,:,None]
                    fields = {q: fields[q] + alpha[..., None]*(v-fields[q])
                              for q, v in controls.items()}
                    statistics.append(dict(axis=axis, limited_coefficients=int(np.sum(alpha < 1.)),
                        minimum_progress_increment=float(np.diff(fields['Y']@table['progress_weights'],axis=2).min())))
                knots.append(self.knots/(shape[2]-2))
                fields, guard = guard_chart(fields, table, shape, knots, self.strides,
                    table['bilger_weights'],table['progress_weights'])
                statistics.append(dict(chart_guard=guard))
                cache = dict(fields={q: v.reshape(-1, v.shape[-1]) for q, v in fields.items()},
                             knots=knots, limiter=statistics)
                caches[self.curvature_limit] = cache
                model._reduced_tensor_caches = caches
            self.fields, self.axis_knots = cache['fields'], cache['knots']
            if 'packed' not in cache:
                cache['packed'] = np.column_stack([self.fields['Y'],
                    *[self.fields[q].ravel() for q in ('h','T','omega_C','qdot')]])
                cache['progress_coefficients'] = (self.fields['Y']@table['progress_weights']).reshape(tuple(self.shape))
            self.packed_fields = cache['packed']
            self.progress_coefficients = cache['progress_coefficients']
            self.limiter_statistics = cache['limiter']
        if not hasattr(model, '_reduced_raw_fields'):
            model._reduced_raw_fields = np.column_stack([table['Y'],
                *[table[q].ravel() for q in ('h','T','omega_C','qdot')]])
        self.raw_fields = model._reduced_raw_fields
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

    def state(self, x, *, geometry=None, temperature_cache=None, changed_nodes=None, require_grid=True):
        guide_geometry = None
        if isinstance(geometry, tuple) and len(geometry) == 3 and geometry[0] == 'closure_blend':
            geometry, guide_geometry = geometry[1:]
        x = np.asarray(x).reshape(-1, 3)
        if (require_grid and len(x) != len(self.z)) or not np.isfinite(x).all():
            raise ValueError('Three finite logical coordinates per spatial node are required')
        if geometry is None and (np.any(x < 0.) or np.any(x > 1.)):
            raise ValueError('Trial state outside the resolved logical mesh')
        logical = x * (self.shape-1)
        values = None
        if self.interpolation == 'bounded_tensor' and self.kernel_backend == 'compiled':
            values, _ = evaluate_tensor(x, self.axis_knots, self.strides, self.packed_fields)
            active_geometry = True
        elif self.interpolation == 'bounded_tensor':
            parts = []
            for k in range(3):
                basis = BSpline.design_matrix(x[:, k], self.axis_knots[k], 2,
                                               extrapolate=geometry is not None).tocsr()
                parts.append((basis.data.reshape(-1, 3), basis.indices.reshape(-1, 3)))
            nodes = (parts[0][1][:, :, None, None]*self.strides[0]
                     + parts[1][1][:, None, :, None]*self.strides[1]
                     + parts[2][1][:, None, None, :]).reshape(-1, 27)
            bary = (parts[0][0][:, :, None, None]*parts[1][0][:, None, :, None]
                    * parts[2][0][:, None, None, :]).reshape(-1, 27)
            active_geometry = True
        elif self.interpolation == 'quadratic_progress':
            base = (np.minimum(np.floor(logical[:, :2]).astype(int), self.shape[:2]-2)
                    if geometry is None else geometry)
            if self.kernel_backend == 'compiled':
                values, _ = evaluate_progress(x, base, self.knots/(self.shape[2]-2),
                    self.shape, self.strides, self.raw_fields)
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
        fields = self.fields if self.fields is not None else t
        if values is not None:
            Y = values[:, :self.mech.n_species]
            h, guess, omega, heat = values[:, self.mech.n_species:self.mech.n_species+4].T
        else:
            Y = np.einsum('ni,nij->nj', bary, fields['Y'][nodes])
            h = np.einsum('ni,ni->n', bary, np.asarray(fields['h']).reshape(-1)[nodes])
            guess = np.einsum('ni,ni->n', bary, np.asarray(fields['T']).reshape(-1)[nodes])
            omega = np.einsum('ni,ni->n', bary, np.asarray(fields['omega_C']).reshape(-1)[nodes])
            heat = np.einsum('ni,ni->n', bary, np.asarray(fields['qdot']).reshape(-1)[nodes])
        blend = getattr(self, 'closure_blend', 1.)
        if blend < 1.:
            guide = self.initialization_problem.state(x, geometry=guide_geometry)
            Y = blend*Y + (1.-blend)*guide['Y'].T
            h = blend*h + (1.-blend)*guide['h']
            guess = blend*guess + (1.-blend)*guide['T']
            omega = blend*omega + (1.-blend)*guide['omega_C']
            heat = blend*heat + (1.-blend)*guide['qdot']
            active_geometry = ('closure_blend', active_geometry, guide['geometry'])
        if temperature_cache is None:
            T = self.model._recover_temperature(h, Y.T, guess)
        else:
            # Colored finite differences change only one third of the nodes.
            # Keep the exactly unchanged temperatures instead of repeatedly
            # solving their thermodynamic inversions.
            T = np.asarray(temperature_cache).copy()
            T[changed_nodes] = self.model._recover_temperature(
                h[changed_nodes], Y[changed_nodes].T, guess[changed_nodes])
        result = dict(Y=Y.T, T=T, h=h,
                    Z=t['bilger_weights'] @ Y.T + float(t['bilger_offset']),
                    C=t['progress_weights'] @ Y.T,
                    omega_C=omega, qdot=heat,
                    geometry=active_geometry)
        return result

    def state_derivatives(self, x):
        """Analytic species/enthalpy chart tangents, including continuation."""
        x = np.asarray(x).reshape(-1, 3)
        if self.interpolation == 'bounded_tensor':
            _, gradient = evaluate_tensor(x, self.axis_knots, self.strides, self.packed_fields,
                                           differentiated=self.mech.n_species+1)
        elif self.interpolation == 'quadratic_progress':
            base = np.minimum(np.floor(x[:,:2]*(self.shape[:2]-1)).astype(int), self.shape[:2]-2)
            _, gradient = evaluate_progress(x, base, self.knots/(self.shape[2]-2),
                self.shape, self.strides, self.raw_fields, differentiated=self.mech.n_species+1)
        else:
            raise ValueError('Analytic tangents require a differentiable closure')
        blend = getattr(self, 'closure_blend', 1.)
        dY, dh = gradient[:,:,:self.mech.n_species].transpose(0,2,1), gradient[:,:,-1]
        if blend < 1.:
            guide_Y, guide_h = self.initialization_problem.state_derivatives(x)
            dY, dh = blend*dY+(1.-blend)*guide_Y, blend*dh+(1.-blend)*guide_h
        return dY, dh

    def species_sources(self, x):
        """Query full frozen kinetics only where the progress projection needs it."""
        x = np.asarray(x).reshape(-1,3)
        if self.interpolation == 'bounded_tensor':
            values,_ = evaluate_tensor(x,self.axis_knots,self.strides,self.fields['omega_Y'])
        else:
            base=np.minimum(np.floor(x[:,:2]*(self.shape[:2]-1)).astype(int),self.shape[:2]-2)
            values,_ = evaluate_progress(x,base,self.knots/(self.shape[2]-2),self.shape,self.strides,self.model.table['omega_Y'])
        blend=getattr(self,'closure_blend',1.)
        if blend<1.:
            values=blend*values+(1.-blend)*self.initialization_problem.species_sources(x).T
        return values.T

    def principal_data(self, x, state):
        dY, dh = self.state_derivatives(x)
        T,Y,mw=state['T'],state['Y'],self.mech.molecular_weights
        hk = self.model.thermo.partial_molar_enthalpies(T)/mw[:,None]
        cp = self.model.thermo.cp_mass(T,Y)
        rho,D,lam,W=self.transport.eval_faces_poly_fast(T,self.pressure,Y,1./mw)
        dT=(dh-np.einsum('nj,njk->nk',hk.T,dY))/cp[:,None]
        dW=-W[:,None]**2*np.einsum('j,njk->nk',1./mw,dY)
        dX=(W[:,None,None]*dY+Y.T[:,:,None]*dW[:,None,:])/mw[None,:,None]
        J=-rho[:,None,None]*(mw[None,:,None]/W[:,None,None])*D.T[:,:,None]*dX
        J-=Y.T[:,:,None]*J.sum(axis=1)[:,None,:]
        table=self.model.table
        M=np.stack([np.einsum('j,njk->nk',table['bilger_weights'],dY),
                    np.einsum('j,njk->nk',table['progress_weights'],dY),dh],axis=1)
        B=np.stack([np.einsum('j,njk->nk',table['bilger_weights'],J),
                    np.einsum('j,njk->nk',table['progress_weights'],J),
                    -lam[:,None]*dT+np.einsum('nj,njk->nk',hk.T,J)],axis=1)
        margin=1e-3*np.minimum((rho[None,:]*D).min(axis=0),lam/cp)
        return dY,dh,hk,cp,J,M,B,margin

    def projection_multipliers(self, x, state, *, cache=None, changed_nodes=None):
        x=np.asarray(x).reshape(-1,3)
        if cache is not None:
            alpha,beta,gamma=[v.copy() for v in cache]
            subset=np.asarray(changed_nodes,dtype=int)
            selected_state=dict(Y=state['Y'][:,subset],T=state['T'][subset])
            a,b,g=self.projection_multipliers(x[subset],selected_state)
            alpha[:,subset],beta[subset],gamma[subset]=a,b,g
            return alpha,beta,gamma
        dY,dh,hk,cp,J,M,B,margin=self.principal_data(x,state)
        metric=self.scales/self.mass_flux
        base=np.linalg.solve(M/metric[None,:,None],-B/metric[None,:,None])
        active=(~stable_matrix(base,margin) if self.progress_equation=='adaptive_projection' else np.ones(len(x),dtype=bool))
        alpha=np.broadcast_to(self.model.table['progress_weights'][:,None],state['Y'].shape).copy()
        beta=np.zeros(len(x));gamma=np.zeros(len(x))
        if active.any():
            a,b=progress_projection(state['Y'][:,active],state['T'][active],dY[active],dh[active],hk[:,active],cp[active],
                self.mech.molecular_weights,self.model.table['bilger_weights'],self.model.table['progress_weights'],metric_floor=self.metric_floor)
            Mp,Bp=M[active].copy(),B[active].copy()
            Mp[:,1]=np.einsum('nj,njk->nk',a.T,dY[active])+b[:,None]*dh[active]
            Bp[:,1]=np.einsum('nj,njk->nk',a.T,J[active])+b[:,None]*Bp[:,2]
            end=np.linalg.solve(Mp/metric[None,:,None],-Bp/metric[None,:,None])
            chosen=(stable_blend(base[active],end,margin[active]) if self.progress_equation=='adaptive_projection' else np.ones(active.sum()))
            alpha[:,active]=(1.-chosen)[None,:]*alpha[:,active]+chosen[None,:]*a
            beta[active]=chosen*b;gamma[active]=chosen
        blend = getattr(self, 'closure_blend', 1.)
        return blend*alpha+(1.-blend)*self.model.table['progress_weights'][:,None], blend*beta,blend*gamma

    def initial_progress(self, target, row, *, composition_index=None, loss_index=None):
        """Invert the strictly monotone row C for the chosen approximation."""
        if self.interpolation == 'barycentric':
            return np.interp(target, row, np.linspace(0., 1., len(row)))
        if self.interpolation == 'bounded_tensor':
            if composition_index is None or loss_index is None:
                raise ValueError('Tensor progress inversion requires the training row indices')
            partial = self.progress_coefficients
            for axis, index in enumerate((composition_index, loss_index)):
                u = index/(self.shape[axis]-1)
                partial = BSpline(self.axis_knots[axis], partial, 2, axis=0)(u)
            spline = BSpline(self.axis_knots[2], partial, 2)
            factor = 1.
        else:
            spline = BSpline(self.knots, row, 2)
            factor = len(row)-2
        lo, hi = np.zeros_like(target), np.ones_like(target)
        for _ in range(35):
            mid = .5*(lo+hi)
            below = spline(mid*factor) < target
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
        species_total = self.mass_flux*((1.-weight)[None,:]*Y[:,:-1]+weight[None,:]*Y[:,1:])+J
        return dict(total=F, species=J, species_total=species_total, conduction=conduction,
                    enthalpy_diffusion=enthalpy_diffusion, conductivity=conductivity)

    def residual(self, x, *, geometry=None, temperature_cache=None, changed_nodes=None, evaluated_state=None, projection_cache=None):
        self.residual_evaluations += 1
        state = (self.state(x, geometry=geometry, temperature_cache=temperature_cache, changed_nodes=changed_nodes)
                 if evaluated_state is None else evaluated_state)
        flux = self.fluxes(state)
        out = np.empty((len(self.z), 3))
        out[1:-1] = np.diff(flux['total'], axis=0)
        out[1:-1, 1] -= state['omega_C'][1:-1]*self.volumes
        self._last_projection_coefficients = None
        if self.progress_equation != 'conservative' and getattr(self, 'closure_blend', 1.) > 0.:
            alpha,beta,gamma=self.projection_multipliers(x,state,cache=projection_cache,changed_nodes=changed_nodes)
            self._last_projection_coefficients=(alpha,beta,gamma)
            active=np.flatnonzero(gamma[1:-1]>0.)+1
            if len(active):
                species_residual=flux['species_total'][:,active]-flux['species_total'][:,active-1]
                species_residual-=self.species_sources(np.asarray(x).reshape(-1,3)[active])*self.volumes[active-1][None,:]
                out[active,1]=np.sum(alpha[:,active]*species_residual,axis=0)+beta[active]*out[active,2]
        out[0, 0] = flux['total'][0, 0]-self.mass_flux*self.Z_feed
        out[0, 1] = flux['total'][0, 1]-self.mass_flux*self.C_feed
        out /= self.scales
        logical = np.asarray(x).reshape(-1, 3)
        out[0, 2] = logical[0, 2]  # exactly T(0)=T_burner
        out[-1] = logical[-1]-logical[-2]  # zero gradients of T/Y/Z/C/h
        return out.ravel()

    def diagnostics(self, x):
        state = self.state(x)
        flux = self.fluxes(state)
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
                    physical_progress_integral_balance_error=abs(progress_integral-progress_out)/max(abs(progress_out), 1e-12),
                    inlet_species_flux_error_over_mdot=float(np.max(abs(inlet_species))/self.mass_flux),
                    inlet_enthalpy_flux_error_W_m2=float(inlet_energy-self.mass_flux*self.h_feed),
                    sum_Y_error=float(np.max(abs(state['Y'].sum(axis=0)-1.))),
                    burner_temperature_error_K=float(abs(state['T'][0]-self.T_burner)),
                    maximum_scaled_residual=float(np.max(abs(self.residual(x)))))

    def diffusion_diagnostics(self, x, *, interior_only=True):
        """Check the local principal diffusion matrix of the reduced closure.

        Positive parent species diffusivities do not guarantee a parabolic
        oblique projection onto Z/C/h. A negative real eigenvalue is a closure
        failure, not something to conceal by increasing the Newton budget.
        This numerical diagnostic is at the solved nodes, not a global proof.
        """
        x = np.asarray(x).reshape(-1, 3)
        state = self.state(x,require_grid=False)
        Y, T, mw = state['Y'], state['T'], self.mech.molecular_weights
        rho, D, lam, W = self.transport.eval_faces_poly_fast(T, self.pressure, Y, 1./mw)
        coeff = rho[None, :]*(mw[:, None]/W[None, :])*D
        hk = self.model.thermo.partial_molar_enthalpies(T)/mw[:, None]
        controls, diffusion, species_derivatives, species_diffusion, enthalpy_derivatives = [], [], [], [], []
        analytic = self.interpolation in ('bounded_tensor','quadratic_progress') and getattr(self, 'closure_blend', 1.) == 1.
        derivatives = None
        if analytic and (self.kernel_backend == 'compiled' or self.interpolation == 'quadratic_progress'):
            dYs, dhs = self.state_derivatives(x)
            derivatives = np.concatenate([dYs.transpose(0,2,1),dhs[:,:,None]],axis=2)
            cp = self.model.thermo.cp_mass(T, Y)
        elif analytic:
            parts = []
            for k in range(3):
                basis = BSpline.design_matrix(x[:,k], self.axis_knots[k], 2).tocsr()
                indices = basis.indices.reshape(-1, 3)
                derivative = BSpline(self.axis_knots[k], np.eye(self.shape[k]), 2)(x[:,k], nu=1)
                parts.append((basis.data.reshape(-1, 3), indices,
                              np.take_along_axis(derivative, indices, axis=1)))
            nodes = (parts[0][1][:,:,None,None]*self.strides[0]
                     +parts[1][1][:,None,:,None]*self.strides[1]
                     +parts[2][1][:,None,None,:]).reshape(-1, 27)
            cp = self.model.thermo.cp_mass(T, Y)
        for k in range(3):
            if analytic:
                if derivatives is not None:
                    dY = derivatives[:, k, :self.mech.n_species].T
                    dh = derivatives[:, k, self.mech.n_species]
                else:
                    weights = [parts[j][2 if j == k else 0] for j in range(3)]
                    weights = (weights[0][:,:,None,None]*weights[1][:,None,:,None]
                               *weights[2][:,None,None,:]).reshape(-1, 27)
                    dY = np.einsum('ni,nij->nj', weights, self.fields['Y'][nodes]).T
                    dh = np.einsum('ni,ni->n', weights, self.fields['h'].ravel()[nodes])
                dT = (dh-np.sum(hk*dY, axis=0))/cp
                dW = -W**2*np.sum(dY/mw[:,None], axis=0)
                dX = (W[None,:]*dY+Y*dW[None,:])/mw[:,None]
                J = -coeff*dX
                J -= Y*np.sum(J, axis=0)[None,:]
                controls.append(np.column_stack([self.model.table['bilger_weights']@dY,
                    self.model.table['progress_weights']@dY, dh]))
            else:
                trial = x.copy()
                trial[:, k] += 1e-7
                other = self.state(trial, geometry=state['geometry'],require_grid=False)
                controls.append(np.column_stack([(other[q]-state[q])/1e-7 for q in ('Z','C','h')]))
                J = _corrected_flux_frozen(Y, other['Y'], coeff, np.ones(len(T)), mw, 'molar')/1e-7
                dT = (other['T']-T)/1e-7
            diffusion.append(np.column_stack([self.model.table['bilger_weights']@J,
                self.model.table['progress_weights']@J, -lam*dT+np.sum(hk*J, axis=0)]))
            species_derivatives.append(dY.T if analytic else ((other['Y']-Y)/1e-7).T)
            species_diffusion.append(J.T)
            enthalpy_derivatives.append(dh if analytic else (other['h']-state['h'])/1e-7)
        try:
            metric = self.scales/self.mass_flux
            physical_controls = np.stack(controls, axis=-1)
            orientation = np.linalg.det(physical_controls/metric[None,:,None])
            principal_controls = physical_controls.copy()
            principal_diffusion = np.stack(diffusion, axis=-1)
            gamma = np.zeros(len(x))
            if self.progress_equation != 'conservative':
                dYs = np.stack(species_derivatives, axis=-1)
                dhs = np.stack(enthalpy_derivatives, axis=-1)
                alpha, beta, gamma = self.projection_multipliers(x,state)
                principal_controls[:,1] = np.einsum('nj,njk->nk',alpha.T,dYs)+beta[:,None]*dhs
                principal_diffusion[:,1] = np.einsum('nj,njk->nk',alpha.T,np.stack(species_diffusion,axis=-1))+beta[:,None]*principal_diffusion[:,2]
            eigenvalues = np.linalg.eigvals(-(principal_diffusion/metric[None,:,None])
                        @ np.linalg.inv(principal_controls/metric[None,:,None]))
        except np.linalg.LinAlgError:
            return dict(passed=False, reason='singular_control_tangent')
        tested = eigenvalues[1:-1] if interior_only else eigenvalues
        tested_orientation = orientation[1:-1] if interior_only else orientation
        minimum = float(np.min(tested.real))
        return dict(passed=bool(np.isfinite(eigenvalues).all() and minimum >= -1e-10
                                and np.all(tested_orientation > 0.)),
                    minimum_real_eigenvalue_kg_m_s=minimum,
                    negative_nodes=int(np.sum(tested.real.min(axis=1) < -1e-10)),
                    negative_node_indices=(np.flatnonzero(tested.real.min(axis=1) < -1e-10)
                                           +int(interior_only)).tolist(),
                    boundary_nodes_excluded=bool(interior_only),
                    progress_equation=self.progress_equation,
                    projected_nodes=int(np.count_nonzero(gamma)), maximum_projection_blend=float(gamma.max()),
                    nonpositive_chart_nodes=int(np.count_nonzero(tested_orientation <= 0.)),
                    minimum_scaled_chart_determinant=float(tested_orientation.min()),
                    diagnostic='nodal_principal_diffusion_matrix',
                    derivatives='analytic_spline_and_enthalpy' if analytic else 'finite_difference',
                    finite_difference_step=None if analytic else 1e-7)


def _newton(problem, initial, *, residual_tolerance, max_iterations, verbose):
    x = np.clip(np.asarray(initial, dtype=float).ravel(), 1e-7, 1.-1e-5)
    x[2] = 0.
    n = len(problem.z)
    pattern = lil_matrix((3*n, 3*n), dtype=int)
    for i in range(n):
        pattern[3*i:3*i+3, max(0, 3*(i-1)):min(3*n, 3*(i+2))] = 1
    pattern = pattern.tocsc()
    columns = np.repeat(np.arange(3*n), np.diff(pattern.indptr))
    color_entries = [np.flatnonzero(columns % 9 == color) for color in range(9)]
    history, reason = [], 'iteration_budget_exhausted'
    for iteration in range(max_iterations+1):
        base_state = problem.state(x)
        f = problem.residual(x, evaluated_state=base_state)
        projection_cache = problem._last_projection_coefficients
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
        geometry = base_state['geometry']
        data = np.empty_like(pattern.data, dtype=float)
        for color in range(9):
            selected = np.arange(color, len(x), 9)
            steps = np.full(len(selected), 1e-7)
            plus = x.copy()
            plus[selected] += steps
            difference = problem.residual(plus, geometry=geometry,
                temperature_cache=base_state['T'], changed_nodes=selected//3, projection_cache=projection_cache)-f
            entries = color_entries[color]
            data[entries] = difference[pattern.indices[entries]]/1e-7
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
                difference = problem.residual(perturbed, temperature_cache=base_state['T'], changed_nodes=selected//3, projection_cache=projection_cache)-f
                entries = color_entries[color]
                data[entries] = difference[pattern.indices[entries]]/steps[(columns[entries]-color)//9]
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


def _continue_closure(problem, guide, x, *, residual_tolerance, max_iterations, verbose):
    """Adapt the interpolation closure from the cheap guide to the final map.

    Only fully converged intermediate steps advance the continuation parameter.
    The final result always solves the unblended higher-order balances.
    """
    problem.initialization_problem = guide
    parameter, increment, stages, history = 0., .25, [], []
    while parameter < 1. and len(stages) < 40:
        remaining = max_iterations-sum(s['iterations'] for s in stages)
        if remaining <= 0:
            break
        target = min(1., parameter+increment)
        problem.closure_blend = target
        trial, local, reason = _newton(problem, x, residual_tolerance=residual_tolerance,
            max_iterations=min(remaining, 40), verbose=False)
        stages.append(dict(parameter=target, accepted=reason == 'converged', reason=reason,
                           iterations=len(local)-1, maximum_scaled_residual=local[-1]['maximum_scaled_residual']))
        if verbose:
            print(f'FGM closure continuation {target:.4g}: {reason}', flush=True)
        if reason == 'converged':
            x, parameter = trial, target
            for record in local:
                history.append(dict(record, iteration=len(history), closure_parameter=target))
            increment = min(.5, increment*(1.5 if len(local) < 8 else 1.1))
        else:
            increment *= .5
            if increment < 1e-4:
                break
    problem.closure_blend = 1.
    if not history:
        f = problem.residual(x)
        history = [dict(iteration=0, maximum_scaled_residual=float(np.max(abs(f))),
                        norm=float(np.linalg.norm(f)), closure_parameter=1.)]
    return x, history, ('converged' if parameter == 1. else 'closure_continuation_failed'), stages


def solve_reduced_burner_fgm(model, *, phi, mass_flux, seed_profile, seed_row,
                             width=None, grid=None, max_spacing=1.25e-4,
                             initial_solution=None, residual_tolerance=1e-7,
                             max_iterations=180, max_energy_error=.02,
                             interpolation='bounded_tensor', curvature_limit=.75,
                             progress_equation='conservative', metric_floor=1e-30,
                             kernel_backend='compiled', verbose=False):
    """Solve a burner using a frozen FGM and a verified training warm start.

    ``seed_row`` is an output name from the table metadata. A detailed withheld
    solution is not a seed. The seed fixes only the initial guess and grid;
    all final controls are obtained by solving the reduced balances.
    A previous reduced iterate from the same frozen table may supply a start.
    This initial implementation uses the table's fixed pressure/temperature.
    """
    total_started = time.perf_counter()
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
    problem = ReducedBurnerProblem(model, grid, mass_flux, feed, interpolation=interpolation,
                                    curvature_limit=curvature_limit, progress_equation=progress_equation,
                                    metric_floor=metric_floor,kernel_backend=kernel_backend)
    i, j = record['composition_index'], record['loss_index']
    C_seed = model.table['progress_weights'] @ seed['Y']
    C_row = model.table['C'].reshape(tuple(problem.shape))[i, j]
    if np.any(np.diff(C_row) <= 0.):
        raise ValueError('Training progress samples must increase strictly')
    progress = np.minimum(.985, problem.initial_progress(np.interp(problem.z, seed['z'], C_seed), C_row,
        composition_index=i, loss_index=j))
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
    initialization_started = time.perf_counter()
    guide_report = None
    if initial_solution is None and (interpolation == 'bounded_tensor' or progress_equation != 'conservative'):
        guide = ReducedBurnerProblem(model, problem.z, mass_flux, feed, interpolation='quadratic_progress',
                                     kernel_backend=kernel_backend)
        guide_initial = initial.copy()
        guide_initial[:, 2] = np.minimum(.985, guide.initial_progress(
            np.interp(problem.z, seed['z'], C_seed), C_row))
        guide_started = time.perf_counter()
        guide_x, guide_history, guide_reason = _newton(guide, guide_initial,
            residual_tolerance=residual_tolerance, max_iterations=min(max_iterations, 40), verbose=False)
        initial = guide_x.reshape(-1, 3)
        guide_report = dict(reason=guide_reason, iterations=len(guide_history)-1,
                            maximum_scaled_residual=guide_history[-1]['maximum_scaled_residual'],
                            elapsed_seconds=time.perf_counter()-guide_started,
                            backend='reduced_FGM_low_order_initialization', detailed_source_evaluations=0)
    if initial_solution is not None:
        if str(initial_solution.get('table_sha256', '')) != model.table_sha256:
            raise ValueError('Previous iterate must identify the same frozen FGM table')
        if str(initial_solution.get('interpolation', '')) != interpolation:
            raise ValueError('Previous iterate must use the same interpolation coordinates')
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
    initialization_seconds = started-initialization_started
    setup_seconds = started-total_started
    continuation = None
    remaining = max_iterations-(guide_report['iterations'] if guide_report is not None else 0)
    if guide_report is not None and guide_report['reason'] == 'converged':
        x, history, reason, continuation = _continue_closure(problem, guide, initial.ravel(),
            residual_tolerance=residual_tolerance, max_iterations=max(0,remaining), verbose=verbose)
    else:
        x, history, reason = _newton(problem, initial, residual_tolerance=residual_tolerance,
                                     max_iterations=max(0,remaining), verbose=verbose)
    state = problem.state(x)
    flux = problem.fluxes(state)
    diagnostics = problem.diagnostics(x)
    parabolicity = problem.diffusion_diagnostics(x)
    face_x = .5*(x.reshape(-1,3)[:-1]+x.reshape(-1,3)[1:])
    face_parabolicity = problem.diffusion_diagnostics(face_x,interior_only=False)
    accepted = (reason == 'converged' and diagnostics['relative_energy_closure_error'] <= max_energy_error
                and np.max(state['T'])-problem.T_burner > 50.
                and diagnostics['sum_Y_error'] < 1e-10 and parabolicity['passed'] and face_parabolicity['passed'])
    if reason == 'converged' and not accepted:
        reason = 'physical_acceptance_failed'
    profile = {k: v for k, v in state.items() if k != 'geometry'}
    profile.update(z=problem.z, logical_coordinates=x.reshape(-1, 3),
                   table_sha256=np.array(model.table_sha256),
                   interpolation=np.array(interpolation),
                   z_face=.5*(problem.z[:-1]+problem.z[1:]),
                   total_enthalpy_flux=flux['total'][:, 2],
                   conductive_heat_flux=flux['conduction'], diffusive_enthalpy_flux=flux['enthalpy_diffusion'],
                   u=mass_flux/model.thermo.density(state['T'], problem.pressure, state['Y']))
    iterations = sum(s['iterations'] for s in continuation) if continuation is not None else len(history)-1
    report = dict(accepted=bool(accepted), reason=reason, phi=float(phi), mass_flux_kg_m2_s=float(mass_flux),
                  width_m=width, nodes=len(problem.z), elapsed_seconds=time.perf_counter()-started,
                  total_elapsed_seconds=time.perf_counter()-total_started, setup_seconds=setup_seconds,
                  iterations=iterations, total_newton_iterations=iterations+(guide_report['iterations'] if guide_report else 0),
                  max_iterations=max_iterations, residual_tolerance=residual_tolerance,
                  initialization_seconds=initialization_seconds,
                  initialization_guide=guide_report,
                  closure_continuation=continuation,
                  max_energy_error=max_energy_error, seed_row=seed_row, seed_sha256=digest,
                  backend='reduced_FGM_native_transport', controls=['Z', 'C', 'h'],
                  source='frozen_table', interpolation=interpolation,
                  curvature_limit=curvature_limit,
                  progress_equation=progress_equation,metric_floor=metric_floor,kernel_backend=kernel_backend,
                  detailed_source_evaluations=0, residual_evaluations=problem.residual_evaluations,
                  jacobian_colors=9, advection='Peclet_fitted_finite_volume',
                  solver_source_sha256=_SOURCE_SHA256,
                  parabolicity=parabolicity,
                  face_parabolicity=face_parabolicity,limiter_statistics=problem.limiter_statistics if interpolation=='bounded_tensor' else None,
                  table_sha256=model.table_sha256,
                  diagnostics=diagnostics, history=history)
    if not accepted:
        raise ReducedConvergenceError(report, profile)
    return profile, report
