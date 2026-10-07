"""Native axisymmetric counterflow diffusion flames with two inflow boundaries.

The unknowns are axial mass flux, radial spread rate, temperature, all mass
fractions, and one radial pressure eigenvalue. Chemistry and transport use the
same native kernels as premixed flames; no reference solution seeds the solve.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import erfc

from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.initialization import hp_equilibrium, oxygen_demand
from kflame.flame.equations import _corrected_flux_frozen
from kflame.flame.solver import AdaptiveRefiner
from kflame.flame.diffusion_kernels import assemble_residual

_THERMO_BATCH_BUDGET_BYTES=96*1024**2

@dataclass
class DiffusionCase:
    mech: str
    fuel_Y: np.ndarray
    oxidizer_Y: np.ndarray
    fuel_temperature: float = 300.
    oxidizer_temperature: float = 300.
    fuel_mass_flux: float = .12
    oxidizer_mass_flux: float = .24
    P: float = 101325.
    width: float = .02
    transport_model: str = 'mixture-averaged'
    soret_enabled: bool = False
    flux_gradient_basis: str = 'molar'


def upwind_gradient(values, z, mass_flux):
    """Both signs of axial flow occur; switch the donor at stagnation."""
    left = (values[...,1:-1]-values[...,:-2])/np.diff(z)[:-1]
    right = (values[...,2:]-values[...,1:-1])/np.diff(z)[1:]
    return np.where(mass_flux[1:-1] >= 0.,left,right)


def enthalpy_face_balance(z,m,rho,V,h,diffusive_heat_flux):
    """Face-to-face diagnostic compatible with trapezoidal continuity.

    Integrate the linearly reconstructed radial mass export to each midpoint;
    averaging nodal m*h would break invariance to a common enthalpy offset.
    This diagnostic does not remove truncation error from the energy scheme.
    """
    dz=np.diff(z);zface=.5*(z[:-1]+z[1:]);radial_mass=2.*rho*V
    mface=.5*(m[:-1]+m[1:])+dz*(radial_mass[1:]-radial_mass[:-1])/8.
    totalflux=mface*.5*(h[:-1]+h[1:])+diffusive_heat_flux
    radial=radial_mass*h
    integration_z=np.r_[zface[0],z[1:-1],zface[-1]]
    export=float(np.trapezoid(np.interp(integration_z,z,radial),integration_z))
    defect=float(totalflux[-1]-totalflux[0]+export)
    return zface,totalflux,radial,export,defect


class DiffusionFlameProblem:
    def __init__(self, case, mech, z):
        self.case, self.mech_data, self.P = case, mech, case.P
        self.transport_model, self.soret_enabled = case.transport_model, case.soret_enabled
        self.flux_gradient_basis = 'molar'
        self.assume_finite_y = True
        self.backend = NativeSpeciesBackend(self,mech)
        self.thermo = self.backend.thermo
        self.ns, self.nv = mech.n_species, mech.n_species+3
        self.rhof = float(self.thermo.density(case.fuel_temperature,case.P,case.fuel_Y))
        self.rhoo = float(self.thermo.density(case.oxidizer_temperature,case.P,case.oxidizer_Y))
        self.mref = max(case.fuel_mass_flux,case.oxidizer_mass_flux)
        self.rhoref = .5*(self.rhof+self.rhoo)
        self.aref = (case.fuel_mass_flux/self.rhof+case.oxidizer_mass_flux/self.rhoo)/case.width
        self.lref = self.rhoref*self.aref**2
        demand = oxygen_demand(mech)*mech.inv_molecular_weights
        df, do = float(demand@case.fuel_Y), float(demand@case.oxidizer_Y)
        if df <= 0. or do >= 0.:
            raise ValueError('The fuel and oxidizer streams must define a combustible mixture')
        self.Zst = -do/(df-do)
        Yst = self.Zst*case.fuel_Y+(1.-self.Zst)*case.oxidizer_Y
        hst = (self.Zst*self.thermo.enthalpy_mass(case.fuel_temperature,case.fuel_Y)
               +(1.-self.Zst)*self.thermo.enthalpy_mass(case.oxidizer_temperature,case.oxidizer_Y))
        Tst = self.Zst*case.fuel_temperature+(1.-self.Zst)*case.oxidizer_temperature
        for _ in range(8):
            h, cp = self.thermo.enthalpy_mass_cp_mass(Tst,Yst)
            Tst -= float((h-hst)/cp)
        self.Teq, self.Yeq, _ = hp_equilibrium(mech,Tst,case.P,Yst)
        self.Tref = max(self.Teq,case.fuel_temperature,case.oxidizer_temperature)
        self.cpref = float(self.thermo.cp_mass(self.Teq,self.Yeq))
        self.species_scale = self.mref/case.width
        self.energy_scale = self.species_scale*self.cpref*self.Tref
        self.set_grid(z)

    def set_grid(self,z):
        self.z = np.asarray(z,dtype=float)
        self.n = len(z)
        self.dz = np.diff(z)
        self.volumes = .5*(self.dz[:-1]+self.dz[1:])
        self.excess = np.argmax(self.initial_fields()['Y'],axis=0)
        self.column_rows = []
        for j in range(self.n):
            rows = np.arange(max(j-1,0)*self.nv,min(j+2,self.n)*self.nv)
            if j >= self.n-2:
                rows = np.r_[rows,self.n*self.nv]
            self.column_rows.append(rows)
        self.color_rows=[]
        for color in range(3):
            nodes=np.arange(color,self.n,3)
            lengths=np.array([len(self.column_rows[j]) for j in nodes])
            rows=np.concatenate([self.column_rows[j] for j in nodes])
            self.color_rows.append((nodes,lengths,rows))

    def initial_fields(self):
        c, z = self.case, self.z
        uf, uo = c.fuel_mass_flux/self.rhof, c.oxidizer_mass_flux/self.rhoo
        momentum_f, momentum_o = np.sqrt(c.fuel_mass_flux*uf),np.sqrt(c.oxidizer_mass_flux*uo)
        xstagnation = c.width*momentum_f/(momentum_f+momentum_o)
        _, D, _, _ = self.backend.eval_faces(np.array([self.Teq]),self.Yeq[:,None])
        diffusivity = float(np.mean(D[:,0]))
        mixing = .5*erfc(np.sqrt(self.aref/(2.*diffusivity))*(z-xstagnation))
        rich = np.maximum((mixing-self.Zst)/(1.-self.Zst),0.)
        lean = np.minimum(mixing/self.Zst,1.)
        Y = np.where((mixing > self.Zst)[None,:],
            self.Yeq[:,None]+(c.fuel_Y-self.Yeq)[:,None]*rich[None,:],
            c.oxidizer_Y[:,None]+(self.Yeq-c.oxidizer_Y)[:,None]*lean[None,:])
        T = np.where(mixing > self.Zst,self.Teq+(c.fuel_temperature-self.Teq)*rich,
                     c.oxidizer_temperature+(self.Teq-c.oxidizer_temperature)*lean)
        T[0],T[-1] = c.fuel_temperature,c.oxidizer_temperature
        u = uf-(uf+uo)*z/c.width
        rho = self.thermo.density(T,self.P,Y)
        m = rho*u
        m[0],m[-1] = c.fuel_mass_flux,-c.oxidizer_mass_flux
        V = np.interp(z,[0.,xstagnation,c.width],[0.,self.aref,0.])
        return dict(T=T,Y=Y,m=m,V=V,Lambda=-self.lref)

    def pack(self,fields):
        blocks = np.column_stack([fields['m']/self.mref,fields['V']/self.aref,
                                  fields['T']/self.Tref,fields['Y'].T])
        return np.r_[blocks.ravel(),fields['Lambda']/self.lref]

    def equilibrated_guess(self):
        """Native HP mixing-line seed when an infinitely-fast seed stalls.

        Each initial parcel retains its two-stream mixture and enthalpy; this
        changes only the guess, not the final equations or inlet conditions.
        """
        f=self.initial_fields()
        demand=oxygen_demand(self.mech_data)*self.mech_data.inv_molecular_weights
        df,do=demand@self.case.fuel_Y,demand@self.case.oxidizer_Y
        Z=np.clip((demand@f['Y']-do)/(df-do),0.,1.)
        for j,mixture in enumerate(Z):
            Y=mixture*self.case.fuel_Y+(1.-mixture)*self.case.oxidizer_Y
            temperature=mixture*self.case.fuel_temperature+(1.-mixture)*self.case.oxidizer_temperature
            target=(mixture*self.thermo.enthalpy_mass(self.case.fuel_temperature,self.case.fuel_Y)
                    +(1.-mixture)*self.thermo.enthalpy_mass(self.case.oxidizer_temperature,self.case.oxidizer_Y))
            for _ in range(8):
                h,cp=self.thermo.enthalpy_mass_cp_mass(temperature,Y)
                temperature-=float((h-target)/cp)
            if j not in (0,self.n-1):
                f['T'][j],f['Y'][:,j],_=hp_equilibrium(self.mech_data,temperature,self.P,Y)
        return f

    def unpack(self,x):
        blocks = np.asarray(x[:-1]).reshape(self.n,self.nv)
        return dict(m=blocks[:,0]*self.mref,V=blocks[:,1]*self.aref,
                    T=blocks[:,2]*self.Tref,Y=blocks[:,3:].T,Lambda=x[-1]*self.lref)

    def properties(self,fields, *, nodal_thermo=None):
        T,Y = fields['T'],fields['Y']
        if nodal_thermo is None:
            omega,hk = np.empty_like(Y),np.empty_like(Y)
            rho,cp = self.backend.eval_grid_thermo_kinetics_into(T,Y,omega,hk)
            hk*=self.mech_data.inv_molecular_weights[:,None]
        else:
            rho,cp,omega,hk=nodal_thermo
        Tf,Yf = .5*(T[:-1]+T[1:]),.5*(Y[:,:-1]+Y[:,1:])
        if self.backend.uses_multicomponent_flux:
            rhof,lam,Wf,multi,dthermal = self.backend.eval_multicomponent_face_transport(Tf,Yf)
            W = self.mech_data.molecular_weights
            coeff = rhof[None,None,:]*W[:,None,None]*multi*W[None,:,None]/Wf[None,None,:]**2
            X = self.thermo.Y_to_X(Y)
            J = np.einsum('klf,lf->kf',coeff,np.diff(X,axis=1)/self.dz)
            J -= dthermal/Tf[None,:]*np.diff(T)[None,:]/self.dz
        else:
            rhof,D,lam,Wf = self.backend.eval_faces(Tf,Yf)
            coeff = rhof[None,:]*self.mech_data.molecular_weights[:,None]/Wf[None,:]*D
            J = _corrected_flux_frozen(Y[:,:-1],Y[:,1:],coeff,self.dz,self.mech_data.molecular_weights,'molar')
            if self.soret_enabled:
                J -= self.backend.eval_mixture_thermal_diffusion(Tf,Yf)/Tf[None,:]*np.diff(T)[None,:]/self.dz
        viscosity = self.backend.transport.viscosity(Tf,self.thermo.Y_to_X(Yf))
        return rho,cp,omega,hk,lam,J,viscosity

    def residual(self,x, *, fixed_temperature=None, previous=None, dt=None,cached_properties=None,nodal_thermo=None):
        f = self.unpack(x)
        m,V,T,Y,L = (f[k] for k in ('m','V','T','Y','Lambda'))
        rho,cp,omega,hk,lam,J,mu = self.properties(f,nodal_thermo=nodal_thermo) if cached_properties is None else cached_properties
        old=f if previous is None else self.unpack(previous)
        return assemble_residual(m,V,T,Y,L,rho,cp,omega,hk,lam,J,mu,self.dz,
            self.case.fuel_Y,self.case.oxidizer_Y,self.case.fuel_mass_flux,self.case.oxidizer_mass_flux,
            self.case.fuel_temperature,self.case.oxidizer_temperature,
            self.mref,self.aref,self.Tref,self.lref,self.species_scale,self.energy_scale,self.excess,
            previous is not None,old['V'],old['T'],old['Y'],1. if dt is None else dt,
            fixed_temperature is not None,T if fixed_temperature is None else fixed_temperature)

    def _perturbed_node_thermo(self,x,nodes):
        """Reuse the premixed backend's exact [base,T,Y...] thermal batches."""
        nv=self.ns+2
        fields=self.unpack(x)
        temperatures=np.repeat(fields['T'][nodes],nv)
        compositions=np.repeat(fields['Y'][:,nodes],nv,axis=1)
        tcolumns=nodes*self.nv+2
        temperatures[1::nv]=(x[tcolumns]+1e-7*np.maximum(abs(x[tcolumns]),1.))*self.Tref
        species=np.arange(self.ns)[:,None]
        columns=np.arange(len(nodes))[None,:]*nv+2+species
        compositions[species,columns]+=1e-10+1e-5*abs(fields['Y'][:,nodes])
        omega=np.empty_like(compositions);hk=np.empty_like(compositions)
        rho,cp=self.backend.eval_jacobian_thermo_kinetics_into(temperatures,compositions,omega,hk)
        hk*=self.mech_data.inv_molecular_weights[:,None]
        return rho.reshape(-1,nv),cp.reshape(-1,nv),omega.reshape(self.ns,-1,nv),hk.reshape(self.ns,-1,nv)

    def jacobian(self,x,fun,base, *, flow_fun=None,thermo_fun=None,base_thermo=None):
        """Color node triples; the number of residual batches is mesh-independent."""
        rows,cols,values = [],[],[]
        for nodes,lengths,rr in self.color_rows:
            # Drop the previous color before allocating the next thermal bank.
            bank=None
            # Bound temporary memory for large mechanisms; the ordinary path
            # retains identical equations and full derivatives when exceeded.
            bytes_needed=8*len(nodes)*(self.ns+2)*(3*self.ns+3)
            bank=(self._perturbed_node_thermo(x,nodes) if thermo_fun is not None
                  and base_thermo is not None and bytes_needed<=_THERMO_BATCH_BUDGET_BYTES else None)
            for component in range(self.nv):
                indices = nodes*self.nv+component
                if not len(indices):
                    continue
                step = (1e-10+1e-5*abs(x[indices]) if component>=3
                        else 1e-7*np.maximum(abs(x[indices]),1.))
                trial = x.copy();trial[indices] += step
                if component<2 and flow_fun is not None:
                    difference=flow_fun(trial)-base
                elif bank is not None and component>=2:
                    thermal=[value.copy() for value in base_thermo]
                    column=component-1
                    thermal[0][nodes]=bank[0][:,column]
                    thermal[1][nodes]=bank[1][:,column]
                    thermal[2][:,nodes]=bank[2][:,:,column]
                    thermal[3][:,nodes]=bank[3][:,:,column]
                    difference=thermo_fun(trial,thermal)-base
                    del thermal
                else:
                    difference=fun(trial)-base
                rows.append(rr)
                cols.append(np.repeat(indices,lengths))
                values.append(difference[rr]/np.repeat(step,lengths))
        trial=x.copy();increment=1e-7*max(abs(x[-1]),1.);trial[-1]+=increment
        difference=((fun(trial) if flow_fun is None else flow_fun(trial))-base)/increment
        rr=np.flatnonzero(difference)
        rows.append(rr);cols.append(np.full(len(rr),len(x)-1));values.append(difference[rr])
        return csc_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(x),len(x)))


def _newton(problem,x,deadline, *, fixed_temperature=None,previous=None,dt=None,max_iterations=25):
    fun = lambda value:problem.residual(value,fixed_temperature=fixed_temperature,previous=previous,dt=dt)
    factor=None;age=0;base=None
    for iteration in range(max_iterations):
        if time.perf_counter() > deadline:
            return x,False,'time_limit'
        if base is None:
            base = fun(x)
        if not np.isfinite(base).all():
            return x,False,'nonfinite_residual'
        norm = float(np.max(abs(base)))
        if norm < 1e-7:
            return x,True,'converged'
        if factor is None or age>=4:
            try:
                if isinstance(problem,DiffusionFlameProblem):
                    # m, V and Lambda do not change T or Y. Reuse exact
                    # properties only for these columns, including a donor
                    # switch at stagnation; all T/Y derivatives stay complete.
                    cached=problem.properties(problem.unpack(x))
                    flow_fun=lambda value:problem.residual(value,fixed_temperature=fixed_temperature,
                        previous=previous,dt=dt,cached_properties=cached)
                    thermo_fun=lambda value,thermal:problem.residual(value,fixed_temperature=fixed_temperature,
                        previous=previous,dt=dt,nodal_thermo=thermal)
                    matrix=problem.jacobian(x,fun,base,flow_fun=flow_fun,
                        thermo_fun=thermo_fun,base_thermo=cached[:4])
                else:
                    matrix=problem.jacobian(x,fun,base)
                factor=splu(matrix);age=0
            except RuntimeError:
                return x,False,'singular_jacobian'
        age+=1
        step = factor.solve(-base)
        if not np.isfinite(step).all():
            return x,False,'singular_jacobian'
        b,delta = x[:-1].reshape(problem.n,problem.nv),step[:-1].reshape(problem.n,problem.nv)
        # Use the original correction before the trace-species bound changes it.
        component_scale=1e-9+1e-4*np.mean(abs(b),axis=0)
        scales=np.r_[np.tile(component_scale,problem.n),1e-4*max(abs(x[-1]),1.)]
        oldnorm=np.linalg.norm(step/scales)
        # At a zero trace-species bound an unconstrained cold Newton step can
        # point out of the physical simplex. Release that component again as
        # soon as its direction points inward; acceptance still requires the
        # complete original residual, never a projected-residual shortcut.
        if previous is None:
            blocked=(b[:,3:]<=1e-10)&(b[:,3:]+delta[:,3:]<-1e-7)
            delta[:,3:][blocked]=0.
        limits = []
        for component,lower,upper in ((2,200./problem.Tref,6000./problem.Tref),(slice(3,None),-1e-7,1.000001)):
            current,change = b[:,component],delta[:,component]
            below=current+change<lower
            above=current+change>upper
            if below.any():
                limits.append(np.min((lower-current)[below]/change[below]))
            if above.any():
                limits.append(np.min((upper-current)[above]/change[above]))
        length = min(1.,.99*min(limits,default=2.))
        if length <= 1e-10:
            return x,False,'bounds'
        # Reuse the factorization for error-based damping, as in MultiNewton.
        # Raw chemistry rates span many decades and are a poor merit norm.
        for _ in range(16):
            trial = x+length*step
            candidate = fun(trial)
            if np.isfinite(candidate).all():
                # A root reached on the last allowed step is still converged.
                if np.max(abs(candidate))<1e-7:
                    return trial,True,'converged'
                if np.linalg.norm(factor.solve(candidate)/scales)<oldnorm*(1.-1e-4*length):
                    x=trial;base=candidate;break
            length*=.5
        else:
            if age>1:
                factor=None
                continue
            return x,False,'line_search'
    return x,False,'iteration_limit'


def solve_diffusion_problem(problem, *, initial=None,max_time=180.,ratio=3.,slope=.04,curve=.08,max_points=1600,refine=True):
    started = time.perf_counter();deadline=started+max_time
    x = problem.pack(problem.initial_fields() if initial is None else initial)
    history=[]
    refiner=AdaptiveRefiner(ratio=ratio,slope=slope,curve=curve,prune=-1.,max_points=max_points)
    reason='not_started';grid_converged=False
    for mesh_round in range(32):
        x,ok,reason=_newton(problem,x,deadline)
        if not ok and mesh_round==0 and initial is None:
            x=problem.pack(problem.equilibrated_guess())
            x,ok,reason=_newton(problem,x,deadline)
            temperature=problem.unpack(x)['T'].copy()
            if not ok:
                candidate,frozen_ok,_=_newton(problem,x,deadline,fixed_temperature=temperature,max_iterations=35)
                if frozen_ok:
                    x=candidate
                    x,ok,reason=_newton(problem,x,deadline)
        if not ok:
            # Backward-Euler continuation retains algebraic flow and inflow constraints.
            blocks=x[:-1].reshape(problem.n,problem.nv)
            guess=np.maximum(blocks[:,3:],0.)
            blocks[:,3:]=guess/guess.sum(axis=1,keepdims=True)
            dt=1e-7
            for transient in range(150):
                if time.perf_counter()>deadline:
                    reason='time_limit';break
                prior=x.copy()
                candidate,success,status=_newton(problem,x,deadline,previous=prior,dt=dt,max_iterations=15)
                if success:
                    x=candidate;dt=min(dt*1.5,.05)
                else:
                    blocks=x[:-1].reshape(problem.n,problem.nv)
                    guess=np.maximum(blocks[:,3:],0.)
                    blocks[:,3:]=guess/guess.sum(axis=1,keepdims=True)
                    dt*=.25
                    if dt<1e-12:
                        reason=status;break
                if transient%5==4:
                    candidate,ok,reason=_newton(problem,x,deadline)
                    if ok:
                        x=candidate;break
        maximum=float(np.max(abs(problem.residual(x))))
        history.append(dict(nodes=problem.n,converged=ok,residual=maximum,reason=reason))
        if not ok:
            break
        fields=problem.unpack(x)
        if not refine:
            break
        profiles={'T':fields['T'],'m':fields['m'],'V':fields['V']}
        profiles.update({f'Y{k}':row for k,row in enumerate(fields['Y'])})
        insert,_=refiner.analyze(problem.z,profiles)
        if not insert:
            grid_converged=problem.n<max_points;reason='converged' if grid_converged else 'grid_limit';break
        if problem.n+len(insert)>max_points:
            reason='grid_limit';break
        oldz=problem.z.copy()
        newz=np.sort(np.r_[oldz,[.5*(oldz[j]+oldz[j+1]) for j in sorted(insert)]])
        interpolated={k:np.interp(newz,oldz,fields[k]) for k in ('m','V','T')}
        interpolated['Y']=np.array([np.interp(newz,oldz,row) for row in fields['Y']])
        interpolated['Lambda']=fields['Lambda']
        problem.set_grid(newz)
        x=problem.pack(interpolated)
    fields=problem.unpack(x)
    rho,cp,omega,hk,lam,J,_=problem.properties(fields)
    fields.update(z=problem.z.copy(),u=fields['m']/rho,rho=rho,cp_mass=cp,
                  omega_Y=omega,qdot=-np.sum(hk*omega,axis=0),h=problem.thermo.enthalpy_mass(fields['T'],fields['Y']),
                  species_diffusive_flux=J,conductive_heat_flux=-lam*np.diff(fields['T'])/problem.dz,
                  species_names=np.array(problem.mech_data.species_names))
    # Total h includes formation enthalpy. Counterflow exports enthalpy
    # radially: a planar constant-mdot outlet-deficit identity is inapplicable.
    hkface=problem.thermo.partial_molar_enthalpies(.5*(fields['T'][:-1]+fields['T'][1:]))*problem.mech_data.inv_molecular_weights[:,None]
    qspecies=np.sum(hkface*J,axis=0)
    zface,totalflux,radial,export,defect=enthalpy_face_balance(problem.z,fields['m'],rho,
        fields['V'],fields['h'],qspecies+fields['conductive_heat_flux'])
    balance_scale=problem.mref*problem.cpref*problem.Tref
    fields.update(z_face=zface,diffusive_enthalpy_flux=qspecies,total_enthalpy_flux=totalflux,
                  radial_enthalpy_export_density=radial)
    maximum=float(np.max(abs(problem.residual(x))))
    burning=bool(fields['T'].max()>max(problem.case.fuel_temperature,problem.case.oxidizer_temperature)+100.)
    report=dict(converged=bool(ok),grid_converged=grid_converged,
                grid_mode='adaptive' if refine else 'fixed',burning=burning,
                maximum_scaled_residual=maximum,reason=reason,history=history,
                stoichiometric_mixture_fraction=problem.Zst,
                maximum_species_sum_error=float(np.max(abs(fields['Y'].sum(axis=0)-1.))),
                minimum_mass_fraction=float(fields['Y'].min()),
                strain_rate_mean_s1=float((fields['u'][0]-fields['u'][-1])/problem.case.width),
                strain_rate_max_s1=float(np.max(abs(np.gradient(fields['u'],problem.z)))),
                enthalpy_definition='total_sensible_plus_formation',
                enthalpy_balance=dict(radial_export_between_first_and_last_faces_W_m2=export,
                    axial_flux_change_W_m2=float(totalflux[-1]-totalflux[0]),
                    finite_grid_defect_W_m2=defect,
                    relative_defect=abs(defect)/balance_scale,
                    normalization_W_m2=balance_scale,
                    boundary_conduction_outward_W_m2=[float(-fields['conductive_heat_flux'][0]),float(fields['conductive_heat_flux'][-1])],
                    note='Continuity-compatible face mass flux and radial export; finite-grid estimate. '
                         'Relative defect normalized by mref*cpref*Tref, independent of the enthalpy zero.'),
                runtime_s=time.perf_counter()-started)
    accepted=bool(ok and (grid_converged or not refine) and burning and maximum<1e-7
                  and report['minimum_mass_fraction']>=-1e-7 and report['maximum_species_sum_error']<1e-7)
    report['accepted']=accepted
    return fields,accepted,report
