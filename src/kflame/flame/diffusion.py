"""Native axisymmetric counterflow diffusion flames with two inflow boundaries.

The unknowns are axial mass flux, radial spread rate, temperature, all mass
fractions, and one radial pressure eigenvalue. Chemistry and transport use the
same native kernels as premixed flames; no reference solution seeds the solve.
"""
from dataclasses import dataclass
import time

import numba
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import erfc

from kflame.chemistry.analytic import species_partials
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.initialization import hp_equilibrium, oxygen_demand
from kflame.flame.equations import _corrected_flux_frozen
from kflame.flame.solver import AdaptiveRefiner
from kflame.flame import block_linear, diffusion_kernels
from kflame.flame.diffusion_kernels import assemble_residual

_THERMO_BATCH_BUDGET_BYTES=96*1024**2
_EXACT_MULTICOMPONENT_SPECIES=20


def _split(function,count,*arguments,minimum=24):
    """Run ``function(start, stop, *arguments)`` over [0, count) in threads.

    The compiled range kernels release the GIL, so the Numba thread count
    (NUMBA_NUM_THREADS) sets the number of concurrent ranges. Plain threads
    avoid compiling a parallel region over very large argument tuples.
    """
    threads=max(1,min(numba.get_num_threads(),count//minimum))
    if threads==1:
        function(0,count,*arguments)
        return
    bounds=[count*i//threads for i in range(threads+1)]
    tasks=[block_linear.pool().submit(function,bounds[i],bounds[i+1],*arguments) for i in range(1,threads)]
    function(bounds[0],bounds[1],*arguments)
    for task in tasks:
        task.result()


def face_properties(problem,T,Y):
    """lambda, Wilke viscosity and species fluxes at all faces (compiled)."""
    data=problem._compiled_data()
    ns,n=Y.shape
    lam,mu,J=np.empty(n-1),np.empty(n-1),np.empty((ns,n-1))
    _split(diffusion_kernels.face_range,n-1,T,Y,problem.dz,float(problem.P),data['invW'],data['mw'],
           data['mixture'],data['multi'],data['multicomponent'],bool(problem.soret_enabled),lam,mu,J,minimum=4)
    return lam,mu,J

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
    # Optically thin H2O/CO2 radiation (Cantera Flow1D model); off = adiabatic.
    radiation_enabled: bool = False
    radiation_scale: float = 1.
    boundary_emissivities: tuple = (0., 0.)


def upwind_gradient(values, z, mass_flux):
    """Both signs of axial flow occur; switch the donor at stagnation."""
    left = (values[...,1:-1]-values[...,:-2])/np.diff(z)[:-1]
    right = (values[...,2:]-values[...,1:-1])/np.diff(z)[1:]
    return np.where(mass_flux[1:-1] >= 0.,left,right)


def enthalpy_face_balance(z,m,rho,V,h,diffusive_heat_flux,sink=None):
    """Face-to-face diagnostic compatible with trapezoidal continuity.

    Integrate the linearly reconstructed radial mass export to each midpoint;
    averaging nodal m*h would break invariance to a common enthalpy offset.
    This diagnostic does not remove truncation error from the energy scheme.
    A volumetric loss ``sink`` [W/m3] (radiation) is integrated over the same
    interval and added to the defect; ``export`` excludes it.
    """
    dz=np.diff(z);zface=.5*(z[:-1]+z[1:]);radial_mass=2.*rho*V
    mface=.5*(m[:-1]+m[1:])+dz*(radial_mass[1:]-radial_mass[:-1])/8.
    totalflux=mface*.5*(h[:-1]+h[1:])+diffusive_heat_flux
    radial=radial_mass*h
    integration_z=np.r_[zface[0],z[1:-1],zface[-1]]
    export=float(np.trapezoid(np.interp(integration_z,z,radial),integration_z))
    lost=0. if sink is None else float(np.trapezoid(np.interp(integration_z,z,sink),integration_z))
    defect=float(totalflux[-1]-totalflux[0]+export+lost)
    return zface,totalflux,radial,export,defect


class DiffusionFlameProblem:
    def __init__(self, case, mech, z):
        self.case, self.mech_data, self.P = case, mech, case.P
        self.transport_model, self.soret_enabled = case.transport_model, case.soret_enabled
        self.flux_gradient_basis = 'molar'
        self.assume_finite_y = True
        # Analytic production-rate columns; grouped differences otherwise.
        self.analytic_chemistry = True
        # Multicomponent composition columns: 'exact' re-solves the face
        # system per column (O(ns^4) per face); 'frozen' keeps its base
        # coefficients (O(ns^3)); 'auto' is exact up to 20 species.
        self.multicomponent_jacobian = 'auto'
        # Compiled face properties for the residual; False keeps the NumPy path.
        self.compiled_faces = True
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
        names = list(mech.species_names)
        eps_fuel, eps_oxidizer = (float(value) for value in case.boundary_emissivities)
        background = diffusion_kernels.STEFAN_BOLTZMANN*(eps_fuel*case.fuel_temperature**4
                                                         +eps_oxidizer*case.oxidizer_temperature**4)
        self.radiation = (bool(case.radiation_enabled), float(case.radiation_scale), float(case.P),
                          names.index('H2O') if 'H2O' in names else -1,
                          names.index('CO2') if 'CO2' in names else -1, float(background),
                          np.ascontiguousarray(mech.inv_molecular_weights, dtype=float))
        self.set_grid(z)

    def set_grid(self,z):
        self.z = np.asarray(z,dtype=float)
        self.n = len(z)
        self._linear_model = None
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
        if self.compiled_faces:
            # One parallel pass per face: lambda, Wilke viscosity and fluxes
            # (corrected mixture-averaged or Dixon--Lewis, optional Soret).
            lam,viscosity,J=face_properties(self,np.ascontiguousarray(T),np.ascontiguousarray(Y))
            return rho,cp,omega,hk,lam,J,viscosity
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
            fixed_temperature is not None,T if fixed_temperature is None else fixed_temperature,
            self.radiation)

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

    def _compiled_data(self):
        """Mechanism constants for the compiled local Jacobian, built once."""
        data=getattr(self,'_compiled',None)
        if data is not None:
            return data
        backend,transport=self.backend,self.backend.transport
        contiguous=lambda value,dtype=np.float64:np.ascontiguousarray(value,dtype=dtype)
        mw=contiguous(self.mech_data.molecular_weights)
        ratio=mw[None,:]/mw[:,None]
        mixture=(contiguous(transport._cond_poly),contiguous(transport._diff_poly),
                 contiguous(transport._visc_poly),contiguous(ratio**.25),
                 contiguous(np.sqrt(8.*(1.+1./ratio))),contiguous(transport._eps_pair),
                 contiguous(transport._cstar_poly),float(transport._tiny))
        thermo=backend.thermo
        nasa=(contiguous(thermo._lo),contiguous(thermo._hi),contiguous(thermo._Tmid))
        multi=backend.multicomponent_transport
        if multi is not None:
            multi_data=(contiguous(multi._log_eps_pair),contiguous(multi.epsilon_k),contiguous(multi.crot),
                        contiguous(multi.zrot),contiguous(multi.frot_298),contiguous(multi._viscosity_fit),
                        contiguous(multi._diffusion_fit),contiguous(multi._astar_poly),
                        contiguous(multi._bstar_poly),contiguous(multi._cstar_poly),
                        multi.linear_solver=='dense',*nasa)
        else:
            # Typed placeholders; the multicomponent branch is never executed.
            multi_data=(np.zeros((1,1)),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros((1,5)),
                        np.zeros((1,1,5)),np.zeros((1,1,9)),np.zeros((1,1,9)),np.zeros((1,1,9)),False,*nasa)
        k=backend.kinetics
        kinetics=(*nasa,contiguous(backend.W),k._nb_A_hi,k._nb_b_hi,k._nb_Ea_hi,k._nb_A_lo,k._nb_b_lo,
                  k._nb_Ea_lo,k._nb_is_three_body,k._nb_is_falloff,k._nb_is_reversible,k._nb_has_troe,
                  k._nb_troe_A,k._nb_troe_T3,k._nb_troe_T1,k._nb_troe_T2,k._sp_r_idx,k._sp_r_nu,
                  k._sp_r_count,k._sp_p_idx,k._sp_p_nu,k._sp_p_count,k._sp_r_plan,k._sp_p_plan,
                  k._sp_net_idx,k._sp_net_nu,k._sp_net_count,k._sp_eff_idx,k._sp_eff_delta,
                  k._sp_eff_count,k._nb_delta_nu)
        self._compiled=data=dict(mw=mw,invW=contiguous(self.mech_data.inv_molecular_weights),
            mixture=mixture,multi=multi_data,kinetics=kinetics,multicomponent=multi is not None)
        return data

    def _frozen_multicomponent(self):
        mode=self.multicomponent_jacobian
        if mode not in ('auto','exact','frozen'):
            raise ValueError('multicomponent_jacobian must be auto, exact or frozen')
        return bool(self.backend.uses_multicomponent_flux and
                    (mode=='frozen' or (mode=='auto' and self.ns>_EXACT_MULTICOMPONENT_SPECIES)))

    def compile_kernels(self,x):
        """Load or JIT-compile every kernel of a solve without solving.

        Range kernels are called on empty ranges and the block elimination
        on a one-node system, so a cold Numba cache never consumes the
        solver time budget. Results are not used.
        """
        self.residual(x)
        self.jacobian_blocks(x,compile_only=True)
        one=np.ones((1,1,1))
        for kernel in (block_linear.twisted_top,block_linear.twisted_bottom,block_linear.twisted_middle):
            kernel(one,one,one,np.zeros((1,1)),0,np.empty((1,1,1)),np.empty((1,1,1)))
        block_linear.twisted_solve(one,one,one,one,0,np.ones(1))

    def jacobian_blocks(self,x, *, fixed_temperature=None,compile_only=False):
        """Compiled steady Jacobian in bordered block-tridiagonal form.

        Each column uses the same finite-difference step as ``jacobian``; only
        the perturbed node, its two faces and three row blocks are recomputed.
        The Lambda column is exact: one in every interior momentum row.
        Returns the blocks and the diagonal multiplying 1/dt in transient rows.
        """
        data=self._compiled_data()
        f=self.unpack(x)
        T,Y=np.ascontiguousarray(f['T']),np.ascontiguousarray(f['Y'])
        m,V=np.ascontiguousarray(f['m']),np.ascontiguousarray(f['V'])
        omega,hk=np.empty_like(Y),np.empty_like(Y)
        rho,cp=self.backend.eval_grid_thermo_kinetics_into(T,Y,omega,hk)
        hk*=self.mech_data.inv_molecular_weights[:,None]
        soret=bool(self.soret_enabled)
        lam0,mu0,J0=face_properties(self,T,Y)
        c=self.case
        scalars=(float(self.mref),float(self.aref),float(self.Tref),float(self.lref),float(self.species_scale),
                 float(self.energy_scale),float(c.fuel_mass_flux),float(c.oxidizer_mass_flux),
                 float(c.fuel_temperature),float(c.oxidizer_temperature))
        fixed=fixed_temperature is not None
        Tfixed=np.ascontiguousarray(T if not fixed else fixed_temperature,dtype=float)
        analytic=False;jy=np.zeros((1,1,1));jt=np.zeros((1,1))
        if self.analytic_chemistry:
            try:
                jy,jt=species_partials(self.backend,T,Y,_temperature=True)
                analytic=True
            except ValueError:
                # Non-elementary orders or a clipped mixture: grouped finite differences.
                pass
        ns,n,nv=self.ns,self.n,self.nv
        lower,diag,upper=np.zeros((n-1,nv,nv)),np.zeros((n,nv,nv)),np.zeros((n-1,nv,nv))
        border=np.zeros((2,nv))
        _split(diffusion_kernels.jacobian_range,0 if compile_only else n,np.ascontiguousarray(x),m,V,T,Y,
            np.ascontiguousarray(rho),np.ascontiguousarray(cp),omega,hk,lam0,mu0,J0,self.dz,float(f['Lambda']),
            scalars,np.ascontiguousarray(c.fuel_Y,dtype=float),np.ascontiguousarray(c.oxidizer_Y,dtype=float),
            np.ascontiguousarray(self.excess,dtype=np.int64),fixed,Tfixed,float(self.P),data['invW'],data['mw'],
            data['mixture'],data['multi'],data['multicomponent'],soret,data['kinetics'],analytic,jy,jt,
            self.radiation,self._frozen_multicomponent(),lower,diag,upper,border,minimum=4)
        mass=np.zeros((self.n,self.nv))
        mass[1:-1,1]=rho[1:-1]*self.aref/self.lref
        if not fixed:
            mass[1:-1,2]=rho[1:-1]*cp[1:-1]*self.Tref/self.energy_scale
        mass[1:-1,3:]=(rho[1:-1]/self.species_scale)[:,None]
        return BorderedBlocks(lower,diag,upper,border,mass.ravel())

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


class BorderedBlocks:
    """Steady counterflow Jacobian [[A, c], [r, 0]] with A block tridiagonal.

    c is one in interior momentum rows (the exact Lambda column) and r is the
    final continuity row, nonzero only in the last two node blocks. A
    transient matrix adds ``mass/dt`` to the diagonal of A.
    """

    def __init__(self,lower,diag,upper,border,mass):
        self.lower,self.diag,self.upper,self.border,self.mass=lower,diag,upper,border,mass
        self.n,self.nv=diag.shape[0],diag.shape[1]

    def column(self):
        c=np.zeros(self.n*self.nv)
        c[self.nv+1:(self.n-1)*self.nv:self.nv]=1.
        return c

    def tocsc(self,rdt=0.):
        """Assembled sparse matrix, used by the fallback and in tests."""
        n,nv=self.n,self.nv
        index=np.arange(nv)
        rows=[];cols=[];vals=[]
        diag=self.diag.copy()
        diag[:,index,index]+=rdt*self.mass.reshape(n,nv)
        for blocks,dr,dc in ((diag,0,0),(self.lower,1,0),(self.upper,0,1)):
            count=blocks.shape[0]
            r=(np.arange(count)[:,None,None]+dr)*nv+index[None,:,None]
            q=(np.arange(count)[:,None,None]+dc)*nv+index[None,None,:]
            rows.append(np.broadcast_to(r,blocks.shape).ravel());cols.append(np.broadcast_to(q,blocks.shape).ravel())
            vals.append(blocks.ravel())
        c=np.flatnonzero(self.column())
        rows.append(c);cols.append(np.full(len(c),n*nv));vals.append(np.ones(len(c)))
        rows.append(np.full(2*nv,n*nv));cols.append(np.arange((n-2)*nv,n*nv));vals.append(self.border.ravel())
        size=n*nv+1
        return csc_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(size,size))


class BorderedFactor:
    """Block elimination of A with a Schur complement for Lambda.

    A is eliminated from both ends towards its middle block with compiled
    LAPACK/BLAS block kernels; on large grids the two independent sweeps run
    in two threads. A singular or non-finite pivot block, or a vanishing Schur
    complement, falls back to sparse LU of the same assembled matrix; the
    linear system itself is never modified.
    """

    def __init__(self,blocks,rdt=0.):
        n,nv=blocks.n,blocks.nv
        self.rdt=rdt;self.n,self.nv=n,nv
        self.sparse=None
        self.lower,self.upper=blocks.lower,blocks.upper
        try:
            self.inverse,self.coupling,self.k=block_linear.factor_two_ended(
                blocks.lower,blocks.diag,blocks.upper,rdt*blocks.mass.reshape(n,nv))
            self.v=block_linear.twisted_solve(self.lower,self.upper,self.inverse,self.coupling,self.k,
                                              blocks.column())
            self.r=blocks.border.ravel()
            with np.errstate(all='raise'):
                self.schur=-float(self.r@self.v[(n-2)*nv:])
            if not np.isfinite(self.v).all() or not abs(self.schur)>1e-300:
                raise FloatingPointError('Lambda Schur complement')
        except (FloatingPointError,np.linalg.LinAlgError,ZeroDivisionError):
            self.inverse=self.coupling=None
            self.sparse=splu(blocks.tocsc(rdt))

    def solve(self,rhs):
        if self.sparse is not None:
            return self.sparse.solve(rhs)
        n,nv=self.n,self.nv
        u=block_linear.twisted_solve(self.lower,self.upper,self.inverse,self.coupling,self.k,
                                          np.ascontiguousarray(rhs[:-1]))
        delta=(rhs[-1]-self.r@u[(n-2)*nv:])/self.schur
        out=np.empty(n*nv+1)
        np.subtract(u,self.v*delta,out=out[:-1])
        out[-1]=delta
        return out


class _LinearModel:
    """Steady Jacobian at an anchor state; transient rows add ``mass/dt``.

    Kept across Newton calls and pseudo-time steps on one grid and temperature
    mode, as Cantera's MultiJac keeps its Jacobian and only refactorizes when
    dt changes. It shapes Newton directions only: every step is still accepted
    against the complete current residual.
    """

    def __init__(self,problem,x,fixed_temperature):
        self.blocks=problem.jacobian_blocks(x,fixed_temperature=fixed_temperature)
        self.fixed=fixed_temperature is not None
        self.age=0
        self.factors={}
        problem.jacobian_evaluations=getattr(problem,'jacobian_evaluations',0)+1

    def factor(self,rdt):
        factor=self.factors.get(rdt)
        if factor is None:
            # Keep the steady factorization and the current time step only.
            for key in [key for key in self.factors if key not in (0.,)]:
                del self.factors[key]
            factor=self.factors[rdt]=BorderedFactor(self.blocks,rdt)
        return factor


class _SparseModel:
    """Generic problems: one sparse finite-difference matrix per Newton call."""

    def __init__(self,problem,x,fun,base):
        self.matrix=splu(problem.jacobian(x,fun,base))
        self.fixed=None
        self.age=0
        problem.jacobian_evaluations=getattr(problem,'jacobian_evaluations',0)+1

    def factor(self,rdt):
        return self.matrix


def _newton(problem,x,deadline, *, fixed_temperature=None,previous=None,dt=None,max_iterations=25,
            correction_tolerance=False,max_age=4,max_damping=16,refresh_rate=None):
    """Damped Newton on the scaled counterflow residual.

    Steady and fixed-temperature calls converge only on the maximum scaled
    residual ``1e-7``. Pseudo-time steps may instead stop, as in Cantera, when
    the weighted Newton correction is below one (relative 1e-4, absolute
    1e-11); they only advance the continuation and never certify a flame.
    """
    fun = lambda value:problem.residual(value,fixed_temperature=fixed_temperature,previous=previous,dt=dt)
    blocked=hasattr(problem,'jacobian_blocks')
    rdt=0. if previous is None else 1./dt
    model=getattr(problem,'_linear_model',None) if blocked else None
    if model is not None and (model.fixed!=(fixed_temperature is not None) or model.blocks.n!=problem.n):
        model=None
    base=None;last_norm=None
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
        if model is None or model.age>=max_age:
            try:
                model=_LinearModel(problem,x,fixed_temperature) if blocked else _SparseModel(problem,x,fun,base)
            except RuntimeError:
                return x,False,'singular_jacobian'
            if blocked:
                problem._linear_model=model
        try:
            factor=model.factor(rdt)
        except RuntimeError:
            return x,False,'singular_jacobian'
        model.age+=1
        step = factor.solve(-base)
        if not np.isfinite(step).all():
            return x,False,'singular_jacobian'
        b,delta = x[:-1].reshape(problem.n,problem.nv),step[:-1].reshape(problem.n,problem.nv)
        # Use the original correction before the trace-species bound changes it.
        mean=np.mean(abs(b),axis=0)
        scales=np.empty(len(x))
        scales[:-1].reshape(problem.n,problem.nv)[:]=1e-9+1e-4*mean
        scales[-1]=1e-4*max(abs(x[-1]),1.)
        oldnorm=np.linalg.norm(step/scales)
        if (refresh_rate is not None and last_norm is not None and model.age>1
                and oldnorm>refresh_rate*last_norm):
            # A stale Jacobian contracts slowly: rebuild it at this state.
            model=None;last_norm=None
            continue
        last_norm=oldnorm
        if correction_tolerance:
            weights=np.empty(len(x))
            weights[:-1].reshape(problem.n,problem.nv)[:]=1e-11+1e-4*mean
            weights[-1]=1e-11+1e-4*abs(x[-1])
            if np.sqrt(np.mean((step/weights)**2))<1.:
                trial=x+step
                tb=trial[:-1].reshape(problem.n,problem.nv)
                if (tb[:,3:].min()>=-1e-7 and tb[:,3:].max()<=1.000001
                        and 200.<=tb[:,2].min()*problem.Tref and tb[:,2].max()*problem.Tref<=6000.):
                    candidate=fun(trial)
                    if np.isfinite(candidate).all():
                        return trial,True,'converged'
        # At a zero trace-species bound an unconstrained cold Newton step can
        # point out of the physical simplex. Release that component again as
        # soon as its direction points inward; acceptance still requires the
        # complete original residual, never a projected-residual shortcut.
        if previous is None:
            blocked_species=(b[:,3:]<=1e-10)&(b[:,3:]+delta[:,3:]<-1e-7)
            delta[:,3:][blocked_species]=0.
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
            if model.age>1:
                model=None
                continue
            return x,False,'bounds'
        # Reuse the factorization for error-based damping, as in MultiNewton.
        # Raw chemistry rates span many decades and are a poor merit norm.
        for _ in range(max_damping):
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
            if model.age>1:
                model=None
                continue
            return x,False,'line_search'
    return x,False,'iteration_limit'


def _normalized(x,problem):
    x=x.copy()
    blocks=x[:-1].reshape(problem.n,problem.nv)
    guess=np.maximum(blocks[:,3:],0.)
    blocks[:,3:]=guess/guess.sum(axis=1,keepdims=True)
    return x


TRANSIENT=dict(growth=1.5,iterations=25,refresh_rate=.5,max_age=20)
STEADY=dict(max_age=12,refresh_rate=.5)


def _pseudo_transient(problem,x,deadline, *,dt=1e-7,dt_max=.05,schedule=(2,5,10,20),max_steps=400):
    """Backward-Euler continuation with Cantera's time-step policy.

    dt grows only after a step that needed no new Jacobian and halves after
    a failure; a steady solve is attempted after each block of the schedule.
    Failed steady attempts are discarded. Algebraic flow and inflow rows
    are never relaxed. Returns the steady state, success flag and reason.
    """
    x=_normalized(x,problem)
    stage=0;successes=0;reason='transient_limit'
    for _ in range(max_steps):
        if time.perf_counter()>deadline:
            return x,False,'time_limit'
        before=getattr(problem,'jacobian_evaluations',0)
        candidate,success,status=_newton(problem,x,deadline,previous=x.copy(),dt=dt,
                                         max_iterations=TRANSIENT['iterations'],
                                         correction_tolerance=True,max_age=TRANSIENT['max_age'],max_damping=8,
                                         refresh_rate=TRANSIENT['refresh_rate'])
        if success:
            x=candidate;successes+=1
            if getattr(problem,'jacobian_evaluations',0)==before:
                dt=min(dt*TRANSIENT['growth'],dt_max)
        else:
            x=_normalized(x,problem)
            dt*=.5
            if dt<1e-12:
                return x,False,status
        if successes>=schedule[stage]:
            successes=0;stage=min(stage+1,len(schedule)-1)
            candidate,ok,reason=_newton(problem,x,deadline,max_iterations=12,max_age=10,max_damping=6)
            if ok:
                return candidate,True,reason
    return x,False,reason


def solve_diffusion_problem(problem, *, initial=None,max_time=180.,ratio=3.,slope=.04,curve=.08,max_points=1600,refine=True):
    x = problem.pack(problem.initial_fields() if initial is None else initial)
    # JIT compilation (first use or changed sources) is not solver time.
    problem.compile_kernels(x)
    started = time.perf_counter();deadline=started+max_time
    history=[]
    refiner=AdaptiveRefiner(ratio=ratio,slope=slope,curve=curve,prune=-1.,max_points=max_points)
    reason='not_started';grid_converged=False
    # A cold radiative start first finds the adiabatic burning branch: the
    # pseudo-transient from the seed can cool through extinction on the
    # coarse grid. Radiation is then switched on from that burning state.
    radiation=problem.radiation
    adiabatic_start=bool(radiation[0]) and initial is None
    if adiabatic_start:
        problem.radiation=(False,)+tuple(radiation[1:]);problem._linear_model=None
    for mesh_round in range(32):
        start=x.copy()
        x,ok,reason=_newton(problem,x,deadline,max_age=10 if mesh_round==0 else STEADY['max_age'],
                            max_damping=6 if mesh_round==0 else 16,
                            refresh_rate=None if mesh_round==0 else STEADY['refresh_rate'])
        if not ok:
            # A failed steady attempt is discarded, as in Cantera.
            x,ok,reason=_pseudo_transient(problem,start,deadline)
        if not ok and mesh_round==0 and initial is None:
            # Last resort: native HP mixing-line seed and the original continuation.
            x=problem.pack(problem.equilibrated_guess())
            x,ok,reason=_newton(problem,x,deadline)
            temperature=problem.unpack(x)['T'].copy()
            if not ok:
                candidate,frozen_ok,_=_newton(problem,x,deadline,fixed_temperature=temperature,max_iterations=35)
                if frozen_ok:
                    x=candidate
                    x,ok,reason=_newton(problem,x,deadline)
            if not ok:
                x,ok,reason=_pseudo_transient(problem,x,deadline)
        if adiabatic_start and problem.radiation is not radiation:
            history.append(dict(nodes=problem.n,converged=ok,reason=reason,stage='adiabatic_start',
                                residual=float(np.max(abs(problem.residual(x))))))
            problem.radiation=radiation;problem._linear_model=None
            if ok:
                start=x.copy()
                x,ok,reason=_newton(problem,x,deadline,max_age=10,max_damping=6)
                if not ok:
                    x,ok,reason=_pseudo_transient(problem,start,deadline)
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
    qrad=diffusion_kernels.radiation_profile(np.ascontiguousarray(fields['T']),
                                            np.ascontiguousarray(fields['Y']),problem.radiation)
    qrad[[0,-1]]=0.  # Inlet rows are Dirichlet; the loss acts on interior energy rows.
    fields['radiative_heat_loss']=qrad
    zface,totalflux,radial,export,defect=enthalpy_face_balance(problem.z,fields['m'],rho,
        fields['V'],fields['h'],qspecies+fields['conductive_heat_flux'],qrad)
    integration_z=np.r_[zface[0],problem.z[1:-1],zface[-1]]
    radiated=float(np.trapezoid(np.interp(integration_z,problem.z,qrad),integration_z))
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
                radiation=dict(enabled=bool(problem.case.radiation_enabled),
                               scale=float(problem.case.radiation_scale),
                               boundary_emissivities=[float(e) for e in problem.case.boundary_emissivities],
                               model='optically thin, Planck-mean H2O/CO2 fits of Cantera Flow1D',
                               peak_loss_W_m3=float(qrad.max()),integrated_loss_W_m2=radiated),
                enthalpy_balance=dict(radial_export_between_first_and_last_faces_W_m2=export,
                    axial_flux_change_W_m2=float(totalflux[-1]-totalflux[0]),
                    finite_grid_defect_W_m2=defect,
                    relative_defect=abs(defect)/balance_scale,
                    normalization_W_m2=balance_scale,
                    boundary_conduction_outward_W_m2=[float(-fields['conductive_heat_flux'][0]),float(fields['conductive_heat_flux'][-1])],
                    radiative_loss_between_first_and_last_faces_W_m2=radiated,
                    note='Continuity-compatible face mass flux and radial export; finite-grid estimate. '
                         'Relative defect normalized by mref*cpref*Tref, independent of the enthalpy zero.'),
                runtime_s=time.perf_counter()-started)
    accepted=bool(ok and (grid_converged or not refine) and burning and maximum<1e-7
                  and report['minimum_mass_fraction']>=-1e-7 and report['maximum_species_sum_error']<1e-7)
    report['accepted']=accepted
    return fields,accepted,report
