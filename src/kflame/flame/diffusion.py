"""Native axisymmetric counterflow diffusion flames with two inflow boundaries.

The unknowns are axial mass flux, radial spread rate, temperature, all mass
fractions, and one radial pressure eigenvalue. Chemistry and transport use the
same native kernels as premixed flames; no reference solution seeds the solve.
"""
from dataclasses import dataclass, replace
import os
import time

import numba
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import erfc

from kflame.chemistry.analytic import prepare_species_partials, species_partials, species_partials_range
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.initialization import hp_equilibrium, oxygen_demand
from kflame.flame.equations import _corrected_flux_frozen
from kflame.flame.solver import AdaptiveRefiner
from kflame.flame import block_linear
from kflame.flame import diffusion_kernels
from kflame.flame import fast_transport
from kflame.flame.diffusion_kernels import assemble_residual
from kflame.flame.structured_jacobian import structured_blocks
from kflame.flame.structured_linear import StructuredFactor
from kflame.flame.buffer_pool import pool_of
from kflame.flame.krylov import gmres

_THERMO_BATCH_BUDGET_BYTES=96*1024**2
_EXACT_MULTICOMPONENT_SPECIES=20
_CHEMISTRY_TILE_NODES=32
_CHEMISTRY_TILE_BUDGET_BYTES=1024**2


def heavy_threads():
    """Ranges for O(ns^2)-per-item kernels of large mechanisms.

    ``KFLAME_THREADS`` overrides. By default two physical cores (half of the
    logical CPUs) are left free so the machine stays responsive; never fewer
    than the Numba thread count.
    """
    configured=os.environ.get('KFLAME_THREADS')
    if configured:
        return max(1,int(configured))
    return max(numba.get_num_threads(),(os.cpu_count() or 2)//2-2)


def _split(function,count,*arguments,minimum=24,threads=None):
    """Run ``function(start, stop, *arguments)`` over [0, count) in threads.

    The compiled range kernels release the GIL, so the Numba thread count
    (NUMBA_NUM_THREADS) sets the number of concurrent ranges unless
    ``threads`` is given. Plain threads avoid compiling a parallel region
    over very large argument tuples.
    """
    threads=max(1,min(numba.get_num_threads() if threads is None else threads,count//minimum))
    if threads==1:
        function(0,count,*arguments)
        return
    bounds=[count*i//threads for i in range(threads+1)]
    tasks=[block_linear.pool().submit(function,bounds[i],bounds[i+1],*arguments) for i in range(1,threads)]
    function(bounds[0],bounds[1],*arguments)
    for task in tasks:
        task.result()


def _species_layout(Y):
    """Borrow contiguous C/F inputs; pack strided node columns only once.

    Range kernels only read Y and own their scratch, so no global ownership
    or writeability of the borrowed input is required.
    """
    if Y.flags.c_contiguous or Y.flags.f_contiguous:
        return Y
    return np.asfortranarray(Y)


# Smaller mechanisms keep the column Jacobian and scalar faces: per-call
# overheads exceed the O(ns^2) savings (H2-O2: 1.6 s against 0.42 s).
STRUCTURED_MIN_SPECIES=20


def _fast_faces(problem):
    """Vectorized face-blocked mixture transport (with or without Soret)."""
    mode=getattr(problem,'fast_transport','auto')
    if mode not in ('auto',True,False):
        raise ValueError('fast_transport must be auto, True or False')
    return (bool(mode) and (mode is True or problem.ns>=STRUCTURED_MIN_SPECIES)
            and problem.transport_model=='mixture-averaged' and not problem.backend.uses_multicomponent_flux)


def face_properties(problem,T,Y,fast=True):
    """lambda, Wilke viscosity and species fluxes at all faces (compiled).

    ``fast=False`` keeps the scalar kernels shared with the column-wise
    finite-difference Jacobian, so its base and trial faces round alike.
    """
    data=problem._compiled_data()
    ns,n=Y.shape
    if fast and _fast_faces(problem) and 'fast_mixture' in data:
        lam,mu,J=np.empty(n-1),np.empty(n-1),np.empty((ns,n-1))
        coef=np.empty((n-1,ns))
        threads=dict(minimum=2 if ns>=100 else 64,threads=heavy_threads() if ns>=100 else None)
        if problem.soret_enabled:
            _split(fast_transport.face_range_fast_soret,n-1,T,Y,problem.dz,float(problem.P),data['invW'],
                   data['mw'],data['fast_mixture'],data['fast_soret'],lam,mu,J,coef,np.empty((n-1,ns)),**threads)
        else:
            _split(fast_transport.face_range_fast,n-1,T,Y,problem.dz,float(problem.P),data['invW'],data['mw'],
                   data['fast_mixture'],lam,mu,J,coef,**threads)
        return lam,mu,J
    # Keep public J in C order: downstream NumPy reductions depend on layout.
    lam,mu,J=np.empty(n-1),np.empty(n-1),np.empty((ns,n-1))
    # Multicomponent faces of large mechanisms are O(ns^2)-O(ns^3) each: use
    # the heavy-kernel thread count (other calls keep the default split).
    heavy=dict(threads=heavy_threads()) if data['multicomponent'] and ns>=100 else {}
    _split(diffusion_kernels.face_range,n-1,T,Y,problem.dz,float(problem.P),data['invW'],data['mw'],
           data['mixture'],data.get('multi_trace',data.get('multi_certified',data['multi'])),data['multicomponent'],
           bool(problem.soret_enabled),lam,mu,J,
           minimum=24 if ns<=20 and not data["multicomponent"] else 4,**heavy)
    return lam,mu,J

_FACE_BANK_BUDGET_BYTES=96*1024**2


def _face_bank_key(problem):
    return (id(problem),id(problem.backend),id(problem._compiled_data()),float(problem.P),
        bool(problem.soret_enabled),problem.transport_model,problem.flux_gradient_basis,
        bool(problem.compiled_faces),bool(problem.analytic_chemistry),int(problem.chemistry_tile_nodes),
        bool(problem._frozen_multicomponent()),bool(problem.temperature_pair_symmetry),
        bool(problem.thermal_transport_tangents),float(problem.Tref))


def _face_bank_available(problem):
    if (problem.ns<256 or not problem.compiled_faces or not problem._frozen_multicomponent()
        or getattr(problem,'trace_tail',None) is not None
        or not problem.analytic_chemistry or problem.chemistry_tile_nodes<=0):
        return False
    size=8*(problem.n-1)*(problem.ns*problem.ns+problem.ns+1)
    if problem.thermal_transport_tangents:
        size+=8*(problem.n-1)*(2*problem.ns+3)
    return size<=_FACE_BANK_BUDGET_BYTES and not problem._compiled_data()['multi'][10]


def _same_float64_bits(left,right):
    # Integer views compare signed zero as well as every other input bit;
    # ndarray equality alone would accept a changed +0/-0 species state.
    return (isinstance(left,np.ndarray) and isinstance(right,np.ndarray)
        and left.dtype==right.dtype==np.float64 and left.shape==right.shape
        and np.array_equal(left.view(np.uint64),right.view(np.uint64)))


def _valid_face_bank(problem,bank,T,Y,cached_properties):
    if bank is None or cached_properties is None or not _face_bank_available(problem):
        return False
    if (type(bank) is not dict or bank.get('owner') is not problem
        or bank.get('key')!=_face_bank_key(problem)):
        return False
    if not (_same_float64_bits(T,bank.get('T')) and _same_float64_bits(Y,bank.get('Y'))
            and _same_float64_bits(problem.z,bank.get('z')) and _same_float64_bits(problem.dz,bank.get('dz'))
            and _same_float64_bits(cached_properties[4],bank.get('lambda'))):
        return False
    ns,n=problem.ns,problem.n
    for name,shape in (('multi',(n-1,ns,ns)),('thermal',(n-1,ns)),('lambda',(n-1,))):
        array=bank.get(name)
        if (not isinstance(array,np.ndarray) or array.shape!=shape or array.dtype!=np.float64
            or not array.flags.c_contiguous or array.flags.writeable or not np.isfinite(array).all()):
            return False
    return True


def face_properties_bank(problem,T,Y,bank):
    data=problem._compiled_data()
    ns,n=Y.shape
    lam,mu,J=np.empty(n-1),np.empty(n-1),np.empty((ns,n-1))
    arguments=(T,Y,problem.dz,float(problem.P),data['invW'],data['mw'],
        data['mixture'],data.get('multi_certified',data['multi']),True,bool(problem.soret_enabled),
        lam,mu,J,bank['multi'],bank['thermal'],bank['lambda'])
    if problem.thermal_transport_tangents:
        _split(diffusion_kernels.face_range_bank_thermal,n-1,*arguments,float(problem.Tref),
               *bank['temperature_tangents'],minimum=4)
    else:
        _split(diffusion_kernels.face_range_bank,n-1,*arguments,minimum=4)
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
    # Multicomponent transport of large mechanisms: 'auto' omits collisions
    # between two trace species (total mole fraction <= 1e-8) from 200
    # species on; None keeps the complete Dixon--Lewis kernel.
    multicomponent_trace_tail: object = 'auto'


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
        # Large mechanisms stream derivatives into the local assembly. Zero
        # keeps the full-grid path for diagnostics; the arithmetic is the same.
        self.chemistry_tile_nodes = _CHEMISTRY_TILE_NODES
        # Multicomponent composition columns: 'exact' re-solves the face
        # system per column (O(ns^4) per face); 'frozen' keeps its base
        # coefficients (O(ns^3)); 'auto' is exact up to 20 species.
        self.multicomponent_jacobian = 'auto'
        # Opt-in analytic composition rows with an admitted frozen face bank.
        # The physical residual and the temperature columns are unchanged.
        self.analytic_frozen_transport = False
        # Opt-in algebraic finite trials; analytic rows take priority when valid.
        self.algebraic_frozen_transport = False
        # Experimental transport-only temperature derivatives; guarded per face.
        self.thermal_transport_tangents = False
        # Compiled face properties for the residual; False keeps the NumPy path.
        self.compiled_faces = True
        # Retain only the current Newton base's already evaluated properties.
        # False keeps the independent residual/Jacobian preparation for checks.
        self.reuse_base_properties = True
        # Analytic frozen-transport Jacobian in diag + low-rank coupling form
        # (mixture-averaged transport): 'auto', True or False.
        self.structured_jacobian = 'auto'
        self.temperature_pair_symmetry = True
        self.backend = NativeSpeciesBackend(self,mech)
        self.thermo = self.backend.thermo
        # Tail of the pruned multicomponent model, or None (complete kernel).
        multi = self.backend.multicomponent_transport
        self.trace_tail = None if multi is None else multi.trace_tail
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
        mixing = .5*erfc((z-xstagnation)/self.mixing_layer_thickness())
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

    def mixing_layer_thickness(self):
        """sqrt(2 D/a) of the burnt equilibrium mixture at the nominal strain aref."""
        _, D, _, _ = self.backend.eval_faces(np.array([self.Teq]),self.Yeq[:,None])
        return float(np.sqrt(2.*float(np.mean(D[:,0]))/self.aref))

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
        if self.compiled_faces:
            # Pack a strided state once for both chemistry and face transport.
            Y = _species_layout(Y)
        if nodal_thermo is None:
            omega,hk = np.empty_like(Y),np.empty_like(Y)
            rho,cp = self.backend.eval_grid_thermo_kinetics_into(T,Y,omega,hk)
            hk*=self.mech_data.inv_molecular_weights[:,None]
        else:
            rho,cp,omega,hk=nodal_thermo
        if self.compiled_faces:
            # One parallel pass per face: lambda, Wilke viscosity and fluxes
            # (corrected mixture-averaged or Dixon--Lewis, optional Soret).
            lam,viscosity,J=face_properties(self,np.ascontiguousarray(T),_species_layout(Y))
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

    def properties_for_newton(self,fields, *, nodal_thermo=None):
        """Explicit local coefficients for this invocation; ordinary API still returns seven."""
        if not _face_bank_available(self):
            return self.properties(fields,nodal_thermo=nodal_thermo),None
        T=np.ascontiguousarray(fields['T'])
        Y=_species_layout(fields['Y'])
        ns,n=Y.shape
        bank=dict(multi=np.empty((n-1,ns,ns)),thermal=np.empty((n-1,ns)))
        bank['lambda']=np.empty(n-1)
        if self.thermal_transport_tangents:
            bank['temperature_tangents']=(np.zeros((n-1,ns)),np.zeros((n-1,ns)),
                                          np.zeros((n-1,3)))
        if nodal_thermo is None:
            omega,hk=np.empty_like(Y),np.empty_like(Y)
            rho,cp=self.backend.eval_grid_thermo_kinetics_into(T,Y,omega,hk)
            hk*=self.mech_data.inv_molecular_weights[:,None]
        else:
            rho,cp,omega,hk=nodal_thermo
        lam,viscosity,J=face_properties_bank(self,T,Y,bank)
        props=(rho,cp,omega,hk,lam,J,viscosity)
        for name in ('multi','thermal','lambda'):
            bank[name].flags.writeable=False
        for array in bank.get('temperature_tangents',()):
            array.flags.writeable=False
        bank.update(key=_face_bank_key(self),owner=self,T=T.copy(),Y=Y.copy(order='C'),
                    z=self.z.copy(),dz=self.dz.copy())
        for name in ('T','Y','z','dz'):
            bank[name].flags.writeable=False
        return props,bank

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
            temperature_inputs,certified=multi.export_temperature_snapshot()
            (snapshot_mw,log_eps_pair,epsilon,zrot,frot298,viscosity_fit,diffusion_fit,
             astar_fit,bstar_fit,cstar_fit)=temperature_inputs
            same_weights=(mw.shape==snapshot_mw.shape and mw.dtype==snapshot_mw.dtype
                          and mw.tobytes()==snapshot_mw.tobytes())
            certified=bool(certified and same_weights and self.temperature_pair_symmetry)
            if certified:
                mw=snapshot_mw
            multi_data=(log_eps_pair,epsilon,contiguous(multi.crot),zrot,frot298,viscosity_fit,
                        diffusion_fit,astar_fit,bstar_fit,cstar_fit,multi.linear_solver=='dense',
                        *nasa)
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
        if k._plog_data is not None or k._sri_data is not None:
            kinetics=(*kinetics,k._plog_data,k._sri_data)
        self._compiled=data=dict(mw=mw,invW=contiguous(self.mech_data.inv_molecular_weights),
            mixture=mixture,multi=multi_data,kinetics=kinetics,multicomponent=multi is not None)
        if multi is None or self.trace_tail is not None:
            # Mixture tables also serve as the structured preconditioner of the
            # pruned multicomponent model.
            data['fast_mixture']=fast_transport.mixture_tables(transport,mw)
            data['fast_soret']=fast_transport.soret_tables(transport)
        if multi is not None:
            # Keep the established fourteen-entry tuple for existing/raw
            # consumers. Only owned compiled face paths use this tagged packet.
            data['multi_certified']=(*multi_data,certified)
            if self.trace_tail is not None and certified and not multi_data[10]:
                # Residual faces of the pruned model (multicomponent_trace).
                data['multi_trace']=(*multi_data,certified,float(self.trace_tail))
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
        if _face_bank_available(self):
            fields=self.unpack(x)
            T=np.ascontiguousarray(fields['T']);Y=_species_layout(fields['Y'])
            data=self._compiled_data();ns=self.ns
            # Empty export outputs have the same Numba layouts/writeability
            # as the real producer, and require no full-grid bank allocation.
            diffusion_kernels.face_range_bank(0,0,T,Y,self.dz,float(self.P),data['invW'],data['mw'],
                data['mixture'],data.get('multi_certified',data['multi']),True,bool(self.soret_enabled),
                np.empty(0),np.empty(0),np.empty((ns,0)),
                np.empty((0,ns,ns)),np.empty((0,ns)),np.empty(0))
            if self.thermal_transport_tangents:
                diffusion_kernels.face_range_bank_thermal(0,0,T,Y,self.dz,float(self.P),data['invW'],data['mw'],
                    data['mixture'],data.get('multi_certified',data['multi']),True,bool(self.soret_enabled),
                    np.empty(0),np.empty(0),np.empty((ns,0)),np.empty((0,ns,ns)),np.empty((0,ns)),np.empty(0),
                    float(self.Tref),np.empty((0,ns)),np.empty((0,ns)),np.empty((0,3)))
        self.jacobian_blocks(x,compile_only=True)
        if _use_structured(self):
            try:
                StructuredFactor(structured_blocks(self,x)).solve(np.zeros(len(x)))
                StructuredFactor(structured_blocks(self,x,lazy=True)).solve(np.zeros(len(x)))
            except ValueError:
                pass
        one=np.ones((1,1,1))
        for kernel in (block_linear.twisted_top,block_linear.twisted_bottom,block_linear.twisted_middle):
            kernel(one,one,one,np.zeros((1,1)),0,np.empty((1,1,1)),np.empty((1,1,1)))
        block_linear.twisted_solve(one,one,one,one,0,np.ones(1))

    def jacobian_blocks(self,x, *, fixed_temperature=None,compile_only=False,cached_properties=None,transport_bank=None):
        """Compiled steady Jacobian in bordered block-tridiagonal form.

        The ordinary route uses the same finite-difference steps as
        ``jacobian`` and recomputes only the local node/faces/neighbor rows.
        With ``analytic_frozen_transport=True`` and an admitted face bank,
        valid composition columns directly differentiate the frozen closure,
        ideal-gas thermochemistry, advection and radiation. Alternatively,
        ``algebraic_frozen_transport=True`` evaluates the original finite
        face trials with algebraic updates. Valid analytic rows take priority
        when both options are enabled. Unsafe algebraic trials fall back to
        the ordered helper; absent/stale banks retain the ordinary route.
        ``thermal_transport_tangents=True`` supplies guarded analytic transport
        tangents to the temperature trial. Nodal thermo and row differences
        retain their previous implementation; this option defaults to False.
        All three options are experimental and disabled by default. Flow
        columns and the complete physical residual are unchanged. Temperature
        columns retain their ordinary route when the thermal option is off.
        The Lambda column is exact: one in every interior momentum row.
        Returns the blocks and the diagonal multiplying 1/dt in transient rows.
        ``cached_properties`` must come from ``properties(unpack(x))`` on this
        problem's compiled path with unchanged state, grid and configuration.
        The Newton driver owns that tuple locally; no identity-based cache is
        retained on the problem or reused across calls/grids.
        """
        data=self._compiled_data()
        f=self.unpack(x)
        T,Y=np.ascontiguousarray(f['T']),np.ascontiguousarray(f['Y'])
        m,V=np.ascontiguousarray(f['m']),np.ascontiguousarray(f['V'])
        if cached_properties is not None:
            rho,cp,omega,hk,lam0,J0,mu0=cached_properties
        # The fused native kernel has fixed ordered node reductions. Its
        # NumPy fallback can round differently for F residual inputs versus C
        # Jacobian inputs; retain that original nodal evaluation in that case.
        fused_nodal=(type(self.backend) is NativeSpeciesBackend
                     and bool(getattr(self,'use_fused_numba_thermo_kinetics',True))
                     and bool(getattr(self.backend.kinetics,'_sparse_numba_available',False)))
        if cached_properties is None or not fused_nodal:
            omega,hk=np.empty_like(Y),np.empty_like(Y)
            rho,cp=self.backend.eval_grid_thermo_kinetics_into(T,Y,omega,hk)
            hk*=self.mech_data.inv_molecular_weights[:,None]
        else:
            # The local assembler historically receives C nodal outputs even
            # though residual evaluation packs node columns in Fortran order.
            omega,hk=np.ascontiguousarray(omega),np.ascontiguousarray(hk)
        if cached_properties is None or _fast_faces(self):
            # Column trials use the scalar face kernels: so must their base.
            lam0,mu0,J0=face_properties(self,T,Y,fast=False)
        soret=bool(self.soret_enabled)
        c=self.case
        scalars=(float(self.mref),float(self.aref),float(self.Tref),float(self.lref),float(self.species_scale),
                 float(self.energy_scale),float(c.fuel_mass_flux),float(c.oxidizer_mass_flux),
                 float(c.fuel_temperature),float(c.oxidizer_temperature))
        fixed=fixed_temperature is not None
        Tfixed=np.ascontiguousarray(T if not fixed else fixed_temperature,dtype=float)
        analytic=False;jy=np.zeros((1,1,1));jt=np.zeros((1,1))
        tile_nodes=int(self.chemistry_tile_nodes) if self.ns>20 else 0
        if tile_nodes>0:
            bytes_per_node=8*self.ns*(self.ns+1)
            tile_nodes=min(tile_nodes,max(1,_CHEMISTRY_TILE_BUDGET_BYTES//bytes_per_node))
        if self.analytic_chemistry:
            try:
                if tile_nodes>0:
                    T,Y=prepare_species_partials(self.backend,T,Y)
                else:
                    jy,jt=species_partials(self.backend,T,Y,_temperature=True)
                analytic=True
            except ValueError:
                # Non-elementary orders or a clipped mixture: grouped finite differences.
                pass
        ns,n,nv=self.ns,self.n,self.nv
        lower,diag,upper=np.zeros((n-1,nv,nv)),np.zeros((n,nv,nv)),np.zeros((n-1,nv,nv))
        border=np.zeros((2,nv))
        arguments=(np.ascontiguousarray(x),m,V,T,Y,
            np.ascontiguousarray(rho),np.ascontiguousarray(cp),omega,hk,lam0,mu0,J0,self.dz,float(f['Lambda']),
            scalars,np.ascontiguousarray(c.fuel_Y,dtype=float),np.ascontiguousarray(c.oxidizer_Y,dtype=float),
            np.ascontiguousarray(self.excess,dtype=np.int64),fixed,Tfixed,float(self.P),data['invW'],data['mw'],
            data['mixture'],data.get('multi_certified',data['multi']),data['multicomponent'],
            soret,data['kinetics'],analytic)
        outputs=(self.radiation,self._frozen_multicomponent(),lower,diag,upper,border)
        use_base_bank = analytic and tile_nodes>0 and _valid_face_bank(self,transport_bank,T,Y,cached_properties)
        base_bank=(transport_bank['multi'],transport_bank['thermal'],transport_bank['lambda']) if use_base_bank else None
        thermal_bank=None
        if use_base_bank and self.thermal_transport_tangents:
            tangents=transport_bank.get('temperature_tangents')
            if (isinstance(tangents,tuple) and len(tangents)==3
                and all(isinstance(a,np.ndarray) and a.dtype==np.float64
                        and a.shape==shape and a.flags.c_contiguous
                        and not a.flags.writeable and np.isfinite(a).all()
                        for a,shape in zip(tangents,((n-1,ns),(n-1,ns),(n-1,3))))
                and np.isin(tangents[2][:,2],(0.,1.)).all()):
                thermal_bank=tangents
        warm_bank=None
        warm_thermal=None
        if compile_only and _face_bank_available(self):
            warm_bank=(np.empty((0,ns,ns)),np.empty((0,ns)),np.empty(0))
            for array in warm_bank:
                array.flags.writeable=False
            if self.thermal_transport_tangents:
                warm_thermal=(np.empty((0,ns)),np.empty((0,ns)),np.empty((0,3)))
                for array in warm_thermal:
                    array.flags.writeable=False
        if analytic and tile_nodes>0:
            temperature_cache_path = ns>=256 and outputs[1] and not data['multi'][10]
            def tiled_range(start,stop):
                use_temperature_cache = use_base_bank or (temperature_cache_path and (compile_only or
                    diffusion_kernels._temperature_cache_eligible(start,stop,x,T,self.Tref,nv)))
                # Large frozen systems otherwise lose face reuse at every one-node tile.
                carried_ws=((diffusion_kernels._node_workspace_temperature_cache(ns,use_base_bank)
                            if use_temperature_cache else diffusion_kernels._node_workspace(ns,True))
                            if ns>=256 and outputs[1] else None)
                # Compile both serial kernels even for the empty warmup range.
                for first in range(start,max(start+1,stop),tile_nodes):
                    last=min(first+tile_nodes,stop)
                    dy,dt=species_partials_range(self.backend,T,Y,first,last,_validated=True)
                    if carried_ws is None:
                        diffusion_kernels.jacobian_range(first,last,*arguments,dy,dt,*outputs,True,first)
                    elif use_base_bank:
                        carried_ws=diffusion_kernels.jacobian_range_bank(
                            first,last,*arguments,dy,dt,*outputs,
                            carried_ws,first>start,base_bank,True,first,self.analytic_frozen_transport,self.algebraic_frozen_transport,thermal_bank)
                    elif use_temperature_cache:
                        carried_ws=diffusion_kernels.jacobian_range_temperature_cache(
                            first,last,*arguments,dy,dt,*outputs,
                            carried_ws,first>start,True,first)
                    else:
                        carried_ws=diffusion_kernels.jacobian_range_workspace(
                            first,last,*arguments,dy,dt,*outputs,
                            carried_ws,first>start,True,first)
                    if compile_only:
                        # Empty/single-node arrays appear C-contiguous to Numba,
                        # while ordinary node-packed tiles have layout A. Load
                        # both consumer specializations outside the solve timer.
                        packed=np.empty((2,self.ns,self.ns)).transpose(1,0,2)
                        diffusion_kernels.jacobian_range(0,0,*arguments,packed,dt,*outputs,True,0)
                        if carried_ws is not None:
                            ordinary_ws=diffusion_kernels._node_workspace(ns,True)
                            for partials in (dy,packed):
                                diffusion_kernels.jacobian_range_workspace(
                                    0,0,*arguments,partials,dt,*outputs,ordinary_ws,False,True,0)
                                if temperature_cache_path:
                                    diffusion_kernels.jacobian_range_temperature_cache(
                                        0,0,*arguments,partials,dt,*outputs,carried_ws,False,True,0)
                                if warm_bank is not None:
                                    diffusion_kernels.jacobian_range_bank(
                                        0,0,*arguments,partials,dt,*outputs,carried_ws,False,warm_bank,True,0,
                                        self.analytic_frozen_transport,self.algebraic_frozen_transport,warm_thermal)
                        if temperature_cache_path:
                            diffusion_kernels._temperature_cache_eligible(0,0,x,T,self.Tref,nv)
                    del dy,dt  # Release this bank before allocating the next.
            # Small grids of large frozen systems need more independent ranges;
            # retain the existing partition for larger grids and other paths.
            _split(tiled_range,0 if compile_only else n,minimum=2 if ns>=256 and outputs[1] and 8<=n<16 else 4)
        else:
            _split(diffusion_kernels.jacobian_range,0 if compile_only else n,*arguments,jy,jt,*outputs,minimum=4)
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


# Largest normwise backward error accepted from the block elimination; a
# stable factorization gives ~1e-15, a pivot-free elimination breakdown O(1).
BACKWARD_ERROR_LIMIT=1e-8


def bordered_matvec(blocks,rdt,x):
    """[[A + rdt M, c], [r, 0]] x for dense (BorderedBlocks) or structured blocks."""
    n,nv=blocks.n,blocks.nv
    u=x[:-1].reshape(n,nv)
    y=np.einsum('nij,nj->ni',blocks.diag,u)+rdt*blocks.mass.reshape(n,nv)*u
    if hasattr(blocks,'lowD'):
        y[1:]+=blocks.lowD*u[:-1]+np.einsum('nar,nr->na',blocks.lowP,np.einsum('nbr,nb->nr',blocks.lowQ,u[:-1]))
        y[:-1]+=blocks.upD*u[1:]+np.einsum('nar,nr->na',blocks.upP,np.einsum('nbr,nb->nr',blocks.upQ,u[1:]))
    else:
        y[1:]+=np.einsum('nij,nj->ni',blocks.lower,u[:-1])
        y[:-1]+=np.einsum('nij,nj->ni',blocks.upper,u[1:])
    y[1:-1,1]+=x[-1]
    out=np.empty(n*nv+1)
    out[:-1]=y.ravel()
    out[-1]=blocks.border.ravel()@x[(n-2)*nv:n*nv]
    return out


def _matrix_norm(blocks,rdt):
    """Upper bound of the infinity norm of the bordered matrix."""
    n,nv=blocks.n,blocks.nv
    rows=abs(blocks.diag).sum(axis=2)+abs(rdt*blocks.mass.reshape(n,nv))
    if hasattr(blocks,'lowD'):
        rows[1:]+=abs(blocks.lowD)+np.einsum('nar,nr->na',abs(blocks.lowP),abs(blocks.lowQ).sum(axis=1))
        rows[:-1]+=abs(blocks.upD)+np.einsum('nar,nr->na',abs(blocks.upP),abs(blocks.upQ).sum(axis=1))
    else:
        rows[1:]+=abs(blocks.lower).sum(axis=2)
        rows[:-1]+=abs(blocks.upper).sum(axis=2)
    rows[1:-1,1]+=1.
    return max(float(rows.max()),float(abs(blocks.border).sum()))


def backward_error(blocks,rdt,x,b):
    """Normwise backward error of x as a solution of the bordered system for b."""
    if not np.isfinite(x).all():
        return np.inf
    residual=float(np.max(abs(bordered_matvec(blocks,rdt,x)-b)))
    scale=_matrix_norm(blocks,rdt)*float(np.max(abs(x)))+float(np.max(abs(b)))
    return residual/scale if scale>0. else (0. if residual==0. else np.inf)


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
        # Block elimination pivots only inside blocks: the first solve is
        # verified against the matrix and falls back to sparse LU if needed.
        self.blocks=blocks;self.backward_error=None

    def solve(self,rhs):
        if self.sparse is not None:
            return self.sparse.solve(rhs)
        out=self._eliminate(rhs)
        if self.backward_error is None:
            self.backward_error=backward_error(self.blocks,self.rdt,out,rhs)
            if not self.backward_error<=BACKWARD_ERROR_LIMIT:
                self.inverse=self.coupling=None
                self.sparse=splu(self.blocks.tocsc(self.rdt))
                return self.sparse.solve(rhs)
        return out

    def determinant_sign(self):
        """Sign of the bordered determinant (pivot inverses keep the sign of their pivots)."""
        if self.sparse is not None:
            return _lu_determinant_sign(self.sparse)
        signs,_=np.linalg.slogdet(self.inverse)
        return int(np.prod(np.sign(signs)))*int(np.sign(self.schur))

    def _eliminate(self,rhs):
        n,nv=self.n,self.nv
        u=block_linear.twisted_solve(self.lower,self.upper,self.inverse,self.coupling,self.k,
                                          np.ascontiguousarray(rhs[:-1]))
        delta=(rhs[-1]-self.r@u[(n-2)*nv:])/self.schur
        out=np.empty(n*nv+1)
        np.subtract(u,self.v*delta,out=out[:-1])
        out[-1]=delta
        return out


def _permutation_sign(order):
    """+1 for an even permutation, -1 for an odd one (cycle decomposition)."""
    order=np.asarray(order);seen=np.zeros(len(order),dtype=bool);sign=1
    for start in range(len(order)):
        if seen[start]:
            continue
        length=0;j=start
        while not seen[j]:
            seen[j]=True;j=order[j];length+=1
        if length%2==0:
            sign=-sign
    return sign


def _lu_determinant_sign(lu):
    """Sign of det A from SuperLU factors Pr A Pc = L U (unit lower L)."""
    diagonal=lu.U.diagonal()
    if not np.all(diagonal):
        return 0
    return (int(np.prod(np.sign(diagonal)))*_permutation_sign(lu.perm_r)
            *_permutation_sign(lu.perm_c))


# 1/dt [1/s] at which the pseudo-transient matrix J + M/dt is dominated by M.
STABILITY_SHIFT=1e12


def branch_stability(problem,x):
    """Parity of the real unstable modes of a converged counterflow flame.

    The pseudo-transient M dx/dt = -F(x) linearized about a steady state has
    a real growth rate lambda > 0 exactly where det(J + s M) = 0 at s =
    lambda. Comparing the sign of det J (s = 0) with det(J + s M) for large s
    gives the parity of the real unstable modes: the burning and extinct
    branches of the S-curve have none, and the middle branch, between the
    extinction and ignition turning points, one. Unlike the slope of Tmax
    against strain (which changes sign on the burning branch with heat loss
    to the inlets or Lewis-number effects), the parity does not depend on
    the flame. Each sign comes from the factorization: the LU of every pivot.

    Returns dict(stable, steady_sign, transient_sign, time_s), or None when a
    factorization fails (the flame is then not judged).
    """
    started=time.perf_counter()
    model=None
    try:
        model=_LinearModel(problem,x,None)
        steady=model.factor(0.).determinant_sign()
        transient=model.factor(STABILITY_SHIFT).determinant_sign()
    except (np.linalg.LinAlgError,RuntimeError,FloatingPointError,ValueError,ZeroDivisionError):
        return None
    finally:
        model=None
    if steady==0 or transient==0:
        return None
    return dict(stable=bool(steady==transient),steady_sign=int(steady),transient_sign=int(transient),
                time_s=time.perf_counter()-started)


# A cold start that ends extinct or on the unstable middle branch is solved
# again on finer initial grids. The coarse grid's numerical diffusion can
# quench a strained flame that burns on finer grids (GRI CH4/air at 10x the
# default fluxes: 31 nodes extinct, 61 nodes middle branch at 1628 K, 121
# nodes burning at 1917 K, as continuation gives). The first retry resolves
# the mixing layer sqrt(2D/a) with this many intervals (at least four times
# the previous count); later ones double the count. Extinction on a grid
# that already resolves the mixing layer is accepted as physical.
COLD_START_RETRIES=2
MIXING_LAYER_INTERVALS=16


def _use_structured(problem):
    mode=getattr(problem,'structured_jacobian',False)
    if mode not in ('auto',True,False):
        raise ValueError('structured_jacobian must be auto, True or False')
    if not (bool(mode) and type(problem) is DiffusionFlameProblem and problem.compiled_faces):
        return False
    if _trace_preconditioned(problem):
        return True
    return (problem.transport_model=='mixture-averaged' and not problem.backend.uses_multicomponent_flux
            and (mode is True or problem.ns>=STRUCTURED_MIN_SPECIES))


def _trace_preconditioned(problem):
    """Pruned multicomponent model: structured mixture-form preconditioner and GMRES.

    Its O(ns^2) composition coupling has no low-rank form; the analytic
    structured Jacobian with mixture-averaged face coefficients around the
    multicomponent base fluxes preconditions Newton--Krylov on the complete
    residual (matrix-free J v), and every step is accepted on that residual.
    """
    return (getattr(problem,'trace_tail',None) is not None and problem.compiled_faces
            and 'multi_trace' in problem._compiled_data())


class _LinearModel:
    """Steady Jacobian at an anchor state; transient rows add ``mass/dt``.

    Kept across Newton calls and pseudo-time steps on one grid and temperature
    mode, as Cantera's MultiJac keeps its Jacobian and only refactorizes when
    dt changes. It shapes Newton directions only: every step is still accepted
    against the complete current residual.
    """

    def __init__(self,problem,x,fixed_temperature, *, cached_properties=None,transport_bank=None):
        self.problem=problem
        self.structured=None
        if _use_structured(problem):
            try:
                # Large systems assemble their diagonal blocks chunk by chunk
                # inside the factorization and never hold all of them.
                large=(8*problem.n*problem.nv*problem.nv>=INPLACE_FACTOR_BYTES
                       or problem.nv>=LAZY_MIN_VARIABLES)
                self.structured=structured_blocks(problem,x,fixed_temperature=fixed_temperature,
                                                  cached_properties=cached_properties,lazy=large)
                self.state=(x.copy(),None if fixed_temperature is None else np.array(fixed_temperature))
            except ValueError:
                self.structured=None
        if self.structured is not None:
            self.blocks=self.structured
        elif _fast_faces(problem):
            # Vectorized residual faces round differently from the scalar
            # column kernels: rebuild the base with the latter.
            self.blocks=problem.jacobian_blocks(x,fixed_temperature=fixed_temperature)
        elif transport_bank is not None:
            self.blocks=problem.jacobian_blocks(x,fixed_temperature=fixed_temperature,
                                               cached_properties=cached_properties,transport_bank=transport_bank)
        elif cached_properties is None:
            self.blocks=problem.jacobian_blocks(x,fixed_temperature=fixed_temperature)
        else:
            self.blocks=problem.jacobian_blocks(x,fixed_temperature=fixed_temperature,
                                                cached_properties=cached_properties)
        self.fixed=fixed_temperature is not None
        self.age=0
        self.factors={}
        problem.jacobian_evaluations=getattr(problem,'jacobian_evaluations',0)+1

    def factor(self,rdt):
        factor=self.factors.get(rdt)
        if factor is None and rdt>0. and self.structured is not None:
            # As CVODE reuses M = I - gamma J while gamma/gamma_old stays
            # close to one, a transient step whose 1/dt is within a factor
            # TRANSIENT_FACTOR_REUSE of the cached one keeps that factor. It
            # only shapes the correction; each step is still accepted on
            # the true transient residual. Steady factors are always exact.
            limit=KRYLOV_FACTOR_REUSE if _use_krylov(self.problem) else TRANSIENT_FACTOR_REUSE
            for key,cached in self.factors.items():
                if key>0. and max(rdt/key,key/rdt)<=limit:
                    return cached
        if factor is None:
            # Keep the steady factorization and the current time step only.
            for key in [key for key in self.factors if key not in (0.,)]:
                del self.factors[key]
            if self.structured is None:
                factor=self.factors[rdt]=BorderedFactor(self.blocks,rdt)
            else:
                factor=self.factors[rdt]=self._structured_factor(rdt)
        return factor

    def _structured_factor(self,rdt):
        pool=pool_of(self.problem)
        if self.structured.lazy:
            # One factor at a time: its pivot inverses are the largest array
            # of the solve. Assembly is repeated for another time step.
            self.factors.clear()
            try:
                return StructuredFactor(self.structured,rdt,pool=pool)
            except np.linalg.LinAlgError:
                x,fixed=self.state
                self.structured=self.blocks=structured_blocks(self.problem,x,fixed_temperature=fixed)
                return StructuredFactor(self.structured,rdt,pool=pool)
        inplace=self.structured.diag.nbytes>=INPLACE_FACTOR_BYTES
        if inplace:
            self.factors.clear()
        if self.structured.consumed:
            # Recompute the analytic blocks at the same anchor state.
            x,fixed=self.state
            self.structured=self.blocks=None
            self.structured=self.blocks=structured_blocks(self.problem,x,fixed_temperature=fixed)
        try:
            return StructuredFactor(self.structured,rdt,inplace=inplace,pool=pool)
        except np.linalg.LinAlgError:
            x,fixed=self.state
            self.structured=self.blocks=structured_blocks(self.problem,x,fixed_temperature=fixed)
            return StructuredFactor(self.structured,rdt,pool=pool)


class _SparseModel:
    """Generic problems: one sparse finite-difference matrix per Newton call."""

    def __init__(self,problem,x,fun,base):
        self.matrix=splu(problem.jacobian(x,fun,base))
        self.fixed=None
        self.age=0
        problem.jacobian_evaluations=getattr(problem,'jacobian_evaluations',0)+1

    def factor(self,rdt):
        return self.matrix


# Right-preconditioned GMRES, shared with the premixed solver.
_gmres=gmres


# Inexact Newton-Krylov acceleration of the factored (frozen-transport,
# possibly aged) linear model: matrix-free J v from the complete residual.
KRYLOV=dict(rtol=1e-2,iterations=12)


# Below this size a fresh factorization is cheaper than Krylov residuals.
KRYLOV_MIN_SPECIES=400


def _use_krylov(problem):
    mode=getattr(problem,'newton_krylov','auto')
    if mode is True:
        return True
    if bool(mode) and _use_structured(problem) and _trace_preconditioned(problem):
        return True
    return bool(mode) and _use_structured(problem) and problem.ns>=KRYLOV_MIN_SPECIES


def _newton(problem,x,deadline, *, fixed_temperature=None,previous=None,dt=None,max_iterations=25,
            correction_tolerance=False,max_age=4,max_damping=16,refresh_rate=None):
    """Damped Newton on the scaled counterflow residual.

    Steady and fixed-temperature calls converge only on the maximum scaled
    residual ``1e-7``. Pseudo-time steps may instead stop, as in Cantera, when
    the weighted Newton correction is below one (relative 1e-4, absolute
    1e-11); they only advance the continuation and never certify a flame.
    """
    reuse_properties=(type(problem) is DiffusionFlameProblem and problem.compiled_faces
                      and problem.reuse_base_properties)
    reuse_bank=reuse_properties and _face_bank_available(problem)
    evaluated_properties=None;evaluated_bank=None
    def fun(value,export_bank=False):
        nonlocal evaluated_properties,evaluated_bank
        evaluated_bank=None
        if reuse_properties:
            if reuse_bank and export_bank:
                evaluated_properties,evaluated_bank=problem.properties_for_newton(problem.unpack(value))
            else:
                evaluated_properties=problem.properties(problem.unpack(value))
            return problem.residual(value,fixed_temperature=fixed_temperature,previous=previous,dt=dt,
                                    cached_properties=evaluated_properties)
        return problem.residual(value,fixed_temperature=fixed_temperature,previous=previous,dt=dt)
    blocked=hasattr(problem,'jacobian_blocks')
    krylov=blocked and _use_krylov(problem)
    rdt=0. if previous is None else 1./dt
    model=getattr(problem,'_linear_model',None) if blocked else None
    if model is not None and (model.fixed!=(fixed_temperature is not None) or model.blocks.n!=problem.n):
        model=None
    base=None;base_properties=None;base_bank=None;last_norm=None;factor=None
    for iteration in range(max_iterations):
        if time.perf_counter() > deadline:
            return x,False,'time_limit'
        if base is None:
            base = fun(x,export_bank=model is None or model.age>=max_age)
            base_properties=evaluated_properties;base_bank=evaluated_bank;evaluated_bank=None
        if not np.isfinite(base).all():
            return x,False,'nonfinite_residual'
        norm = float(np.max(abs(base)))
        if norm < 1e-7:
            return x,True,'converged'
        if model is None or model.age>=max_age:
            if blocked:
                # Drop the previous Jacobian and its factors before building the
                # next one: large mechanisms must never hold both (GBs each).
                model=None;factor=None;problem._linear_model=None
            try:
                if blocked:
                    if base_bank is None:
                        model=_LinearModel(problem,x,fixed_temperature,cached_properties=base_properties)
                    else:
                        model=_LinearModel(problem,x,fixed_temperature,cached_properties=base_properties,
                                           transport_bank=base_bank)
                else:
                    model=_SparseModel(problem,x,fun,base)
                # Coefficients live only until this base's blocks are assembled.
                # An unexpected refresh uses the ordinary exact path; it never
                # borrows the coefficients of a rejected or earlier state.
                base_bank=None
            except RuntimeError:
                return x,False,'singular_jacobian'
            if blocked:
                problem._linear_model=model
        try:
            factor=None  # never pin a previous factor while a new one is built
            factor=model.factor(rdt)
        except RuntimeError:
            return x,False,'singular_jacobian'
        model.age+=1
        step = factor.solve(-base)
        if not np.isfinite(step).all():
            return x,False,'singular_jacobian'
        b = x[:-1].reshape(problem.n,problem.nv)
        # Use the original correction before the trace-species bound changes it.
        mean=np.mean(abs(b),axis=0)
        scales=np.empty(len(x))
        scales[:-1].reshape(problem.n,problem.nv)[:]=1e-9+1e-4*mean
        scales[-1]=1e-4*max(abs(x[-1]),1.)
        # Merit norm of the factored linear model (also used by damping).
        oldnorm=np.linalg.norm(step/scales)
        if krylov:
            def apply_jacobian(v):
                peak=float(np.max(abs(v)/scales))
                if not peak>0.:
                    return np.zeros_like(v)
                eps=1e-3/peak
                return (problem.residual(x+eps*v,fixed_temperature=fixed_temperature,previous=previous,dt=dt)
                        -base)/eps
            accelerated,solved=_gmres(apply_jacobian,factor.solve,-base,step,KRYLOV['rtol'],KRYLOV['iterations'])
            if np.isfinite(accelerated).all():
                step=accelerated
            if not solved and model.age>1:
                # The factored model no longer preconditions this state.
                model=None;last_norm=None
                continue
        delta = step[:-1].reshape(problem.n,problem.nv)
        # With Krylov acceleration the stale model is judged by GMRES itself.
        if (refresh_rate is not None and not krylov and last_norm is not None and model.age>1
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
            candidate = fun(trial,export_bank=model.age>=max_age)
            if np.isfinite(candidate).all():
                # A root reached on the last allowed step is still converged.
                if np.max(abs(candidate))<1e-7:
                    return trial,True,'converged'
                if np.linalg.norm(factor.solve(candidate)/scales)<oldnorm*(1.-1e-4*length):
                    x=trial;base=candidate;base_properties=evaluated_properties;base_bank=evaluated_bank;evaluated_bank=None;break
            length*=.5
        else:
            if model.age>1:
                model=None
                continue
            return x,False,'line_search'
    return x,False,'iteration_limit'


@numba.njit(cache=True)
def _refine_ratios_core(z,values,slope,curve,thresh,min_range):
    """Largest slope/curvature criterion ratio of every interval (refiner rules)."""
    n=z.shape[0]
    out=np.zeros(n-1)
    dz=np.empty(n-1)
    for j in range(n-1):
        dz[j]=z[j+1]-z[j]
    slopes=np.empty(n-1)
    for p in range(values.shape[0]):
        v=values[p]
        for j in range(n-1):
            slopes[j]=(v[j+1]-v[j])/dz[j]
        vmin,vmax=np.min(v),np.max(v)
        smin,smax=np.min(slopes),np.max(slopes)
        if vmax-vmin>min_range*max(abs(vmax),abs(vmin)):
            change=slope*(vmax-vmin)
            for j in range(n-1):
                out[j]=max(out[j],abs(v[j+1]-v[j])/(change+thresh))
        if smax-smin>min_range*max(abs(smax),abs(smin)):
            change=curve*(smax-smin)
            for j in range(n-2):
                r=abs(slopes[j+1]-slopes[j])/(change+thresh/dz[j])
                out[j]=max(out[j],r)
                out[j+1]=max(out[j+1],r)
    return out


def _refine_ratios(z,profiles,refiner):
    rows=[np.asarray(v,dtype=float) for v in profiles.values()]
    return _refine_ratios_core(np.asarray(z,dtype=float),np.ascontiguousarray(np.stack(rows)),
                               refiner.slope,refiner.curve,refiner._thresh,refiner._min_range)


def _normalized(x,problem):
    x=x.copy()
    blocks=x[:-1].reshape(problem.n,problem.nv)
    guess=np.maximum(blocks[:,3:],0.)
    blocks[:,3:]=guess/guess.sum(axis=1,keepdims=True)
    return x


TRANSIENT=dict(growth=1.5,iterations=25,refresh_rate=.5,max_age=20)
# Largest 1/dt ratio for which a cached transient factorization is reused.
TRANSIENT_FACTOR_REUSE=2.
# Structured factors at least this large are computed in place, one at a time.
INPLACE_FACTOR_BYTES=256*1024**2
# From this block size the diagonal blocks are assembled chunk by chunk in
# the factorization and never held besides the pivot inverses.
LAZY_MIN_VARIABLES=200
# With Krylov acceleration the factor is only a preconditioner: reuse longer.
KRYLOV_FACTOR_REUSE=8.
STEADY=dict(max_age=12,refresh_rate=.5)
WARM=dict(iterations=25)


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


def _mixture_bootstrap(problem,max_time,ratio,slope,curve,max_points,refine):
    """Accepted mixture-averaged flame on its own adaptive grid, or None.

    Multicomponent and Soret flames start from it (as the premixed
    ``multicomponent_bootstrap``): the cold pseudo-transient, the most
    expensive part of a solve, then runs with the cheap transport. Only the
    initial state changes; the requested flame is solved, refined and
    accepted on its own complete residual.
    """
    twin=DiffusionFlameProblem(replace(problem.case,transport_model='mixture-averaged',soret_enabled=False),
                               problem.mech_data,problem.z)
    twin.refine_split=getattr(problem,'refine_split',2)
    fields,accepted,report=solve_diffusion_problem(twin,max_time=max_time,ratio=ratio,slope=slope,curve=curve,
                                                   max_points=max_points,refine=refine)
    summary=dict(accepted=bool(accepted),nodes=len(fields['z']),runtime_s=float(report['runtime_s']),
                 reason=report['reason'])
    if not accepted:
        return None,summary
    return dict(z=fields['z'],**{key:fields[key] for key in ('m','V','T','Y','Lambda')}),summary


def solve_diffusion_problem(problem, *, initial=None,max_time=180.,ratio=3.,slope=.04,curve=.08,max_points=1600,refine=True,
                            cold_retries=None):
    entered=time.perf_counter()
    retries=COLD_START_RETRIES if cold_retries is None else cold_retries
    bootstrap=None;bootstrap_time=0.
    if (initial is None and getattr(problem,'transport_bootstrap',True) and type(problem) is DiffusionFlameProblem
            and (problem.backend.uses_multicomponent_flux or problem.soret_enabled)):
        seed,bootstrap=_mixture_bootstrap(problem,max_time,ratio,slope,curve,max_points,refine)
        bootstrap_time=bootstrap['runtime_s']
        if seed is not None:
            problem.set_grid(seed.pop('z'))
            initial=seed
    cold_start=initial is None
    cold_grid=problem.z.copy()

    span=float(cold_grid[-1]-cold_grid[0])
    resolved=int(np.ceil(MIXING_LAYER_INTERVALS*span/problem.mixing_layer_thickness()))
    def retry_cold(history,burning,stability):
        """Cold start again on a finer uniform initial grid."""
        intervals=len(cold_grid)-1
        nodes=(max(4*intervals,resolved) if intervals<resolved else 2*intervals)+1
        problem.set_grid(np.linspace(cold_grid[0],cold_grid[-1],nodes))
        attempt=dict(stage='cold_start_retry',nodes=int(len(cold_grid)),burning=bool(burning),branch=stability,
                     runtime_s=time.perf_counter()-entered,next_initial_nodes=int(nodes))
        fields,accepted,report=solve_diffusion_problem(problem,initial=None,
            max_time=max_time-(time.perf_counter()-entered),ratio=ratio,slope=slope,curve=curve,
            max_points=max_points,refine=refine,cold_retries=retries-1)
        report['history']=list(history)+[attempt]+list(report['history'])
        report['runtime_s']=time.perf_counter()-entered
        return fields,accepted,report

    x = problem.pack(problem.initial_fields() if initial is None else initial)
    # JIT compilation (first use or changed sources) is not solver time.
    problem.compile_kernels(x)
    started = time.perf_counter()-bootstrap_time;deadline=started+max_time
    history=[] if bootstrap is None else [dict(stage='mixture_averaged_bootstrap',**bootstrap)]
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
        cold=mesh_round==0 and initial is None
        x,ok,reason=_newton(problem,x,deadline,max_age=10 if cold else STEADY['max_age'],
                            max_damping=6 if cold else 16,max_iterations=25 if cold else WARM['iterations'],
                            refresh_rate=None if cold else STEADY['refresh_rate'])
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
        if (mesh_round==0 and cold_start and retries>0
                and max_time-(time.perf_counter()-entered)>0.):
            # The branch is decided by the cold start: judge it on this coarse
            # grid, before refining an extinct or unstable state.
            burning=bool(fields['T'].max()>max(problem.case.fuel_temperature,problem.case.oxidizer_temperature)+100.)
            stability=branch_stability(problem,x) if burning else None
            if ((not burning and len(cold_grid)-1<resolved)
                    or (stability is not None and not stability['stable'])):
                problem._linear_model=None;problem._buffer_pool=None
                return retry_cold(history,burning,stability)
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
        split=int(getattr(problem,'refine_split',2))
        if split>2:
            # Multi-point refinement: an interval whose worst criterion ratio
            # is r receives min(split, ceil(r)) equal sub-intervals at once,
            # so fewer intermediate grids (and their factorizations) are
            # needed. The acceptance test is unchanged: the final grid must
            # again produce no insertion with the standard criteria.
            ratios=_refine_ratios(problem.z,profiles,refiner)
            points=[]
            for j in sorted(insert):
                pieces=int(min(split,max(2,np.ceil(ratios[j]))))
                points.extend(oldz[j]+(oldz[j+1]-oldz[j])*np.arange(1,pieces)/pieces)
            if problem.n+len(points)>max_points:
                points=[.5*(oldz[j]+oldz[j+1]) for j in sorted(insert)]
            newz=np.sort(np.r_[oldz,points])
        else:
            newz=np.sort(np.r_[oldz,[.5*(oldz[j]+oldz[j+1]) for j in sorted(insert)]])
        interpolated={k:np.interp(newz,oldz,fields[k]) for k in ('m','V','T')}
        interpolated['Y']=np.array([np.interp(newz,oldz,row) for row in fields['Y']])
        interpolated['Lambda']=fields['Lambda']
        problem.set_grid(newz)
        x=problem.pack(interpolated)
    # Release the last Jacobian/factors and the reusable buffers (GBs for
    # large mechanisms) before the diagnostics; the problem may outlive us.
    problem._linear_model=None
    problem._buffer_pool=None
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
    # Branch of the S-curve: a converged burning flame must be stable.
    stability=branch_stability(problem,x) if ok and burning else None
    problem._linear_model=None;problem._buffer_pool=None
    report=dict(converged=bool(ok),grid_converged=grid_converged,
                grid_mode='adaptive' if refine else 'fixed',burning=burning,branch=stability,
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
    stable=stability is None or stability['stable']
    accepted=bool(ok and (grid_converged or not refine) and burning and stable and maximum<1e-7
                  and report['minimum_mass_fraction']>=-1e-7 and report['maximum_species_sum_error']<1e-7)
    report['accepted']=accepted
    if (cold_start and ok and retries>0 and max_time-(time.perf_counter()-entered)>0.
            and ((not burning and len(cold_grid)-1<resolved) or (burning and not stable))):
        # Refinement moved the flame to another branch: cold start again.
        return retry_cold([],burning,stability)
    return fields,accepted,report
