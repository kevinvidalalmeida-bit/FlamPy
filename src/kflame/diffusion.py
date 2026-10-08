"""Public native nonpremixed workflow; reference tools load explicitly."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from kflame.api import _output, _stream
from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.fgm.nonadiabatic3d import bilger_coefficients
from kflame.flame.diffusion import DiffusionCase, DiffusionFlameProblem, solve_diffusion_problem


def solve_diffusion_flame(*, mechanism='gri30.yaml', fuel='CH4', oxidizer='O2:1,N2:3.76',
                          fuel_temperature=300., oxidizer_temperature=300., pressure=101325.,
                          fuel_mass_flux=.12, oxidizer_mass_flux=.24, width=.02, grid=None,
                          initial_points=31, transport='mixture-averaged', soret=False,
                          ratio=3., slope=.01, curve=.02, max_points=2400, max_time=600.,
                          initial_solution=None, refine=True, output=None, plots=False,
                          radiation=False, radiation_scale=1., boundary_emissivities=(0., 0.)):
    """Solve a native axisymmetric diffusion flame with two separate inflows.

    Streams are mole-basis strings/dictionaries. Both mass fluxes are positive
    magnitudes [kg/(m² s)]; oxidizer velocity is negative. Returns all species,
    sources, u, spread rate V, pressure eigenvalue, Bilger Z, h and diagnostics.
    Saves NPZ, CSV and metadata. A rejected/extinguished solve saves its state
    before raising RuntimeError. No premixed flame speed is assigned. Cantera
    is never loaded by this entry point. Output must not already exist.

    ``radiation=True`` adds the optically thin H2O/CO2 radiative loss of
    Cantera's Flow1D to the energy equation (non-adiabatic flame); the default
    is adiabatic. ``radiation_scale`` multiplies that loss to generate deeper
    enthalpy defects for tabulation (1 is the physical model), and
    ``boundary_emissivities`` are the (fuel, oxidizer) inlet emissivities.
    """
    for name,value in dict(fuel_temperature=fuel_temperature,oxidizer_temperature=oxidizer_temperature,
            pressure=pressure,fuel_mass_flux=fuel_mass_flux,oxidizer_mass_flux=oxidizer_mass_flux,
            width=width,max_time=max_time).items():
        if not np.isfinite(value) or value<=0.:
            raise ValueError(f'{name} must be finite and positive in SI units')
    if transport not in ('mixture-averaged','multicomponent'):
        raise ValueError('Unknown transport model')
    emissivities=tuple(float(value) for value in boundary_emissivities)
    if (not np.isfinite(radiation_scale) or radiation_scale<0. or len(emissivities)!=2
            or not all(0.<=value<=1. for value in emissivities)):
        raise ValueError('Require radiation_scale >= 0 and two emissivities in [0, 1]')
    if output is not None and Path(output).exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not isinstance(initial_points,int) or not isinstance(max_points,int) or not 5<=initial_points<=max_points:
        raise ValueError('Require integer 5 <= initial_points <= max_points')
    if not np.isfinite(ratio) or ratio<2. or not 0.<slope<=1. or not 0.<curve<=1.:
        raise ValueError('Require ratio >= 2 and 0 < slope, curve <= 1')
    mechanism=resolve_mechanism(str(mechanism));mech=load_mechanism(mechanism)
    fuel,oxidizer=_stream(fuel),_stream(oxidizer)
    feeds=[]
    for text in (fuel,oxidizer):
        amounts=_composition(text,mech.species_names)*mech.molecular_weights
        feeds.append(amounts/amounts.sum())
    seed=None
    if initial_solution is not None:
        seed_folder=Path(initial_solution)
        meta=json.loads((seed_folder/'metadata.json').read_text(encoding='utf-8'))
        if (not meta.get('accepted') or meta.get('backend')!='native_cpu'
                or meta.get('flow_type')!='counterflow_diffusion'
                or meta.get('mechanism_sha256')!=hashlib.sha256(Path(mechanism).read_bytes()).hexdigest()
                or meta.get('transport')!=transport or meta.get('soret')!=bool(soret)
                or not np.isclose(meta['pressure'],pressure,rtol=1e-12)
                or not np.isclose(meta['fuel_temperature'],fuel_temperature,rtol=1e-12)
                or not np.isclose(meta['oxidizer_temperature'],oxidizer_temperature,rtol=1e-12)
                or not np.allclose(meta['fuel_Y'],feeds[0],rtol=1e-12,atol=1e-15)
                or not np.allclose(meta['oxidizer_Y'],feeds[1],rtol=1e-12,atol=1e-15)):
            raise ValueError('Continuation requires an accepted native diffusion flame with the same streams and transport')
        with np.load(seed_folder/'flame.npz',allow_pickle=False) as saved:
            seed={key:saved[key].copy() for key in ('z','T','Y','m','V','Lambda','species_names')}
        if list(seed['species_names'])!=list(mech.species_names):
            raise ValueError('Continuation species order differs')
        if (not np.isfinite(seed['z']).all() or np.any(np.diff(seed['z'])<=0.)
                or seed['Y'].shape!=(mech.n_species,len(seed['z']))
                or any(not np.isfinite(seed[k]).all() for k in ('T','Y','m','V','Lambda'))):
            raise ValueError('Continuation must contain finite aligned profiles')
        if grid is None:
            grid=seed['z']*width/seed['z'][-1]
            # A changed strain on a fully refined grid makes the first
            # nonlinear solve unnecessarily stiff and expensive. Preserve
            # the native profile as a guess on a bounded coarse subset, then
            # certify the target with the normal full adaptive refinement.
            changed=(not np.isclose(meta['fuel_mass_flux'],fuel_mass_flux,rtol=1e-12)
                     or not np.isclose(meta['oxidizer_mass_flux'],oxidizer_mass_flux,rtol=1e-12)
                     or not np.isclose(meta['width'],width,rtol=1e-12))
            continuation_points=max(initial_points,min(201,max_points))
            if changed and len(grid)>continuation_points:
                indices=np.unique(np.linspace(0,len(grid)-1,continuation_points).astype(int))
                grid=grid[indices]
    z=np.linspace(0.,width,initial_points) if grid is None else np.asarray(grid,dtype=float)
    if (z.ndim!=1 or not 5<=len(z)<=max_points or not np.isfinite(z).all()
            or z[0]!=0. or not np.isclose(z[-1],width,rtol=1e-12,atol=0.) or np.any(np.diff(z)<=0.)):
        raise ValueError('grid must contain increasing finite nodes from 0 to width')
    case=DiffusionCase(mechanism,*feeds,fuel_temperature,oxidizer_temperature,
                       fuel_mass_flux,oxidizer_mass_flux,pressure,width,transport,bool(soret),
                       radiation_enabled=bool(radiation),radiation_scale=float(radiation_scale),
                       boundary_emissivities=emissivities)
    source_root=Path(__file__).parent
    digest=hashlib.sha256()
    for source in sorted(source_root.rglob('*.py')):
        digest.update(source.relative_to(source_root).as_posix().encode())
        digest.update(source.read_bytes())
    source_sha256=digest.hexdigest()
    started=time.perf_counter()
    problem=DiffusionFlameProblem(case,mech,z)
    initial=None
    if seed is not None:
        scaledz=seed['z']*width/seed['z'][-1]
        factor=np.sqrt(fuel_mass_flux*oxidizer_mass_flux/(meta['fuel_mass_flux']*meta['oxidizer_mass_flux']))
        initial={key:np.interp(z,scaledz,seed[key]) for key in ('T','m','V')}
        initial['m']*=factor;initial['V']*=factor*seed['z'][-1]/width
        initial['Y']=np.array([np.interp(z,scaledz,row) for row in seed['Y']])
        initial['Lambda']=float(seed['Lambda'])*factor**2*(seed['z'][-1]/width)**2
    fields,accepted,report=solve_diffusion_problem(problem,initial=initial,max_time=max_time,
        ratio=ratio,slope=slope,curve=curve,max_points=max_points,refine=refine)
    runtime=time.perf_counter()-started
    coeff,offset=bilger_coefficients(mech,fuel,oxidizer)
    fields['Z']=coeff@fields['Y']+offset
    folder=_output(output,'diffusion')
    np.savez_compressed(folder/'flame.npz',**fields)
    metadata=dict(backend='native_cpu',flow_type='counterflow_diffusion',accepted=accepted,
        mechanism=mechanism,mechanism_sha256=hashlib.sha256(Path(mechanism).read_bytes()).hexdigest(),
        fuel=fuel,oxidizer=oxidizer,fuel_Y=feeds[0].tolist(),oxidizer_Y=feeds[1].tolist(),
        fuel_temperature=fuel_temperature,oxidizer_temperature=oxidizer_temperature,pressure=pressure,
        fuel_mass_flux=fuel_mass_flux,oxidizer_mass_flux=oxidizer_mass_flux,width=width,
        transport=transport,soret=bool(soret),radiation=bool(radiation),radiation_scale=float(radiation_scale),
        boundary_emissivities=list(emissivities),refine=bool(refine),report=report,runtime_s=runtime,
        source_sha256=source_sha256,
        numerical_settings=dict(initial_points=len(z),ratio=ratio,slope=slope,curve=curve,
                                max_points=max_points,max_time=max_time))
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    with (folder/'profiles.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.writer(out)
        writer.writerow(['z_m','T_K','u_m_s','V_s1','Z','h_J_kg','qdot_W_m3','qrad_W_m3',*mech.species_names])
        writer.writerows(np.column_stack([fields[k] for k in ('z','T','u','V','Z','h','qdot','radiative_heat_loss')]
                                         +[fields['Y'].T]))
    if plots:
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(2,1,sharex=True)
        axes[0].plot(fields['z'],fields['T']);axes[0].set_ylabel('T [K]')
        axes[1].plot(fields['z'],fields['Z']);axes[1].set_ylabel('Z');axes[1].set_xlabel('z [m]')
        figure.tight_layout();figure.savefig(folder/'profiles.png',dpi=150);figure.savefig(folder/'profiles.pdf');plt.close(figure)
    if not accepted:
        raise RuntimeError(f'Diffusion flame rejected ({report["reason"]}, burning={report["burning"]}); diagnostics: {folder}')
    return dict(fields,accepted=accepted,report=report,output=folder,runtime_s=runtime)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=solve_diffusion_flame.__doc__)
    parser.add_argument('--mechanism',default='gri30.yaml')
    parser.add_argument('--fuel',default='CH4')
    parser.add_argument('--oxidizer',default='O2:1,N2:3.76')
    for name,default in (('fuel-temperature',300.),('oxidizer-temperature',300.),('pressure',101325.),
                         ('fuel-mass-flux',.12),('oxidizer-mass-flux',.24),('width',.02),
                         ('slope',.01),('curve',.02),('max-time',600.)):
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--transport',choices=['mixture-averaged','multicomponent'],default='mixture-averaged')
    parser.add_argument('--soret',action='store_true')
    parser.add_argument('--radiation',action='store_true',help='Optically thin H2O/CO2 radiative loss')
    parser.add_argument('--radiation-scale',type=float,default=1.)
    parser.add_argument('--plots',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--initial-solution',type=Path)
    parser.add_argument('--max-points',type=int,default=2400)
    result=solve_diffusion_flame(**vars(parser.parse_args(argv)))
    print(result['report']);print(result['output'])
