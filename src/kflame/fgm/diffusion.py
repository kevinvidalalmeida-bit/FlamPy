"""Connected nonpremixed manifolds in physical local (Z,C) controls.

Unlike a premixed family, each diffusion flame spans the mixture coordinate;
the second training direction is inlet-flow scaling (strain), not inlet phi.
No flame is sorted on reaction progress, which rises then falls along z.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from kflame.api import _output
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.diffusion import solve_diffusion_flame
from kflame.fgm.nonadiabatic3d import SimplexMesh,NonAdiabaticFGM,OutsideManifoldError


def _progress_weights(Y,provided,names):
    if provided is not None:
        from kflame.fgm.common import parse_progress_weights
        mapping=parse_progress_weights(provided) if isinstance(provided,str) else provided
        missing=set(mapping)-set(names)
        if missing:
            raise ValueError(f'Progress species absent from mechanism: {sorted(missing)}')
        weights=np.array([mapping.get(name,0.) for name in names],dtype=float)
        if not np.isfinite(weights).all() or not np.any(weights):
            raise ValueError('Progress weights must be finite and nonzero')
        return weights
    # One fixed linear combination separates adjacent strains at fixed Z.
    # Signed coefficients are allowed; normalization prevents arbitrary gains.
    differences=np.diff(Y,axis=0).reshape(-1,Y.shape[-1])
    norms=np.sum(abs(differences),axis=1)
    directions=differences[norms>1e-8]/norms[norms>1e-8,None]
    if not len(directions):
        raise ValueError('The selected strains do not span an independent chemical direction')
    ns=Y.shape[-1]
    A=np.column_stack([-directions,directions,np.ones(len(directions))])
    A=np.vstack([A,np.r_[np.ones(2*ns),0.]])
    b=np.r_[np.zeros(len(directions)),1.]
    objective=np.r_[np.zeros(2*ns),-1.]
    answer=linprog(objective,A_ub=A,b_ub=b,bounds=[(0.,None)]*(2*ns+1),method='highs')
    if not answer.success or answer.x[-1]<=1e-8:
        raise ValueError('No unique linear progress parametrizes this strain family; inspect its branches or add physical controls')
    return answer.x[:ns]-answer.x[ns:2*ns]


def build_diffusion_manifold(profiles,metadata,output, *, mixture_points=241,progress_weights=None):
    """Tabulate accepted native flamelets preserving their physical adjacency."""
    if len(profiles)<2 or not isinstance(mixture_points,int) or mixture_points<3:
        raise ValueError('Require at least two flamelets and >=3 mixture samples')
    columns=[];excluded_spatial=[]
    for profile in profiles:
        Z=np.asarray(profile['Z'])
        # Differential diffusion can produce local Bilger Z outside the two
        # feed values. The mixing chart explicitly covers [0,1]; do not clip
        # those states onto the chart, or silently sort a folded trajectory.
        admissible=np.flatnonzero((Z>=0.)&(Z<=1.))
        excluded_spatial.append(int(len(Z)-len(admissible)))
        if len(admissible)<3:
            raise ValueError('A flamelet must span a resolved mixture fraction range')
        if np.any(np.diff(Z[admissible])>1e-10):
            raise ValueError('Local Bilger Z folds along a flamelet; a single Z trajectory cannot be tabulated')
        order=admissible[::-1]
        keep=np.r_[True,np.diff(Z[order])>1e-12]
        order=order[keep]
        if len(order)<3:
            raise ValueError('A flamelet must span a resolved mixture fraction range')
        columns.append((Z[order],order,profile))
    lower=max(row[0][0] for row in columns);upper=min(row[0][-1] for row in columns)
    if upper<=lower:
        raise ValueError('Flamelets have no shared mixture-fraction domain')
    # Union of physical knots resolves steep reaction fronts without hundreds
    # of uniform samples in chemically inactive tails.
    Zgrid=np.unique(np.r_[np.linspace(lower,upper,mixture_points),
                         *[Z[(Z>lower)&(Z<upper)] for Z,_,_ in columns]])
    field_rows={name:[] for name in ('Y','h','T','omega_Y','qdot')}
    for Z,order,profile in columns:
        for name in field_rows:
            values=profile[name]
            field_rows[name].append(np.array([np.interp(Zgrid,Z,row[order]) for row in values]).T
                if name in ('Y','omega_Y') else np.interp(Zgrid,Z,values[order]))
    fields={name:np.asarray(rows) for name,rows in field_rows.items()}
    names=list(profiles[0]['species_names'])
    weights=_progress_weights(fields['Y'],progress_weights,names)
    C=fields['Y']@weights
    nc,nz=C.shape
    controls=np.column_stack([np.tile(Zgrid,nc),C.ravel()])
    indices=np.arange(nc*nz).reshape(nc,nz)
    a,b=indices[:-1,:-1].ravel(),indices[1:,:-1].ravel()
    c,d=indices[:-1,1:].ravel(),indices[1:,1:].ravel()
    cells=np.stack([np.stack([a,b,d],axis=1),np.stack([a,d,c],axis=1)],axis=1).reshape(-1,3)
    span=np.ptp(controls,axis=0)
    if np.any(span<=0.):
        raise ValueError('Degenerate diffusion manifold')
    scaled=(controls-controls.min(axis=0))/span
    determinants=np.linalg.det((scaled[cells[:,1:]]-scaled[cells[:,:1]]).transpose(0,2,1))
    nondegenerate=abs(determinants)>1e-13
    signs=np.sign(determinants[nondegenerate])
    if len(signs) and np.any(signs!=signs[0]):
        raise ValueError('The (Z,C) chart folds between strain rows; retain diagnostics and change the physical parametrization')
    cells=cells[nondegenerate]
    if not len(cells):
        raise ValueError('Degenerate diffusion manifold')
    table={name:value.reshape((-1,)+value.shape[2:]) for name,value in fields.items()}
    table.update(controls=controls,cells=cells,Z_grid=Zgrid,progress_weights=weights,species_names=np.array(names))
    table['omega_C']=table['omega_Y']@weights
    # Detect overlaps at cell interiors, never fill a convex hull across holes.
    mesh=SimplexMesh(controls,cells)
    _,_,covered=mesh.locate_batch(controls[cells].mean(axis=1))
    if not covered.all():
        raise ValueError('Diffusion manifold has uncovered cell interiors')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'diffusion_fgm.npz'
    np.savez_compressed(target,**table)
    mechanism_sha256=hashlib.sha256(Path(metadata['mechanism']).read_bytes()).hexdigest()
    meta=dict(metadata,mechanism_sha256=mechanism_sha256,flow_type='counterflow_diffusion',controls=['Z','C'],table_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              table_builder_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              species_names=names,progress_weights={name:float(weight) for name,weight in zip(names,weights) if weight!=0.},
              geometry=dict(vertices=len(controls),triangles=len(cells),excluded_degenerate_cells=int((~nondegenerate).sum()),
                            excluded_folded_cells=0,excluded_spatial_points_outside_feed_Z=excluded_spatial),
              mixture_fraction_range=[float(lower),float(upper)],
              note='Native flamelets, connected adjacent strain/Z cells, no progress sorting or extrapolation. '
                   'The two-control chart contains the enthalpy histories of its training family, not arbitrary independent heat losses.')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta


def generate_diffusion_fgm(*,flow_scales=(.75,1.,1.25),mixture_points=241,progress_weights=None,
                           initial_solution=None,output=None,**flame_settings):
    """Generate native strained diffusion flamelets and a connected (Z,C) FGM.

    Flow scales multiply both base inlet mass fluxes. Steady burning branches
    only; an extinguished or rejected flamelet aborts rather than being hidden.
    Independent held-out strain validation is required for accuracy claims.
    """
    scales=np.asarray(flow_scales,dtype=float)
    if scales.ndim!=1 or len(scales)<2 or not np.isfinite(scales).all() or np.any(scales<=0.) or np.any(np.diff(scales)<=0.):
        raise ValueError('flow_scales must be finite, positive and strictly increasing')
    forbidden=set(flame_settings)&{'output','initial_solution','grid','refine'}
    if forbidden:
        raise ValueError(f'Family controls these flame settings: {sorted(forbidden)}')
    folder=_output(output,'diffusion_fgm')
    settings=dict(flame_settings)
    fuel_flux=settings.pop('fuel_mass_flux',.12);oxidizer_flux=settings.pop('oxidizer_mass_flux',.24)
    profiles=[];previous=initial_solution
    for i,scale in enumerate(scales):
        flame=solve_diffusion_flame(**settings,fuel_mass_flux=float(fuel_flux*scale),
            oxidizer_mass_flux=float(oxidizer_flux*scale),initial_solution=previous,output=folder/f'flamelet_{i:03d}')
        profiles.append(flame);previous=flame['output']
    meta=json.loads((previous/'metadata.json').read_text(encoding='utf-8'))
    meta.update(flow_scales=scales.tolist(),base_fuel_mass_flux=fuel_flux,base_oxidizer_mass_flux=oxidizer_flux,
                all_final_accepted=True,flamelet_count=len(profiles))
    result=build_diffusion_manifold(profiles,meta,folder,mixture_points=mixture_points,progress_weights=progress_weights)
    return dict(output=folder,metadata=result,flamelets=profiles)


class DiffusionFGM:
    """Batched lookup on actual connected cells; no production chemistry calls."""
    def __init__(self,folder):
        folder=Path(folder)
        self.metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        path=folder/'diffusion_fgm.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.metadata['table_sha256']:
            raise ValueError('Frozen diffusion table fingerprint differs')
        with np.load(path,allow_pickle=False) as saved:
            self.table={key:saved[key].copy() for key in saved.files}
        if hashlib.sha256(Path(self.metadata['mechanism']).read_bytes()).hexdigest()!=self.metadata['mechanism_sha256']:
            raise ValueError('Frozen diffusion mechanism fingerprint differs')
        mech=load_mechanism(self.metadata['mechanism'])
        self.thermo=NativeThermo(mech)
        self.max_temperature=2.*mech.max_temperature
        self.mesh=SimplexMesh(self.table['controls'],self.table['cells'])

    def lookup_batch(self, *, Z,C):
        Z,C=np.broadcast_arrays(np.asarray(Z,dtype=float),np.asarray(C,dtype=float))
        shape=Z.shape
        controls=np.column_stack([Z.ravel(),C.ravel()])
        nodes,weights,covered=self.mesh.locate_batch(controls)
        if not covered.all():
            raise OutsideManifoldError('Requested diffusion state is outside adjacent training flamelets')
        result={name:np.sum(weights[:,:,None]*self.table[name][nodes],axis=1).T
                for name in ('Y','omega_Y')}
        result.update({name:np.sum(weights*self.table[name][nodes],axis=1) for name in ('h','T','omega_C','qdot')})
        result['T']=NonAdiabaticFGM._recover_temperature(self,result['h'],result['Y'],result['T'])
        result.update(Z=Z.ravel(),C=C.ravel())
        return {name:values.reshape((values.shape[0],)+shape) if name in ('Y','omega_Y') else values.reshape(shape)
                for name,values in result.items()}

    def lookup(self, *, Z,C):
        return self.lookup_batch(Z=Z,C=C)


def validate_diffusion_fgm(table,native,output, *, limits=None):
    """Audit a withheld native strain, explicitly reporting uncovered states.

    Cold degenerate cells are excluded from the chart. Coverage of reaction
    points (qdot > 1% of its positive peak) must pass separately from accuracy.
    Sources come from interpolation, never from recomputed detailed chemistry.
    """
    fgm=DiffusionFGM(table)
    native=Path(native)
    meta=json.loads((native/'metadata.json').read_text(encoding='utf-8'))
    if not meta.get('accepted') or meta.get('flow_type')!='counterflow_diffusion':
        raise ValueError('An accepted withheld native diffusion flame is required')
    for key in ('mechanism_sha256','transport','soret','pressure','fuel_temperature','oxidizer_temperature','width','fuel_Y','oxidizer_Y'):
        if meta.get(key)!=fgm.metadata.get(key):
            raise ValueError(f'Withheld flame differs in {key}')
    with np.load(native/'flame.npz',allow_pickle=False) as saved:
        profile={key:saved[key].copy() for key in saved.files}
    Z=profile['Z'];C=fgm.table['progress_weights']@profile['Y']
    _,_,covered=fgm.mesh.locate_batch(np.column_stack([Z,C]))
    active=profile['qdot']>.01*max(float(profile['qdot'].max()),1.)
    if not covered.any() or not active.any():
        raise ValueError('Withheld flame has no covered states or no resolved reaction zone')
    predicted=fgm.lookup_batch(Z=Z[covered],C=C[covered])
    source_scale=max(float(abs(profile['omega_Y']).max()),1e-12)
    metrics=dict(temperature_K=float(abs(predicted['T']-profile['T'][covered]).max()),
        species_absolute=float(abs(predicted['Y']-profile['Y'][:,covered]).max()),
        species_source_relative=float(abs(predicted['omega_Y']-profile['omega_Y'][:,covered]).max()/source_scale),
        heat_release_relative=float(abs(predicted['qdot']-profile['qdot'][covered]).max()/max(float(abs(profile['qdot']).max()),1.)),
        enthalpy_J_kg=float(abs(predicted['h']-profile['h'][covered]).max()),
        enthalpy_temperature_equivalent_K=float((abs(predicted['h']-profile['h'][covered])/
            fgm.thermo.cp_mass(profile['T'][covered],profile['Y'][:,covered])).max()),
        species_sum_error=float(abs(predicted['Y'].sum(axis=0)-1.).max()),
        thermodynamic_enthalpy_error_J_kg=float(abs(fgm.thermo.enthalpy_mass(predicted['T'],predicted['Y'])-predicted['h']).max()))
    limits=dict(temperature_K=20.,species_absolute=.005,species_source_relative=.1,
                heat_release_relative=.1,enthalpy_temperature_equivalent_K=20.) if limits is None else dict(limits)
    reaction_coverage=float(covered[active].mean())
    checks={key:metrics[key]<=value for key,value in limits.items()}
    checks['reaction_coverage']=reaction_coverage>=.99
    checks['thermodynamic_consistency']=metrics['thermodynamic_enthalpy_error_J_kg']<1e-4 and metrics['species_sum_error']<1e-7
    report=dict(passed=all(checks.values()),metrics=metrics,limits=limits,checks=checks,
                covered_points=int(covered.sum()),total_points=len(Z),reaction_coverage=reaction_coverage,
                native_metadata=meta,table_sha256=fgm.metadata['table_sha256'],
                scope='Withheld native strain; uncovered states reported explicitly, errors measured only on covered states.')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'comparison.npz',native_Z=Z,native_C=C,covered=covered,active=active,**predicted)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism',default='gri30.yaml');parser.add_argument('--fuel',default='CH4')
    parser.add_argument('--oxidizer',default='O2:1,N2:3.76')
    parser.add_argument('--flow-scales',default='.75,1,1.25')
    parser.add_argument('--fuel-mass-flux',type=float,default=.12);parser.add_argument('--oxidizer-mass-flux',type=float,default=.24)
    for name,default in (('fuel-temperature',300.),('oxidizer-temperature',300.),('pressure',101325.),
                         ('width',.02),('slope',.01),('curve',.02),('max-time',600.)):
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--transport',choices=['mixture-averaged','multicomponent'],default='mixture-averaged')
    parser.add_argument('--soret',action='store_true')
    parser.add_argument('--progress-weights')
    parser.add_argument('--mixture-points',type=int,default=241)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--initial-solution',type=Path)
    args=vars(parser.parse_args(argv));args['flow_scales']=[float(v) for v in args['flow_scales'].split(',')]
    result=generate_diffusion_fgm(**args);print(result['metadata']);print(result['output'])
