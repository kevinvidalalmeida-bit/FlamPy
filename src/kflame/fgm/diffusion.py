"""Connected nonpremixed manifolds in physical local (Z,C) controls.

Unlike a premixed family, each diffusion flame spans the mixture coordinate;
the second training direction is inlet-flow scaling (strain), not inlet phi.
No flame is sorted on reaction progress, which rises then falls along z.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog

from kflame.api import _output
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.diffusion import solve_diffusion_flame
from kflame.fgm.nonadiabatic3d import SimplexMesh,NonAdiabaticFGM,OutsideManifoldError


FOLD_TOLERANCE=1e-6


def _progress_weights(Y,provided,names,time_limit=None):
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
    answer=linprog(objective,A_ub=A,b_ub=b,bounds=[(0.,None)]*(2*ns+1),method='highs',
                   options={} if time_limit is None else {'time_limit':float(time_limit)})
    if not answer.success or answer.x[-1]<=1e-8:
        raise ValueError('No unique linear progress parametrizes this strain family; inspect its branches or add physical controls')
    return answer.x[:ns]-answer.x[ns:2*ns]


def _adaptive_mixture_grid(columns,lower,upper,mixture_points,tolerance,max_points=20000):
    """Smallest knot subset reproducing every flamelet within ``tolerance``.

    Starts from ``mixture_points`` uniform samples and repeatedly inserts, in
    each interval whose piecewise-linear reconstruction misses a flamelet
    state by more than ``tolerance`` (relative to that field's family range),
    the physical knot of largest error. Controlled fields are T, h, qdot and
    every Y_k and omega_k, so the Z sampling error stays explicitly bounded.
    Grid values are interpolated from the full profile, exactly as the table
    stores them; the error is measured on the knots inside the shared domain.
    Returns the grid and the largest relative error reached; a budget too
    small for the tolerance is an error, never a silent coarser table.
    """
    data=[]
    for Z,order,profile in columns:
        rows=np.vstack([profile['T'][order],profile['h'][order],profile['qdot'][order],
                        profile['Y'][:,order],profile['omega_Y'][:,order]])
        inside=(Z>=lower)&(Z<=upper)
        data.append((Z,rows,inside))
    scale=np.max([np.ptp(rows[:,inside],axis=1) for _,rows,inside in data],axis=0)
    scale=np.where(scale>0.,scale,1.)
    grid=np.unique(np.r_[np.linspace(lower,upper,mixture_points),lower,upper])
    while True:
        worst=np.zeros(len(grid)-1);where=np.full(len(grid)-1,np.nan)
        for Z,rows,inside in data:
            sampled=np.array([np.interp(grid,Z,row) for row in rows])
            knots,values=Z[inside],rows[:,inside]
            rebuilt=np.array([np.interp(knots,grid,row) for row in sampled])
            error=np.max(abs(rebuilt-values)/scale[:,None],axis=0)
            interval=np.clip(np.searchsorted(grid,knots,side='right')-1,0,len(grid)-2)
            order=np.lexsort((-error,interval))
            first=order[np.r_[True,np.diff(interval[order])!=0]]
            better=error[first]>worst[interval[first]]
            worst[interval[first][better]]=error[first][better]
            where[interval[first][better]]=knots[first][better]
        achieved=float(worst.max(initial=0.))
        insert=where[(worst>tolerance)&np.isfinite(where)]
        insert=insert[~np.isin(insert,grid)]
        if not len(insert):
            return grid,achieved
        if len(grid)+len(insert)>max_points:
            raise ValueError(f'Mixture sampling needs more than {max_points} points to reach '
                             f'tolerance {tolerance:g} (reached {achieved:.3g})')
        grid=np.unique(np.r_[grid,insert])


def _mixture_columns(profiles,mixture_points,tolerance=None):
    """Admissible monotone Z trajectories and their shared sampling grid.

    Without ``tolerance`` the grid is the union of all physical knots;
    otherwise it is the adaptive subset of ``_adaptive_mixture_grid``.
    """
    columns=[];excluded_spatial=[];dropped=[]
    for profile in profiles:
        Z=np.asarray(profile['Z'])
        # Differential diffusion can produce local Bilger Z outside the two
        # feed values. The mixing chart explicitly covers [0,1]; do not clip
        # those states onto the chart, or silently sort a folded trajectory.
        admissible=np.flatnonzero((Z>=0.)&(Z<=1.))
        excluded_spatial.append(int(len(Z)-len(admissible)))
        if len(admissible)<3:
            raise ValueError('A flamelet must span a resolved mixture fraction range')
        # A genuine fold is rejected: Z drops below its running maximum by
        # more than FOLD_TOLERANCE, in one step or accumulated over several.
        # Smaller reversals, e.g. 1e-8 in the cold tails with Soret and
        # multicomponent diffusion, are rounding-level: the monotone envelope
        # is kept and the dropped states are counted.
        order=admissible[::-1]
        values=Z[order]
        envelope=np.maximum.accumulate(values)
        if np.max(envelope-values)>FOLD_TOLERANCE:
            raise ValueError('Local Bilger Z folds along a flamelet; a single Z trajectory cannot be tabulated')
        keep=np.r_[True,values[1:]>envelope[:-1]+1e-12]
        dropped.append(int((~keep).sum()))
        order=order[keep]
        if len(order)<3:
            raise ValueError('A flamelet must span a resolved mixture fraction range')
        columns.append((Z[order],order,profile))
    lower=max(row[0][0] for row in columns);upper=min(row[0][-1] for row in columns)
    if upper<=lower:
        raise ValueError('Flamelets have no shared mixture-fraction domain')
    # Union of physical knots resolves steep reaction fronts without hundreds
    # of uniform samples in chemically inactive tails.
    sampling_error=None
    if tolerance is None:
        Zgrid=np.unique(np.r_[np.linspace(lower,upper,mixture_points),
                             *[Z[(Z>lower)&(Z<upper)] for Z,_,_ in columns]])
    else:
        Zgrid,sampling_error=_adaptive_mixture_grid(columns,lower,upper,mixture_points,tolerance)
    field_rows={name:[] for name in ('Y','h','T','omega_Y','qdot')}
    for Z,order,profile in columns:
        for name in field_rows:
            values=profile[name]
            field_rows[name].append(np.array([np.interp(Zgrid,Z,row[order]) for row in values]).T
                if name in ('Y','omega_Y') else np.interp(Zgrid,Z,values[order]))
    fields={name:np.asarray(rows) for name,rows in field_rows.items()}
    return (Zgrid,fields,dict(outside=excluded_spatial,nonmonotone=dropped,sampling_error=sampling_error),
            float(lower),float(upper))


def build_diffusion_manifold(profiles,metadata,output, *, mixture_points=241,progress_weights=None):
    """Tabulate accepted native flamelets preserving their physical adjacency."""
    if len(profiles)<2 or not isinstance(mixture_points,int) or mixture_points<3:
        raise ValueError('Require at least two flamelets and >=3 mixture samples')
    Zgrid,fields,excluded,lower,upper=_mixture_columns(profiles,mixture_points)
    excluded_spatial=excluded['outside']
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
                            excluded_folded_cells=0,excluded_spatial_points_outside_feed_Z=excluded_spatial,
                            dropped_rounding_reversals_of_Z=excluded['nonmonotone']),
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


def _heat_loss_progress_weights(Y,h):
    """Linear progress making (strain, loss) -> (C, h) injective at each Z.

    Y has shape (strains, losses, Z samples, species). Two linear families of
    constraints share one margin: C increases with strain at every loss and
    Z, and the 2x2 determinant of (C, h) against (strain, loss) keeps one
    sign in every cell, so no cell folds. Either determinant sign may be the
    attainable one; both linear programs are solved and the larger margin
    kept. Signed coefficients are allowed.
    """
    ns=Y.shape[-1]
    strain=np.diff(Y,axis=0).reshape(-1,ns)
    di,dj=np.diff(Y,axis=0)[:,:-1],np.diff(Y,axis=1)[:-1]
    hi,hj=np.diff(h,axis=0)[:,:-1],np.diff(h,axis=1)[:-1]
    # det = (di.w) hj - (dj.w) hi is linear in the progress weights w.
    determinant=(di*hj[...,None]-dj*hi[...,None]).reshape(-1,ns)
    def normalized(vectors):
        norms=np.sum(abs(vectors),axis=1)
        keep=norms>1e-12*max(float(norms.max()),1e-300)
        return vectors[keep]/norms[keep,None]
    best=None
    for sign in (1.,-1.):
        directions=np.vstack([normalized(strain),normalized(sign*determinant)])
        A=np.vstack([np.column_stack([-directions,directions,np.ones(len(directions))]),np.r_[np.ones(2*ns),0.]])
        # Degenerate (zero-margin) programs can stall the simplex; a bounded
        # solve that does not finish is simply not a candidate.
        answer=linprog(np.r_[np.zeros(2*ns),-1.],A_ub=A,b_ub=np.r_[np.zeros(len(directions)),1.],
                       bounds=[(0.,None)]*(2*ns+1),method='highs',options={'time_limit':10.})
        if answer.success and answer.x[-1]>1e-10 and (best is None or answer.x[-1]>best.x[-1]):
            best=answer
    if best is None:
        raise ValueError('No linear progress parametrizes this strain/loss family without folds; '
                         'provide progress_weights or change the loss levels')
    return best.x[:ns]-best.x[ns:2*ns]


def build_diffusion_heat_loss_manifold(profiles,metadata,output, *, mixture_points=241,progress_weights=None,
                                       mixture_tolerance=1e-3):
    """Tabulate a strain x heat-loss family of flamelets in physical (Z,C,h).

    ``profiles[i][j]`` is the flamelet of strain row i (increasing flow scale)
    and loss level j (increasing radiation scale, j=0 adiabatic). Each one is
    sampled on one shared Z grid; structured Freudenthal tetrahedra connect
    adjacent strains, loss levels and Z samples. Folded or degenerate cells
    are excluded and counted; queries outside the cells are never filled.
    ``mixture_tolerance`` bounds the Z sampling error of every flamelet
    (relative to each field's family range); None keeps all physical knots.
    """
    if (len(profiles)<2 or len({len(row) for row in profiles})!=1 or len(profiles[0])<2
            or not isinstance(mixture_points,int) or mixture_points<3):
        raise ValueError('Require a complete grid of >=2 strains x >=2 loss levels and >=3 mixture samples')
    from kflame.fgm.nonadiabatic3d import _connected_cells
    n_strain,n_loss=len(profiles),len(profiles[0])
    flat=[profile for row in profiles for profile in row]
    Zgrid,fields,excluded,lower,upper=_mixture_columns(flat,mixture_points,mixture_tolerance)
    nz=len(Zgrid)
    fields={name:value.reshape((n_strain,n_loss)+value.shape[1:]) for name,value in fields.items()}
    names=list(flat[0]['species_names'])
    sample=np.unique(np.searchsorted(Zgrid,np.linspace(lower,upper,mixture_points)).clip(0,nz-1))
    Zfield=np.broadcast_to(Zgrid,(n_strain,n_loss,nz))

    def chart(weights):
        C=fields['Y']@weights
        controls=np.column_stack([Zfield.ravel(),C.ravel(),fields['h'].ravel()])
        # The cell orientation is the sign of Z_k (C_i h_j - C_j h_i) over the
        # index axes (strain i, loss j, Z k). Either term can dominate, so both
        # signs are tried and the consistent majority is kept; cells of the
        # other sign are folds. Only the orientation test sees the flipped h.
        options=[_connected_cells((n_strain,n_loss,nz),controls*np.array([1.,1.,sign])) for sign in (1.,-1.)]
        return (C,controls)+max(options,key=lambda item:len(item[0]))

    if progress_weights is not None:
        candidates=[('provided',_progress_weights(fields['Y'],progress_weights,names))]
    else:
        # Generic candidates, judged by the connected chart they produce:
        # sums of oxidation products present in the mechanism and absent
        # from the fuel stream, then the strain-ordering linear programs.
        fuel=np.asarray(metadata.get('fuel_Y',np.zeros(len(names))),dtype=float)
        candidates=[]
        for mapping in ({'CO2':1.,'H2O':1.},{'CO2':1.,'CO':1.,'H2O':1.,'H2':.5}):
            weights=np.array([mapping.get(name,0.) if fuel[k]<=0. else 0. for k,name in enumerate(names)])
            if weights.any() and not any(np.array_equal(weights,w) for _,w in candidates):
                candidates.append(('products:'+'+'.join(n for n,v in zip(names,weights) if v),weights))
        for label,builder in (('injective_linear_program',
                               lambda:_heat_loss_progress_weights(fields['Y'][:,:,sample],fields['h'][:,:,sample])),
                              ('strain_linear_program',lambda:_progress_weights(fields['Y'],None,names,time_limit=10.))):
            try:
                candidates.append((label,builder()))
            except ValueError:
                pass
        if not candidates:
            raise ValueError('No progress definition is available; provide progress_weights')
    scored=[]
    for label,weights in candidates:
        result=chart(weights)
        scored.append((result[4]['excluded_folded_cells'],-len(result[2]),len(scored),label,weights,result))
    _,_,_,progress_rule,weights,(C,controls,cells,triangles,statistics)=min(scored,key=lambda item:item[:3])
    selection=[dict(rule=label,folded_cells=int(folded),tetrahedra=int(-valid))
               for folded,valid,_,label,_,_ in scored]
    if not len(cells):
        raise ValueError('Degenerate diffusion heat-loss manifold')
    mesh=SimplexMesh(controls,cells)
    _,_,covered=mesh.locate_batch(controls[cells].mean(axis=1))
    if not covered.all():
        raise ValueError('Diffusion heat-loss manifold has uncovered cell interiors')
    table={name:value.reshape((-1,)+value.shape[3:]) for name,value in fields.items()}
    table.update(controls=controls,cells=cells,Z_grid=Zgrid,progress_weights=weights,species_names=np.array(names),
                 structured_shape=np.array([n_strain,n_loss,nz]),
                 reference_points=np.column_stack([Zfield[:,0].ravel(),C[:,0].ravel()]),
                 reference_cells=np.asarray(triangles,dtype=int),reference_h=fields['h'][:,0].ravel())
    table['omega_C']=table['omega_Y']@weights
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'diffusion_heat_loss_fgm.npz'
    np.savez_compressed(target,**table)
    adiabatic=fields['h'][:,0]
    deficit=adiabatic[:,None,:]-fields['h']
    meta=dict(metadata,mechanism_sha256=hashlib.sha256(Path(metadata['mechanism']).read_bytes()).hexdigest(),
              flow_type='counterflow_diffusion',controls=['Z','C','h'],
              table_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              table_builder_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              species_names=names,progress_weights={name:float(weight) for name,weight in zip(names,weights) if weight!=0.},
              geometry=dict(statistics,excluded_spatial_points_outside_feed_Z=excluded['outside'],
                            dropped_rounding_reversals_of_Z=excluded['nonmonotone']),
              structured_shape=[n_strain,n_loss,nz],mixture_fraction_range=[lower,upper],
              mixture_sampling=dict(points=int(nz),relative_tolerance=mixture_tolerance,
                                    achieved_relative_error=excluded['sampling_error'],
                                    rule='adaptive physical-knot subset' if mixture_tolerance is not None
                                         else 'union of physical knots'),
              maximum_enthalpy_deficit_J_kg=float(deficit.max()),
              maximum_enthalpy_deficit_definition=('h(adiabatic flamelet of the same strain, Z) - h(flamelet, Z) '
                                                   'on the table Z grid; the lookup delta_h is h_adiabatic(Z,C) - h, '
                                                   'which differs where radiation also changes the composition at fixed Z'),
              progress_rule=progress_rule,progress_candidates=selection,
              h_definition='total_sensible_plus_formation_J_kg',
              note='Native counterflow flamelets connected over strain, radiative loss level and Z. '
                   'delta_h = h_adiabatic(Z,C) - h. No extrapolation outside the connected cells.')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta


def _heat_loss_row(task):
    """Radiative flamelets of one strain, each continuing from the previous one."""
    settings,fuel_flux,oxidizer_flux,losses,seed,folder,i=task
    row=[]
    for j,loss in losses:
        flame=solve_diffusion_flame(**settings,fuel_mass_flux=fuel_flux,oxidizer_mass_flux=oxidizer_flux,
            radiation=True,radiation_scale=float(loss),initial_solution=seed,
            output=folder/f'flamelet_{i:03d}_{j:03d}')
        row.append(flame);seed=flame['output']
    return row


def generate_diffusion_heat_loss_fgm(*,flow_scales=(.75,1.,1.25),radiation_scales=(0.,2.,4.,6.,8.),
                                     mixture_points=241,mixture_tolerance=1e-3,progress_weights=None,
                                     initial_solution=None,output=None,jobs=1,**flame_settings):
    """Generate a strain x radiative-loss flamelet family and its (Z,C,h) FGM.

    Loss level 0 must be adiabatic (radiation scale 0). Positive levels use
    the optically thin H2O/CO2 radiation multiplied by ``radiation_scales``;
    scales above one deepen the enthalpy defect for tabulation. Each flamelet
    continues from its neighbour (lower loss, or adiabatic lower strain). An
    extinguished or rejected flamelet aborts the family instead of being hidden.

    ``jobs`` > 1 solves the loss rows of different strains in separate
    processes after the adiabatic strain chain; every flamelet and its seed
    are the same as with ``jobs=1``.
    """
    if not isinstance(jobs,int) or isinstance(jobs,bool) or jobs<1:
        raise ValueError('jobs must be a positive integer')
    scales=np.asarray(flow_scales,dtype=float);losses=np.asarray(radiation_scales,dtype=float)
    if scales.ndim!=1 or len(scales)<2 or not np.isfinite(scales).all() or np.any(scales<=0.) or np.any(np.diff(scales)<=0.):
        raise ValueError('flow_scales must be finite, positive and strictly increasing')
    if (losses.ndim!=1 or len(losses)<2 or not np.isfinite(losses).all() or losses[0]!=0.
            or np.any(np.diff(losses)<=0.)):
        raise ValueError('radiation_scales must start at 0 (adiabatic) and increase strictly')
    forbidden=set(flame_settings)&{'output','initial_solution','grid','refine','radiation','radiation_scale'}
    if forbidden:
        raise ValueError(f'Family controls these flame settings: {sorted(forbidden)}')
    folder=_output(output,'diffusion_heat_loss_fgm')
    settings=dict(flame_settings)
    fuel_flux=settings.pop('fuel_mass_flux',.12);oxidizer_flux=settings.pop('oxidizer_mass_flux',.24)
    previous=initial_solution;started=time.perf_counter()
    # Adiabatic strain chain first: each loss row continues from its own one.
    adiabatic=[]
    for i,scale in enumerate(scales):
        flame=solve_diffusion_flame(**settings,fuel_mass_flux=float(fuel_flux*scale),
            oxidizer_mass_flux=float(oxidizer_flux*scale),radiation=False,radiation_scale=1.,
            initial_solution=previous,output=folder/f'flamelet_{i:03d}_000')
        adiabatic.append(flame);previous=flame['output']
    tasks=[(settings,float(fuel_flux*scale),float(oxidizer_flux*scale),list(enumerate(losses))[1:],
            adiabatic[i]['output'],folder,i) for i,scale in enumerate(scales)]
    if jobs>1 and len(tasks)>1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(jobs,len(tasks))) as pool:
            rows=list(pool.map(_heat_loss_row,tasks))
    else:
        rows=[_heat_loss_row(task) for task in tasks]
    grid=[[adiabatic[i],*row] for i,row in enumerate(rows)]
    meta=json.loads((grid[0][0]['output']/'metadata.json').read_text(encoding='utf-8'))
    for key in ('radiation','radiation_scale','fuel_mass_flux','oxidizer_mass_flux','report','runtime_s'):
        meta.pop(key,None)
    meta.update(flow_scales=scales.tolist(),radiation_scales=losses.tolist(),base_fuel_mass_flux=fuel_flux,
                base_oxidizer_mass_flux=oxidizer_flux,all_final_accepted=True,flamelet_count=len(scales)*len(losses),
                generation_runtime_s=time.perf_counter()-started,
                flamelet_runtimes_s=[[float(flame['report']['runtime_s']) for flame in row] for row in grid])
    result=build_diffusion_heat_loss_manifold(grid,meta,folder,mixture_points=mixture_points,
                                              progress_weights=progress_weights,
                                              mixture_tolerance=mixture_tolerance)
    return dict(output=folder,metadata=result,flamelets=grid)


class DiffusionHeatLossFGM:
    """Batched (Z,C,h) lookup on connected strain/loss cells; no chemistry calls."""
    def __init__(self,folder):
        folder=Path(folder)
        self.metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        path=folder/'diffusion_heat_loss_fgm.npz'
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
        self._reference=None

    @property
    def reference_mesh(self):
        if self._reference is None:
            self._reference=SimplexMesh(self.table['reference_points'],self.table['reference_cells'])
        return self._reference

    def lookup_batch(self,*,Z,C,h,outside='raise'):
        """Y and omega_Y have shape (species, states); T comes from total h and Y.

        outside='mask' returns NaN and covered=False for unsupported states.
        """
        if outside not in ('raise','mask'):
            raise ValueError("outside must be 'raise' or 'mask'")
        Z,C,h=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (Z,C,h)])
        shape=Z.shape
        controls=np.column_stack([Z.ravel(),C.ravel(),h.ravel()])
        nodes,weights,covered=self.mesh.locate_batch(controls)
        if outside=='raise' and not covered.all():
            raise OutsideManifoldError('Requested (Z,C,h) lies outside adjacent training flamelets')
        n,ns=len(controls),self.thermo.n_sp
        result=dict(Y=np.full((ns,n),np.nan),omega_Y=np.full((ns,n),np.nan),
                    **{name:np.full(n,np.nan) for name in ('T','omega_C','qdot','delta_h')})
        if covered.any():
            chosen,w=nodes[covered],weights[covered]
            for name in ('Y','omega_Y'):
                result[name][:,covered]=np.einsum('ni,nij->jn',w,self.table[name][chosen])
            for name in ('omega_C','qdot'):
                result[name][covered]=np.einsum('ni,ni->n',w,self.table[name][chosen])
            guess=np.einsum('ni,ni->n',w,self.table['T'][chosen])
            result['T'][covered]=NonAdiabaticFGM._recover_temperature(self,controls[covered,2],
                                                                      result['Y'][:,covered],guess)
            ref_nodes,ref_weights,ref_covered=self.reference_mesh.locate_batch(controls[:,:2])
            usable=ref_covered&covered
            result['delta_h'][usable]=(np.einsum('ni,ni->n',ref_weights[usable],
                                                 self.table['reference_h'][ref_nodes[usable]])-controls[usable,2])
        result.update(Z=controls[:,0],C=controls[:,1],h=controls[:,2],covered=covered)
        return {name:value.reshape((value.shape[0],)+shape) if name in ('Y','omega_Y') else value.reshape(shape)
                for name,value in result.items()}

    def lookup(self,*,Z,C,h):
        return self.lookup_batch(Z=Z,C=C,h=h)


def validate_diffusion_heat_loss_fgm(table,native,output,*,limits=None):
    """Audit a withheld (strain, loss) native flamelet in (Z,C,h).

    Errors are measured on covered states only; coverage of the reaction zone
    (qdot > 1% of its peak) is a separate check. Nothing is extrapolated.
    """
    fgm=DiffusionHeatLossFGM(table)
    native=Path(native)
    meta=json.loads((native/'metadata.json').read_text(encoding='utf-8'))
    if not meta.get('accepted') or meta.get('flow_type')!='counterflow_diffusion':
        raise ValueError('An accepted withheld native diffusion flame is required')
    for key in ('mechanism_sha256','transport','soret','pressure','fuel_temperature','oxidizer_temperature','width','fuel_Y','oxidizer_Y'):
        if meta.get(key)!=fgm.metadata.get(key):
            raise ValueError(f'Withheld flame differs in {key}')
    with np.load(native/'flame.npz',allow_pickle=False) as saved:
        profile={key:saved[key].copy() for key in saved.files}
    Z,h=profile['Z'],profile['h'];C=fgm.table['progress_weights']@profile['Y']
    predicted=fgm.lookup_batch(Z=Z,C=C,h=h,outside='mask')
    covered=predicted['covered']
    active=profile['qdot']>.01*max(float(profile['qdot'].max()),1.)
    if not covered.any() or not active.any():
        raise ValueError('Withheld flame has no covered states or no resolved reaction zone')
    source_scale=max(float(abs(profile['omega_Y']).max()),1e-12)
    Yp=predicted['Y'][:,covered]
    metrics=dict(temperature_K=float(abs(predicted['T'][covered]-profile['T'][covered]).max()),
        species_absolute=float(abs(Yp-profile['Y'][:,covered]).max()),
        species_source_relative=float(abs(predicted['omega_Y'][:,covered]-profile['omega_Y'][:,covered]).max()/source_scale),
        heat_release_relative=float(abs(predicted['qdot'][covered]-profile['qdot'][covered]).max()/max(float(abs(profile['qdot']).max()),1.)),
        species_sum_error=float(abs(Yp.sum(axis=0)-1.).max()),
        thermodynamic_enthalpy_error_J_kg=float(abs(fgm.thermo.enthalpy_mass(predicted['T'][covered],Yp)-h[covered]).max()),
        maximum_enthalpy_deficit_J_kg=float(np.nanmax(predicted['delta_h'])) if np.isfinite(predicted['delta_h']).any() else None)
    limits=dict(temperature_K=20.,species_absolute=.005,species_source_relative=.1,heat_release_relative=.1) if limits is None else dict(limits)
    reaction_coverage=float(covered[active].mean())
    checks={key:metrics[key]<=value for key,value in limits.items()}
    checks['reaction_coverage']=reaction_coverage>=.99
    checks['thermodynamic_consistency']=metrics['thermodynamic_enthalpy_error_J_kg']<1e-4 and metrics['species_sum_error']<1e-7
    report=dict(passed=all(checks.values()),metrics=metrics,limits=limits,checks=checks,
                covered_points=int(covered.sum()),total_points=len(Z),reaction_coverage=reaction_coverage,
                native_metadata=meta,table_sha256=fgm.metadata['table_sha256'],
                scope='Withheld native strain/loss flamelet; uncovered states reported, errors on covered states only.')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'comparison.npz',native_Z=Z,native_C=C,native_h=h,active=active,
                        **{key:value for key,value in predicted.items() if key not in ('Z','C','h')})
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


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
    parser.add_argument('--radiation-scales',help='Comma list starting at 0: build the (Z,C,h) heat-loss table')
    parser.add_argument('--mixture-tolerance',type=float,default=1e-3,help='Z sampling tolerance of the (Z,C,h) table')
    parser.add_argument('--jobs',type=int,default=1,help='(Z,C,h) table: processes for the loss rows of different strains')
    args=vars(parser.parse_args(argv));args['flow_scales']=[float(v) for v in args['flow_scales'].split(',')]
    radiation=args.pop('radiation_scales');tolerance=args.pop('mixture_tolerance');jobs=args.pop('jobs')
    if radiation is None:
        result=generate_diffusion_fgm(**args)
    else:
        result=generate_diffusion_heat_loss_fgm(**args,radiation_scales=[float(v) for v in radiation.split(',')],
                                                mixture_tolerance=tolerance,jobs=jobs)
    print(result['metadata']);print(result['output'])
