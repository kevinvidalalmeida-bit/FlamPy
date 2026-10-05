"""Build a budgeted Cartesian library with parallel composition families.

The selection is an explicit development decision. Keep confirmation cases
separate: a changed selection requires new independent physical validation.
Each process owns its native solves, transport caches and continuation chain.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from kflame import solve_flame, solve_burner_flame
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.thermo import NativeThermo
from kflame.chemistry.transport import NativeTransport
from kflame.flame.burner_grid import refine_burner_boundary_grid
import numpy as np
from kflame.fgm.nonadiabatic3d import build_nonadiabatic_table
from examples.validate_reduced_burner import write_json


def family(output,phis,selection,reuse_from):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'generation.json').exists():
        saved=json.loads((output/'generation.json').read_text(encoding='utf-8'))
        if saved['all_final_accepted'] and saved['phis']==phis and saved['mass_flux_fractions']==selection['mass_flux_fractions']:
            return output
    settings=selection['flame_settings'];mech=load_mechanism(settings['mechanism']);thermo=NativeThermo(mech);transport=NativeTransport(mech)
    existing=json.loads((Path(reuse_from)/'generation.json').read_text(encoding='utf-8')) if reuse_from else dict(rows=[])
    rows=[];reused=0;attempts=[]
    def obtain(i,j,phi,fraction,previous=None,mdot=None):
        nonlocal reused
        name=f'phi_{i:03d}_'+('adiabatic' if j==0 else f'loss_{j:03d}')
        path=output/name
        if j and previous is not None and (path/'metadata.json').exists():
            cached_meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
            if not cached_meta['accepted']:
                name+='_continued';path=output/name
        matches=[r for r in existing['rows'] if r['phi']==phi and r['loss_index']==0] if j==0 else [r for r in existing['rows'] if r['phi']==phi and r.get('fraction')==fraction]
        if matches and not path.exists():
            source=Path(reuse_from)/matches[0]['output'];meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
            feed=fresh_mixture(mech,phi,settings['fuel'],settings['oxidizer'])
            if (meta['accepted'] and meta['backend']=='native_cpu' and meta['report']['grid_converged']
                    and meta['temperature']==settings['temperature'] and meta['pressure']==settings['pressure']
                    and meta['transport']==settings['transport'] and meta['soret']==settings['soret']
                    and np.allclose(meta['inlet_Y'],feed,rtol=0.,atol=1e-14)
                    and (j==0 or abs(meta['mass_flux']-mdot)<1e-12)):
                shutil.copytree(source,path);reused+=1
        for level,slope in enumerate([settings.get('slope',.04),.04,.02,.01,.005,.0025]):
            target=path if level==0 else output/(name+('_boundary_pe' if level==1 else f'_boundary_slope_{slope:g}'))
            stem=target.name;restart=0
            while target.exists() and not (target/'metadata.json').exists():
                restart+=1;target=output/(stem+f'_restart_{restart}')
            if not (target/'metadata.json').exists():
                options=dict(settings)
                if level:options.update(slope=slope,curve=2*slope,prune=.001,max_points=6000)
                if level==1 and j and previous is not None:
                    with np.load(previous/'flame.npz',allow_pickle=False) as saved:
                        options['grid']=refine_burner_boundary_grid(saved['z'],saved['T'],saved['Y'],mdot,settings['pressure'],thermo,transport)
                try:
                    if j:solve_burner_flame(phi=phi,mass_flux=mdot,output=target,initial_solution=previous,**options)
                    else:solve_flame(phi=phi,output=target,**options)
                except RuntimeError:
                    if not (target/'metadata.json').exists():raise
            meta=json.loads((target/'metadata.json').read_text(encoding='utf-8'))
            energy=(meta.get('heat_loss') or {}).get('relative_energy_closure_error',0.)
            attempts.append(dict(phi=phi,fraction=fraction,level=level,energy_error=energy,accepted=meta['accepted']))
            write_json(output/'training_attempts.json',attempts)
            if meta['accepted'] and meta['report']['grid_converged'] and energy<=selection.get('max_energy_error',.02):
                return target,meta
            if meta['accepted']:previous=target
        raise RuntimeError(f'Physical training acceptance failed at phi={phi}, fraction={fraction}')
    for i,phi in enumerate(phis):
        ad,meta=obtain(i,0,phi,None)
        feed=np.array(meta['inlet_Y']);mdot=thermo.density(settings['temperature'],settings['pressure'],feed)*meta['Su']
        rows.append(dict(composition_index=i,loss_index=0,phi=phi,kind='adiabatic_reference',output=ad.name,inlet_Y=feed.tolist(),adiabatic_mass_flux=float(mdot),Su=meta['Su']))
        previous=None;strong=None
        anchor=selection.get('continuation_anchor_fraction',.65)
        standard=sorted([f for f in selection['mass_flux_fractions'] if f<=anchor],reverse=True)
        weak=sorted([f for f in selection['mass_flux_fractions'] if f>anchor])
        for fraction in standard+weak:
            j=selection['mass_flux_fractions'].index(fraction)+1
            if weak and fraction==weak[0]:previous=strong
            print(f'LIBRARY phi={phi:g} r={fraction:g}',flush=True)
            path,meta=obtain(i,j,phi,fraction,previous,mdot*fraction);previous=path
            if standard and fraction==standard[0]:strong=path
            rows.append(dict(composition_index=i,loss_index=j,phi=phi,kind='isothermal_burner',output=path.name,inlet_Y=feed.tolist(),fraction=fraction,mass_flux=float(mdot*fraction),adiabatic_mass_flux=float(mdot),inlet_velocity=meta['inlet_velocity'],heat_loss=meta['heat_loss']))
            write_json(output/'training_attempts.json',attempts)
    generation=dict(mechanism=settings['mechanism'],fuel=settings['fuel'],oxidizer=settings['oxidizer'],temperature_K=settings['temperature'],pressure_Pa=settings['pressure'],transport=settings['transport'],soret=settings['soret'],phis=phis,mass_flux_fractions=selection['mass_flux_fractions'],progress_species=selection['progress_species'],max_energy_error=selection.get('max_energy_error',.02),all_final_accepted=True,rows=rows,reused_profiles=reused,backend='native_cpu',flame_settings=settings)
    write_json(output/'generation.json',generation)
    return output


def run(selection,output,reuse_from=None,jobs=4):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    phis=selection['phis'];count=len(phis)*(len(selection['mass_flux_fractions'])+1)
    if count!=selection['flame_budget'] or len(phis)%2:
        raise ValueError('Selection must meet its budget and have an even number of compositions')
    plan=output/'selection.json'
    if plan.exists() and json.loads(plan.read_text(encoding='utf-8'))!=selection:
        raise ValueError('Different frozen selection in output')
    write_json(plan,selection)
    groups=[phis[i:i+2] for i in range(0,len(phis),2)]
    with ProcessPoolExecutor(max_workers=min(jobs,len(groups))) as pool:
        futures=[pool.submit(family,output/f'group_{i:03d}',group,selection,reuse_from) for i,group in enumerate(groups)]
        folders=[f.result() for f in futures]
    generation=None;rows=[];reused=0
    for group,folder in enumerate(folders):
        current=json.loads((folder/'generation.json').read_text(encoding='utf-8'))
        if generation is None:generation=dict(current)
        reused+=current['reused_profiles']
        for row in current['rows']:
            name=f'group_{group:03d}/{row["output"]}'
            rows.append(dict(row,composition_index=group*2+row['composition_index'],output=name))
    generation.update(phis=phis,rows=rows,reused_profiles=reused,selection=selection,
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output/'generation.json',generation)
    build_nonadiabatic_table(output,progress_points=181)
    if reuse_from:
        old_generation=json.loads((Path(reuse_from)/'generation.json').read_text(encoding='utf-8'))
        old_hashes={hashlib.sha256((Path(reuse_from)/r['output']/'flame.npz').read_bytes()).hexdigest() for r in old_generation['rows']}
        metadata=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
        reused=sum(digest in old_hashes for digest in metadata['training_profile_sha256'].values())
        generation['reused_profiles']=metadata['reused_profiles']=reused
        write_json(output/'generation.json',generation);write_json(output/'metadata.json',metadata)
    print(dict(training_flames=len(rows),reused_profiles=reused,new_profiles=len(rows)-reused),flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-from',type=Path)
    p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    run(json.loads(a.selection.read_text(encoding='utf-8')),a.output,a.reuse_from,a.jobs)
