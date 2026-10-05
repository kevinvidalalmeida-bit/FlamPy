"""Spatial grid and domain checks with a frozen reduced manifold.

Previous reduced iterates warm-start reduced solves. Detailed solutions are
independent comparisons and never supply the reduced controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from kflame import solve_burner_flame
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import solve_reduced_burner_fgm, ReducedConvergenceError
from examples.validate_reduced_burner import read_profile, write_json, flux_diagnostics
from examples.validate_nonadiabatic_3d import cantera_reference, reference_input_hash


def run(table, validation, profiles):
    model=NonAdiabaticFGM(table);root=Path(validation);profiles=Path(profiles)
    settings=json.loads((root/'plan.json').read_text(encoding='utf-8'))['settings']
    result=dict(table_sha256=model.table_sha256,grid=[],domain=[])
    for phi,r in [(.815,.575),(1.235,.575),(.985,.285)]:
        folder=root/f'case_{phi:.6f}_{r:.6f}'
        initial=read_profile(folder/'reduced.npz');report=json.loads((folder/'reduced_report.json').read_text(encoding='utf-8'))
        if not report['accepted']:raise ValueError('Convergence requires an accepted baseline reduced solve')
        label=f'φ={phi:.3f}, r={r:.3f}'
        for level in [1,2]:
            grid=np.sort(np.r_[initial['z'],.5*(initial['z'][:-1]+initial['z'][1:])])
            try:
                profile,rpt=solve_reduced_burner_fgm(model,phi=phi,mass_flux=report['mass_flux_kg_m2_s'],
                    seed_profile=profiles/report['seed_row']/'flame.npz',seed_row=report['seed_row'],
                    width=settings['width_m'],grid=grid,initial_solution=initial,max_iterations=80,**settings.get('solver_options',{}))
            except ReducedConvergenceError as e:profile,rpt=e.profile,e.report
            entry=dict(label=label,level=level,nodes=len(grid),accepted=rpt['accepted'],
                temperature_change_K=float(np.max(abs(profile['T']-np.interp(grid,initial['z'],initial['T'])))),
                wall_heat_flux_change_relative=float(abs(profile['conductive_heat_flux'][0]-initial['conductive_heat_flux'][0])/abs(initial['conductive_heat_flux'][0])))
            entry['passed']=(rpt['accepted'] and entry['temperature_change_K']<=settings['limits']['grid_temperature_change_K']
                              and entry['wall_heat_flux_change_relative']<=settings['limits']['grid_wall_heat_flux_change_relative'])
            result['grid'].append(entry);np.savez_compressed(folder/f'grid_{level}.npz',**profile)
            write_json(folder/f'grid_{level}.json',dict(solver=rpt,metrics=entry));initial=profile
            print('GRID',json.dumps(entry,ensure_ascii=True),flush=True)
    phi,r=.985,.285;folder=root/f'case_{phi:.6f}_{r:.6f}';baseline=read_profile(folder/'reduced.npz')
    report=json.loads((folder/'reduced_report.json').read_text(encoding='utf-8'))
    comp=read_profile(folder/'comparison.npz');native_meta=json.loads((folder/'native/metadata.json').read_text(encoding='utf-8'))
    native0=read_profile(folder/'native/flame.npz');reference0=read_profile(folder/'reference.npz')
    entries={name:dict(label=name,widths_m=[settings['width_m']],outlet_temperature_K=[float(p['T'][-1])],
              wall_heat_flux_W_m2=[float(flux_diagnostics(model,p,report['mass_flux_kg_m2_s'],comp['feed_Y'])['wall_heat_flux_W_m2'])],accepted=[True])
             for name,p in [('FGM reducido',baseline),('Detallado nativo',native0),('Cantera',reference0)]}
    for width in [.06,.09]:
        native_path=folder/f'domain_native_{int(width*1000)}_continued_grid'
        if not native_path.exists():
            native_grid=np.r_[native0['z'],np.linspace(native0['z'][-1],width,31)[1:]]
            solve_burner_flame(phi=phi,mass_flux=report['mass_flux_kg_m2_s'],initial_solution=folder/'native',
                grid=native_grid,
                output=native_path,mechanism=model.metadata['mechanism'],fuel=model.metadata['fuel'],oxidizer=model.metadata['oxidizer'],
                temperature=model.metadata['temperature_K'],pressure=model.metadata['pressure_Pa'],transport=model.metadata['transport'],soret=False,
                width=width,initial_points=24,slope=settings['native_slope'],curve=settings['native_curve'],ratio=3.,prune=.005,max_points=1800,max_time=240.)
        meta=json.loads((native_path/'metadata.json').read_text(encoding='utf-8'))
        if not meta['accepted']:raise RuntimeError('Native extended domain did not pass its own acceptance')
        native=read_profile(native_path/'flame.npz')
        reference_path=folder/f'domain_reference_{int(width*1000)}.npz'
        reference=read_profile(reference_path) if reference_path.exists() else {}
        if str(reference.get('input_digest',''))!=reference_input_hash(meta,report['mass_flux_kg_m2_s']):
            reference=cantera_reference(meta,report['mass_flux_kg_m2_s'],model.table['progress_weights']);np.savez_compressed(reference_path,**reference)
        try:
            profile,rpt=solve_reduced_burner_fgm(model,phi=phi,mass_flux=report['mass_flux_kg_m2_s'],
                seed_profile=profiles/report['seed_row']/'flame.npz',seed_row=report['seed_row'],width=width,
                initial_solution=baseline,max_iterations=80,**settings.get('solver_options',{}))
        except ReducedConvergenceError as e:profile,rpt=e.profile,e.report
        np.savez_compressed(folder/f'domain_reduced_{int(width*1000)}.npz',**profile)
        write_json(folder/f'domain_reduced_{int(width*1000)}.json',rpt)
        for name,p,accepted in [('FGM reducido',profile,rpt['accepted']),('Detallado nativo',native,True),('Cantera',reference,True)]:
            entry=entries[name];entry['widths_m'].append(width);entry['outlet_temperature_K'].append(float(p['T'][-1]))
            entry['wall_heat_flux_W_m2'].append(float(flux_diagnostics(model,p,report['mass_flux_kg_m2_s'],comp['feed_Y'])['wall_heat_flux_W_m2']))
            entry['accepted'].append(accepted)
        if rpt['accepted']:baseline=profile
        print('DOMAIN',width,rpt['accepted'],{name:entry['outlet_temperature_K'][-1] for name,entry in entries.items()},flush=True)
    for name,entry in entries.items():
        entry['last_temperature_change_K']=abs(entry['outlet_temperature_K'][-1]-entry['outlet_temperature_K'][-2])
        entry['last_wall_heat_flux_change_relative']=abs(entry['wall_heat_flux_W_m2'][-1]-entry['wall_heat_flux_W_m2'][-2])/abs(entry['wall_heat_flux_W_m2'][-2])
        entry['passed']=(all(entry['accepted']) and entry['last_temperature_change_K']<=settings['limits']['domain_temperature_change_K']
                          and entry['last_wall_heat_flux_change_relative']<=settings['limits']['domain_wall_heat_flux_change_relative'])
        result['domain'].append(entry)
    write_json(root/'convergence.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--table',type=Path,required=True);parser.add_argument('--validation',type=Path,required=True);parser.add_argument('--profiles',type=Path,required=True)
    args=parser.parse_args();run(args.table,args.validation,args.profiles)
