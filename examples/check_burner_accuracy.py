"""Check stricter Newton tolerances and a finer mesh for an accepted burner.

python examples/check_burner_accuracy.py runs/validation3d/case_03_native_fine \
    --output runs/burner_accuracy
The chemistry and imposed mass flux are unchanged. The new runs are solved
and certified from native continuation guesses, not merely resampled.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kflame import solve_burner_flame
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
if __package__:
    from .audit_nonadiabatic_sources import detailed_sources, source_metrics
else:
    from audit_nonadiabatic_sources import detailed_sources, source_metrics


def read(path):
    with np.load(path/'flame.npz',allow_pickle=False) as saved:
        return {k:saved[k] for k in saved.files}


def check(folder,output):
    folder,output=Path(folder),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if not meta['accepted']:
        raise ValueError('An accepted native burner is required')
    mech=load_mechanism(meta['mechanism'])
    weights=np.array([dict(CO2=1.,H2O=1.,CO=1.,H2=.5).get(s,0.) for s in mech.species_names])
    refine=meta['refinement']
    base=read(folder)
    settings=dict(mechanism=meta['mechanism'],mass_flux=meta['mass_flux'],
                  Y=dict(zip(meta['species_names'],meta['inlet_Y'])),
                  temperature=meta['temperature'],pressure=meta['pressure'],
                  width=meta['initial_width'],transport=meta['transport'],soret=meta['soret'],
                  rtol=.01*meta['tolerances']['rtol'],atol=.01*meta['tolerances']['atol'],
                  max_time=240.,**refine)
    records=[]
    previous=folder
    for name,factor in [('tight_tolerance',1.),('tight_fine_mesh',.5)]:
        path=output/name
        if not path.exists():
            solve_burner_flame(initial_solution=previous,output=path,
                               **dict(settings,slope=factor*refine['slope'],curve=factor*refine['curve'],prune=factor*refine['prune']))
        fine=read(path)
        current=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        if not current['accepted']:
            raise RuntimeError('Convergence check did not produce an accepted flame')
        old_mass,old_q=detailed_sources(mech,base['T'],base['Y'],meta['pressure'])
        new_mass,new_q=detailed_sources(mech,fine['T'],fine['Y'],meta['pressure'])
        source_C=source_metrics(base['z'],weights@old_mass,
                                np.interp(base['z'],fine['z'],weights@new_mass))
        source_q=source_metrics(base['z'],old_q,np.interp(base['z'],fine['z'],new_q))
        integrals=(float(np.trapezoid(weights@old_mass,base['z'])),float(np.trapezoid(weights@new_mass,fine['z'])))
        expected=meta['mass_flux']*float(weights@(fine['Y'][:,-1]-np.asarray(meta['inlet_Y'])))
        balance=abs(integrals[1]-expected)/abs(expected)
        record=dict(check=name,reference_nodes=len(base['z']),new_nodes=len(fine['z']),
                    temperature_Linf_K=float(np.max(abs(base['T']-np.interp(base['z'],fine['z'],fine['T'])))),
                    Tmax_change_K=abs(float(np.max(base['T']))-float(np.max(fine['T']))),
                    omega_C=source_C,qdot=source_q,source_integrals=integrals,
                    progress_balance=balance,heat_loss=current['heat_loss'],
                    tolerances=current['tolerances'],refinement=current['refinement'],
                    solver_acceptance={k:current['report'][k] for k in ('Finf_final','weighted_step_norm_final','grid_converged','final_accepted')})
        records.append(record)
        print(json.dumps(record),flush=True)
        previous=path
    result=dict(case=folder.name,baseline_tolerances=meta['tolerances'],checks=records,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                baseline_profile_sha256=hashlib.sha256((folder/'flame.npz').read_bytes()).hexdigest(),
                scope='Compare detailed burner solutions under stricter nonlinear and spatial tolerances.')
    (output/'convergence.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result


def check_reference(folder, reference, output):
    """Refine the independent Cantera solution using only its own initial guess."""
    import cantera as ct
    if __package__:
        from .validate_nonadiabatic_3d import reference_input_hash
    else:
        from validate_nonadiabatic_3d import reference_input_hash
    folder,reference,output=Path(folder),Path(reference),Path(output)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    with np.load(reference,allow_pickle=False) as stored:
        old={k:stored[k] for k in stored.files}
    if str(old.get('input_digest',''))!=reference_input_hash(meta,meta['mass_flux']):
        raise ValueError('Independent reference inputs do not match the native comparison case')
    gas=ct.Solution(resolve_mechanism(meta['mechanism']))
    gas.TPY=meta['temperature'],meta['pressure'],meta['inlet_Y']
    flame=ct.BurnerFlame(gas,grid=old['z'])
    flame.burner.mdot=meta['mass_flux']
    flame.transport_model=meta['transport']
    flame.soret_enabled=meta['soret']
    data=ct.SolutionArray(gas,len(old['z']),extra=dict(grid=old['z'],velocity=old['u']))
    data.TPY=old['T'],meta['pressure'],old['Y'].T
    flame.set_initial_guess(data=data)
    criteria={k:meta['refinement'][k] for k in ('ratio','slope','curve','prune')}
    criteria.update({k:.5*criteria[k] for k in ('slope','curve','prune')})
    flame.set_refine_criteria(**criteria)
    flame.solve(loglevel=0,auto=False)
    weights=np.array([dict(CO2=1.,H2O=1.,CO=1.,H2=.5).get(s,0.) for s in gas.species_names])
    omega=weights@(flame.net_production_rates*gas.molecular_weights[:,None])
    qdot=flame.heat_release_rate
    report=dict(reference='Cantera grid refinement from its independent solution',
                reference_version=ct.__version__,old_nodes=len(old['z']),new_nodes=len(flame.grid),
                T_Linf_K=float(np.max(abs(old['T']-np.interp(old['z'],flame.grid,flame.T)))),
                omega_C=source_metrics(old['z'],old['omega_C'],np.interp(old['z'],flame.grid,omega)),
                qdot=source_metrics(old['z'],old['qdot'],np.interp(old['z'],flame.grid,qdot)),
                source_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'cantera_convergence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    np.savez_compressed(output/'cantera_refined.npz',z=flame.grid,T=flame.T,Y=flame.Y,u=flame.velocity,omega_C=omega,qdot=qdot)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='Independent Cantera reference NPZ to refine separately')
    args=parser.parse_args()
    check(args.folder,args.output)
    if args.reference:
        print(json.dumps(check_reference(args.folder,args.reference,args.output)),flush=True)


if __name__=='__main__':
    main()
