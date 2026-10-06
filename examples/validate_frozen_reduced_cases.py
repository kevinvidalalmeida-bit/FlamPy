"""Revalidate a rebuilt FGM against independent saved burner references.

Only verified training profiles initialize the reduced solver. Archived
reduced solutions supply spatial grids, and native/Cantera profiles supply
comparisons only. Reuse avoids resolving unchanged detailed references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import solve_reduced_burner_fgm, ReducedConvergenceError, _SOURCE_SHA256
from examples.audit_nonadiabatic_sources import detailed_sources
from examples.validate_reduced_burner import comparisons


def read_arrays(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key] for key in data.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def run(table, profiles, references, output, limits, *, case_names=None):
    import cantera as ct
    started=time.perf_counter()
    table,profiles,references,output=map(Path,(table,profiles,references,output))
    model=NonAdiabaticFGM(table)
    mech=load_mechanism(model.metadata['mechanism'])
    folders=sorted(path for path in references.glob('case_*') if path.is_dir())
    if not folders:
        raise ValueError('No independent saved burner cases found')
    names={path.name for path in folders}
    if case_names is not None and not set(case_names)<=names:
        raise ValueError('Requested cases must belong to the frozen reference bank')
    input_hashes={folder.name:{name:digest(folder/name) for name in
        ('reduced_report.json','reduced.npz','comparison.npz','native/metadata.json')} for folder in folders}
    plan=dict(table_sha256=model.table_sha256,solver_source_sha256=_SOURCE_SHA256,
        validator_sha256=digest(__file__),comparison_sha256=digest(comparisons.__code__.co_filename),
        chemistry_auditor_sha256=digest(detailed_sources.__code__.co_filename),cantera_version=ct.__version__,
        limits=limits,reference_hashes=input_hashes,
        initialization='Verified training profiles only; archived reduced states supply grids only')
    output.mkdir(parents=True,exist_ok=True)
    plan_path=output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
            raise ValueError('Frozen validation output differs from the table, references, limits or code')
    elif any(output.iterdir()):
        raise ValueError('A new frozen validation must use an empty output directory')
    write_json(plan_path,plan)
    (output/'validator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    gas=ct.Solution(resolve_mechanism(model.metadata['mechanism']))
    for folder in folders:
        if case_names is not None and folder.name not in case_names:
            continue
        target=output/folder.name;target.mkdir(exist_ok=True)
        result_path=target/'result.json'
        if result_path.exists():
            record=json.loads(result_path.read_text(encoding='utf-8'))
            if (record.get('validation') is not None and 'error' not in record
                    and record.get('profile_sha256') is not None and (target/'reduced.npz').exists()
                    and digest(target/'reduced.npz')==record['profile_sha256']):
                print(folder.name,'cached',flush=True)
                continue
        options=json.loads((folder/'reduced_report.json').read_text(encoding='utf-8'))
        comparison=read_arrays(folder/'comparison.npz')
        grid=read_arrays(folder/'reduced.npz')['z']
        phi,mdot=options['phi'],options['mass_flux_kg_m2_s']
        native_meta=json.loads((folder/'native/metadata.json').read_text(encoding='utf-8'))
        feed=fresh_mixture(mech,phi,model.metadata['fuel'],model.metadata['oxidizer'])
        if (not np.isclose(float(comparison['mass_flux']),mdot,rtol=1e-12,atol=0.)
                or not np.isclose(native_meta['mass_flux'],mdot,rtol=1e-12,atol=0.)
                or not np.allclose(comparison['feed_Y'],feed,rtol=1e-12,atol=1e-15)
                or not np.allclose(native_meta['inlet_Y'],feed,rtol=1e-12,atol=1e-15)
                or native_meta['species_names']!=list(mech.species_names)
                or native_meta['temperature']!=model.metadata['temperature_K']
                or native_meta['pressure']!=model.metadata['pressure_Pa']
                or native_meta['transport']!=model.metadata['transport']
                or native_meta['soret']!=model.metadata['soret']
                or not native_meta.get('accepted') or not native_meta.get('report',{}).get('grid_converged')
                or digest(resolve_mechanism(native_meta['mechanism']))!=digest(resolve_mechanism(model.metadata['mechanism']))):
            raise ValueError(f'{folder.name}: saved references differ from the requested physical model')
        seed=min((row for row in model.metadata['rows'] if row['loss_index']>0),
            key=lambda row:(abs(row['phi']-phi),abs(row['mass_flux']-mdot)))
        record=dict(case=folder.name,phi=phi,mass_flux_kg_m2_s=mdot,passed=False)
        try:
            try:
                reduced,report=solve_reduced_burner_fgm(model,phi=phi,mass_flux=mdot,
                    seed_profile=profiles/seed['output']/'flame.npz',seed_row=seed['output'],
                    width=options['width_m'],grid=grid,max_iterations=options['max_iterations'],
                    residual_tolerance=options['residual_tolerance'],max_energy_error=options['max_energy_error'],
                    interpolation=options['interpolation'],curvature_limit=options['curvature_limit'],
                    progress_equation=options.get('progress_equation','conservative'),
                    metric_floor=options.get('metric_floor',1e-30),kernel_backend=options.get('kernel_backend','compiled'))
            except ReducedConvergenceError as error:
                reduced,report=error.profile,error.report
            record['report']=report
            np.savez_compressed(target/'reduced.npz',**reduced)
            record['profile_sha256']=digest(target/'reduced.npz')
            native={key[7:]:value for key,value in comparison.items() if key.startswith('native_')}
            reference={key[10:]:value for key,value in comparison.items() if key.startswith('reference_')}
            rates,heat=detailed_sources(mech,native['T'],native['Y'],model.metadata['pressure_Pa'])
            native.update(omega_C=model.table['progress_weights']@rates,qdot=heat)
            states=ct.SolutionArray(gas,shape=len(reference['T']))
            states.TPY=reference['T'],model.metadata['pressure_Pa'],reference['Y'].T
            reference.update(omega_C=states.net_production_rates@(model.table['progress_weights']*gas.molecular_weights),
                             qdot=states.heat_release_rate)
            validation=comparisons(model,reduced,native,reference,mdot,feed,limits)
            record.update(validation=validation,passed=bool(report['accepted'] and validation['passed']))
        except Exception as error:
            record['error']=repr(error)
        write_json(result_path,record)
        print(folder.name,'passed' if record['passed'] else 'failed',
            (record.get('validation') or {}).get('metrics',record.get('error')),flush=True)
    records=[json.loads(path.read_text(encoding='utf-8')) for path in sorted(output.glob('case_*/result.json'))]
    runtimes=[record['report']['total_elapsed_seconds'] for record in records if 'report' in record]
    summary=dict(table_sha256=model.table_sha256,planned_cases=len(folders),cases=len(records),
        numerically_accepted=sum(record.get('report',{}).get('accepted',False) for record in records),
        physical_passed=sum(record['passed'] for record in records),
        pending_cases=sorted(names-{record['case'] for record in records}),
        invocation_elapsed_seconds=time.perf_counter()-started,
        total_solver_elapsed_seconds=sum(runtimes),
        median_solver_elapsed_seconds=float(np.median(runtimes)) if runtimes else None,
        records=records)
    write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table',type=Path,required=True)
    parser.add_argument('--profiles',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settings',type=Path,required=True,help='Frozen physical comparison limits')
    parser.add_argument('--case',action='append',dest='case_names')
    args=parser.parse_args()
    limits=json.loads(args.settings.read_text(encoding='utf-8'))['limits']
    result=run(args.table,args.profiles,args.references,args.output,limits,case_names=args.case_names)
    print({key:value for key,value in result.items() if key!='records'})
