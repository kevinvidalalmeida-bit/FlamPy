"""Independent Cantera counterflow references for saved native diffusion flames."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def compare_diffusion_flame(native,output, *, slope=.01,curve=.02,limits=None):
    import cantera as ct
    native,output=Path(native),Path(output)
    meta=json.loads((native/'metadata.json').read_text(encoding='utf-8'))
    if not meta.get('accepted') or meta.get('flow_type')!='counterflow_diffusion' or meta.get('backend')!='native_cpu':
        raise ValueError('An accepted native counterflow diffusion flame is required')
    output.mkdir(parents=True,exist_ok=False)
    with np.load(native/'flame.npz',allow_pickle=False) as saved:
        ours={key:saved[key].copy() for key in saved.files}
    gas=ct.Solution(meta['mechanism'])
    if list(gas.species_names)!=list(ours['species_names']):
        raise ValueError('Reference species order differs')
    flame=ct.CounterflowDiffusionFlame(gas,width=meta['width'])
    flame.P=meta['pressure']
    for boundary,prefix in ((flame.fuel_inlet,'fuel'),(flame.oxidizer_inlet,'oxidizer')):
        boundary.Y=meta[prefix+'_Y'];boundary.T=meta[prefix+'_temperature'];boundary.mdot=meta[prefix+'_mass_flux']
    flame.transport_model=meta['transport'];flame.soret_enabled=meta['soret']
    flame.radiation_enabled=False
    flame.set_max_grid_points(flame.flame,2400)
    flame.set_refine_criteria(ratio=3.,slope=slope,curve=curve,prune=0.)
    started=time.perf_counter();flame.solve(loglevel=0,auto=True);runtime=time.perf_counter()-started
    reference=dict(z=flame.grid,T=flame.T,Y=flame.Y,u=flame.velocity,V=flame.spread_rate,
        Lambda=float(flame.flame.radial_pressure_gradient[0]),rho=flame.density,h=flame.enthalpy_mass,
        qdot=flame.heat_release_rate,omega_Y=flame.net_production_rates*gas.molecular_weights[:,None],
        species_names=np.array(gas.species_names))
    np.savez_compressed(output/'reference.npz',**reference)
    targets=ours['z']
    aligned={key:np.interp(targets,reference['z'],reference[key]) for key in ('T','u','V','h','qdot')}
    aligned['Y']=np.array([np.interp(targets,reference['z'],row) for row in reference['Y']])
    metrics=dict(temperature_K=float(np.max(abs(ours['T']-aligned['T']))),
        species_absolute=float(np.max(abs(ours['Y']-aligned['Y']))),
        velocity_relative=float(np.max(abs(ours['u']-aligned['u']))/max(np.max(abs(reference['u'])),1e-12)),
        pressure_eigenvalue_relative=float(abs(float(ours['Lambda'])-reference['Lambda'])/abs(reference['Lambda'])),
        heat_release_peak_relative=float(abs(ours['qdot'].max()-reference['qdot'].max())/max(abs(reference['qdot']).max(),1.)),
        peak_temperature_K=float(abs(ours['T'].max()-reference['T'].max())))
    limits=dict(temperature_K=20.,species_absolute=.005,velocity_relative=.03,
                pressure_eigenvalue_relative=.02,heat_release_peak_relative=.05) if limits is None else dict(limits)
    checks={key:metrics[key]<=value for key,value in limits.items()}
    report=dict(passed=all(checks.values()),metrics=metrics,limits=limits,checks=checks,
        cantera_version=ct.__version__,reference_nodes=len(flame.grid),reference_runtime_s=runtime,
        independent_initial_guess=True,native_metadata=meta,
        numerical_settings=dict(ratio=3.,slope=slope,curve=curve,max_points=2400),
        scope='Same streams, temperatures, pressure, inlet mass fluxes, separation and transport. '
              'Cantera starts independently; profiles compared at physical z without shifts.')
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    result=compare_diffusion_flame(args.native,args.output)
    print(json.dumps({k:result[k] for k in ('passed','metrics','reference_runtime_s')},indent=2))


if __name__=='__main__':
    main()
