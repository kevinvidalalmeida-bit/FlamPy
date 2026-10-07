"""Sequential native/Cantera timings and native solver cost attribution.

Run in an otherwise idle machine:
python examples/benchmark_diffusion_solver.py --output runs/diffusion_timing
Instrumentation records inclusive timings; nested categories must not be added.
Both solvers start without a reference seed and adapt their own meshes.
"""
import argparse
from contextlib import contextmanager,nullcontext
from functools import wraps
import json
from pathlib import Path
import time
from types import SimpleNamespace

from kflame import solve_diffusion_flame
import numpy as np
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.transport import NativeTransport
from kflame.flame import diffusion
from kflame.reference.diffusion import compare_diffusion_flame


@contextmanager
def instrument_native():
    timings={};originals=[];matrices=[]
    def measured(name,fun):
        @wraps(fun)
        def call(*args,**kwargs):
            started=time.perf_counter()
            try:
                return fun(*args,**kwargs)
            finally:
                entry=timings.setdefault(name,dict(calls=0,time_s=0.))
                entry['calls']+=1;entry['time_s']+=time.perf_counter()-started
        return call
    targets=[(diffusion.DiffusionFlameProblem,'residual','residual_inclusive'),
             (diffusion.DiffusionFlameProblem,'jacobian','jacobian_inclusive'),
             (diffusion.DiffusionFlameProblem,'equilibrated_guess','equilibrium_fallback'),
             (NativeSpeciesBackend,'eval_grid_thermo_kinetics_into','thermo_kinetics'),
             (NativeSpeciesBackend,'eval_jacobian_thermo_kinetics_into','grouped_jacobian_thermo'),
             (NativeSpeciesBackend,'eval_faces','mixture_face_transport'),
             (NativeSpeciesBackend,'eval_multicomponent_face_transport','multicomponent_face_transport'),
             (NativeSpeciesBackend,'eval_mixture_thermal_diffusion','mixture_soret'),
             (NativeTransport,'viscosity','viscosity')]
    for owner,attribute,label in targets:
        original=getattr(owner,attribute);originals.append((owner,attribute,original))
        setattr(owner,attribute,measured(label,original))
    original_lu=diffusion.splu
    def factorize(matrix):
        matrices.append(dict(size=matrix.shape[0],stored_entries=matrix.nnz,
                             exact_nonzero_entries=int(np.count_nonzero(matrix.data))))
        factor=measured('lu_factorization',original_lu)(matrix)
        return SimpleNamespace(solve=measured('linear_solves',factor.solve))
    diffusion.splu=factorize
    try:
        yield timings,matrices
    finally:
        diffusion.splu=original_lu
        for owner,attribute,original in reversed(originals):
            setattr(owner,attribute,original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=['CH4','H2','H2-MS'],default=['CH4','H2','H2-MS'])
    parser.add_argument('--no-profile',action='store_true',help='Measure without native instrumentation')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    settings={'CH4':dict(mechanism='gri30.yaml',fuel='CH4',fuel_mass_flux=.12,oxidizer_mass_flux=.24),
              'H2':dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15),
              'H2-MS':dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15,
                           transport='multicomponent',soret=True)}
    # Warm native compiled kernels without sharing a flame profile with a case.
    solve_diffusion_flame(**settings['H2'],refine=False,output=args.output/'warmup')
    summary=dict(profile_enabled=not args.no_profile,
                 timing_scope='Sequential solve-only timings, compiled native kernels. '
                             'Same physics and refinement thresholds, independent grids/initial guesses. '
                             'One sample per case, not a statistical speed guarantee.',cases={})
    for name in args.cases:
        with (nullcontext(({},[])) if args.no_profile else instrument_native()) as (timings,matrices):
            native=solve_diffusion_flame(**settings[name],output=args.output/name/'native')
        reference=compare_diffusion_flame(native['output'],args.output/name/'cantera')
        entry=dict(native_s=native['report']['runtime_s'],native_api_s=native['runtime_s'],
                   cantera_s=reference['reference_runtime_s'],
                   native_over_cantera=native['report']['runtime_s']/reference['reference_runtime_s'],
                   native_nodes=len(native['z']),cantera_nodes=reference['reference_nodes'],
                   validation_passed=reference['passed'],metrics=reference['metrics'],
                   source_sha256=reference['native_metadata']['source_sha256'],
                   inclusive_timings=timings,jacobian_matrices=matrices,
                   mesh_history=native['report']['history'])
        summary['cases'][name]=entry
        (args.output/'benchmark.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        print(name,json.dumps({k:entry[k] for k in ('native_s','cantera_s','native_over_cantera','validation_passed')}),flush=True)
    if not all(row['validation_passed'] for row in summary['cases'].values()):
        raise SystemExit('At least one physical comparison failed; inspect benchmark.json')


if __name__=='__main__':
    main()
