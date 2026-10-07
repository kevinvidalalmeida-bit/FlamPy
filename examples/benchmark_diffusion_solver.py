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
import os
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

from kflame import solve_diffusion_flame
import numpy as np
import numba
import scipy
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.transport import NativeTransport
from kflame.flame import diffusion
from kflame.reference.diffusion import compare_diffusion_flame


@contextmanager
def instrument_native():
    timings={};originals=[];matrices=[];newton_calls=[]
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
    original_newton=diffusion._newton
    def traced_newton(problem,x,*args,**kwargs):
        started=time.perf_counter();before=len(matrices)
        result=original_newton(problem,x,*args,**kwargs)
        newton_calls.append(dict(nodes=problem.n,
            mode='transient' if kwargs.get('previous') is not None else
                 'fixed_temperature' if kwargs.get('fixed_temperature') is not None else 'steady',
            dt=kwargs.get('dt'),time_s=time.perf_counter()-started,
            jacobians=len(matrices)-before,converged=result[1],reason=result[2]))
        return result
    diffusion._newton=traced_newton
    try:
        yield timings,matrices,newton_calls
    finally:
        diffusion._newton=original_newton
        diffusion.splu=original_lu
        for owner,attribute,original in reversed(originals):
            setattr(owner,attribute,original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=['CH4','H2','H2-MS'],default=['CH4','H2','H2-MS'])
    parser.add_argument('--no-profile',action='store_true',help='Measure without native instrumentation')
    parser.add_argument('--repeats',type=int,default=1,help='Independent solves per case; report medians and ranges')
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    settings={'CH4':dict(mechanism='gri30.yaml',fuel='CH4',fuel_mass_flux=.12,oxidizer_mass_flux=.24),
              'H2':dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15),
              'H2-MS':dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15,
                           transport='multicomponent',soret=True)}
    # Warm native compiled kernels without sharing a flame profile with a case.
    solve_diffusion_flame(**settings['H2'],refine=False,output=args.output/'warmup')
    if 'H2-MS' in args.cases:
        solve_diffusion_flame(**settings['H2-MS'],refine=False,output=args.output/'warmup_multicomponent_soret')
    summary=dict(profile_enabled=not args.no_profile,
                 timing_scope='Sequential solve-only timings, compiled native kernels. '
                             'Same physics and refinement thresholds, independent grids/initial guesses. '
                             'Native runs before Cantera within each pair. Local samples, '
                             'not a statistical speed guarantee.',
                 repeats=args.repeats,environment=dict(python=platform.python_version(),
                    platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),
                    numpy=np.__version__,numba=numba.__version__,scipy=scipy.__version__,
                    numba_threads=numba.get_num_threads(),thread_overrides={name:os.environ.get(name)
                        for name in ('OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
                 cases={},samples={},aggregates={})
    for repeat in range(args.repeats):
        # Rotate case order; every solve starts independently, without profile continuation.
        offset=repeat%len(args.cases)
        for name in args.cases[offset:]+args.cases[:offset]:
            folder=args.output/name if args.repeats==1 else args.output/name/f'repeat_{repeat+1:02d}'
            with (nullcontext(({},[],[])) if args.no_profile else instrument_native()) as (timings,matrices,newton_calls):
                native=solve_diffusion_flame(**settings[name],output=folder/'native')
            reference=compare_diffusion_flame(native['output'],folder/'cantera')
            entry=dict(native_s=native['report']['runtime_s'],native_api_s=native['runtime_s'],
                       cantera_s=reference['reference_runtime_s'],
                       native_over_cantera=native['report']['runtime_s']/reference['reference_runtime_s'],
                       native_nodes=len(native['z']),cantera_nodes=reference['reference_nodes'],
                       validation_passed=reference['passed'],metrics=reference['metrics'],
                       source_sha256=reference['native_metadata']['source_sha256'],
                       cantera_version=reference['cantera_version'],
                       cantera_solver_statistics=reference['solver_statistics'],
                       cantera_convergence=reference['convergence_settings'],
                       native_convergence=dict(criterion='Maximum absolute scaled residual',
                                               threshold=1e-7,value=native['report']['maximum_scaled_residual']),
                       inclusive_timings=timings,jacobian_matrices=matrices,
                       newton_calls=newton_calls,
                       mesh_history=native['report']['history'])
            # Keep the single-run case record compatible with earlier audits.
            summary['cases'][name]=entry
            samples=summary['samples'].setdefault(name,[]);samples.append(entry)
            summary['aggregates'][name]=dict(samples=len(samples),validation_passed=all(row['validation_passed'] for row in samples),
                **{key:dict(median=statistics.median(row[key] for row in samples),
                            minimum=min(row[key] for row in samples),maximum=max(row[key] for row in samples))
                   for key in ('native_s','cantera_s','native_over_cantera')})
            (args.output/'benchmark.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
            print(name,repeat+1,json.dumps({k:entry[k] for k in ('native_s','cantera_s','native_over_cantera','validation_passed')}),flush=True)
    if not all(row['validation_passed'] for row in summary['aggregates'].values()):
        raise SystemExit('At least one physical comparison failed; inspect benchmark.json')


if __name__=='__main__':
    main()
