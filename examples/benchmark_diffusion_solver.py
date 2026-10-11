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
    """Inclusive timings in the same window as native_s.

    solve_diffusion_problem compiles and warms the kernels before starting the
    clock reported as native_s; calls made during compile_kernels are recorded
    under 'preparation:<name>' so the solve categories share native_s's window.
    Inclusive categories overlap and must not be summed.
    """
    timings={};originals=[];matrices=[];newton_calls=[]
    phase=dict(preparation=False)
    def measured(name,fun):
        @wraps(fun)
        def call(*args,**kwargs):
            started=time.perf_counter()
            try:
                return fun(*args,**kwargs)
            finally:
                label=('preparation:'+name) if phase['preparation'] else name
                entry=timings.setdefault(label,dict(calls=0,time_s=0.))
                entry['calls']+=1;entry['time_s']+=time.perf_counter()-started
        return call
    original_compile=diffusion.DiffusionFlameProblem.compile_kernels
    def compile_kernels(self,*args,**kwargs):
        phase['preparation']=True;started=time.perf_counter()
        try:
            return original_compile(self,*args,**kwargs)
        finally:
            phase['preparation']=False
            entry=timings.setdefault('preparation:compile_kernels',dict(calls=0,time_s=0.))
            entry['calls']+=1;entry['time_s']+=time.perf_counter()-started
    diffusion.DiffusionFlameProblem.compile_kernels=compile_kernels
    targets=[(diffusion.DiffusionFlameProblem,'residual','residual_inclusive'),
             (diffusion.DiffusionFlameProblem,'properties','properties_inclusive'),
             (diffusion.DiffusionFlameProblem,'jacobian_blocks','compiled_jacobian_inclusive'),
             (diffusion.DiffusionFlameProblem,'equilibrated_guess','equilibrium_fallback'),
             (diffusion.BorderedFactor,'solve','linear_solves'),
             (diffusion,'face_properties','compiled_face_transport'),
             (diffusion,'species_partials','analytic_chemistry_partials'),
             (NativeSpeciesBackend,'eval_grid_thermo_kinetics_into','thermo_kinetics')]
    for owner,attribute,label in targets:
        original=getattr(owner,attribute);originals.append((owner,attribute,original))
        setattr(owner,attribute,measured(label,original))
    original_init=diffusion.BorderedFactor.__init__
    def factor(self,blocks,rdt=0.):
        started=time.perf_counter()
        original_init(self,blocks,rdt)
        entry=timings.setdefault('bordered_factorization',dict(calls=0,time_s=0.))
        entry['calls']+=1;entry['time_s']+=time.perf_counter()-started
        matrices.append(dict(size=blocks.n*blocks.nv+1,nodes=blocks.n,transient=bool(rdt),
                             sparse_fallback=self.sparse is not None))
    diffusion.BorderedFactor.__init__=factor
    original_newton=diffusion._newton
    def traced_newton(problem,x,*args,**kwargs):
        started=time.perf_counter();before=getattr(problem,'jacobian_evaluations',0)
        result=original_newton(problem,x,*args,**kwargs)
        newton_calls.append(dict(nodes=problem.n,
            mode='transient' if kwargs.get('previous') is not None else
                 'fixed_temperature' if kwargs.get('fixed_temperature') is not None else 'steady',
            dt=kwargs.get('dt'),time_s=time.perf_counter()-started,
            jacobians=getattr(problem,'jacobian_evaluations',0)-before,converged=result[1],reason=result[2]))
        return result
    diffusion._newton=traced_newton
    try:
        yield timings,matrices,newton_calls
    finally:
        diffusion.DiffusionFlameProblem.compile_kernels=original_compile
        diffusion._newton=original_newton
        diffusion.BorderedFactor.__init__=original_init
        for owner,attribute,original in reversed(originals):
            setattr(owner,attribute,original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=['CH4','H2','H2-MS','CH4-MS'],default=['CH4','H2','H2-MS'])
    parser.add_argument('--radiation',action='store_true',
                        help='Optically thin radiation in native and Cantera flames (non-adiabatic)')
    parser.add_argument('--no-profile',action='store_true',help='Measure without native instrumentation')
    parser.add_argument('--repeats',type=int,default=1,help='Independent solves per case; report medians and ranges')
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    settings={'CH4':dict(mechanism='gri30.yaml',fuel='CH4',fuel_mass_flux=.12,oxidizer_mass_flux=.24),
              'H2':dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15),
              'CH4-MS':dict(mechanism='gri30.yaml',fuel='CH4',fuel_mass_flux=.12,oxidizer_mass_flux=.24,
                            transport='multicomponent',soret=True),
              'H2-MS':dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15,
                           transport='multicomponent',soret=True)}
    # Warm native compiled kernels without sharing a flame profile with a case.
    solve_diffusion_flame(**settings['H2'],refine=False,output=args.output/'warmup')
    if 'H2-MS' in args.cases or 'CH4-MS' in args.cases:
        solve_diffusion_flame(**settings['H2-MS'],refine=False,output=args.output/'warmup_multicomponent_soret')
    summary=dict(profile_enabled=not args.no_profile,radiation=args.radiation,
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
                native=solve_diffusion_flame(**settings[name],radiation=args.radiation,output=folder/'native')
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
