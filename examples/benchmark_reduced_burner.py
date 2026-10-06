"""Measure colored thermodynamic-cache savings on an identical frozen problem."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import ReducedBurnerProblem
from examples.validate_reduced_burner import read_profile, write_json
from kflame.fgm.tensor_kernels import evaluate_tensor
from kflame.fgm.reduced_burner import _SOURCE_SHA256
from numba import get_num_threads
from scipy.optimize import brentq


def benchmark_temperature_inversion(model, state, *, repeats=20):
    """Matched reference to the previous separate-property Newton algorithm."""
    target=np.tile(state['h'],repeats)
    Y=np.tile(state['Y'],(1,repeats))
    guess=np.tile(state['T'],repeats)+50.
    def separate():
        T=np.clip(guess,200.,model.max_temperature)
        lower=np.full(len(T),200.);upper=np.full(len(T),model.max_temperature)
        for _ in range(16):
            residual=model.thermo.enthalpy_mass(T,Y)-target
            done=abs(residual)<=1e-6
            if done.all():return T
            lower=np.where(residual<0.,T,lower);upper=np.where(residual>0.,T,upper)
            cp=model.thermo.cp_mass(T,Y);proposed=T-residual/cp
            safe=(cp>0.)&np.isfinite(proposed)&(proposed>lower)&(proposed<upper)
            T=np.where(done,T,np.where(safe,proposed,.5*(lower+upper)))
        for i in np.flatnonzero(abs(model.thermo.enthalpy_mass(T,Y)-target)>1e-6):
            T[i]=brentq(lambda temperature:float(model.thermo.enthalpy_mass(temperature,Y[:,i]))-target[i],
                200.,model.max_temperature,xtol=1e-8)
        return T
    fused=lambda:model._recover_temperature(target,Y,guess)
    reference,actual=separate(),fused()  # Warm up compilation outside timings.
    np.testing.assert_allclose(actual,reference,rtol=0.,atol=1e-7)
    timings={'separate':[],'fused':[]}
    for repetition in range(9):
        pairs=[('separate',separate),('fused',fused)]
        for name,run in pairs[::1 if repetition%2==0 else -1]:
            start=time.perf_counter();run();timings[name].append(time.perf_counter()-start)
    medians={name:float(np.median(samples)) for name,samples in timings.items()}
    return dict(nodes=len(target),initial_temperature_offset_K=50.,median_seconds=medians,
        samples_seconds=timings,speedup=medians['separate']/medians['fused'],
        maximum_temperature_difference_K=float(np.max(abs(actual-reference))),
        maximum_enthalpy_closure_J_kg=float(np.max(abs(model.thermo.enthalpy_mass(actual,Y)-target))))


def benchmark(table, case, output):
    start=time.perf_counter();model=NonAdiabaticFGM(table);load=time.perf_counter()-start
    case=Path(case);profile=read_profile(case/'reduced.npz');comparison=read_profile(case/'comparison.npz')
    options=json.loads((case/'reduced_report.json').read_text(encoding='utf-8'))
    start=time.perf_counter()
    problem=ReducedBurnerProblem(model,profile['z'],float(comparison['mass_flux']),comparison['feed_Y'],
        progress_equation=options.get('progress_equation','conservative'),metric_floor=options.get('metric_floor',1e-30),curvature_limit=options['curvature_limit'])
    construction=time.perf_counter()-start
    x=profile['logical_coordinates'].ravel();state=problem.state(x);perturbations=[]
    problem.residual(x,evaluated_state=state);projection_cache=problem._last_projection_coefficients
    for color in range(9):
        selected=np.arange(color,len(x),9);trial=x.copy();trial[selected]+=1e-7
        perturbations.append((selected//3,trial))
    def evaluate(cached,projected_cache=True):
        return [problem.residual(trial,geometry=state['geometry'],
                     temperature_cache=state['T'] if cached else None,
                     changed_nodes=nodes if cached else None,
                     projection_cache=projection_cache if cached and projected_cache else None) for nodes,trial in perturbations]
    full,cached=evaluate(False),evaluate(True)
    error=float(max(np.max(abs(a-b)) for a,b in zip(full,cached)))
    if error>1e-9:raise RuntimeError('Cache changed the colored residuals')
    timings={}
    for backend,reuse,projected_cache,name in [('numpy',False,False,'numpy_full'),('numpy',True,True,'numpy_cached'),
            ('compiled',False,False,'full'),('compiled',True,False,'temperature_cache'),('compiled',True,True,'cached')]:
        problem.kernel_backend=backend
        actual=evaluate(reuse,projected_cache)
        difference=float(max(np.max(abs(a-b)) for a,b in zip(full,actual)))
        if difference>1e-9:raise RuntimeError('Compiled kernel or cache changed residuals')
        values=[]
        for _ in range(9):
            started=time.perf_counter();evaluate(reuse,projected_cache);values.append(time.perf_counter()-started)
        timings[name]=dict(median_seconds=float(np.median(values)),samples_seconds=values,maximum_residual_difference=difference)
    points=np.random.default_rng(213).uniform(0.,1.,(76636,3));batch={}
    reference,_=evaluate_tensor(points,problem.axis_knots,problem.strides,problem.packed_fields,parallel_threshold=10**9)
    for name,threshold in [('serial',10**9),('parallel',1)]:
        actual,_=evaluate_tensor(points,problem.axis_knots,problem.strides,problem.packed_fields,parallel_threshold=threshold)
        np.testing.assert_array_equal(actual,reference)
        values=[]
        for _ in range(7):
            start=time.perf_counter();evaluate_tensor(points,problem.axis_knots,problem.strides,problem.packed_fields,parallel_threshold=threshold);values.append(time.perf_counter()-start)
        batch[name]=dict(median_seconds=float(np.median(values)),samples_seconds=values)
    reports=[]
    for path in case.parent.glob('case_*/reduced_report.json'):
        reports.append(json.loads(path.read_text(encoding='utf-8')))
    accepted=[r for r in reports if r['accepted']]
    result=dict(table_sha256=model.table_sha256,python=platform.python_version(),platform=platform.system(),
        nodes=len(profile['z']),colors=9,dense_finite_difference_columns=len(x),
        solver_source_sha256=_SOURCE_SHA256,table_loading_seconds=load,closure_construction_seconds=construction,
        logical_processors=os.cpu_count(),
        numba_threads=get_num_threads(),cache_maximum_residual_difference=error, timings=timings,
        cache_speedup=timings['full']['median_seconds']/timings['cached']['median_seconds'],
        compiled_kernel_speedup=timings['numpy_cached']['median_seconds']/timings['cached']['median_seconds'],
        combined_speedup=timings['numpy_full']['median_seconds']/timings['cached']['median_seconds'],
        large_batch=batch,large_batch_points=len(points),large_batch_parallel_speedup=batch['serial']['median_seconds']/batch['parallel']['median_seconds'],
        accepted_case_count=len(accepted),failed_case_count=len(reports)-len(accepted),
        solve_median_seconds=float(np.median([r['elapsed_seconds'] for r in accepted])),
        solve_range_seconds=[min(r['elapsed_seconds'] for r in accepted),max(r['elapsed_seconds'] for r in accepted)],
        total_solve_median_seconds=float(np.median([r['total_elapsed_seconds'] for r in accepted])),
        total_solve_range_seconds=[min(r['total_elapsed_seconds'] for r in accepted),max(r['total_elapsed_seconds'] for r in accepted)],
        detailed_kinetics_evaluations_in_reduced_production=0,
        thermodynamic_inversion=benchmark_temperature_inversion(model,state),
        physical_passed_case_count=sum(json.loads(p.read_text(encoding='utf-8'))['passed']
            for p in case.parent.glob('case_*/validation.json')),
        note='Accepted counts mean numerical convergence. Case times may include concurrent validation processes; matched microbenchmarks use identical inputs. Total solve includes guide and continuation; table loading and construction are separate.')
    write_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--table',type=Path,required=True);parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(benchmark(args.table,args.case,args.output))
