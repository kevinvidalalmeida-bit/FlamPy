"""Instrument selected completed campaign cases, independently of benchmark times.

One excluded warmup and one instrumented solve per solver/case. No production
solver sources are edited. Cantera counters and native counters have explicitly
different scopes; inclusive native subtimers must not be added together.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sweeps',ROOT/'benchmarks/benchmark_flame_sweeps.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def selection(manifest):
    return [c for c in manifest['cases'] if
        (c['phi']==1 and c['pressure_atm']==1 and c['temperature']==300) or
        (c['fuel']=='H2' and c['phi']==1 and c['pressure_atm']==10 and c['temperature']==300) or
        (c['fuel']=='H2' and c['phi']==.9 and c['pressure_atm']==1 and c['temperature']==300
         and c['transport']=='mixture-averaged' and c['soret'])]


def instrument(backend):
    state=dict(active=False)
    original=runner.solve
    if backend=='native':
        import kflame.flame.solver as solver
        import kflame.flame.equations as equations
        # Nonoverlapping time buckets: nested residuals within a Jacobian build
        # belong to that Jacobian; invocation counters remain separately named.
        def timed(func, category):
            def wrapped(*a,**kw):
                if not state['active']:return func(*a,**kw)
                state['calls'][category]+=1
                outer=not state['stack']
                state['stack'].append(category)
                start=time.perf_counter()
                try:return func(*a,**kw)
                finally:
                    dt=time.perf_counter()-start
                    state['stack'].pop()
                    if outer:state['seconds'][category]+=dt
            return wrapped
        for name,category in [('residual','residual'),('build_jacobian_steady','jacobian'),
                              ('factorize','factorize'),('solve_linear','linear_solve')]:
            old=getattr(solver,name)
            wrapper=timed(old,category)
            setattr(solver,name,wrapper)
            if getattr(equations,name,None) is old:setattr(equations,name,wrapper)
        hybrid=solver._hybrid_newton
        def wrapped_hybrid(problem,*a,**kw):
            if not state['active']:return hybrid(problem,*a,**kw)
            phase=dict(label=kw.get('label'),nodes=problem.n_points,width=problem.width,
                       energy=problem.solve_energy,transport=problem.transport_model,
                       soret=problem.case.soret_enabled)
            state['phases'].append(phase)
            before=len(getattr(problem,'_solver_trace',[]))
            start=time.perf_counter()
            try:
                value=hybrid(problem,*a,**kw)
                phase.update(ok=value[1],history=value[2])
                return value
            except Exception as exc:
                phase['interrupted']=type(exc).__name__
                raise
            finally:
                phase['wall_s']=time.perf_counter()-start
                trace=getattr(problem,'_solver_trace',[])
                if len(trace)>before:phase['history']=trace[-1]['history']
        solver._hybrid_newton=wrapped_hybrid
        solve=solver.solve_free_flame
        def traced_solve(problem,*a,**kw):
            if state['active']:
                kw['options']=replace(kw['options'],profile=True,trace_solver=True)
            return solve(problem,*a,**kw)
        solver.solve_free_flame=traced_solve
    else:
        import cantera as ct
        solve=ct.FreeFlame.solve
        def traced_cantera(flame,*a,**kw):
            if not state['active']:return solve(flame,*a,**kw)
            start=time.perf_counter()
            def callback(kind,dt):
                state['events'].append(dict(kind=kind,dt=dt,elapsed_s=time.perf_counter()-start,
                    nodes=len(flame.grid),width=float(flame.grid[-1]-flame.grid[0]),
                    energy=flame.energy_enabled,transport=flame.transport_model,soret=flame.soret_enabled))
                return 0.
            flame.set_time_step_callback(lambda dt:callback('transient_accepted',dt))
            flame.set_steady_callback(lambda dt:callback('steady_accepted',dt))
            kw['loglevel']=2
            return solve(flame,*a,**kw)
        ct.FreeFlame.solve=traced_cantera
    n=0
    def measured(*a,**kw):
        nonlocal n
        n+=1
        state.update(active=n>1,stack=[],calls=Counter(),seconds=Counter(),phases=[],events=[])
        fields,result=original(*a,**kw)
        if state['active']:
            result['instrumentation']=dict(diagnostic_only=True,phases=state['phases'],
                cantera_events=state['events'],function_calls=dict(state['calls']),
                exclusive_root_seconds=dict(state['seconds']),
                bucket_definition='Root-owned timers: Jacobian includes nested residual calls; no overlap among these buckets. Native profile subtimers are inclusive.',
                counter_definition='Native residual counter counts full residual calls including guards; Cantera eval_count_stats excludes finite-difference Jacobian evaluations. Callback counts are accepted steps, not attempts.')
        state['active']=False
        return fields,result
    runner.solve=measured


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3'))
    ap.add_argument('--output', type=Path,
                    default=Path('TESIS_RESUL/corridas/reproduccion/thesis_flames_L3_diagnostics'))
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--worker',type=Path)
    args=ap.parse_args(argv)
    if args.worker:
        for k,v in runner.THREADS.items():os.environ[k]=v
        req=runner.read(args.worker)
        instrument(req['backend'])
        return runner.worker(args.worker)
    root=args.input.resolve();out=args.output.resolve()
    source=runner.read(root/'manifest.json')
    planned=selection(source)
    manifest=dict(diagnostic_only=True,parent_manifest_sha256=runner.digest(root/'manifest.json'),
        parent=str(root),environment=runner.environment(),instrument_sha256=runner.digest(Path(__file__)),
        cases=planned,protocol='One excluded warmup and one instrumented solve each. Timings not added to production sample.',
        settings=runner.settings(3,max_seconds=source['max_seconds']))
    if manifest['environment']!=source['environment']:
        raise ValueError('Current solver code/environment differs from the completed campaign')
    out.mkdir(parents=True,exist_ok=True)
    with runner.campaign_lock(out):
        if (out/'manifest.json').exists():
            if not args.resume or runner.read(out/'manifest.json')!=manifest:
                raise ValueError('Use --resume with identical diagnostic code/configuration, or a new output')
        else:
            if any(p.name!='.campaign.lock' for p in out.iterdir()):raise ValueError('Nonempty output')
            shutil.copytree(root/'inputs',out/'inputs')
            shutil.copy2(Path(__file__),out/'diagnostic_source.py')
            runner.atomic_json(out/'manifest.json',manifest)
        records=[]
        for i,c in enumerate(planned):
            order=['native','cantera'] if i%2==0 else ['cantera','native']
            for backend in order:
                folder=out/c['id']/backend;folder.mkdir(parents=True,exist_ok=True)
                path=folder/'result.json'
                if path.exists():
                    old=runner.read(path)
                    if old.get('diagnostic_committed'):
                        records.append(old);continue
                    archive=folder/f'interrupted-{time.time_ns()}'
                    archive.mkdir()
                    for f in list(folder.iterdir()):
                        if f.is_file():shutil.copy2(f,archive/f.name)
                runner.atomic_json(folder/'request.json',dict(case=c,settings=manifest['settings'],
                    backend=backend,variant=backend,mechanism=str(out/'inputs'/c['mechanism'])))
                print(f'Diagnostic {i+1}/{len(planned)}: {c["id"]} / {backend}',flush=True)
                start=time.perf_counter()
                with (folder/'console.log').open('w',encoding='utf-8') as log:
                    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(folder/'request.json')],
                        env=dict(os.environ,**runner.THREADS),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    try:code=process.wait(timeout=2*source['max_seconds']+120)
                    except (KeyboardInterrupt,subprocess.TimeoutExpired):
                        process.kill();process.wait();raise
                result=runner.read(path) if path.exists() else dict(accepted=False,status='worker_failed')
                result.update(diagnostic_committed=True,diagnostic_only=True,returncode=code,
                    process_s=time.perf_counter()-start,folder=str(folder.relative_to(out)),
                    usable=bool(result.get('accepted') and result.get('diagnostics_complete')))
                if backend=='cantera':
                    log=(folder/'console.log').read_text(encoding='utf-8')
                    result['log_counts']=dict(newton_attempts=len(re.findall(r'Attempt Newton solution',log)),
                        newton_successes=len(re.findall(r'Newton steady-state solve succeeded',log)),
                        newton_failures=len(re.findall(r'Newton steady-state solve failed',log)))
                runner.atomic_json(path,result);records.append(result)
                runner.atomic_json(out/'index.json',records)
                print(f'  {result.get("status")}; instrumented seconds={result.get("time_s")}',flush=True)
        runner.atomic_json(out/'index.json',records)
    return 0 if all(r['usable'] for r in records) else 2


if __name__=='__main__':raise SystemExit(main())
