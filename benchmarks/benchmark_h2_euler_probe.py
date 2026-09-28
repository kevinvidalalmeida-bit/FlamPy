"""One-case diagnostic: native PTC-SER, native direct BE, and Cantera.

The worker selects the internal pseudo_time_method option. Production defaults
and scientific campaigns stay unchanged; source and effective options are archived. No timing confidence intervals are inferred from a single run.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sweeps',ROOT/'benchmarks/benchmark_flame_sweeps.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def install_variant(folder, variant):
    import kflame.flame.solver as solver
    source=inspect.getsource(solver._hybrid_newton)
    (folder/'hybrid_original.py').write_text(source,encoding='utf-8')
    if variant=='euler':
        runner.STRATEGIES['h2o2.yaml']='backward-euler'
    else:
        runner.STRATEGIES['h2o2.yaml']='ptc-ser'
    original_newton=solver.newton_solve
    counters={}
    def counted(*args,**kwargs):
        if kwargs.get('rdt',0):
            key='ptc' if kwargs.get('residual_damping',False) else 'backward_euler'
            if variant=='euler' and key=='ptc':
                raise AssertionError('PTC executed in BE-only experiment')
        else:
            key='steady_newton'
        counters[key]=counters.get(key,0)+1
        return original_newton(*args,**kwargs)
    solver.newton_solve=counted
    original_solve=runner.solve
    def solve(*args,**kwargs):
        counters.clear()
        fields,result=original_solve(*args,**kwargs)
        result['probe_newton_calls']=dict(counters)
        result['probe_strategy']=variant
        return fields,result
    runner.solve=solve


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/h2_euler_probe_L4_p10'))
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args(argv)
    if args.worker:
        for k,v in runner.THREADS.items():os.environ[k]=v
        request=runner.read(args.worker)
        if request['backend']=='native':install_variant(args.worker.parent,request['variant'])
        return runner.worker(args.worker)
    out=args.output.resolve()
    if out.exists():parser.error('Use a new diagnostic output directory')
    out.mkdir(parents=True)
    (out/'inputs').mkdir()
    for name in ('h2o2.yaml','collision_integrals_mm.json'):
        shutil.copy2(ROOT/'src/kflame/chemistry/data'/name,out/'inputs'/name)
    shutil.copy2(ROOT/'src/kflame/flame/solver.py',out/'inputs/solver_original.py')
    shutil.copy2(Path(__file__),out/'inputs/benchmark_h2_euler_probe.py')
    c=runner.case('H2',pressure=10)
    s=runner.settings(4,max_seconds=60.)
    manifest=dict(kind='isolated single-observation diagnostic',case=c,settings=s,
        environment=runner.environment(),probe_sha256=runner.digest(Path(__file__)),
        order=['native','euler','cantera'],measured_repetitions=1,
        timing='Warmup excluded; cold profile; cleared molecular caches; full construction and solve. Lightweight Newton call counter on both native variants.',
        interpretation='Diagnostic only; not five-pair scientific campaign or statistical speedup evidence')
    runner.atomic_json(out/'manifest.json',manifest)
    records=[]
    for variant in manifest['order']:
        folder=out/variant; folder.mkdir()
        backend='cantera' if variant=='cantera' else 'native'
        request=dict(case=c,settings=s,backend=backend,variant=variant,
                     mechanism=str(out/'inputs/h2o2.yaml'))
        runner.atomic_json(folder/'request.json',request)
        print(f'One measured solve + warmup: {variant}',flush=True)
        started=time.perf_counter()
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(folder/'request.json')],
                cwd=ROOT,env=dict(os.environ,**runner.THREADS),stdout=log,stderr=subprocess.STDOUT)
            try:
                code=process.wait(timeout=240)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                process.kill();process.wait()
                runner.atomic_json(folder/'interruption.json',dict(status='interrupted_or_timeout',process_s=time.perf_counter()-started))
                raise
        path=folder/'result.json'
        result=runner.read(path) if path.exists() else dict(status='worker_failed',accepted=False)
        result.update(returncode=code,folder=variant,process_s=time.perf_counter()-started,
                      usable=bool(result.get('accepted') and result.get('diagnostics_complete')))
        runner.atomic_json(path,result);records.append(result)
        print({k:result.get(k) for k in ('status','usable','time_s','Su','nodes','probe_newton_calls')},flush=True)
        runner.atomic_json(out/'index.json',records)
    sys.path.insert(0,str(ROOT/'tools'))
    from postprocess_flame_sweeps import comparison
    by_name={r['folder']:r for r in records}
    comparisons={}
    for left,right in [('native','euler'),('euler','cantera'),('native','cantera')]:
        if by_name[left]['usable'] and by_name[right]['usable']:
            comparisons[f'{left}_vs_{right}']=dict(
                time_ratio_left_over_right=by_name[left]['time_s']/by_name[right]['time_s'],
                **comparison(out,by_name[left],by_name[right]))
    runner.atomic_json(out/'summary.json',dict(manifest=manifest,
        results=[{k:r.get(k) for k in ('folder','status','usable','time_s','Su','nodes','width','probe_newton_calls')} for r in records],
        comparisons=comparisons))
    print(f'Diagnostic saved: {out}',flush=True)
    return 0 if all(r['usable'] for r in records) else 2


if __name__=='__main__':
    raise SystemExit(main())
