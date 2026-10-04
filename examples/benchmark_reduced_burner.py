"""Measure colored thermodynamic-cache savings on an identical frozen problem."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import ReducedBurnerProblem
from examples.validate_reduced_burner import read_profile, write_json


def benchmark(table, case, output):
    start=time.perf_counter();model=NonAdiabaticFGM(table);load=time.perf_counter()-start
    case=Path(case);profile=read_profile(case/'reduced.npz');comparison=read_profile(case/'comparison.npz')
    problem=ReducedBurnerProblem(model,profile['z'],float(comparison['mass_flux']),comparison['feed_Y'])
    x=profile['logical_coordinates'].ravel();state=problem.state(x);perturbations=[]
    for color in range(9):
        selected=np.arange(color,len(x),9);trial=x.copy();trial[selected]+=1e-7
        perturbations.append((selected//3,trial))
    def evaluate(cached):
        return [problem.residual(trial,geometry=state['geometry'],
                     temperature_cache=state['T'] if cached else None,
                     changed_nodes=nodes if cached else None) for nodes,trial in perturbations]
    full,cached=evaluate(False),evaluate(True)
    error=float(max(np.max(abs(a-b)) for a,b in zip(full,cached)))
    if error>1e-9:raise RuntimeError('Cache changed the colored residuals')
    timings={}
    for reuse in [False,True]:
        values=[]
        for _ in range(9):
            started=time.perf_counter();evaluate(reuse);values.append(time.perf_counter()-started)
        timings['cached' if reuse else 'full']=dict(median_seconds=float(np.median(values)),samples_seconds=values)
    reports=[]
    for path in case.parent.glob('case_*/reduced_report.json'):
        reports.append(json.loads(path.read_text(encoding='utf-8')))
    accepted=[r for r in reports if r['accepted']]
    result=dict(table_sha256=model.table_sha256,python=platform.python_version(),platform=platform.system(),
        nodes=len(profile['z']),colors=9,dense_finite_difference_columns=len(x),
        table_loading_seconds=load,cache_maximum_residual_difference=error, timings=timings,
        cache_speedup=timings['full']['median_seconds']/timings['cached']['median_seconds'],
        accepted_case_count=len(accepted),failed_case_count=len(reports)-len(accepted),
        solve_median_seconds=float(np.median([r['elapsed_seconds'] for r in accepted])),
        solve_range_seconds=[min(r['elapsed_seconds'] for r in accepted),max(r['elapsed_seconds'] for r in accepted)],
        detailed_kinetics_evaluations_in_reduced_production=0,
        note='Single-machine timing; table load, construction and detailed validation excluded from solve times.')
    write_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--table',type=Path,required=True);parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(benchmark(args.table,args.case,args.output))
