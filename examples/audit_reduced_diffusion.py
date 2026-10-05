"""Audit principal diffusion at every logical cell centre of a frozen table.

This is a finite sample audit, not a proof throughout each cell. Parabolicity
is also checked at every node of each production solution. The audit detects
an unsuitable reactive coordinate independently of withheld burner profiles.
"""
import argparse
import hashlib
import json
import itertools
from pathlib import Path
import numpy as np
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import ReducedBurnerProblem, _SOURCE_SHA256


def audit(table, output, interpolation='bounded_tensor',*,progress_equation='conservative',samples='centre',metric_floor=1e-30,curvature_limit=.75):
    model = NonAdiabaticFGM(table)
    shape = np.asarray(model.table['structured_shape'])
    cells=np.stack(np.meshgrid(*[np.arange(n-1) for n in shape],indexing='ij'),axis=-1).reshape(-1,3)
    offsets=[(.5,)*3]
    if samples=='centre_gauss':offsets+=list(itertools.product([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)],repeat=3))
    mech = load_mechanism(model.metadata['mechanism'])
    feed = fresh_mixture(mech, 1., model.metadata['fuel'], model.metadata['oxidizer'])
    reports, minimum = [], np.inf
    for probe,offset in enumerate(offsets):
        points=(cells+offset)/(shape-1)
        for start in range(0,len(points),2048):
            x=points[start:start+2048]
            problem=ReducedBurnerProblem(model,np.linspace(0.,.03,len(x)),.1,feed,interpolation=interpolation,
                progress_equation=progress_equation,metric_floor=metric_floor,curvature_limit=curvature_limit)
            entry=problem.diffusion_diagnostics(x,interior_only=False)
            entry.update(first_sample=probe*len(points)+start,probe_offset=list(offset))
            reports.append(entry);minimum=min(minimum,entry.get('minimum_real_eigenvalue_kg_m_s',-np.inf))
        print('DIFFUSION',probe+1,len(offsets),minimum,flush=True)
    result = dict(table_sha256=model.table_sha256, interpolation=interpolation,
                  solver_source_sha256=_SOURCE_SHA256,progress_equation=progress_equation,metric_floor=metric_floor,curvature_limit=curvature_limit,
                  samples=len(points)*len(offsets),logical_cells=len(points),sample_definition=samples,
                  passed=all(r['passed'] for r in reports),
                  minimum_real_eigenvalue_kg_m_s=minimum,
                  negative_samples=sum(r.get('negative_nodes', 0) for r in reports), batches=reports,
                  nonpositive_chart_samples=sum(r.get('nonpositive_chart_nodes',0) for r in reports),
                  projected_samples=sum(r.get('projected_nodes',0) for r in reports),
                  chart_guard=problem.limiter_statistics[-1] if interpolation=='bounded_tensor' else None,
                  limitation='Finite sample audit; does not prove positivity inside every cell.')
    Path(output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print({k:v for k,v in result.items() if k!='batches'}, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interpolation', default='bounded_tensor')
    p.add_argument('--progress-equation',default='conservative',choices=['conservative','adaptive_projection','entropy_projection'])
    p.add_argument('--samples',default='centre',choices=['centre','centre_gauss'])
    p.add_argument('--metric-floor',default=1e-30,type=float)
    p.add_argument('--curvature-limit',default=.75,type=float)
    a = p.parse_args()
    audit(a.table,a.output,a.interpolation,progress_equation=a.progress_equation,samples=a.samples,metric_floor=a.metric_floor,curvature_limit=a.curvature_limit)
