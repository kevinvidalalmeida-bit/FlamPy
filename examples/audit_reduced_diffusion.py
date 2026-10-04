"""Audit principal diffusion at every logical cell centre of a frozen table.

This is a finite sample audit, not a proof throughout each cell. Parabolicity
is also checked at every node of each production solution. The audit detects
an unsuitable reactive coordinate independently of withheld burner profiles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import ReducedBurnerProblem


def audit(table, output, interpolation='bounded_tensor'):
    model = NonAdiabaticFGM(table)
    shape = np.asarray(model.table['structured_shape'])
    axes = [(np.arange(n-1)+.5)/(n-1) for n in shape]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    mech = load_mechanism(model.metadata['mechanism'])
    feed = fresh_mixture(mech, 1., model.metadata['fuel'], model.metadata['oxidizer'])
    reports, minimum = [], np.inf
    for start in range(0, len(points), 512):
        x = points[start:start+512]
        problem = ReducedBurnerProblem(model, np.linspace(0., .03, len(x)), .1, feed,
                                        interpolation=interpolation)
        entry = problem.diffusion_diagnostics(x, interior_only=False)
        entry['first_sample'] = start
        reports.append(entry)
        minimum = min(minimum, entry.get('minimum_real_eigenvalue_kg_m_s', -np.inf))
        if start % 8192 == 0:
            print('DIFFUSION', start, len(points), minimum, flush=True)
    result = dict(table_sha256=model.table_sha256, interpolation=interpolation,
                  solver_source_sha256=hashlib.sha256(Path('src/kflame/fgm/reduced_burner.py').read_bytes()).hexdigest(),
                  samples=len(points), sample_definition='every_logical_cell_centre',
                  passed=all(r['passed'] for r in reports),
                  minimum_real_eigenvalue_kg_m_s=minimum,
                  negative_samples=sum(r.get('negative_nodes', 0) for r in reports), batches=reports,
                  limitation='Finite sample audit; does not prove positivity inside every cell.')
    Path(output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print({k:v for k,v in result.items() if k!='batches'}, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interpolation', default='bounded_tensor')
    a = p.parse_args()
    audit(a.table, a.output, a.interpolation)
