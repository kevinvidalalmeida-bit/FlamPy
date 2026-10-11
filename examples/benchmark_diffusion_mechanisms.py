"""Complete native counterflow flames for every installed mechanism.

Each (mechanism, mode) runs in its own process, so memory and JIT state do
not leak between cases; JIT compilation is excluded from the solver time.

Modes
  new      production defaults: analytic structured Jacobian, structured
           factorization, vectorized faces and (large mechanisms) Newton-Krylov
  legacy   column-wise finite-difference Jacobian with dense blocks and the
           scalar face kernels (the previous production path)
  split4   ``new`` plus multi-point grid refinement (refine_split=4)

python examples/benchmark_diffusion_mechanisms.py --output runs/diffusion_mechanisms.json
python examples/benchmark_diffusion_mechanisms.py --mechanisms gri30.yaml --modes new legacy --cantera
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MODES = {
    'new': {},
    'legacy': dict(structured_jacobian=False, fast_transport=False, newton_krylov=False),
    'split4': dict(refine_split=4),
}

CHILD = r'''
import json, sys, time
import numpy as np
import kflame
from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion as D
spec = json.loads(sys.argv[1])
mechanism = resolve_mechanism(spec['mechanism'])
mech = load_mechanism(mechanism)
fuel = 'CH4:1' if 'CH4' in mech.species_names else 'H2:1'
feeds = []
for text in (fuel, 'O2:1,N2:3.76'):
    amounts = _composition(text, mech.species_names)*mech.molecular_weights
    feeds.append(amounts/amounts.sum())
case = D.DiffusionCase(mechanism, *feeds, transport_model=spec['transport'], soret_enabled=spec['soret'])
problem = D.DiffusionFlameProblem(case, mech, np.linspace(0., case.width, 31))
for key, value in spec['options'].items():
    setattr(problem, key, value)
fields, accepted, report = D.solve_diffusion_problem(problem, max_time=spec['max_time'], ratio=3., slope=.01,
                                                     curve=.02, max_points=2400)
out = dict(accepted=bool(accepted), reason=report['reason'], solver_s=report['runtime_s'], nodes=int(problem.n),
           levels=[h['nodes'] for h in report['history']], jacobians=int(getattr(problem, 'jacobian_evaluations', 0)),
           Tmax=float(fields['T'].max()), residual=report['maximum_scaled_residual'])
if spec.get('save'):
    np.savez_compressed(spec['save'], z=fields['z'], T=fields['T'], Y=fields['Y'])
print('RESULT '+json.dumps(out))
'''


def run(spec, timeout):
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1')
    try:
        done = subprocess.run([sys.executable, '-c', CHILD, json.dumps(spec)], capture_output=True, text=True,
                              timeout=timeout, env=environment)
    except subprocess.TimeoutExpired:
        return dict(status='timeout', timeout_s=timeout)
    for line in done.stdout.splitlines():
        if line.startswith('RESULT '):
            return dict(status='ok', **json.loads(line[7:]))
    return dict(status='failed', returncode=done.returncode, stderr=done.stderr[-2000:])


def main():
    from kflame.chemistry.catalog import available_mechanisms
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mechanisms', nargs='+', help='Catalog filenames; default: all installed')
    parser.add_argument('--modes', nargs='+', default=['new', 'legacy'], choices=sorted(MODES))
    parser.add_argument('--transport', default='mixture-averaged', choices=['mixture-averaged', 'multicomponent'])
    parser.add_argument('--soret', action='store_true')
    parser.add_argument('--max-time', type=float, default=1800.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {args.output}')
    entries = [e for e in available_mechanisms() if e['installed']]
    if args.mechanisms:
        entries = [e for e in entries if e['filename'] in args.mechanisms]
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), transport=args.transport, soret=args.soret,
                  settings=dict(initial_points=31, ratio=3., slope=.01, curve=.02, max_points=2400,
                                fuel='CH4 (H2 for h2o2)', oxidizer='air', pressure_Pa=101325.),
                  numba_threads=os.environ.get('NUMBA_NUM_THREADS', 'kflame default'), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='kflame-bench-'))
    for entry in entries:
        for mode in args.modes:
            spec = dict(mechanism=entry['filename'], transport=args.transport, soret=args.soret,
                        options=MODES[mode], max_time=args.max_time,
                        save=str(folder/f"{Path(entry['filename']).stem}-{mode}.npz"))
            print(f"{entry['name']:16s} {mode:7s} ...", end=' ', flush=True)
            result = run(spec, timeout=args.max_time+600)
            print(result.get('solver_s', result['status']), flush=True)
            report['cases'].append(dict(name=entry['name'], filename=entry['filename'], species=entry['species'],
                                        reactions=entry['reactions'], mode=mode, **result))
            args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Profiles: {folder}')


if __name__ == '__main__':
    main()
