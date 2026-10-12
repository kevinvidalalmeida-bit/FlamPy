"""GMRES with the mixture-form preconditioner on a multicomponent counterflow state.

Run from the repository root:

    python examples/audit_multicomponent_krylov.py gri30.yaml --points 61
    python examples/audit_multicomponent_krylov.py jetsurf2.yaml --points 121
    python examples/audit_multicomponent_krylov.py jetsurf2.yaml --refine     # ~1500-node grid

Starts from the accepted mixture-averaged flame (as the multicomponent solve
does), builds the structured mixture-form Jacobian of the pruned problem and
reports: linearity of the matrix-free products (pruned vs complete kernel),
GMRES iterations for the first Newton step and how far J M^-1 is from the
identity. On 61-121 nodes GRI and JetSurF converge in 2 iterations; on the
refined JetSurF grid the solver stalled after 12 (open item in
docs/estado-mecanismos.md).
"""
import argparse
import time

import numpy as np

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame.diffusion import DiffusionCase, DiffusionFlameProblem, StructuredFactor, solve_diffusion_problem
from kflame.flame.krylov import gmres
from kflame.flame.structured_jacobian import structured_blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mechanism')
    parser.add_argument('--points', type=int, default=61)
    parser.add_argument('--refine', action='store_true', help='Use the adaptive grid of the mixture flame')
    parser.add_argument('--iterations', type=int, nargs='+', default=[12, 60])
    args = parser.parse_args()
    path = resolve_mechanism(args.mechanism)
    mech = load_mechanism(path)
    feeds = []
    for text in ('CH4', 'O2:1,N2:3.76'):
        amounts = _composition(text, mech.species_names)*mech.molecular_weights
        feeds.append(amounts/amounts.sum())
    mixture = DiffusionFlameProblem(DiffusionCase(path, *feeds), mech, np.linspace(0., .02, args.points))
    fields, ok, _ = solve_diffusion_problem(mixture, refine=args.refine, max_time=900., slope=.01, curve=.02)
    z = fields['z']
    print(f'mixture flame: accepted {ok}, {len(z)} nodes, Tmax {fields["T"].max():.2f} K', flush=True)

    def problem_with(tail):
        p = DiffusionFlameProblem(DiffusionCase(path, *feeds, transport_model='multicomponent',
                                                multicomponent_trace_tail=tail), mech, z)
        p.transport_bootstrap = False
        return p
    pruned, complete = problem_with(1e-8), problem_with(1e-300)
    x = pruned.pack({k: fields[k] for k in ('m', 'V', 'T', 'Y', 'Lambda')})
    F = pruned.residual(x)
    b = x[:-1].reshape(pruned.n, pruned.nv)
    scales = np.empty(len(x))
    scales[:-1].reshape(pruned.n, pruned.nv)[:] = 1e-9+1e-4*np.mean(abs(b), axis=0)
    scales[-1] = 1e-4*max(abs(x[-1]), 1.)

    def matvec(p, v, factor=1e-3):
        eps = factor/float(np.max(abs(v)/scales))
        return (p.residual(x+eps*v)-p.residual(x))/eps
    v = np.random.default_rng(0).standard_normal(len(x))*scales
    for p, label in ((pruned, 'pruned'), (complete, 'complete')):
        base = matvec(p, v)
        drift = [np.linalg.norm(matvec(p, v, f)-base)/np.linalg.norm(base) for f in (1e-2, 1e-4)]
        print(f'{label}: product drift with the step (1e-2, 1e-4): {drift[0]:.1e}, {drift[1]:.1e}', flush=True)
    tic = time.perf_counter()
    M = StructuredFactor(structured_blocks(pruned, x))
    print(f'mixture-form preconditioner: {time.perf_counter()-tic:.1f} s', flush=True)
    for maxit in args.iterations:
        count = [0]

        def A(w):
            count[0] += 1
            return matvec(pruned, w)
        step, converged = gmres(A, M.solve, -F, M.solve(-F), 1e-2, maxit)
        print(f'GMRES (rtol 1e-2, at most {maxit}): converged {converged} after {count[0]} products, '
              f'residual {np.linalg.norm(matvec(complete, step)+F)/np.linalg.norm(F):.2e}', flush=True)


if __name__ == '__main__':
    main()
