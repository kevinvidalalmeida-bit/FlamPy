"""Sparsity of the chemistry Jacobian and LU fill-in of a node block, per mechanism.

Run from the repository root (no flame is solved):

    python examples/audit_chemistry_sparsity.py gri30.yaml jetsurf2.yaml aramcomech3.yaml

d omega/d C splits into the sparse stoichiometric part (species of a
reaction x its reactants, and products when reversible) and a low-rank part:
third-body sums (a column of ones plus a few collider columns), density and
temperature. The script reports the stoichiometric density, the collider
columns, and the fill-in of a sparse LU of that pattern (plus a dominant
diagonal) for three orderings, against a dense inverse of the same size.
It informs the sparse + low-rank pivot study of docs/estado-mecanismos.md.
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism


def audit(name):
    mech = load_mechanism(resolve_mechanism(name))
    ns = mech.n_species
    pattern = np.zeros((ns, ns), bool)
    colliders = set()
    third = 0
    for rx in mech.reactions:
        rows = np.flatnonzero(rx.net_stoich)
        cols = list(rx.reactant_indices)+(list(rx.product_indices) if rx.reversible else [])
        for j in cols:
            pattern[rows, j] = True
        if rx.efficiencies is not None:
            third += 1
            colliders |= set(np.flatnonzero(np.asarray(rx.efficiencies) != 1.))
    np.fill_diagonal(pattern, True)
    rng = np.random.default_rng(0)
    A = sp.csc_matrix(np.where(pattern, rng.standard_normal((ns, ns)), 0.)+np.diag(ns*np.ones(ns)))
    print(f'{name}: {ns} species, {mech.n_reactions} reactions, {third} third-body/falloff, '
          f'{len(colliders)} collider columns; stoichiometric density {100*pattern.mean():.1f} %')
    for spec in ('MMD_AT_PLUS_A', 'COLAMD', 'NATURAL'):
        tic = time.perf_counter()
        lu = splu(A, permc_spec=spec, diag_pivot_thresh=0.1)
        elapsed = time.perf_counter()-tic
        print(f'  sparse LU {spec:14s}: fill {100*(lu.L.nnz+lu.U.nnz)/ns**2:5.1f} % of dense, {1e3*elapsed:6.2f} ms')
    dense = A.toarray()
    tic = time.perf_counter()
    for _ in range(3):
        np.linalg.inv(dense)
    print(f'  dense inverse: {1e3*(time.perf_counter()-tic)/3:.2f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mechanisms', nargs='+')
    for name in parser.parse_args().mechanisms:
        audit(name)


if __name__ == '__main__':
    main()
