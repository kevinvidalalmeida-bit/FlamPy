"""Nonadiabatic premixed manifold with physical controls (Z, C, h).

Z is local Bilger mixture fraction, C is one unscaled species combination,
and h is total specific enthalpy. Structured tetrahedra preserve adjacency
of inlet compositions, burner mass fluxes, and flamelet samples.
"""
import itertools
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from kflame.chemistry.initialization import _composition, oxygen_demand
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.thermo import NativeThermo


class OutsideManifoldError(ValueError):
    """The requested controls lie outside the sampled valid cells."""


class AmbiguousManifoldError(ValueError):
    """Multiple nonadjacent cells map to the same physical controls."""


def bilger_coefficients(mech, fuel, oxidizer):
    """Affine Z = coefficients @ Y + offset; streams specified in mole basis.

    The factor two in the traditional Bilger beta cancels on normalization.
    Evaluate the local fractions without clipping or substituting inlet Z.
    """
    beta = oxygen_demand(mech) * mech.inv_molecular_weights
    values = []
    for text in (fuel, oxidizer):
        amounts = _composition(text, mech.species_names) * mech.molecular_weights
        values.append(float(beta @ (amounts / amounts.sum())))
    span = values[0] - values[1]
    if span <= 0.:
        raise ValueError('Fuel and oxidizer must define a positive Bilger span')
    return beta / span, -values[1] / span


def _trajectory_samples(profile, weights, samples):
    progress = weights @ profile['Y']
    span = float(progress[-1] - progress[0])
    if span <= 1e-10 or np.any(np.diff(progress) < -1e-6 * span):
        raise ValueError('Progress is degenerate or nonmonotone; revise species weights')
    keep = []
    for j, value in enumerate(progress):
        if keep and value <= progress[keep[-1]] + 1e-12 * span:
            # Preserve the physical inlet even on a near-flat initial plateau;
            # the burned plateau keeps its final node instead.
            if len(keep) > 1:
                keep[-1] = j
        else:
            keep.append(j)
    x = progress[keep]
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError('Progress cannot parameterize this flamelet')
    # s is a sampling coordinate only. The lookup control C is never normalized
    # separately for each flamelet, and the actual endpoints remain in C space.
    target = x[0] + samples * (x[-1] - x[0])
    y = np.array([np.interp(target, x, species[keep]) for species in profile['Y']])
    T = np.interp(target, x, profile['T'][keep])
    conductivity = np.interp(target, x, profile['conductivity'][keep])
    return y, T, conductivity


class SimplexMesh:
    """Locate points on an explicitly connected mesh, without a convex hull fill."""
    def __init__(self, points, cells):
        self.points = np.asarray(points, dtype=float)
        self.cells = np.asarray(cells, dtype=int)
        self.offset = self.points.min(axis=0)
        self.scale = np.ptp(self.points, axis=0)
        if np.any(self.scale <= 0):
            raise ValueError('Degenerate manifold control range')
        vertices = (self.points[self.cells] - self.offset) / self.scale
        self.origin = vertices[:, 0]
        matrix = (vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)
        determinant = np.linalg.det(matrix)
        usable = abs(determinant) > 1e-13
        self.cells = self.cells[usable]
        vertices = vertices[usable]
        self.origin = self.origin[usable]
        self.inverse = np.linalg.inv(matrix[usable])
        self.lower, self.upper = vertices.min(axis=1), vertices.max(axis=1)
        if not self.cells.size:
            raise ValueError('No nondegenerate cells in the manifold')
        self.centers = vertices.mean(axis=1)
        self.radii = np.max(np.linalg.norm(vertices - self.centers[:, None], axis=2), axis=1)
        self.tree = cKDTree(self.centers)
        self.search_radius = float(np.max(self.radii))

    def locate(self, controls):
        point = (np.asarray(controls, dtype=float) - self.offset) / self.scale
        if not np.isfinite(point).all():
            raise ValueError('Manifold controls must be finite')
        # A simplex lies inside its vertex bounding sphere. This search is
        # complete, unlike selecting a fixed number of nearest centroids.
        possible = np.asarray(self.tree.query_ball_point(point, self.search_radius + 1e-9), dtype=int)
        possible = possible[np.all(point >= self.lower[possible] - 1e-10, axis=1)
                            & np.all(point <= self.upper[possible] + 1e-10, axis=1)]
        if not possible.size:
            raise OutsideManifoldError('Controls outside the sampled manifold')
        last = np.einsum('nij,nj->ni', self.inverse[possible], point - self.origin[possible])
        barycentric = np.column_stack((1. - last.sum(axis=1), last))
        inside = np.flatnonzero(np.all(barycentric >= -1e-9, axis=1)
                               & np.all(barycentric <= 1. + 1e-9, axis=1))
        if not inside.size:
            raise OutsideManifoldError('Controls outside adjacent resolved flamelets')
        interior = inside[np.all(barycentric[inside] > 1e-7, axis=1)]
        if interior.size > 1:
            raise AmbiguousManifoldError('Nonadjacent flamelet cells overlap in control space')
        selected = int(interior[0] if interior.size else inside[0])
        return self.cells[possible[selected]], barycentric[selected]


def build_nonadiabatic_table(folder, *, progress_points=181):
    """Build connected cells from accepted profiles listed in generation.json."""
    folder = Path(folder)
    if not isinstance(progress_points, int) or progress_points < 3:
        raise ValueError('progress_points must be an integer >= 3')
    generation = json.loads((folder / 'generation.json').read_text(encoding='utf-8'))
    if not generation['all_final_accepted']:
        raise ValueError('All training flames must be accepted before tabulation')
    mech = load_mechanism(generation['mechanism'])
    thermo, kinetics = NativeThermo(mech), NativeKinetics(mech)
    weights = np.array([generation['progress_species'].get(name, 0.) for name in mech.species_names])
    z_weights, z_offset = bilger_coefficients(mech, generation['fuel'], generation['oxidizer'])
    samples = .5 * (1. - np.cos(np.linspace(0., np.pi, progress_points)))
    n_phi, n_loss = len(generation['phis']), len(generation['mass_flux_fractions']) + 1
    shape = (n_phi, n_loss, progress_points)
    expected = set(itertools.product(range(n_phi), range(n_loss)))
    recorded = [(r['composition_index'], r['loss_index']) for r in generation['rows']]
    if len(recorded) != len(expected) or set(recorded) != expected:
        raise ValueError('Training rows must contain each composition/loss pair exactly once')
    fields = {name: np.empty(shape) for name in ('Z', 'C', 'h', 'T', 'rho', 'cp_mass', 'conductivity', 'omega_C', 'qdot')}
    fields['Y'] = np.empty((*shape, mech.n_species))
    profile_hashes = {}
    for record in generation['rows']:
        i, j = record['composition_index'], record['loss_index']
        profile_path = folder / record['output'] / 'flame.npz'
        flame_meta = json.loads((profile_path.parent / 'metadata.json').read_text(encoding='utf-8'))
        if (not flame_meta.get('accepted') or flame_meta.get('backend') != 'native_cpu'
                or not flame_meta['report'].get('grid_converged')):
            raise ValueError(f'Training profile is not accepted on a converged grid: {record["output"]}')
        if (j and (flame_meta['heat_loss']['relative_energy_closure_error'] > generation['max_energy_error'])):
            raise ValueError(f'Training profile exceeds energy closure tolerance: {record["output"]}')
        profile_hashes[record['output']] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        with np.load(profile_path, allow_pickle=False) as profile:
            if list(profile['species_names']) != list(mech.species_names):
                raise ValueError('Training profile and mechanism species order differ')
            y, T, conductivity = _trajectory_samples(profile, weights, samples)
        if j == 0:
            y[:, 0] = record['inlet_Y']
            T[0] = generation['temperature_K']
        rho = thermo.density(T, generation['pressure_Pa'], y)
        h = thermo.enthalpy_mass(T, y)
        concentrations = rho[None, :] * y * mech.inv_molecular_weights[:, None]
        omega = kinetics.net_production_rates(T, concentrations, thermo.g_RT(T))
        fields['Y'][i, j], fields['T'][i, j] = y.T, T
        fields['Z'][i, j] = z_weights @ y + z_offset
        fields['C'][i, j] = weights @ y
        fields['h'][i, j], fields['rho'][i, j] = h, rho
        fields['cp_mass'][i, j] = thermo.cp_mass(T, y)
        fields['conductivity'][i, j] = conductivity
        fields['omega_C'][i, j] = weights @ (omega * mech.molecular_weights[:, None])
        fields['qdot'][i, j] = -np.sum(thermo.partial_molar_enthalpies(T) * omega, axis=0)
    if any(not np.isfinite(value).all() for value in fields.values()):
        raise ValueError('Nonfinite property in resolved flamelet samples')
    controls = np.column_stack([fields[name].ravel() for name in ('Z', 'C', 'h')])
    indices = np.arange(np.prod(shape)).reshape(shape)
    cells, orientation = [], []
    for i, j, k in itertools.product(range(n_phi - 1), range(n_loss - 1), range(progress_points - 1)):
        base = np.array([i, j, k])
        for permutation in itertools.permutations(range(3)):
            corners = [base.copy()]
            for axis in permutation:
                corner = corners[-1].copy()
                corner[axis] += 1
                corners.append(corner)
            cell = [int(indices[tuple(corner)]) for corner in corners]
            cells.append(cell)
            orientation.append(np.linalg.det((np.array(corners[1:]) - corners[0]).T))
    cells = np.asarray(cells, dtype=int)
    scaled = (controls - controls.min(axis=0)) / np.ptp(controls, axis=0)
    matrix = (scaled[cells[:, 1:]] - scaled[cells[:, :1]]).transpose(0, 2, 1)
    signed = np.linalg.det(matrix) / np.asarray(orientation)
    folded = signed < -1e-13
    degenerate = abs(signed) <= 1e-13
    # A reversed cell is excluded and reported, never silently used for lookup.
    # Remaining nonadjacent overlaps are rejected at query time.
    cells = cells[~folded & ~degenerate]
    reference_points = np.column_stack((fields['Z'][:, 0].ravel(), fields['C'][:, 0].ravel()))
    reference_indices = np.arange(n_phi * progress_points).reshape(n_phi, progress_points)
    triangles = []
    for i, k in itertools.product(range(n_phi - 1), range(progress_points - 1)):
        a, b = reference_indices[i, k], reference_indices[i + 1, k]
        c, d = reference_indices[i, k + 1], reference_indices[i + 1, k + 1]
        triangles.extend([(a, b, d), (a, d, c)])
    payload = {name: value.reshape((-1, mech.n_species)) if name == 'Y' else value.ravel()
               for name, value in fields.items()}
    payload.update(controls=controls, cells=cells, progress_weights=weights,
                   bilger_weights=z_weights, bilger_offset=np.array(z_offset),
                   species_names=np.asarray(mech.species_names), structured_shape=np.array(shape),
                   reference_points=reference_points, reference_cells=np.asarray(triangles, dtype=int),
                   reference_h=fields['h'][:, 0].ravel(), sampling_coordinate=samples)
    np.savez_compressed(folder / 'nonadiabatic_fgm.npz', **payload)
    metadata = dict(generation, format='FlamPy_nonadiabatic_fgm_v1', controls=['Z', 'C', 'h'],
                    Z_definition='local_Bilger_with_mole_basis_streams_no_clipping',
                    C_definition='unscaled_common_weighted_species_mass_fractions',
                    h_definition='total_sensible_plus_formation_J_kg',
                    source_units=dict(omega_C='kg/(m^3 s)', qdot='W/m^3'), progress_points=progress_points,
                    training_profile_sha256=profile_hashes,
                    mechanism_sha256=hashlib.sha256(Path(resolve_mechanism(generation['mechanism'])).read_bytes()).hexdigest(),
                    mesh=dict(vertices=len(controls), tetrahedra=len(cells),
                              excluded_folded_cells=int(folded.sum()), excluded_degenerate_cells=int(degenerate.sum())),
                    interpolation='barycentric_on_adjacent_flamelets_with_enthalpy_temperature_recovery',
                    limitations='Steady planar burner library; no conjugate solid heat transfer, radiation or transient wall quenching.')
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8', newline='\n')
    return folder


class NonAdiabaticFGM:
    """Query local Bilger Z, unscaled progress C, and total enthalpy h [J/kg]."""
    def __init__(self, folder):
        folder = Path(folder)
        self.metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        with np.load(folder / 'nonadiabatic_fgm.npz', allow_pickle=False) as saved:
            self.table = {name: saved[name] for name in saved.files}
        mech = load_mechanism(self.metadata['mechanism'])
        fingerprint = self.metadata.get('mechanism_sha256')
        if fingerprint and fingerprint != hashlib.sha256(Path(resolve_mechanism(self.metadata['mechanism'])).read_bytes()).hexdigest():
            raise ValueError('Table and current mechanism fingerprints differ')
        if list(self.table['species_names']) != mech.species_names:
            raise ValueError('Mechanism and table species differ')
        self.thermo = NativeThermo(mech)
        self.max_temperature = 2. * mech.max_temperature
        self.mesh = SimplexMesh(self.table['controls'], self.table['cells'])
        self.reference_mesh = SimplexMesh(self.table['reference_points'], self.table['reference_cells'])

    def reference_enthalpy(self, *, Z, C):
        nodes, barycentric = self.reference_mesh.locate([Z, C])
        return float(barycentric @ self.table['reference_h'][nodes])

    def lookup(self, *, Z, C, h):
        nodes, barycentric = self.mesh.locate([Z, C, h])
        Y = barycentric @ self.table['Y'][nodes]
        T = brentq(lambda temperature: float(self.thermo.enthalpy_mass(temperature, Y)) - h,
                   200., self.max_temperature, xtol=1e-8)
        try:
            delta_h = self.reference_enthalpy(Z=Z, C=C) - h
        except OutsideManifoldError:
            delta_h = None
        pressure = self.metadata['pressure_Pa']
        return dict(Z=float(Z), C=float(C), h=float(h), delta_h=delta_h, T=T, Y=Y,
                    rho=float(self.thermo.density(T, pressure, Y)),
                    cp_mass=float(self.thermo.cp_mass(T, Y)),
                    **{name: float(barycentric @ self.table[name][nodes])
                       for name in ('omega_C', 'qdot', 'conductivity')})
