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

from kflame.chemistry.initialization import _composition, oxygen_demand
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.fgm.search import build_hierarchy, locate_many


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
        self.hierarchy = build_hierarchy(self.lower,self.upper)

    def locate(self, controls):
        nodes,weights,covered=self.locate_batch(np.asarray(controls,dtype=float)[None,:])
        if not covered[0]:
            raise OutsideManifoldError('Controls outside adjacent resolved flamelets')
        return nodes[0],weights[0]

    def locate_batch(self,controls):
        controls=np.asarray(controls,dtype=float)
        if controls.ndim!=2 or controls.shape[1]!=len(self.offset):
            raise ValueError('Manifold controls must have shape (states, dimensions)')
        points=np.ascontiguousarray((controls-self.offset)/self.scale)
        if not np.isfinite(points).all():
            raise ValueError('Manifold controls must be finite')
        selected,weights,status=locate_many(points,self.origin,self.inverse,self.lower,self.upper,*self.hierarchy)
        if np.any(status==2):
            raise AmbiguousManifoldError('Nonadjacent flamelet cells overlap in control space')
        covered=status==0
        nodes=np.full((len(points),self.cells.shape[1]),-1,dtype=int)
        nodes[covered]=self.cells[selected[covered]]
        return nodes,weights,covered


def _connected_cells(shape, controls):
    """Vectorized Freudenthal tetrahedra, with the original orientation check."""
    n_phi, n_loss, progress_points = shape
    indices = np.arange(np.prod(shape)).reshape(shape)
    bases = indices[:-1,:-1,:-1].ravel()
    strides = np.array([n_loss*progress_points,progress_points,1])
    offsets, orientation = [], []
    for permutation in itertools.permutations(range(3)):
        offsets.append(np.r_[0,np.cumsum(strides[list(permutation)])])
        corners = np.vstack((np.zeros(3),np.cumsum(np.eye(3)[list(permutation)],axis=0)))
        orientation.append(np.linalg.det((corners[1:]-corners[0]).T))
    cells = (bases[:,None,None]+np.asarray(offsets)[None,:,:]).reshape(-1,4)
    scaled = (controls-controls.min(axis=0))/np.ptp(controls,axis=0)
    matrix = (scaled[cells[:,1:]]-scaled[cells[:,:1]]).transpose(0,2,1)
    signed = np.linalg.det(matrix)/np.tile(orientation,len(bases))
    folded, degenerate = signed < -1e-13, abs(signed) <= 1e-13
    cells = cells[~folded & ~degenerate]
    ref = np.arange(n_phi*progress_points).reshape(n_phi,progress_points)
    a, b = ref[:-1,:-1].ravel(), ref[1:,:-1].ravel()
    c, d = ref[:-1,1:].ravel(), ref[1:,1:].ravel()
    triangles = np.stack((np.stack((a,b,d),axis=1),np.stack((a,d,c),axis=1)),axis=1).reshape(-1,3)
    return cells, triangles, dict(vertices=len(controls), tetrahedra=len(cells),
        excluded_folded_cells=int(folded.sum()), excluded_degenerate_cells=int(degenerate.sum()))


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
    cells, triangles, mesh_statistics = _connected_cells(shape, controls)
    reference_points = np.column_stack((fields['Z'][:, 0].ravel(), fields['C'][:, 0].ravel()))
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
                    mesh=mesh_statistics,
                    interpolation='barycentric_on_adjacent_flamelets_with_enthalpy_temperature_recovery',
                    limitations='Steady planar burner library; no conjugate solid heat transfer, radiation or transient wall quenching.')
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8', newline='\n')
    return folder


def subset_nonadiabatic_table(source, output, *, phis, mass_flux_fractions):
    """Reconstruct a coordinate subset from stored native vertex fields.

    No flame is solved and no property is interpolated. Connectivity is
    rebuilt between the retained profiles and rechecked for folded cells.
    The selection itself requires separate error/coverage validation.
    """
    source = Path(source)
    meta = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    with np.load(source/'nonadiabatic_fgm.npz',allow_pickle=False) as saved:
        original = {k:saved[k] for k in saved.files}
    phis = np.asarray(phis,dtype=float)
    fractions = np.sort(np.asarray(mass_flux_fractions,dtype=float))[::-1]
    if (phis.ndim!=1 or len(phis)<2 or np.any(np.diff(phis)<=0.)
            or fractions.ndim!=1 or not len(fractions) or np.any(np.diff(fractions)>=0.)):
        raise ValueError('Subset coordinates must be distinct; phis must increase')
    def indices(values, available):
        result = []
        for value in values:
            matches = np.flatnonzero(np.asarray(available)==value)
            if len(matches)!=1:
                raise ValueError('Every subset coordinate must exist in the source table')
            result.append(int(matches[0]))
        return result
    p = indices(phis,meta['phis'])
    r = [0]+[j+1 for j in indices(fractions,meta['mass_flux_fractions'])]
    shape = tuple(original['structured_shape'])
    selected = np.arange(np.prod(shape)).reshape(shape)[np.ix_(p,r,np.arange(shape[2]))].ravel()
    payload = {k:v.copy() for k,v in original.items()}
    for name in ('Y','Z','C','h','T','rho','cp_mass','conductivity','omega_C','qdot','controls'):
        payload[name] = original[name][selected]
    new_shape = (len(p),len(r),shape[2])
    cells, triangles, statistics = _connected_cells(new_shape,payload['controls'])
    payload.update(cells=cells,structured_shape=np.asarray(new_shape),reference_cells=triangles,
        reference_points=original['reference_points'].reshape(shape[0],shape[2],2)[p].reshape(-1,2),
        reference_h=original['reference_h'].reshape(shape[0],shape[2])[p].ravel())
    rows = []
    for row in meta['rows']:
        i,j = row['composition_index'],row['loss_index']
        if i in p and j in r:
            rows.append(dict(row,composition_index=p.index(i),loss_index=r.index(j)))
    public = dict(meta,phis=phis.tolist(),mass_flux_fractions=fractions.tolist(),rows=rows,mesh=statistics,
                  source_table_sha256=hashlib.sha256((Path(source)/'nonadiabatic_fgm.npz').read_bytes()).hexdigest(),
                  training_profile_sha256={row['output']:meta['training_profile_sha256'][row['output']] for row in rows})
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'nonadiabatic_fgm.npz',**payload)
    (output/'metadata.json').write_text(json.dumps(public,indent=2)+'\n',encoding='utf-8',newline='\n')
    return output


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

    def lookup_batch(self, *, Z, C, h, outside='raise'):
        """Query 1D broadcastable arrays; Y has shape (states, species).

        outside='mask' returns covered flags and NaNs for unsupported states.
        Interior overlaps always raise. T is recovered from total h and Y.
        """
        if outside not in ('raise','mask'):
            raise ValueError("outside must be 'raise' or 'mask'")
        values=np.broadcast_arrays(*[np.atleast_1d(np.asarray(v,dtype=float)) for v in (Z,C,h)])
        if values[0].ndim!=1:
            raise ValueError('Batch controls must be one-dimensional')
        controls=np.column_stack(values)
        nodes,bary,covered=self.mesh.locate_batch(controls)
        if outside=='raise' and not covered.all():
            raise OutsideManifoldError('Controls outside adjacent resolved flamelets')
        n=len(controls)
        result={k:np.full(n,np.nan) for k in ('T','rho','cp_mass','omega_C','qdot','conductivity','delta_h')}
        result.update(Z=values[0].copy(),C=values[1].copy(),h=values[2].copy(),covered=covered,
                      Y=np.full((n,self.thermo.n_sp),np.nan))
        if not covered.any():
            return result
        chosen=nodes[covered];weights=bary[covered]
        Y=np.einsum('ni,nij->nj',weights,self.table['Y'][chosen])
        target=values[2][covered]
        guess=(np.einsum('ni,ni->n',weights,self.table['T'][chosen])
               if 'T' in self.table else np.full(len(Y),1200.))
        T=self._recover_temperature(target,Y.T,guess)
        result['Y'][covered]=Y;result['T'][covered]=T
        result['rho'][covered]=self.thermo.density(T,self.metadata['pressure_Pa'],Y.T)
        result['cp_mass'][covered]=self.thermo.cp_mass(T,Y.T)
        for key in ('omega_C','qdot','conductivity'):
            result[key][covered]=np.einsum('ni,ni->n',weights,self.table[key][chosen])
        ref_nodes,ref_weights,ref_covered=self.reference_mesh.locate_batch(controls[:,:2])
        result['delta_h'][ref_covered]=np.einsum('ni,ni->n',ref_weights[ref_covered],self.table['reference_h'][ref_nodes[ref_covered]])-values[2][ref_covered]
        result['delta_h'][~covered]=np.nan
        return result

    def _recover_temperature(self,h,Y,guess):
        """Safeguarded vector Newton iteration; scalar Brent fallback.

        The 1e-6 J/kg closure is an internal inversion tolerance, independent
        of the user's allowed tabulation error in kelvin.
        """
        T=np.clip(np.asarray(guess,dtype=float),200.,self.max_temperature)
        lower=np.full(len(T),200.);upper=np.full(len(T),self.max_temperature)
        for _ in range(16):
            residual=self.thermo.enthalpy_mass(T,Y)-h
            done=abs(residual)<=1e-6
            if done.all():
                return T
            lower=np.where(residual<0.,T,lower)
            upper=np.where(residual>0.,T,upper)
            cp=self.thermo.cp_mass(T,Y)
            proposed=T-residual/cp
            safe=(cp>0.) & np.isfinite(proposed) & (proposed>lower) & (proposed<upper)
            T=np.where(done,T,np.where(safe,proposed,.5*(lower+upper)))
        bad=abs(self.thermo.enthalpy_mass(T,Y)-h)>1e-6
        for i in np.flatnonzero(bad):
            T[i]=brentq(lambda temperature:float(self.thermo.enthalpy_mass(temperature,Y[:,i]))-h[i],
                        200.,self.max_temperature,xtol=1e-8)
        return T
