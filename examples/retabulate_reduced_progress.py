"""Change the reactive coordinate of an existing family without new flames.

This is an offline operation: all Y, T, h and heat-release samples are retained;
the new C, its native source, and the connected physical mesh are rebuilt.
Weights are supplied as data, not hard-coded in the transport solver.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM, _connected_cells


def retabulate(source, output, weights):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a new output directory for a new frozen progress definition')
    model = NonAdiabaticFGM(source)
    mech, table = load_mechanism(model.metadata['mechanism']), model.table
    if not weights or any(n not in mech.species_names for n in weights):
        raise ValueError('Progress species must belong to the mechanism')
    w = np.asarray([weights.get(n, 0.) for n in mech.species_names], dtype=float)
    if not np.isfinite(w).all() or not np.any(w):
        raise ValueError('Progress weights must be finite and nonzero')
    Y, T = table['Y'].T, table['T']
    rho = model.thermo.density(T, model.metadata['pressure_Pa'], Y)
    rates = NativeKinetics(mech).net_production_rates(T,
        rho[None, :]*Y*mech.inv_molecular_weights[:, None], model.thermo.g_RT(T))
    payload = {k: v.copy() for k, v in table.items()}
    mass_sources = rates*mech.molecular_weights[:, None]
    payload.update(C=w@Y, omega_C=w@mass_sources, omega_Y=mass_sources.T, progress_weights=w)
    shape = tuple(table['structured_shape'])
    minimum_growth = float(np.diff(payload['C'].reshape(shape), axis=2).min())
    if minimum_growth <= 0.:
        raise ValueError('New progress must increase strictly in every flame and reactor tail')
    controls = np.column_stack([payload[q] for q in ('Z', 'C', 'h')])
    cells, triangles, statistics = _connected_cells(shape, controls)
    if statistics['excluded_folded_cells'] or statistics['excluded_degenerate_cells']:
        raise ValueError('New progress creates a folded or degenerate physical mesh')
    payload.update(controls=controls, cells=cells, reference_cells=triangles,
        reference_points=np.column_stack([payload[q].reshape(shape)[:, 0].ravel() for q in ('Z', 'C')]))
    audit = dict(source_table_sha256=model.table_sha256, minimum_progress_increment=minimum_growth,
                 additional_flames=0, preserved_fields=['Y','T','h','Z','qdot','rho','cp_mass','conductivity'],
                 source_recomputed_offline=True, full_species_sources_stored=True,
                 builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    meta = dict(model.metadata, progress_species=weights, mesh=statistics, progress_retabulation=audit)
    output.mkdir(parents=True)
    np.savez_compressed(output/'nonadiabatic_fgm.npz', **payload)
    (output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8', newline='\n')
    for name in ('reactor_tail.npz','reactor_audit.json','equilibrium_endpoints.npz'):
        if (source/name).exists():
            shutil.copyfile(source/name, output/name)
    print(audit, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('examples/reduced_progress_weights.json'))
    args = parser.parse_args()
    retabulate(args.source, args.output, json.loads(args.weights.read_text(encoding='utf-8')))
