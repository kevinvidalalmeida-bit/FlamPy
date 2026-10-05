"""Rebuild the budgeted physical table from exact stored vertex fields.

345 rows come unchanged from the published 450-flame library; 87 newly solved
rows are bundled as a delta. No flame or chemical reference is solved here.
Retabulation and native reactor continuation are separate offline operations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from kflame.fgm.nonadiabatic3d import _connected_cells


def reconstruct(source, data, output):
    source, data, output = map(Path, (source, data, output))
    if output.exists():
        raise ValueError('Use a new directory for a frozen reconstruction')
    old_meta = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    meta = json.loads((data/'body_metadata.json').read_text(encoding='utf-8'))
    with np.load(source/'nonadiabatic_fgm.npz', allow_pickle=False) as saved:
        old = {k:saved[k] for k in saved.files}
    with np.load(data/'library_delta.npz', allow_pickle=False) as saved:
        delta = {k:saved[k] for k in saved.files}
    digest = hashlib.sha256((source/'nonadiabatic_fgm.npz').read_bytes()).hexdigest()
    if digest != str(delta['source_table_sha256']):
        raise ValueError('Published source fingerprint differs')
    shape = (len(meta['phis']), len(meta['mass_flux_fractions'])+1, meta['progress_points'])
    old_shape = tuple(old['structured_shape'])
    fields = ['Z','C','h','T','rho','cp_mass','conductivity','omega_C','qdot','Y']
    payload = {k:np.empty((*shape,*old[k].shape[1:]), dtype=old[k].dtype) for k in fields}
    indices = {tuple(ij):n for n,ij in enumerate(delta['row_indices'])}
    for row in meta['rows']:
        i,j = row['composition_index'],row['loss_index']
        if (i,j) in indices:
            for k in fields:payload[k][i,j] = delta[k][indices[i,j]]
            continue
        matches = [r for r in old_meta['rows'] if r['phi']==row['phi'] and
                   (r['loss_index']==0 if j==0 else r.get('fraction')==row.get('fraction'))]
        if len(matches)!=1:raise ValueError('Missing or ambiguous retained physical row')
        previous = matches[0]
        for k in fields:
            payload[k][i,j] = old[k].reshape((*old_shape,*old[k].shape[1:]))[previous['composition_index'],previous['loss_index']]
    payload = {k:v.reshape((-1,*v.shape[3:])) for k,v in payload.items()}
    controls = np.column_stack([payload[k] for k in ('Z','C','h')])
    cells,triangles,statistics = _connected_cells(shape,controls)
    if statistics!=meta['mesh']:raise ValueError('Reconstructed mesh statistics differ')
    payload.update(controls=controls,cells=cells,progress_weights=old['progress_weights'],
        bilger_weights=old['bilger_weights'],bilger_offset=old['bilger_offset'],
        species_names=old['species_names'],structured_shape=np.array(shape),
        reference_points=np.column_stack([payload[k].reshape(shape)[:,0].ravel() for k in ('Z','C')]),
        reference_cells=triangles,reference_h=payload['h'].reshape(shape)[:,0].ravel(),
        sampling_coordinate=old['sampling_coordinate'])
    output.mkdir(parents=True)
    np.savez_compressed(output/'nonadiabatic_fgm.npz',**payload)
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(dict(rows=len(meta['rows']),new_rows=len(indices),table_sha256=hashlib.sha256((output/'nonadiabatic_fgm.npz').read_bytes()).hexdigest()))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('docs/assets/nonadiabatic3d'))
    p.add_argument('--data',type=Path,default=Path('docs/assets/reduced-burner-hpc'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reconstruct(a.source,a.data,a.output)
