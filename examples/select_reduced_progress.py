"""Select progress from a physical training table and rebuild its controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.progress_selection import select_progress_weights
from examples.retabulate_reduced_progress import retabulate


def select(source, output, *, objective='maximum_margin'):
    model = NonAdiabaticFGM(source)
    if objective not in ('maximum_margin','nearest_reference'):
        raise ValueError('Unknown progress-selection objective')
    if 'omega_Y' not in model.table:
        raise ValueError('First tabulate full native species sources offline')
    weights, audit = select_progress_weights(model.table['Y'],model.table['omega_Y'],
        model.table['structured_shape'],normalization=float(np.sum(abs(model.table['progress_weights']))),
        reference_weights=model.table['progress_weights'] if objective=='nearest_reference' else None)
    names = {str(name):float(value) for name,value in zip(model.table['species_names'],weights) if value > 0.}
    retabulate(source,output,names)
    output = Path(output)
    audit.update(source_table_sha256=model.table_sha256,weights=names,
                 selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 algorithm_sha256=hashlib.sha256(Path(select_progress_weights.__code__.co_filename).read_bytes()).hexdigest())
    (output/'progress_selection.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    meta = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    meta['progress_selection'] = audit
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--objective',choices=('maximum_margin','nearest_reference'),default='maximum_margin')
    args=parser.parse_args()
    select(args.source,args.output,objective=args.objective)
