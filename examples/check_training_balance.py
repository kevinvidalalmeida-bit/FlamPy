"""Check the source/flux balance of every accepted native burner profile.

python examples/check_training_balance.py runs/nonadiabatic \
    --output runs/training_progress_balance.json
Uses raw nonuniform spatial meshes, rather than FGM trajectory samples.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kflame.chemistry.mechanism import load_mechanism
if __package__:
    from .audit_nonadiabatic_sources import detailed_sources
else:
    from audit_nonadiabatic_sources import detailed_sources


def check(folder,output):
    folder,output=Path(folder),Path(output)
    generation=json.loads((folder/'generation.json').read_text(encoding='utf-8'))
    mech=load_mechanism(generation['mechanism'])
    weights=np.array([generation['progress_species'].get(s,0.) for s in mech.species_names])
    records=[]
    for row in generation['rows']:
        if row['loss_index']==0:
            continue
        path=folder/row['output']
        metadata=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        if not metadata['accepted']:
            raise RuntimeError('Training balance check requires accepted native profiles')
        with np.load(path/'flame.npz',allow_pickle=False) as profile:
            mass,_=detailed_sources(mech,profile['T'],profile['Y'],generation['pressure_Pa'])
            integral=float(np.trapezoid(weights@mass,profile['z']))
            expected=metadata['mass_flux']*float(weights@(profile['Y'][:,-1]-np.asarray(metadata['inlet_Y'])))
            records.append(dict(phi=row['phi'],fraction=row['fraction'],nodes=len(profile['z']),
                                progress_balance=abs(integral-expected)/abs(expected)))
    if not records:
        raise ValueError('No burner profiles to check')
    result=dict(burner_flames=len(records),maximum_progress_balance=max(r['progress_balance'] for r in records),
                cases=records,generation_sha256=hashlib.sha256((folder/'generation.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                definition='Trapezoidal source integral compared with imposed fresh progress flux and zero outlet diffusive flux',
                scope='Independent integral diagnostic on the raw spatial grids; not a Newton residual or an FGM interpolation error.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=check(args.folder,args.output)
    print(report['burner_flames'],report['maximum_progress_balance'])


if __name__=='__main__':
    main()
