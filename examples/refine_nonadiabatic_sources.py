"""Refine composition/cooling intervals without adding withheld cases.

python examples/refine_nonadiabatic_sources.py runs/nonadiabatic \
    --output runs/nonadiabatic_refined --composition-subdivisions 3
Uses native burner continuation and reuses only matching accepted profiles.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from kflame import generate_nonadiabatic_fgm


def subdivide(values, count):
    if count < 1:
        raise ValueError('Subdivision count must be at least one')
    values = sorted(values)
    refined = []
    for a, b in zip(values[:-1], values[1:]):
        refined.append(a)
        refined.extend(a + (b-a)*i/count for i in range(1,count))
    return [*refined,values[-1]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--composition-subdivisions',type=int,default=3)
    parser.add_argument('--loss-subdivisions',type=int,default=1)
    parser.add_argument('--loss-interval',type=float,nargs=2,action='append',default=[],
                        metavar=('LOW','HIGH'),help='Trisect selected existing cooling intervals')
    parser.add_argument('--progress-points',type=int,default=361)
    args=parser.parse_args()
    metadata=json.loads((args.folder/'generation.json').read_text(encoding='utf-8'))
    phis=subdivide(metadata['phis'],args.composition_subdivisions)
    fractions=subdivide(metadata['mass_flux_fractions'],args.loss_subdivisions)[::-1]
    for low,high in args.loss_interval:
        original=sorted(metadata['mass_flux_fractions'])
        if not any(np.isclose(low,a) and np.isclose(high,b) for a,b in zip(original[:-1],original[1:])):
            raise ValueError('Selected cooling interval must be adjacent in the original family')
        fractions.extend([low+(high-low)/3.,low+2.*(high-low)/3.])
    fractions=sorted(set(fractions),reverse=True)
    progress=','.join(f'{k}:{v}' for k,v in metadata['progress_species'].items())
    result=generate_nonadiabatic_fgm(
        phis=phis,mass_flux_fractions=fractions,progress_species=progress,
        progress_points=args.progress_points,initial_points=24,max_time=240.,
        mechanism=metadata['mechanism'],fuel=metadata['fuel'],oxidizer=metadata['oxidizer'],
        temperature=metadata['temperature_K'],pressure=metadata['pressure_Pa'],
        transport=metadata['transport'],soret=metadata['soret'],
        output=args.output,reuse_from=args.folder)
    print(result)


if __name__=='__main__':
    main()
