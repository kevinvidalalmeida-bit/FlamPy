"""Reconstruct a published selection without repeating native simulations."""
import argparse
import hashlib
import json
from pathlib import Path
from kflame.fgm.nonadiabatic3d import subset_nonadiabatic_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/assets/nonadiabatic3d'))
    parser.add_argument('--selection',type=Path,default=Path('docs/assets/adaptive-fgm/selection.json'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    fingerprint=selection.get('source_table_sha256')
    if fingerprint and fingerprint!=hashlib.sha256((args.source/'nonadiabatic_fgm.npz').read_bytes()).hexdigest():
        raise ValueError('Source table differs from the published selection')
    print(subset_nonadiabatic_table(args.source,args.output,phis=selection['phis'],
                                   mass_flux_fractions=selection['mass_flux_fractions']))


if __name__=='__main__':
    main()
