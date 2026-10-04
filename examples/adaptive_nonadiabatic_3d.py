"""Generate a connected FGM using caller-selected candidates and tolerances.

python examples/adaptive_nonadiabatic_3d.py examples/adaptive_fgm_settings.json \
    --tolerances examples/fgm_tolerances.json --output runs/adaptive
Optional --reuse-from points to accepted raw profiles with matching physics
and solver settings. Cached studies measure selection, not saved solver time.
"""
import argparse
import json
from pathlib import Path

from kflame import generate_adaptive_nonadiabatic_fgm
from kflame.fgm.accuracy import FGMTolerances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('settings', type=Path, help='JSON with candidate coordinates and flame settings')
    parser.add_argument('--tolerances', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-from', type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding='utf-8'))
    folder = generate_adaptive_nonadiabatic_fgm(**settings,
        tolerances=FGMTolerances.from_file(args.tolerances), output=args.output, reuse_from=args.reuse_from)
    print('Table:', folder)
    print('Run independent withheld-profile validation before using this table.')


if __name__ == '__main__':
    main()
