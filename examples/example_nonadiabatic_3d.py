"""Generate a native (Z,C,h) manifold for a planar isothermal CH4-air burner.

python examples/example_nonadiabatic_3d.py --output runs/nonadiabatic
python examples/example_nonadiabatic_3d.py --build-only runs/nonadiabatic
Then run validate_nonadiabatic_3d.py and plot_nonadiabatic_3d.py.
"""
import argparse
from pathlib import Path

from kflame import generate_nonadiabatic_fgm
from kflame.fgm.nonadiabatic3d import build_nonadiabatic_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--build-only', type=Path)
    parser.add_argument('--phis', type=float, nargs='+', default=[.7, .85, 1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
    parser.add_argument('--mass-flux-fractions', type=float, nargs='+', default=[.65, .45, .25, .20, .16, .12, .10, .08, .06])
    parser.add_argument('--progress-points', type=int, default=181)
    parser.add_argument('--raw-only', action='store_true')
    parser.add_argument('--reuse-from', type=Path)
    args = parser.parse_args()
    if args.build_only is not None:
        path = build_nonadiabatic_table(args.build_only, progress_points=args.progress_points)
    else:
        path = generate_nonadiabatic_fgm(phis=args.phis, mass_flux_fractions=args.mass_flux_fractions,
                                       progress_points=args.progress_points, raw_only=args.raw_only,
                                       initial_points=24, max_time=240., output=args.output, reuse_from=args.reuse_from)
    print(path)


if __name__ == '__main__':
    main()
