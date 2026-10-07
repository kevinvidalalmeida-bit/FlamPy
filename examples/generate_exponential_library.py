"""Regenerate a nonadiabatic training library with second-order convection.

Composition families, mass-flux fractions and progress species are copied
from an existing table's metadata, so the new library is directly
comparable. Families are solved in parallel processes (jobs); every flame is
identical to a serial run. Writes raw profiles; tabulate afterwards with
build_nonadiabatic_table.
"""
import argparse
import json
import time
from pathlib import Path

from kflame import generate_nonadiabatic_fgm

SETTINGS = dict(initial_points=24, ratio=3., slope=0.02, curve=0.04, prune=0.005,
                max_points=1800, max_time=240., convection='exponential',
                time_step_ladder=2**0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--like', required=True, help='metadata.json of the table to reproduce')
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    meta = json.loads(Path(args.like).read_text(encoding='utf-8'))
    weights = ','.join(f'{name}:{value:g}' for name, value in meta['progress_species'].items())
    started = time.perf_counter()
    folder = generate_nonadiabatic_fgm(
        phis=tuple(meta['phis']), mass_flux_fractions=tuple(meta['mass_flux_fractions']),
        progress_species=weights, max_energy_error=meta['max_energy_error'],
        raw_only=True, output=args.output, jobs=args.jobs,
        mechanism=meta['mechanism'], fuel=meta['fuel'], oxidizer=meta['oxidizer'],
        temperature=meta['temperature_K'], pressure=meta['pressure_Pa'],
        transport=meta['transport'], soret=meta['soret'], **SETTINGS)
    print(f'library {folder} in {time.perf_counter()-started:.1f} s', flush=True)


if __name__ == '__main__':
    main()
