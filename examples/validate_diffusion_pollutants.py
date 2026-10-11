"""Reproducible methane/air pollutant audit with GRI-Mech 3.0.

python examples/validate_diffusion_pollutants.py --output runs/gri30_pollutants
--fgm also tests a withheld strain against a native (Z,C) table and Cantera.
All detailed reference flames start independently. Native family continuation
is explicit and is not included in independent solve timing comparisons.
"""
import argparse
import json
from pathlib import Path

from kflame import solve_diffusion_flame, generate_diffusion_fgm, validate_diffusion_fgm
from kflame.pollutants import validate_diffusion_pollutants
from kflame.reference.diffusion import compare_diffusion_flame


def plot_profiles(folder, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with np.load(folder / 'pollutants.npz', allow_pickle=False) as data:
        names = [key.removesuffix('_predicted_ppm') for key in data.files if key.endswith('_predicted_ppm')]
        figure, axes = plt.subplots(len(names), 2, figsize=(10, 2.5 * len(names)), squeeze=False)
        for i, name in enumerate(names):
            for j, field in enumerate(('ppm', 'source')):
                for role, style in (('reference', '--'), ('predicted', '-')):
                    axes[i, j].plot(1000. * data['z'], data[f'{name}_{role}_{field}'], style,
                                    label=role)
                axes[i, j].set_title(name)
                axes[i, j].set_xlabel('z [mm]')
                axes[i, j].set_ylabel('Wet mole ppm' if field == 'ppm' else
                    ('Net source [mol/(m3 s)]' if name == 'NOx' else 'Net source [kg/(m3 s)]'))
                axes[i, j].legend()
        figure.suptitle(label)
        figure.tight_layout()
        figure.savefig(folder / 'profiles.png', dpi=160)
        figure.savefig(folder / 'profiles.pdf')
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fgm', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(mechanism='gri30.yaml', fuel='CH4', oxidizer='O2:1,N2:3.76',
                    fuel_mass_flux=.12, oxidizer_mass_flux=.24,
                    fuel_temperature=300., oxidizer_temperature=300.,
                    pressure=101325., width=.02, transport='mixture-averaged', soret=False)
    # Warm compiled native kernels without sharing a flame guess.
    solve_diffusion_flame(mechanism='h2o2.yaml', fuel='H2', fuel_mass_flux=.05,
                         oxidizer_mass_flux=.15, refine=False, output=args.output / 'warmup')
    summary = dict(settings=settings,
        scope='Local flame ppm and net sources, not integrated exhaust emissions. '
              'Screening limits: ppm 1 + 5% of own peak; sources 1e-8 + 10% of own absolute peak.')

    def save():
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    def reference_audit(flame, folder):
        comparison = compare_diffusion_flame(flame['output'], folder / 'cantera')
        report = validate_diffusion_pollutants(flame['output'], folder / 'cantera', folder / 'pollutants')
        plot_profiles(folder / 'pollutants', 'Native detailed flame vs independent Cantera')
        timings = dict(native_s=flame['report']['runtime_s'], cantera_s=comparison['reference_runtime_s'],
            native_initialization='native continuation' if folder.name == 'withheld' else 'independent cold start')
        print(folder.name, json.dumps(dict(passed=report['passed'], timings=timings,
            errors_ppm={name: row['ppm']['maximum_absolute_error'] for name, row in report['pollutants'].items()})), flush=True)
        return dict(timings=timings, flow=comparison, pollutants=report)

    print('Solving independent native GRI30 methane/air flame', flush=True)
    baseline = solve_diffusion_flame(**settings, output=args.output / 'baseline' / 'native')
    summary['baseline'] = reference_audit(baseline, args.output / 'baseline')
    save()
    if args.fgm:
        print('Training native FGM at flow scales 0.75, 1, 1.25', flush=True)
        family = generate_diffusion_fgm(**settings, initial_solution=baseline['output'],
                                       output=args.output / 'fgm')
        summary['training'] = dict(metadata=family['metadata'],
            runtime_s=[flame['report']['runtime_s'] for flame in family['flamelets']])
        save()
        withheld_settings = dict(settings, fuel_mass_flux=.132, oxidizer_mass_flux=.264)
        print('Solving withheld flow scale 1.1', flush=True)
        withheld = solve_diffusion_flame(**withheld_settings, initial_solution=family['flamelets'][1]['output'],
                                        output=args.output / 'withheld' / 'native')
        summary['withheld'] = reference_audit(withheld, args.output / 'withheld')
        generic = validate_diffusion_fgm(family['output'], withheld['output'], args.output / 'fgm_validation')
        pollutants = validate_diffusion_pollutants(withheld['output'], args.output / 'fgm_validation',
                                                   args.output / 'fgm_pollutants', kind='fgm')
        plot_profiles(args.output / 'fgm_pollutants', 'FGM interpolation vs withheld native flame')
        summary['fgm'] = dict(flow=generic, pollutants=pollutants)
        print('FGM', json.dumps(dict(passed=pollutants['passed'], coverage=pollutants['coverage'],
            errors_ppm={name: row['ppm']['maximum_absolute_error'] for name, row in pollutants['pollutants'].items()})), flush=True)
        save()
    if not all(summary[key]['pollutants']['passed'] for key in ('baseline', 'withheld', 'fgm') if key in summary):
        raise SystemExit('A pollutant screening comparison failed; results retained in summary.json')


if __name__ == '__main__':
    main()
