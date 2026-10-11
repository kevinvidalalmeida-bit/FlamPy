"""Counterflow (Z,C,h) heat-loss table and a withheld-flamelet validation.

python examples/example_diffusion_heat_loss.py --case H2 --output runs/h2_heat_loss
python examples/example_diffusion_heat_loss.py --case CH4 --output runs/ch4_heat_loss

The family varies both inlet mass fluxes (strain) and the optically thin
radiation scale (enthalpy loss). The withheld flamelet uses a strain and a
loss level absent from the training family and is never used to build it.
"""
import argparse
import json
from pathlib import Path
import time

from kflame import solve_diffusion_flame
from kflame.fgm.diffusion import generate_diffusion_heat_loss_fgm, validate_diffusion_heat_loss_fgm

CASES = {
    'H2': dict(settings=dict(mechanism='h2o2.yaml', fuel='H2', fuel_mass_flux=.05, oxidizer_mass_flux=.15),
               radiation_scales=(0., 10., 30., 60.), withheld=(1.1, 20.)),
    'CH4': dict(settings=dict(mechanism='gri30.yaml', fuel='CH4', fuel_mass_flux=.12, oxidizer_mass_flux=.24),
                radiation_scales=(0., 2., 4., 6., 8.), withheld=(1.1, 5.)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=sorted(CASES), default='H2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transport', choices=['mixture-averaged', 'multicomponent'], default='mixture-averaged')
    parser.add_argument('--soret', action='store_true')
    args = parser.parse_args()
    case = CASES[args.case]
    settings = dict(case['settings'], transport=args.transport, soret=args.soret)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    family = generate_diffusion_heat_loss_fgm(**settings, flow_scales=(.75, 1., 1.25),
                                              radiation_scales=case['radiation_scales'],
                                              output=args.output/'table')
    generation = time.perf_counter()-started
    flow, loss = case['withheld']
    withheld = solve_diffusion_flame(**{**settings, 'fuel_mass_flux': settings['fuel_mass_flux']*flow,
                                        'oxidizer_mass_flux': settings['oxidizer_mass_flux']*flow},
                                     radiation=True, radiation_scale=loss,
                                     initial_solution=family['flamelets'][1][1]['output'],
                                     output=args.output/'withheld')
    report = validate_diffusion_heat_loss_fgm(args.output/'table', withheld['output'], args.output/'validation')
    table = args.output/'table'/'diffusion_heat_loss_fgm.npz'
    meta = family['metadata']
    summary = dict(case=args.case, transport=args.transport, soret=args.soret,
                   flow_scales=meta['flow_scales'], radiation_scales=meta['radiation_scales'],
                   withheld=dict(flow_scale=flow, radiation_scale=loss),
                   generation_wall_s=generation, flamelet_runtimes_s=meta['flamelet_runtimes_s'],
                   table_megabytes=table.stat().st_size/1e6, geometry=meta['geometry'],
                   mixture_sampling=meta['mixture_sampling'], progress_rule=meta['progress_rule'],
                   progress_candidates=meta['progress_candidates'],
                   maximum_enthalpy_deficit_J_kg=meta['maximum_enthalpy_deficit_J_kg'],
                   validation=dict(passed=report['passed'], metrics=report['metrics'], limits=report['limits'],
                                   checks=report['checks'], reaction_coverage=report['reaction_coverage'],
                                   covered_points=report['covered_points'], total_points=report['total_points']))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('generation_wall_s', 'table_megabytes', 'progress_rule')}))
    print(json.dumps(summary['validation']['metrics'], indent=1))
    if not report['passed']:
        raise SystemExit('Withheld validation failed; inspect validation.json')


if __name__ == '__main__':
    main()
