"""Counterflow (Z,C,h) heat-loss table and a withheld-flamelet validation.

python examples/example_diffusion_heat_loss.py --case H2 --output runs/h2_heat_loss
python examples/example_diffusion_heat_loss.py --case CH4 --output runs/ch4_heat_loss
python examples/example_diffusion_heat_loss.py --plot-only runs/ch4_heat_loss

The family varies both inlet mass fluxes (strain) and the optically thin
radiation scale (enthalpy loss). The withheld flamelet uses a strain and a
loss level absent from the training family and is never used to build it.
The figure (heat_loss_fgm.png/.pdf) is drawn only from the saved outputs;
--plot-only redraws it from an existing output folder.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from kflame import solve_diffusion_flame
from kflame.fgm.diffusion import generate_diffusion_heat_loss_fgm, validate_diffusion_heat_loss_fgm

CASES = {
    'H2': dict(settings=dict(mechanism='h2o2.yaml', fuel='H2', fuel_mass_flux=.05, oxidizer_mass_flux=.15),
               radiation_scales=(0., 10., 30., 60.), withheld=(1.1, 20.)),
    'CH4': dict(settings=dict(mechanism='gri30.yaml', fuel='CH4', fuel_mass_flux=.12, oxidizer_mass_flux=.24),
                radiation_scales=(0., 2., 4., 6., 8.), withheld=(1.1, 5.)),
}


def plot(folder):
    """Family at the nominal strain and the withheld-flamelet comparison, from saved files."""
    import matplotlib.pyplot as plt
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    scales = summary['radiation_scales']
    nominal = summary['flow_scales'].index(1.) if 1. in summary['flow_scales'] else len(summary['flow_scales'])//2
    colors = plt.cm.plasma(np.linspace(0., .85, len(scales)))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout='constrained')
    adiabatic = None
    for loss, (scale, color) in enumerate(zip(scales, colors)):
        with np.load(folder/'table'/f'flamelet_{nominal:03d}_{loss:03d}'/'flame.npz') as saved:
            Z, T, h = saved['Z'], saved['T'], saved['h']
        order = np.argsort(Z)
        Z, T, h = Z[order], T[order], h[order]
        if adiabatic is None:
            adiabatic = (Z, h)
        label = 'adiabatic' if scale == 0 else f'radiation x{scale:g}'
        axes[0, 0].plot(Z, T, color=color, label=label)
        axes[0, 1].plot(Z, (np.interp(Z, *adiabatic)-h)/1e3, color=color, label=label)
    zmax = .3 if summary['case'] == 'CH4' else .6
    axes[0, 0].set(xlabel='Mixture fraction Z', ylabel='Temperature [K]', xlim=(0., zmax),
                   title=f'Training family at the nominal strain ({len(scales)} loss levels)')
    axes[0, 1].set(xlabel='Mixture fraction Z', ylabel=r'Enthalpy defect $h_{ad}-h$ [kJ/kg]', xlim=(0., zmax),
                   title='Radiative enthalpy defect')
    axes[0, 0].legend(fontsize='small')
    with np.load(folder/'withheld'/'flame.npz') as saved:
        native = {key: saved[key] for key in ('Z', 'T', 'qdot')}
    with np.load(folder/'validation'/'comparison.npz') as saved:
        fgm = {key: saved[key] for key in ('native_Z', 'T', 'qdot', 'covered')}
    covered = fgm['covered'].astype(bool)
    withheld = summary['withheld']
    metrics = summary['validation']['metrics']
    for ax, key, label, scale in ((axes[1, 0], 'T', 'Temperature [K]', 1.), (axes[1, 1], 'qdot', r'Heat release $\dot q$ [MW/m$^3$]', 1e-6)):
        order = np.argsort(native['Z'])
        ax.plot(native['Z'][order], scale*native[key][order], color='black', lw=2, label='withheld native flamelet')
        zf = fgm['native_Z'][covered]
        order = np.argsort(zf)
        ax.plot(zf[order], scale*fgm[key][covered][order], color='tab:orange', ls='--', lw=1.6, label='(Z,C,h) table')
        ax.set(xlabel='Mixture fraction Z', ylabel=label, xlim=(0., zmax))
        ax.legend(fontsize='small')
    axes[1, 0].set_title(f"Withheld: strain x{withheld['flow_scale']:g}, radiation x{withheld['radiation_scale']:g}; "
                         f"max dT = {metrics['temperature_K']:.2f} K")
    axes[1, 1].set_title(f"Heat release: error {100*metrics['heat_release_relative']:.1f} % of the peak")
    figure.suptitle(f"Counterflow {summary['case']}/air (Z,C,h) heat-loss table: "
                    f"validation {'passed' if summary['validation']['passed'] else 'FAILED'}")
    for suffix in ('.png', '.pdf'):
        figure.savefig(folder/f'heat_loss_fgm{suffix}', dpi=160)
    plt.close(figure)
    return folder/'heat_loss_fgm.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=sorted(CASES), default='H2')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--transport', choices=['mixture-averaged', 'multicomponent'], default='mixture-averaged')
    parser.add_argument('--soret', action='store_true')
    parser.add_argument('--jobs', type=int, default=3, help='processes for the loss rows (same flamelets as 1)')
    parser.add_argument('--plot-only', type=Path, help='redraw the figure of an existing output folder')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        print(f'Figure: {plot(args.plot_only)}')
        return
    if args.output is None:
        parser.error('--output is required unless --plot-only is given')
    case = CASES[args.case]
    settings = dict(case['settings'], transport=args.transport, soret=args.soret)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    family = generate_diffusion_heat_loss_fgm(**settings, flow_scales=(.75, 1., 1.25),
                                              radiation_scales=case['radiation_scales'],
                                              output=args.output/'table', jobs=args.jobs)
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
    if not args.no_plots:
        print(f'Figure: {plot(args.output)}')
    if not report['passed']:
        raise SystemExit('Withheld validation failed; inspect validation.json')


if __name__ == '__main__':
    main()
