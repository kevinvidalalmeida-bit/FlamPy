"""Generate a native burner FGM and plot conductive heat-loss effects.

Run: python examples/example_heat_loss.py --output runs/burner_example
"""
import argparse
import json

import numpy as np

from kflame import generate_burner_fgm
from kflame.fgm.nonadiabatic import BurnerFGM


def plot_results(folder):
    import matplotlib.pyplot as plt
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    model = BurnerFGM(folder)
    table = model.table
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'xtick.direction': 'in', 'ytick.direction': 'in'}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        valid = table['valid'][0]
        axes[0, 1].plot(table['c'][valid], table['h'][0, valid] / 1000., 'k--', label='Adiabatic')
        fluxes, deficits, heat_fluxes = [], [], []
        for row, record in enumerate(metadata['rows'][1:], start=1):
            flux = record['mass_flux_kg_m2_s']
            color = colors[(row - 1) % len(colors)]
            with np.load(folder / record['output'] / 'flame.npz') as profile:
                axes[0, 0].plot(1000. * profile['z'], profile['T'], color=color,
                                label=rf'$\dot{{m}}={flux:g}$')
            valid = table['valid'][row]
            axes[0, 1].plot(table['c'][valid], table['h'][row, valid] / 1000., color=color,
                            label=rf'$\dot{{m}}={flux:g}$')
            fluxes.append(flux)
            deficits.append(record['heat_loss']['burned_enthalpy_deficit_J_kg'] / 1000.)
            heat_fluxes.append(record['heat_loss']['burner_heat_loss_W_m2'] / 1000.)
        order = np.argsort(fluxes)
        axes[1, 0].plot(np.array(fluxes)[order], np.array(deficits)[order], 'o-', color=colors[0])
        axes[1, 1].plot(np.array(fluxes)[order], np.array(heat_fluxes)[order], 'o-', color=colors[1])
        axes[0, 0].set(xlabel='Distance from burner [mm]', ylabel='Temperature [K]', xlim=(0., 3.))
        axes[0, 1].set(xlabel=r'Progress $c$ [$-$]', ylabel=r'Total enthalpy $h$ [kJ/kg]')
        axes[1, 0].set(xlabel=r'Imposed mass flux [kg/(m$^2$ s)]', ylabel=r'Burned deficit $h_{feed}-h_b$ [kJ/kg]')
        axes[1, 1].set(xlabel=r'Imposed mass flux [kg/(m$^2$ s)]', ylabel=r'Heat flux to burner [kW/m$^2$]')
        for ax in axes.flat:
            ax.grid(alpha=.2)
        axes[0, 0].legend(title=r'$\dot{m}$ [kg/(m$^2$ s)]', fontsize=8)
        axes[0, 1].legend(fontsize=8)
        for extension in ('png', 'pdf'):
            fig.savefig(folder / f'burner_heat_loss.{extension}', dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--phi', type=float, default=1.)
    parser.add_argument('--mass-fluxes', type=float, nargs='+', default=[.04, .08, .12])
    parser.add_argument('--progress-points', type=int, default=241)
    parser.add_argument('--max-time', type=float, default=240.)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    folder = generate_burner_fgm(
        mass_fluxes=args.mass_fluxes, progress_points=args.progress_points, phi=args.phi,
        initial_points=24, max_time=args.max_time, output=args.output)
    if not args.no_plots:
        plot_results(folder)
    print(folder)


if __name__ == '__main__':
    main()
