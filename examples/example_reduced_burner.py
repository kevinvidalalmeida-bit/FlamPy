"""Run only the reduced production solver from published physical inputs and plot it.

    python -m examples.example_reduced_burner --table runs/hpc_table --output runs/hpc_example
    python -m examples.example_reduced_burner --table runs/regen_table --profiles runs/regen_exponential --output runs/regen_example

The case (phi, mass flux, solver settings) comes from the published data in
``--data``. With the published table the published seed row is used; with
another table (for instance the regenerated library) ``--profiles`` names its
training library and the nearest burner training row of that table seeds the
solve. The figure overlays the published detailed native flame of the case.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from kflame import solve_reduced_burner_fgm
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import ReducedConvergenceError


def seed_for(model, record, data, profiles):
    """Seed row and profile: published, or the table's nearest training row."""
    if profiles is None:
        return record['seed_row'], data/'seeds'/record['seed_row']/'flame.npz'
    row = min((row for row in model.metadata['rows'] if row['loss_index'] > 0),
              key=lambda row: (abs(row['phi']-record['phi']), abs(row['mass_flux']-record['mass_flux_kg_m2_s'])))
    return row['output'], Path(profiles)/row['output']/'flame.npz'


def plot(profile, model, native_path, record, output):
    import matplotlib.pyplot as plt
    weights = np.asarray(model.table['progress_weights'])
    native = None
    if native_path.is_file():
        with np.load(native_path) as saved:
            native = {key: saved[key] for key in ('z', 'T', 'Y', 'h_mass')}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    z = 1e3*profile['z']
    panels = (('T', 'Temperature [K]', profile['T'], None if native is None else native['T']),
              ('C', 'Progress C', weights@profile['Y'], None if native is None else weights@native['Y']),
              ('h', 'Total enthalpy [kJ/kg]', profile['h']/1e3, None if native is None else native['h_mass']/1e3))
    for ax, (_, label, reduced, detailed) in zip(axes, panels):
        if detailed is not None:
            ax.plot(1e3*native['z'], detailed, color='black', lw=2, label='detailed native (published)')
        ax.plot(z, reduced, color='tab:orange', ls='--', lw=1.6, label='reduced FGM')
        ax.set(xlabel='Distance from the burner [mm]', ylabel=label, xlim=(0., min(z[-1], 6.)))
        ax.legend(fontsize='small')
    figure.suptitle(f"Reduced burner: $\\phi$ = {record['phi']}, "
                    f"mass flux {record['mass_flux_kg_m2_s']:.4f} kg m$^{{-2}}$ s$^{{-1}}$")
    for suffix in ('.png', '.pdf'):
        figure.savefig(output/f'reduced_profiles{suffix}', dpi=160)
    plt.close(figure)
    return output/'reduced_profiles.png'


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--table',type=Path,required=True)
    parser.add_argument('--case',default='case_0.985000_0.285000')
    parser.add_argument('--data',type=Path,default=Path('docs/assets/reduced-burner-hpc'))
    parser.add_argument('--profiles',type=Path,help='training library of --table (seeds from that table)')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    settings=json.loads((args.data/'plan.json').read_text(encoding='utf-8'))['settings']
    record=json.loads((args.data/args.case/'reduced_report.json').read_text(encoding='utf-8'))
    model=NonAdiabaticFGM(args.table)
    seed_row, seed_profile = seed_for(model, record, args.data, args.profiles)
    try:
        profile,report=solve_reduced_burner_fgm(model,phi=record['phi'],mass_flux=record['mass_flux_kg_m2_s'],
            seed_profile=seed_profile,seed_row=seed_row,
            width=settings['width_m'],max_spacing=settings['max_spacing_m'],
            residual_tolerance=settings['residual_tolerance'],max_iterations=settings['max_iterations'],
            max_energy_error=settings['limits']['energy_closure_relative'],**settings.get('solver_options',{}))
    except ReducedConvergenceError as error:
        profile,report=error.profile,error.report
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'reduced.npz',**profile)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(dict(accepted=report['accepted'],reason=report['reason'],iterations=report['iterations'],
               elapsed_seconds=report['elapsed_seconds'],flames=len(model.metadata['rows']),seed_row=seed_row))
    if not args.no_plots:
        print(f"Figure: {plot(profile, model, args.data/args.case/'native'/'flame.npz', record, args.output)}")
    if not report['accepted']:raise SystemExit(1)


if __name__=='__main__':main()
