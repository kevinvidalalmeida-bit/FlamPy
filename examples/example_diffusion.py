"""Native diffusion flames, FGM and withheld-strain/reference validation.

Run: python examples/example_diffusion.py --output runs/diffusion_example --reference
Cantera is optional and used only when --reference is supplied.
The figure (diffusion_fgm.png/.pdf) is drawn only from the saved outputs;
--plot-only runs/diffusion_example redraws it without solving.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from kflame import solve_diffusion_flame,generate_diffusion_fgm,validate_diffusion_fgm


def plot(folder):
    """Native flame, (Z,C) table and withheld-strain comparison for each fuel."""
    import matplotlib.pyplot as plt
    folder=Path(folder)
    fuels=[fuel for fuel in ('CH4','H2') if (folder/fuel/'fgm_validation'/'comparison.npz').is_file()]
    figure,axes=plt.subplots(len(fuels),3,figsize=(15,4.4*len(fuels)),layout='constrained',squeeze=False)
    for row,fuel in zip(axes,fuels):
        with np.load(folder/fuel/'native'/'flame.npz') as saved:
            native={key:saved[key] for key in ('z','T','Z')}
        with np.load(folder/fuel/'fgm'/'diffusion_fgm.npz') as saved:
            cells=len(saved['cells'])
        family=sorted((folder/fuel/'fgm').glob('flamelet_*/flame.npz'))
        meta=json.loads((folder/fuel/'fgm'/'metadata.json').read_text(encoding='utf-8'))
        with np.load(folder/fuel/'withheld'/'flame.npz') as saved:
            withheld={key:saved[key] for key in ('Z','T')}
        with np.load(folder/fuel/'fgm_validation'/'comparison.npz') as saved:
            lookup={key:saved[key] for key in ('native_Z','covered','T')}
        report=json.loads((folder/fuel/'fgm_validation'/'validation.json').read_text(encoding='utf-8'))
        zmax=.25 if fuel=='CH4' else .15
        ax=row[0]
        ax.plot(1e3*native['z'],native['T'],color='tab:red')
        ax.set(xlabel='Distance from the fuel inlet [mm]',ylabel='Temperature [K]',title=f'{fuel}/air native counterflow flame')
        twin=ax.twinx();twin.plot(1e3*native['z'],native['Z'],color='tab:blue',ls='--');twin.set_ylabel('Mixture fraction Z',color='tab:blue')
        ax=row[1]
        colors=plt.cm.viridis(np.linspace(0.,.85,len(family)))
        for path,scale,color in zip(family,meta['flow_scales'],colors):
            with np.load(path) as saved:
                order=np.argsort(saved['Z'])
                ax.plot(saved['Z'][order],saved['T'][order],color=color,label=f'strain x{scale:g}')
        ax.set(xlabel='Mixture fraction Z',ylabel='Temperature [K]',xlim=(0.,zmax),
               title=f'Training family: {len(family)} flamelets, {cells} connected (Z,C) cells')
        ax.legend(fontsize='small')
        ax=row[2]
        order=np.argsort(withheld['Z'])
        ax.plot(withheld['Z'][order],withheld['T'][order],color='black',lw=2,label='withheld native flamelet')
        covered=lookup['covered'].astype(bool)
        zf=lookup['native_Z'][covered];order=np.argsort(zf)
        ax.plot(zf[order],lookup['T'][order],color='tab:orange',ls='--',lw=1.6,label='(Z,C) table')
        ax.set(xlabel='Mixture fraction Z',ylabel='Temperature [K]',xlim=(0.,zmax),
               title=f"Withheld strain x1.1: max dT = {report['metrics']['temperature_K']:.2f} K "
                     f"({'passed' if report['passed'] else 'FAILED'})")
        ax.legend(fontsize='small')
    for suffix in ('.png','.pdf'):
        figure.savefig(folder/f'diffusion_fgm{suffix}',dpi=160)
    plt.close(figure)
    return folder/'diffusion_fgm.png'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fuel',choices=['CH4','H2','both'],default='both')
    parser.add_argument('--reference',action='store_true')
    parser.add_argument('--plot-only',type=Path,help='redraw the figure of an existing output folder')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        print(f'Figure: {plot(args.plot_only)}')
        return
    if args.output is None:
        parser.error('--output is required unless --plot-only is given')
    args.output.mkdir(parents=True,exist_ok=False)
    cases=dict(CH4=dict(mechanism='gri30.yaml',fuel='CH4',fuel_mass_flux=.12,oxidizer_mass_flux=.24),
               H2=dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15))
    summary={}
    for fuel,settings in cases.items():
        if args.fuel not in ('both',fuel):
            continue
        folder=args.output/fuel
        flame=solve_diffusion_flame(**settings,output=folder/'native')
        family=generate_diffusion_fgm(**settings,initial_solution=flame['output'],output=folder/'fgm')
        withheld=dict(settings)
        for key in ('fuel_mass_flux','oxidizer_mass_flux'):
            withheld[key]*=1.1
        test=solve_diffusion_flame(**withheld,initial_solution=family['flamelets'][1]['output'],output=folder/'withheld')
        report=validate_diffusion_fgm(family['output'],test['output'],folder/'fgm_validation')
        summary[fuel]=dict(native=flame['report'],fgm=report)
        if args.reference:
            from kflame.reference.diffusion import compare_diffusion_flame
            summary[fuel]['reference']=compare_diffusion_flame(flame['output'],folder/'cantera')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        print(fuel,report['passed'],report['metrics'],flush=True)
    if not args.no_plots:
        print(f'Figure: {plot(args.output)}',flush=True)
    if any(not row['fgm']['passed'] or not row.get('reference',{'passed':True})['passed'] for row in summary.values()):
        raise SystemExit('A diffusion validation failed; inspect summary.json')


if __name__=='__main__':
    main()
