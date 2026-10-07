"""Native diffusion flames, FGM and withheld-strain/reference validation.

Run: python examples/example_diffusion.py --output runs/diffusion_example --reference
Cantera is optional and used only when --reference is supplied.
"""
import argparse
import json
from pathlib import Path

from kflame import solve_diffusion_flame,generate_diffusion_fgm,validate_diffusion_fgm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fuel',choices=['CH4','H2','both'],default='both')
    parser.add_argument('--reference',action='store_true')
    args=parser.parse_args()
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
    if any(not row['fgm']['passed'] or not row.get('reference',{'passed':True})['passed'] for row in summary.values()):
        raise SystemExit('A diffusion validation failed; inspect summary.json')


if __name__=='__main__':
    main()
