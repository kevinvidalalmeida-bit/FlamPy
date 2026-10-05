"""Run only the reduced production solver from published physical inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from kflame import solve_reduced_burner_fgm
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.fgm.reduced_burner import ReducedConvergenceError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table',type=Path,required=True)
    parser.add_argument('--case',default='case_0.985000_0.285000')
    parser.add_argument('--data',type=Path,default=Path('docs/assets/reduced-burner-hpc'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    settings=json.loads((args.data/'plan.json').read_text(encoding='utf-8'))['settings']
    record=json.loads((args.data/args.case/'reduced_report.json').read_text(encoding='utf-8'))
    model=NonAdiabaticFGM(args.table)
    try:
        profile,report=solve_reduced_burner_fgm(model,phi=record['phi'],mass_flux=record['mass_flux_kg_m2_s'],
            seed_profile=args.data/'seeds'/record['seed_row']/'flame.npz',seed_row=record['seed_row'],
            width=settings['width_m'],max_spacing=settings['max_spacing_m'],
            residual_tolerance=settings['residual_tolerance'],max_iterations=settings['max_iterations'],
            max_energy_error=settings['limits']['energy_closure_relative'],**settings.get('solver_options',{}))
    except ReducedConvergenceError as error:
        profile,report=error.profile,error.report
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'reduced.npz',**profile)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(dict(accepted=report['accepted'],reason=report['reason'],iterations=report['iterations'],
               elapsed_seconds=report['elapsed_seconds'],flames=len(model.metadata['rows'])))
    if not report['accepted']:raise SystemExit(1)


if __name__=='__main__':main()
