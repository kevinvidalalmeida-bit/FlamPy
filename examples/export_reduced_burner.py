"""Publish reviewable heat-loss arrays, verified seeds and numerical reports.

Raw private work folders and thesis files are never copied. Metadata paths
to bundled mechanisms are made portable; numerical input fingerprints remain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from examples.validate_reduced_burner import write_json


def export(validation, reserved, table, destination):
    destination,table=Path(destination),Path(table)
    destination.mkdir(parents=True,exist_ok=True)
    source=Path(validation);extra=Path(reserved)
    def copy_public(path,target):
        # Canonical LF avoids manifest changes after Git's text normalization.
        if Path(path).suffix=='.json':
            write_json(target,json.loads(Path(path).read_text(encoding='utf-8')))
        else:shutil.copyfile(path,target)
    for name in ['plan.json','convergence.json','benchmark.json','parabolicity_audit.json','before_after.json','sensitivity.json']:
        if (source/name).exists():copy_public(source/name,destination/name)
    copy_public(extra/'plan.json',destination/'reserved_plan.json')
    rows=[]
    for root in [source,extra]:
        for folder in sorted(root.glob('case_*')):
            if not (folder/'validation.json').exists():continue
            target=destination/folder.name;target.mkdir(exist_ok=True)
            for name in ['comparison.npz','reduced.npz','reduced_report.json','validation.json','reference.npz']:
                copy_public(folder/name,target/name)
            native=target/'native';native.mkdir(exist_ok=True)
            shutil.copyfile(folder/'native/flame.npz',native/'flame.npz')
            meta=json.loads((folder/'native/metadata.json').read_text(encoding='utf-8'))
            meta['mechanism']=Path(meta['mechanism']).name
            if 'continuation_from' in meta:meta['continuation_from']='accepted_native_profile'
            write_json(native/'metadata.json',meta)
            rows.append(json.loads((folder/'validation.json').read_text(encoding='utf-8')))
        for seed in (root/'seeds').rglob('flame.npz'):
            target=destination/'seeds'/seed.relative_to(root/'seeds');target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(seed,target)
    meta=json.loads((table/'metadata.json').read_text(encoding='utf-8'))
    shutil.copyfile(table/'reactor_tail.npz',destination/'reactor_tail.npz')
    write_json(destination/'extended_metadata.json',meta)
    summary=dict(cases=rows,cases_total=len(rows),converged=sum(v['accepted'] for v in rows),
                 passed=sum(v['passed'] for v in rows),all_passed=all(v['passed'] for v in rows),
                 development_case_count=len(json.loads((source/'summary.json').read_text(encoding='utf-8'))['cases']),
                 reserved_case_count=len(json.loads((extra/'summary.json').read_text(encoding='utf-8'))['cases']),
                 table_flames=len(meta['rows']),table_progress_points=meta['progress_points'],
                 progress_species=meta['progress_species'],
                 table_sha256=hashlib.sha256((table/'nonadiabatic_fgm.npz').read_bytes()).hexdigest())
    write_json(destination/'summary.json',summary)
    manifest=dict(species_names=meta['species_names'] if 'species_names' in meta else list(np.load(table/'nonadiabatic_fgm.npz')['species_names']),
                  table_sha256=summary['table_sha256'],files={})
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name!='manifest.json':
            manifest['files'][path.relative_to(destination).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    repository = Path(__file__).resolve().parents[1]
    code_paths = [
        'src/kflame/fgm/reduced_burner.py', 'src/kflame/fgm/nonadiabatic3d.py',
        'src/kflame/fgm/tensor_kernels.py','src/kflame/fgm/projection.py',
        'src/kflame/fgm/chart_guard.py','src/kflame/flame/burner_grid.py',
        'src/kflame/chemistry/thermo.py', 'src/kflame/chemistry/transport.py',
        'src/kflame/flame/equations.py', 'examples/extend_reduced_fgm_tail.py',
        'examples/retabulate_reduced_progress.py', 'examples/reduced_progress_weights.json',
        'examples/validate_reduced_burner.py', 'examples/audit_reduced_diffusion.py',
        'examples/benchmark_reduced_burner.py', 'examples/check_reduced_burner_convergence.py',
        'examples/plot_reduced_burner.py', 'examples/export_reduced_burner.py',
        'examples/reduced_burner_development_settings.json',
        'examples/reduced_burner_confirmation_settings.json',
        'examples/reduced_burner_hpc_development_settings.json',
        'examples/reduced_burner_hpc_confirmation_settings.json',
        'examples/reduced_library_432_hpc_selection.json',
        'examples/redistribute_reduced_library.py','examples/cache_reduced_library_tail.py',
        'examples/reconstruct_hpc_table.py','examples/validate_nonadiabatic_3d.py',
        'tests/test_reduced_burner.py','tests/test_tensor_kernels.py',
        'tests/test_progress_projection.py','tests/test_burner_grid.py',
        'tests/test_reactor_branch.py',
    ]
    manifest['code_sha256'] = {name: hashlib.sha256((repository/name).read_bytes()).hexdigest()
                               for name in code_paths}
    write_json(destination/'manifest.json',manifest)
    print({k:v for k,v in summary.items() if k!='cases'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['validation','reserved','table','destination']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();export(args.validation,args.reserved,args.table,args.destination)
