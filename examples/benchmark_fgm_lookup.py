"""Compare identical states against the committed pre-optimization code.

python examples/benchmark_fgm_lookup.py --output runs/fgm_benchmark.json
Requires a Git checkout containing --baseline-ref (default 14af4ae).
First calls compile/warm up outside timed repetitions. Loading is timed once.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import kflame  # configure default BLAS threads before importing NumPy
import numpy as np
import scipy
import numba

from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo


def benchmark(bundle, output, baseline_ref='14af4ae'):
    root = Path(__file__).resolve().parents[1]
    bundle = Path(bundle)
    with tempfile.TemporaryDirectory(prefix='flampy_benchmark_') as temporary:
        def old_module(name, source):
            path = Path(temporary)/(name+'.py')
            path.write_bytes(subprocess.check_output(['git','show',f'{baseline_ref}:{source}'],cwd=root))
            spec = importlib.util.spec_from_file_location(name,path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        old = old_module('fgm_benchmark_baseline','src/kflame/fgm/nonadiabatic3d.py')
        old_chem = old_module('kinetics_benchmark_baseline','src/kflame/chemistry/kinetics.py')
        start = time.perf_counter(); baseline = old.NonAdiabaticFGM(bundle); old_load = time.perf_counter()-start
        start = time.perf_counter(); model = NonAdiabaticFGM(bundle); new_load = time.perf_counter()-start
        controls = []
        names = ['case_00_comparison.npz','case_03_comparison.npz','confirmatory_02_comparison.npz']
        for name in names:
            with np.load(bundle/name,allow_pickle=False) as d:
                controls.extend(np.column_stack([d[k][d['covered']][::4] for k in ('Z','C','h')]))
        controls = np.asarray(controls)
        def scalar(m):
            return [m.lookup(Z=Z,C=C,h=h) for Z,C,h in controls]
        batch = lambda: model.lookup_batch(Z=controls[:,0],C=controls[:,1],h=controls[:,2])
        reference, single, many = scalar(baseline), scalar(model), batch()
        differences = {}
        for key in ('T','Y','omega_C','qdot','rho','cp_mass','conductivity'):
            truth = np.asarray([r[key] for r in reference])
            current = np.asarray([r[key] for r in single])
            np.testing.assert_allclose(current,truth,rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(many[key],truth,rtol=1e-9,atol=1e-5)
            differences[key] = dict(scalar_absolute_max=float(np.max(abs(current-truth))),
                                    batch_absolute_max=float(np.max(abs(many[key]-truth))))
        def median(fn, repeats=5):
            times = []
            for _ in range(repeats):
                start = time.perf_counter(); fn(); times.append(time.perf_counter()-start)
            return dict(median_s=float(np.median(times)),runs_s=times)
        result = dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
            versions=dict(numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),
            OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
            baseline_commit=subprocess.check_output(['git','rev-parse',baseline_ref],cwd=root,text=True).strip(),
            query_states=len(controls),query_profiles=names,warmup_excluded=True,
            load_s=dict(baseline=old_load,optimized=new_load),differences=differences,
            lookup={name:median(fn) for name,fn in [
                ('baseline_scalar',lambda:scalar(baseline)),('optimized_scalar',lambda:scalar(model)),('optimized_batch',batch)]})
        with np.load(bundle/'case_00_comparison.npz',allow_pickle=False) as d:
            T,Y = d['T'],d['Y']
        mech = load_mechanism(model.metadata['mechanism'])
        thermo = NativeThermo(mech)
        concentration = thermo.density(T,model.metadata['pressure_Pa'],Y)[None,:]*Y*mech.inv_molecular_weights[:,None]
        gibbs = thermo.g_RT(T)
        slow, fast = old_chem.NativeKinetics(mech), NativeKinetics(mech)
        np.testing.assert_array_equal(slow.net_production_rates(T,concentration,gibbs),fast.net_production_rates(T,concentration,gibbs))
        result.update(chemistry_profile_nodes=len(T),chemistry_bitwise_equal=True,
            chemistry={name:median(lambda:k.net_production_rates(T,concentration,gibbs),3)
                       for name,k in [('baseline',slow),('optimized',fast)]},
            source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in [
                'src/kflame/chemistry/kinetics.py','src/kflame/fgm/search.py','src/kflame/fgm/nonadiabatic3d.py',
                'examples/benchmark_fgm_lookup.py']})
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path('docs/assets/nonadiabatic3d'))
    parser.add_argument('--baseline-ref',default='14af4ae')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.bundle,args.output,args.baseline_ref),indent=2))


if __name__=='__main__':
    main()
