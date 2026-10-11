"""Measure native multicomponent transport and validate each installed input.

python examples/benchmark_mechanisms.py --faces 32 --threads 4 --validate-cantera --output timings.json
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--faces', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--mechanisms', nargs='+', help='Catalog filenames; default: all installed')
    parser.add_argument('--validate-cantera', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.faces, args.threads, args.repeats) < 1:
        parser.error('faces, threads and repeats must be positive')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {args.output}')
    # Standalone process: set policy before importing NumPy/Numba/Cantera.
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['NUMBA_NUM_THREADS'] = str(args.threads)
    import numpy as np
    import numba
    from kflame.chemistry.backend import NativeSpeciesBackend
    from kflame.chemistry.catalog import available_mechanisms
    from kflame.chemistry.mechanism import load_mechanism
    ct = None
    if args.validate_cantera:
        import cantera as ct
    entries = available_mechanisms()
    if args.mechanisms:
        unknown = set(args.mechanisms)-{entry['filename'] for entry in entries}
        if unknown:
            parser.error(f'Unknown catalog filenames: {sorted(unknown)}')
        entries = [entry for entry in entries if entry['filename'] in args.mechanisms]
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Whole native multicomponent/Soret property evaluation; setup/JIT separate; no flame solves',
        faces=args.faces, numba_threads=numba.get_num_threads(), blas_threads_requested=1,
        pressure_Pa=101325., requested_temperatures_K=[600., 2300.],
        temperature_policy='Intersection of requested interval and common NASA validity interval', mechanisms=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def write():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for entry in entries:
        print(f"Benchmarking {entry['name']}...", flush=True)
        row = dict(name=entry['name'], filename=entry['filename'], species=entry['species'], reactions=entry['reactions'])
        if not entry['installed']:
            row['status'] = 'not-installed'
        else:
            try:
                path = Path(entry['path'])
                row['mechanism_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                started = time.perf_counter()
                mech = load_mechanism(path)
                backend = NativeSpeciesBackend(SimpleNamespace(P=101325., assume_finite_y=True,
                    case=SimpleNamespace(mech=str(path), transport_model='multicomponent', soret_enabled=True)), mech)
                row['parse_and_backend_setup_s'] = time.perf_counter()-started
                x = np.full(mech.n_species, 1e-8)
                for name, amount in {'H2': .15, 'CH4': .05, 'CO': .05, 'O2': .2, 'N2': .53,
                                     'H': .005, 'O': .003, 'OH': .004, 'H2O': .008}.items():
                    if name in mech.species_names:
                        x[mech.species_names.index(name)] = amount
                y = x*mech.molecular_weights
                y /= y.sum()
                lower = max(600., mech.min_temperature)
                upper = min(2300., mech.max_temperature)
                if lower >= upper:
                    raise ValueError('Requested temperatures do not overlap the common NASA interval')
                row['temperature_range_K'] = [lower, upper]
                row['common_thermo_range_K'] = [mech.min_temperature, mech.max_temperature]
                T = np.linspace(lower, upper, args.faces)
                Y = np.repeat(y[:, None], args.faces, axis=1)
                evaluate = lambda: backend.eval_multicomponent_face_transport(T, Y)
                started = time.perf_counter(); evaluate()
                row['first_call_including_jit_s'] = time.perf_counter()-started
                samples = []
                for _ in range(args.repeats):
                    started = time.perf_counter(); evaluate()
                    samples.append(time.perf_counter()-started)
                row['samples_s'] = samples
                row['warm_median_s'] = statistics.median(samples)
                if ct is not None:
                    gas = ct.Solution(str(path), transport_model='multicomponent')
                    actual = evaluate()
                    omega, hk = np.empty_like(Y), np.empty_like(Y)
                    backend.eval_grid_thermo_kinetics_into(T, Y, omega, hk)
                    for face in range(args.faces):
                        gas.TPY = T[face], 101325., Y[:, face]
                        np.testing.assert_allclose(actual[0][face], gas.density, rtol=2e-12)
                        np.testing.assert_allclose(actual[1][face], gas.thermal_conductivity, rtol=3e-7, atol=1e-12)
                        np.testing.assert_allclose(actual[3][:, :, face], gas.multi_diff_coeffs, rtol=3e-7, atol=1e-12)
                        np.testing.assert_allclose(actual[4][:, face], gas.thermal_diff_coeffs, rtol=3e-7, atol=1e-12)
                        np.testing.assert_allclose(omega[:, face], gas.net_production_rates*gas.molecular_weights,
                                                   rtol=3e-9, atol=1e-8)
                    row['cantera_gate'] = True
                row['status'] = 'passed'
            except Exception as error:
                row['status'] = 'failed'
                row['error'] = f'{type(error).__name__}: {error}'
        report['mechanisms'].append(row)
        write()
        print(entry['name'], row['status'], flush=True)
    report['status'] = 'failed' if any(row['status']=='failed' for row in report['mechanisms']) else 'passed-installed-mechanisms'
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write()
    return int(report['status']=='failed')


if __name__ == '__main__':
    raise SystemExit(main())
