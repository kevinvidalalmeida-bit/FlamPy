"""Check installed catalog inputs through the native backend against Cantera.

This is a property/rate integration check, not a converged-flame benchmark.
Run from the repository root: python examples/validate_mechanisms.py --output report.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np

from kflame.chemistry.analytic import species_partials, species_partials_range
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.catalog import available_mechanisms
from kflame.chemistry.mechanism import load_mechanism


def check_entry(entry):
    import cantera as ct

    path = Path(entry['path'])
    mechanism = load_mechanism(path)
    assert (mechanism.n_species, mechanism.n_reactions) == (entry['species'], entry['reactions'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gas = ct.Solution(str(path), transport_model='mixture-averaged')
    problem = SimpleNamespace(P=ct.one_atm, case=SimpleNamespace(mech=str(path)),
                              assume_finite_y=True)
    backend = NativeSpeciesBackend(problem, mechanism)
    # Representative stable species and radicals; every other species stays
    # positive so the complete reaction network is exercised.
    fractions = np.full(mechanism.n_species, 1e-8)
    for name, fraction in {'H2': .15, 'CH4': .05, 'CO': .05, 'O2': .2,
                            'N2': .53, 'H': .005, 'O': .003, 'OH': .004,
                            'H2O': .008}.items():
        if name in mechanism.species_names:
            fractions[mechanism.species_names.index(name)] = fraction
    temperatures = np.array([500., 1100., 2300.])
    gas.TPX = temperatures[0], problem.P, fractions
    composition = np.repeat(gas.Y[:, None], len(temperatures), axis=1)
    density, diffusion, cp, conductivity, omega, enthalpy = backend.eval_grid_native(
        temperatures, composition)
    fused, fused_enthalpy = np.empty_like(composition), np.empty_like(composition)
    backend.eval_grid_thermo_kinetics_into(temperatures, composition, fused, fused_enthalpy)
    for column, temperature in enumerate(temperatures):
        gas.TPY = temperature, problem.P, composition[:, column]
        np.testing.assert_allclose(density[column], gas.density, rtol=2e-12)
        np.testing.assert_allclose(cp[column], gas.cp_mass, rtol=2e-12)
        np.testing.assert_allclose(enthalpy[:, column], gas.partial_molar_enthalpies,
                                   rtol=2e-12, atol=1e-7)
        np.testing.assert_allclose(diffusion[:, column], gas.mix_diff_coeffs, rtol=2e-8)
        np.testing.assert_allclose(conductivity[column], gas.thermal_conductivity, rtol=2e-8)
        reference = gas.net_production_rates * gas.molecular_weights
        np.testing.assert_allclose(omega[:, column], reference, rtol=3e-9, atol=1e-8)
        np.testing.assert_allclose(fused[:, column], reference, rtol=3e-9, atol=1e-8)
        np.testing.assert_allclose(fused_enthalpy[:, column], enthalpy[:, column],
                                   rtol=2e-12, atol=1e-7)
    # Admission through the actual analytic entry point catches unsupported
    # reaction orders. Compare full and tiled traversal byte for byte.
    T, Y = temperatures[1:2].copy(), composition[:, 1:2].copy()
    whole = species_partials(backend, T, Y, _temperature=True)
    tiled = species_partials_range(backend, T, Y, 0, 1)
    for first, second in zip(whole, tiled):
        assert np.isfinite(first).all()
        assert first.tobytes() == second.tobytes()
    return dict(name=entry['name'], filename=path.name, location=entry['location'],
                species=mechanism.n_species, reactions=mechanism.n_reactions,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status='passed', native_transport=True, native_fused_kinetics=True,
                analytic_full_tiled_byte_exact=True,
                cantera_warnings=[str(warning.message) for warning in caught])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite validation evidence: {args.output}')
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Native mixture transport, thermo, fused kinetics and analytic admission; no flame solves',
                  temperatures_K=[500., 1100., 2300.], pressure_Pa=101325., mechanisms=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry in available_mechanisms():
        print(f"Checking {entry['name']}...", flush=True)
        if not entry['installed']:
            result = dict(name=entry['name'], status='not-installed')
        else:
            try:
                result = check_entry(entry)
            except Exception as error:
                result = dict(name=entry['name'], status='failed',
                              error=f'{type(error).__name__}: {error}')
        report['mechanisms'].append(result)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f"{entry['name']}: {result['status']}", flush=True)
    failed = any(entry['status'] == 'failed' for entry in report['mechanisms'])
    report['status'] = 'failed' if failed else 'passed-installed-mechanisms'
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
