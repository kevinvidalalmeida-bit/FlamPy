"""Refine accepted flame families against their smallest enthalpy-loss budget.

Every changed free-flame speed changes the family's prescribed burner mass
fluxes. Re-solve those burners from accepted native profiles, then check the
budget again. This operation never shifts h or inserts artificial species.
It uses training profiles only; withheld validation cases are not inputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from kflame import solve_flame, solve_burner_flame
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.fgm.nonadiabatic3d import build_nonadiabatic_table, training_enthalpy_quality


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def profile(path, thermo, feed_temperature, *, require_accepted=True):
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if require_accepted and (not meta.get('accepted') or meta.get('backend') != 'native_cpu'
                            or not meta.get('report', {}).get('grid_converged')):
        raise ValueError(f'An accepted native profile on a converged grid is required: {path}')
    with np.load(path/'flame.npz', allow_pickle=False) as data:
        hend = float(thermo.enthalpy_mass(data['T'][-1], data['Y'][:, -1]))
    hfeed = float(thermo.enthalpy_mass(feed_temperature, np.asarray(meta['inlet_Y'])))
    return meta, hfeed-hend


def family(source, output, generation, index, safety_factor, max_levels, max_points, reuse_from=None, max_time=60.):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completed = output/'family.json'
    mech = load_mechanism(generation['mechanism'])
    thermo = NativeThermo(mech)
    Tfeed, pressure = generation['temperature_K'], generation['pressure_Pa']
    tolerance = generation['max_energy_error']
    training = sorted((r for r in generation['rows'] if r['composition_index'] == index),
                      key=lambda r: r['loss_index'])
    phi = float(generation['phis'][index])
    current_ad = source/training[0]['output']
    current_burners = [source/r['output'] for r in training[1:]]
    if len(training) != len(generation['mass_flux_fractions'])+1:
        raise ValueError('Each family must have one reference and every requested loss row')
    attempts = []

    cached_family = Path(reuse_from)/output.name if reuse_from is not None else output
    cached_report = cached_family/'family.json'
    if cached_report.exists():
        saved = json.loads(cached_report.read_text(encoding='utf-8'))
        if saved['phi'] != phi or len(saved['rows']) != len(training):
            raise ValueError('Cached family must identify the same composition and every loss row')
        candidates = [cached_family/r['output'] for r in saved['rows']]
        validated = [profile(p, thermo, Tfeed) for p in candidates]
        ad_meta = validated[0][0]
        reference_flux = float(thermo.density(Tfeed, pressure, np.asarray(ad_meta['inlet_Y'])))*ad_meta['Su']
        expected_feed = np.asarray(training[0]['inlet_Y'])
        for j, ((meta, _), row, cached_row) in enumerate(zip(validated, training, saved['rows'])):
            if (cached_row['loss_index'] != row['loss_index']
                    or meta['temperature'] != Tfeed or meta['pressure'] != pressure
                    or meta['transport'] != generation['transport'] or meta['soret'] != generation['soret']
                    or not np.allclose(meta['inlet_Y'], expected_feed, rtol=1e-12, atol=1e-15)
                    or Path(resolve_mechanism(meta['mechanism'])).read_bytes() != Path(resolve_mechanism(generation['mechanism'])).read_bytes()
                    or meta['flow_type'] != ('free_flame' if j == 0 else 'isothermal_burner')):
                raise ValueError('Cached profiles differ from the requested physical family')
            if j and (not np.isclose(meta['mass_flux'], reference_flux*row['fraction'], rtol=1e-12)
                      or meta['heat_loss']['relative_energy_closure_error'] > tolerance):
                raise ValueError('Cached burners do not conserve the requested mass flux or energy')
        quality = training_enthalpy_quality(np.zeros(1), -np.array([d for m,d in validated])[None,:], tolerance)
        if not quality['passed']:
            raise ValueError('Cached family fails the current enthalpy budget')
        if cached_family != output:
            for row, origin in zip(saved['rows'], candidates):
                shutil.copytree(origin, output/row['output'])
            saved['reused_from'] = str(cached_family)
            saved['quality'] = quality
            write_json(completed, saved)
        return saved

    def options(meta):
        r = meta['refinement']
        return dict(mechanism=generation['mechanism'],
                    Y=dict(zip(mech.species_names, meta['inlet_Y'])), temperature=Tfeed,
                    pressure=pressure, width=meta['final_width'], transport=generation['transport'],
                    soret=generation['soret'], ratio=r['ratio'], slope=r['slope'], curve=r['curve'],
                    prune=0., max_points=max(max_points, r['max_points']),
                    rtol=meta['tolerances']['rtol'], atol=meta['tolerances']['atol'], max_time=max_time, species=())

    def solve_or_read(target, previous, settings, flux=None):
        if not (target/'metadata.json').exists():
            try:
                if flux is None:
                    solve_flame(output=target, initial_solution=previous, **settings)
                else:
                    solve_burner_flame(output=target, initial_solution=previous, mass_flux=flux, **settings)
            except RuntimeError:
                if not (target/'metadata.json').exists():
                    raise
        return profile(target, thermo, Tfeed, require_accepted=False)

    for cycle in range(max_levels):
        ad_meta, ad_defect = profile(current_ad, thermo, Tfeed)
        burner_data = [profile(p, thermo, Tfeed) for p in current_burners]
        weakest = min(d for m, d in burner_data)
        if weakest <= 0.:
            raise ValueError(f'Nonpositive burner enthalpy loss at phi={phi:g}')
        # Reserve margin for the change in losses when Su is recomputed.
        while abs(ad_defect) > safety_factor*tolerance*weakest:
            level = sum(a['kind'] == 'adiabatic' for a in attempts)+1
            if level > max_levels:
                raise RuntimeError(f'Adiabatic enthalpy budget exhausted at phi={phi:g}: {ad_defect:g} J/kg')
            settings = options(ad_meta)
            settings.update(slope=.5*ad_meta['refinement']['slope'],
                            curve=.5*ad_meta['refinement']['curve'])
            target = output/f'adiabatic_{level:02d}'
            ad_meta, ad_defect = solve_or_read(target, current_ad, settings)
            if not ad_meta['accepted'] or not ad_meta['report'].get('grid_converged'):
                raise RuntimeError(f'Adiabatic continuation failed numerical acceptance: {target}')
            current_ad = target
            attempts.append(dict(kind='adiabatic', level=level, defect_J_kg=ad_defect,
                                 nodes=ad_meta['nodes'], runtime_s=ad_meta['runtime_s']))
            write_json(output/'attempts.json', attempts)
            print(f'ENTHALPY phi={phi:g} adiabatic level={level}: {ad_defect:.3g} J/kg', flush=True)
        feed = np.asarray(ad_meta['inlet_Y'])
        mdot_ad = float(thermo.density(Tfeed, pressure, feed))*ad_meta['Su']
        next_burners = []
        for j, (row, previous, (meta, deficit)) in enumerate(zip(training[1:], current_burners, burner_data), 1):
            flux = mdot_ad*float(row['fraction'])
            for level in range(max_levels+1):
                if (level == 0 and np.isclose(meta['mass_flux'], flux, rtol=1e-12, atol=0.)
                        and meta['heat_loss']['relative_energy_closure_error'] <= tolerance):
                    break
                settings = options(meta)
                if level:
                    settings.update(slope=.5*meta['refinement']['slope'],
                                    curve=.5*meta['refinement']['curve'])
                target = output/f'burner_{j:02d}_cycle_{cycle:02d}_level_{level:02d}'
                meta, deficit = solve_or_read(target, previous, settings, flux)
                if not meta['accepted'] or not meta['report'].get('grid_converged'):
                    alternate = (next_burners[-1] if next_burners else
                                 current_burners[min(j, len(current_burners)-1)])
                    attempts.append(dict(kind='burner', loss_index=j, cycle=cycle, level=level,
                                         accepted=False, output=target.name, seed=str(previous),
                                         runtime_s=meta['runtime_s']))
                    write_json(output/'attempts.json', attempts)
                    if alternate != previous:
                        # Change the accepted continuation basin before adding
                        # more mesh nodes to the same unsuccessful trajectory.
                        target = output/(target.name+'_neighbor')
                        meta, deficit = solve_or_read(target, alternate, settings, flux)
                error = meta['heat_loss']['relative_energy_closure_error']
                attempts.append(dict(kind='burner', loss_index=j, cycle=cycle, level=level,
                                     mass_flux=flux, energy_error=error, nodes=meta['nodes'],
                                     accepted=meta['accepted'],
                                     output=target.name,
                                     runtime_s=meta['runtime_s']))
                write_json(output/'attempts.json', attempts)
                if meta['accepted'] and meta['report'].get('grid_converged'):
                    previous = target
                elif next_burners:
                    previous = next_burners[-1]
                if meta['accepted'] and meta['report'].get('grid_converged') and error <= tolerance:
                    break
            else:
                raise RuntimeError(f'Burner energy budget exhausted at phi={phi:g}, fraction={row["fraction"]:g}')
            next_burners.append(previous)
            print(f'ENTHALPY phi={phi:g} r={row["fraction"]:g}: closure={meta["heat_loss"]["relative_energy_closure_error"]:.3g}', flush=True)
        current_burners = next_burners
        defects = [profile(current_ad, thermo, Tfeed)[1]]+[profile(p, thermo, Tfeed)[1] for p in current_burners]
        quality = training_enthalpy_quality(np.zeros(1), -np.asarray(defects)[None, :], tolerance)
        if quality['passed']:
            rows = []
            for j, (row, origin) in enumerate(zip(training, [current_ad]+current_burners)):
                # Final rows always identify profiles within the output family.
                destination = output/('accepted_adiabatic' if j == 0 else f'accepted_burner_{j:02d}')
                if not destination.exists():
                    shutil.copytree(origin, destination)
                meta, _ = profile(destination, thermo, Tfeed)
                result = dict(row, output=destination.name, adiabatic_mass_flux=mdot_ad)
                if j == 0:
                    result['Su'] = meta['Su']
                else:
                    result.update(mass_flux=meta['mass_flux'], inlet_velocity=meta['inlet_velocity'],
                                  heat_loss=meta['heat_loss'])
                rows.append(result)
            result = dict(phi=phi, rows=rows, quality=quality, attempts=attempts)
            write_json(completed, result)
            return result
    raise RuntimeError(f'Coupled reference/burner enthalpy budget exhausted at phi={phi:g}')


def refine(source, output, *, jobs=2, safety_factor=.5, max_levels=6, max_points=8000, reuse_from=None, max_time=60.):
    source, output = Path(source).resolve(), Path(output).resolve()
    generation = json.loads((source/'generation.json').read_text(encoding='utf-8'))
    if not generation.get('all_final_accepted'):
        raise ValueError('The source must be a complete accepted native family')
    if (not 0. < safety_factor <= 1. or jobs < 1 or max_levels < 1 or max_points < 3
            or not np.isfinite(max_time) or max_time <= 0.):
        raise ValueError('Require positive budgets and a safety factor in (0,1]')
    source_root = Path(__file__).resolve().parents[1]/'src/kflame'
    fingerprint = hashlib.sha256(b''.join(p.relative_to(source_root).as_posix().encode()+p.read_bytes()
                                         for p in sorted(source_root.rglob('*.py')))).hexdigest()
    manifest = dict(source=str(source), source_generation_sha256=hashlib.sha256((source/'generation.json').read_bytes()).hexdigest(),
                    source_profile_sha256={r['output']:hashlib.sha256((source/r['output']/'flame.npz').read_bytes()).hexdigest()
                                           for r in generation['rows']},
                    mechanism_sha256=hashlib.sha256(Path(resolve_mechanism(generation['mechanism'])).read_bytes()).hexdigest(),
                    solver_source_sha256=fingerprint, safety_factor=safety_factor,
                    max_levels=max_levels, max_points=max_points,
                    max_time=max_time,
                    reuse_from=str(Path(reuse_from).resolve()) if reuse_from is not None else None,
                    refiner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    plan = output/'refinement_plan.json'
    if plan.exists() and json.loads(plan.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Existing refinement output identifies different inputs or source code')
    write_json(plan, manifest)
    shutil.copyfile(Path(__file__), output/'refiner_snapshot.py')
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(family, source, output/f'family_{i:03d}', generation, i,
                               safety_factor, max_levels, max_points, reuse_from, max_time) for i in range(len(generation['phis']))]
        families = [f.result() for f in futures]
    rows = []
    for i, result in enumerate(families):
        rows.extend(dict(r, output=f'family_{i:03d}/{r["output"]}') for r in result['rows'])
    updated = dict(generation, rows=rows, enthalpy_refinement=manifest, all_final_accepted=True)
    write_json(output/'generation.json', updated)
    build_nonadiabatic_table(output, progress_points=181)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--safety-factor', type=float, default=.5)
    parser.add_argument('--max-levels', type=int, default=6)
    parser.add_argument('--max-points', type=int, default=8000)
    parser.add_argument('--reuse-from', type=Path)
    parser.add_argument('--max-time', type=float, default=60.)
    args = parser.parse_args()
    refine(args.source, args.output, jobs=args.jobs, safety_factor=args.safety_factor,
           max_levels=args.max_levels, max_points=args.max_points, reuse_from=args.reuse_from, max_time=args.max_time)
