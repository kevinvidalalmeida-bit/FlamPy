"""Matched, resumable thesis flame sweeps. Large campaigns are user-launched.

Run --dry-run first. Phases: smoke, verify, main, ablation. Output is immutable
per attempt; only indexes are replaced atomically. No FGM tables are generated.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, replace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
THREADS = dict(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='4')
SCHEMA = 2
STRATEGIES = {'gri30.yaml': 'ptc-ser', 'h2o2.yaml': 'backward-euler'}
SPATIAL_PROTOCOL = dict(selected_level=4, reference_level=5, target_relative_change=.005,
                        rule='L4 versus L5 and L4 versus doubled final domain; denominator L4')
SLOPES = [.08, .04, .02, .01, .005, .0025]
PHIS = [.7, .9, 1., 1.1, 1.4]
MODES = [('mixture-averaged', False), ('mixture-averaged', True),
         ('multicomponent', False), ('multicomponent', True)]


def replace_file(source, destination, attempts=12):
    """Retry transient Windows sharing/access locks without deleting the target."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc,'winerror',None) in (5,32,33)):
                raise
            if attempt == attempts-1:
                raise
            time.sleep(min(.05 * 2**attempt, 1.0))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling avoids reopening a .tmp file held by an indexer/sync client.
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return None
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case(fuel, phi=1., pressure=1., temperature=300., transport=None, soret=None):
    transport = transport or ('mixture-averaged' if fuel == 'CH4' else 'multicomponent')
    if soret is None:
        soret = fuel == 'H2'
    c = dict(fuel=fuel, mechanism='gri30.yaml' if fuel == 'CH4' else 'h2o2.yaml',
             phi=float(phi), pressure_atm=float(pressure), temperature=float(temperature),
             transport=transport, soret=bool(soret))
    c['native_strategy'] = STRATEGIES[c['mechanism']]
    c['id'] = f'{fuel}_phi{phi:g}_p{pressure:g}_T{temperature:g}_{transport}_S{int(soret)}'
    return c


def cases():
    result = {}
    for fuel in ('CH4', 'H2'):
        for transport, soret in MODES:
            for phi in PHIS:
                c = case(fuel, phi, transport=transport, soret=soret)
                result[c['id']] = c
        for pressure in (1, 2, 3, 5, 10):
            c = case(fuel, pressure=pressure)
            result[c['id']] = c
        for temperature in (300, 350, 400, 450, 500):
            c = case(fuel, temperature=temperature)
            result[c['id']] = c
    return list(result.values())


def settings(level=4, width=.03, max_seconds=600):
    return dict(level=level, slope=SLOPES[level], curve=2*SLOPES[level], ratio=2.5,
                prune=0., width=width, initial_points=8, max_points=6000,
                rtol=1.e-5, atol=1.e-10, max_seconds=max_seconds,
                gradient_basis='molar', profile_seed=False)


def environment():
    packages = {n: metadata.version(n) for n in ('numpy', 'scipy', 'numba', 'llvmlite', 'cantera', 'PyYAML')}
    source = sorted((ROOT / 'src/kflame').rglob('*.py'))
    source += sorted((ROOT / 'src/kflame/chemistry/data').glob('*'))
    source += [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source if p.is_file()}
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                cpu=platform.processor(), cpu_count=os.cpu_count(), packages=packages,
                threads=THREADS, source_sha256=hashes)


def spatial_protocol(args):
    if getattr(args,'spatial_policy','verified-L4')=='fixed-L3':
        return dict(selected_level=3,reference_level=None,target_relative_change=None,
                    verification_required=False,status='user_selected_without_spatial_verification',
                    rule='Fixed L3 requested by user; retain solver acceptance and diagnostics')
    return SPATIAL_PROTOCOL


def production_selection(out,args):
    if spatial_protocol(args).get('verification_required') is False:
        selected={f:dict(settings=settings(3,max_seconds=args.max_seconds),
                        spatial_verification='not_performed',selection='user_fixed_L3') for f in ('CH4','H2')}
        atomic_json(out/'spatial_selection.json',selected)
        return selected
    path=out/'verification.json'
    selected=read(path) if path.exists() else {}
    if not all(selected.get(f,{}).get('passed') for f in ('CH4','H2')):
        raise ValueError('Run --phase verify successfully before main or ablation.')
    return selected


def ensure_manifest(out, args):
    value = dict(schema=SCHEMA, pairs=args.pairs, smoke=args.phase == 'smoke',
                 max_seconds=args.max_seconds, bootstrap_seed=20260927,
                 environment=environment(), cases=cases(),
                 native_strategies=STRATEGIES, spatial_protocol=spatial_protocol(args),
                 units=dict(z='m', T='K', u='m/s', rho='kg/m3', Y='kg/kg',
                            omega='kg/m3/s', h_species='J/kg', J='kg/m2/s',
                            qdot='W/m3', energy_flux='W/m2', time='s'),
                 protocol='Independent profiles; warm JIT; empty molecular caches; matched paired solves; no FGM')
    path = out / 'manifest.json'
    if path.exists():
        if not args.resume:
            raise ValueError('Output exists. Use --resume or a new directory.')
        if read(path) != value:
            raise ValueError('Incompatible code, environment, pairs or settings. Use a new output directory.')
        for name in ('gri30.yaml', 'h2o2.yaml', 'collision_integrals_mm.json'):
            if digest(out/'inputs'/name) != digest(ROOT/'src/kflame/chemistry/data'/name):
                raise ValueError('Archived physical input was modified: '+name)
    else:
        if any(p.name != '.campaign.lock' for p in out.iterdir()):
            raise ValueError('Output directory is not empty and has no manifest.')
        for name in ('gri30.yaml', 'h2o2.yaml', 'collision_integrals_mm.json'):
            destination = out / 'inputs' / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT / 'src/kflame/chemistry/data' / name, destination)
        atomic_json(path, value)
        # Archive runnable source provenance independently of future edits.
        code = out/'code'
        for relative in value['environment'].get('source_sha256',{}):
            dest=code/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,dest)


def configure_variant(variant):
    import kflame.chemistry.collision_integrals as collision
    import kflame.chemistry.multicomponent as multi
    if variant == 'fits-off':
        def uncached(mech, base):
            fits = collision._build_transport_fits(mech, base)
            for array in fits:
                array.setflags(write=False)
            return fits
        multi.native_transport_fits = uncached


def clear_molecular_caches():
    import kflame.chemistry.collision_integrals as collision
    import kflame.chemistry.transport as transport
    from kflame.chemistry.mechanism import _MECHANISM_CACHE
    collision._FIT_CACHE.clear()
    collision._CONDUCTIVITY_CACHE.clear()
    transport._PAIR_CACHE.clear()
    _MECHANISM_CACHE.clear()


def solve(c, s, backend, mechanism, variant):
    import numpy as np
    from kflame.flame.config import FlameCase
    fc = FlameCase(mech=str(mechanism), fuel=c['fuel'], oxidizer='O2:1, N2:3.76',
                   phi=c['phi'], T_in=c['temperature'], P=101325*c['pressure_atm'],
                   width=s['width'], transport_model=c['transport'], soret_enabled=c['soret'],
                   flux_gradient_basis='molar', steady_rtol=s['rtol'], steady_atol=s['atol'],
                   ratio=s['ratio'], slope=s['slope'], curve=s['curve'], prune=s['prune'])
    if backend == 'native':
        from kflame.benchmarks.soret import benchmark_options
        from kflame.chemistry.backend import NativeSpeciesBackend
        from kflame.flame.problem import FreeFlameProblem
        from kflame.flame.solver import solve_free_flame
        from kflame.flame.state import unpack_state
        opts = replace(benchmark_options(False), max_total_time_s=s['max_seconds'],
                       pseudo_time_method=STRATEGIES[c['mechanism']],
                       max_refine_passes=25, refine_max_points=s['max_points'],
                       refine_ratio=s['ratio'], refine_slope=s['slope'],
                       refine_curve=s['curve'], refine_prune=s['prune'],
                       compiled_block_substitution=variant != 'blocks-off',
                       multicomponent_bootstrap=c['transport'] == 'multicomponent',
                       bootstrap_mesh_factor=2., trace_solver=False)
        start = time.perf_counter()
        problem = FreeFlameProblem(fc, n_points=s['initial_points'])
        problem.assume_finite_y = True
        problem.backend_factory = NativeSpeciesBackend
        state, ok, report = solve_free_flame(problem, options=opts)
        elapsed = time.perf_counter()-start
        u, T, Y = unpack_state(state, problem.n_points, problem.n_species)
        fields = dict(z=problem.z.copy(), u=u.copy(), T=T.copy(), Y=Y.copy(),
                      species_names=np.asarray(problem.species_names))
        accepted = bool(ok and report.get('final_accepted') and report.get('grid_converged'))
        return fields, dict(time_s=elapsed, accepted=accepted, report=report,
                            effective_options=asdict(opts), case=asdict(fc))
    import cantera as ct
    start = time.perf_counter()
    gas = ct.Solution(str(mechanism))
    gas.TP = fc.T_in, fc.P
    gas.set_equivalence_ratio(fc.phi, fc.fuel, fc.oxidizer)
    flame = ct.FreeFlame(gas, width=fc.width)
    flame.transport_model = c['transport']
    flame.flux_gradient_basis = 'molar'
    flame.soret_enabled = c['soret']
    flame.flame.set_steady_tolerances(default=(s['rtol'], s['atol']))
    flame.flame.set_transient_tolerances(default=(s['rtol'], s['atol']))
    flame.set_refine_criteria(ratio=s['ratio'], slope=s['slope'], curve=s['curve'], prune=s['prune'])
    flame.set_max_grid_points(flame.flame, s['max_points'])
    def interrupt(_):
        if time.perf_counter()-start > s['max_seconds']:
            raise TimeoutError('Cantera solve time limit')
        return 0.
    flame.set_interrupt(interrupt)
    flame.solve(loglevel=0, auto=True, refine_grid=True)
    elapsed = time.perf_counter()-start
    assert flame.transport_model == c['transport'] and flame.soret_enabled == c['soret']
    fields = dict(z=flame.grid.copy(), T=flame.T.copy(), Y=flame.Y.copy(),
                  u=flame.velocity.copy(), species_names=np.asarray(gas.species_names))
    stats = {name: getattr(flame, name) for name in ('grid_size_stats', 'jacobian_count_stats',
             'jacobian_time_stats', 'eval_count_stats', 'eval_time_stats', 'time_step_stats')}
    return fields, dict(time_s=elapsed, accepted=True, report=stats, case=asdict(fc),
                        effective_options=dict(s, auto=True, transport=flame.transport_model,
                                               soret=flame.soret_enabled))


def diagnostics(fields, record, backend):
    """Own-backend properties and explicitly reconstructed midpoint balances."""
    import numpy as np
    from types import SimpleNamespace
    from kflame.chemistry.mechanism import load_mechanism, _ATOMIC_WEIGHTS
    from kflame.flame.equations import _corrected_flux, _multicomponent_flux
    if backend == 'native':
        from kflame.chemistry.backend import NativeSpeciesBackend as Backend
    else:
        from kflame.reference.backend import SpeciesBackend as Backend
    fc = SimpleNamespace(**record['case'])
    b = Backend(SimpleNamespace(case=fc, P=fc.P, flux_gradient_basis='molar', soret_enabled=fc.soret_enabled))
    z, T, Y, u = (fields[k] for k in ('z', 'T', 'Y', 'u'))
    n, k = len(z), Y.shape[0]
    rho, cp, lam = np.empty(n), np.empty(n), np.empty(n)
    omega, hmol = np.empty((k,n)), np.empty((k,n))
    for j in range(n):
        rho[j], _, cp[j], lam[j] = b.eval_node_into(T[j], Y[:,j], omega[:,j], hmol[:,j])
    h = hmol / b.W[:,None]
    tf, yf, dz = (T[:-1]+T[1:])/2, (Y[:,:-1]+Y[:,1:])/2, np.diff(z)
    if b.uses_multicomponent_flux:
        rf, lf, wf, multi, thermal = b.eval_multicomponent_face_transport(tf, yf)
        J = _multicomponent_flux(Y[:,:-1], Y[:,1:], T[:-1], T[1:], rf, wf, multi, thermal, dz, b.W)
    else:
        J, lf = np.empty((k,n-1)), np.empty(n-1)
        for j in range(n-1):
            rf, df, lf[j], wf = b.eval_midpoint_full_transport(T[j],T[j+1],Y[:,j],Y[:,j+1])
            J[:,j] = _corrected_flux(Y[:,j],Y[:,j+1],rf,df,dz[j],b.W,wf,'molar')
        if fc.soret_enabled:
            J -= b.eval_mixture_thermal_diffusion(tf,yf) * (np.diff(T)/(tf*dz))[None,:]
    mech = load_mechanism(fc.mech)
    element_matrix = mech.atom_matrix*np.array([_ATOMIC_WEIGHTS[e] for e in mech.element_names])[:,None]/b.W[None,:]
    mdot = rho*u
    species_flux = yf * ((mdot[:-1]+mdot[1:])/2)[None,:] + J
    element_flux = element_matrix @ species_flux
    hf = (h[:,:-1]+h[:,1:])/2
    qcond = -lf*np.diff(T)/dz
    energy_flux = np.sum(hf*species_flux,axis=0)+qcond
    element_inlet = element_matrix @ (mdot[0]*Y[:,0])
    mass_scale = max(abs(mdot[0]),1.e-30)
    energy_scale = max(mass_scale*float(np.max(cp))*abs(T[-1]-T[0]),1.e-30)
    fields.update(rho=rho, cp_mass=cp, conductivity=lam, omega=omega, h_species=h,
                  molecular_weights=b.W, qdot=-np.sum(h*omega,axis=0),
                  z_faces=(z[:-1]+z[1:])/2, J=J, species_flux=species_flux,
                  element_names=np.asarray(mech.element_names), element_matrix=element_matrix,
                  element_flux=element_flux, elemental_source=element_matrix@omega,
                  conductive_heat_flux=qcond, energy_flux=energy_flux)
    metrics = dict(Su=float(u[0]), Tb=float(T[-1]), nodes=n, width=float(z[-1]-z[0]),
                   thickness=float((T[-1]-T[0])/max(np.max(np.gradient(T,z)),1.e-30)),
                   qdot_peak=float(np.max(fields['qdot'])),
                   mass_error=float(np.max(np.abs(mdot-mdot.mean()))/max(abs(mdot.mean()),1.e-30)),
                   species_sum_error=float(np.max(abs(Y.sum(axis=0)-1))), min_Y=float(Y.min()),
                   elemental_error_mass_scaled=float(np.max(abs(element_flux-element_inlet[:,None]))/mass_scale),
                   energy_error_sensible_scaled=float(np.ptp(energy_flux)/energy_scale),
                   diffusive_mass_sum=float(np.max(abs(J.sum(axis=0)))),
                   flux_convention='Own backend; reconstructed midpoint convection + discrete diffusion; energy includes formation enthalpies; mesh-sensitive diagnostic')
    return metrics


def worker(request_path):
    for key, value in THREADS.items():
        os.environ[key] = value
    import numpy as np
    req = read(request_path)
    folder = Path(request_path).parent
    c, s, backend, variant = (req[k] for k in ('case','settings','backend','variant'))
    configure_variant(variant)
    result = dict(case_id=c['id'], backend=backend, variant=variant, accepted=False,
                  status='running', settings=s, condition=c,
                  nonlinear_strategy=STRATEGIES[c['mechanism']] if backend=='native' else 'cantera-default')
    atomic_json(folder/'result.json',result)
    stage='warmup'
    try:
        warm_s = dict(s, slope=.12, curve=.24, prune=0., max_points=min(s['max_points'],1200))
        clear_molecular_caches()
        started = time.perf_counter()
        warm_fields, warm = solve(c,warm_s,backend,req['mechanism'],variant)
        atomic_json(folder/'warmup.json',dict(warm, wall_s=time.perf_counter()-started))
        del warm_fields
        clear_molecular_caches()
        atomic_json(folder/'progress.json',dict(stage='measured',started_utc=time.time()))
        stage='measured'
        started = time.perf_counter()
        fields, measured = solve(c,s,backend,req['mechanism'],variant)
        result.update(measured)
        stage='diagnostics'
        result['status'] = 'accepted' if measured['accepted'] else 'rejected'
        # Save basic profiles before diagnostic reconstruction, even on rejection.
        with (folder/'profile.npz.tmp').open('wb') as f:
            np.savez_compressed(f,**fields)
        replace_file(folder/'profile.npz.tmp',folder/'profile.npz')
        atomic_json(folder/'result.json',result)
        result.update(diagnostics(fields,result,backend))
        with (folder/'profile.npz.tmp').open('wb') as f:
            np.savez_compressed(f,**fields)
        replace_file(folder/'profile.npz.tmp',folder/'profile.npz')
        result['diagnostics_complete'] = True
        result['profile_sha256'] = digest(folder/'profile.npz')
    except Exception as exc:
        traceback.print_exc()
        if 'time_s' in result:
            result.update(diagnostics_complete=False,diagnostic_error=repr(exc))
        else:
            result.update(status='failed',accepted=False,error=repr(exc),failed_stage=stage,
                          failed_stage_time_s=time.perf_counter()-started)
    atomic_json(folder/'result.json',result)
    return 0 if result.get('diagnostics_complete') else 1


def completed_attempt(pair_dir):
    for path in sorted(pair_dir.glob('attempt-*/pair.json'),reverse=True):
        data=read(path)
        if data.get('status') == 'complete':
            return path.parent
    return None


def pair_run(out, phase, c, repetition, s, variants, ordinal):
    directory=out/phase/c['id']/f'pair-{repetition:02d}'
    done=completed_attempt(directory)
    if done:
        info=read(done/'pair.json')
        if info['settings']!=s or set(info['order'])!=set(variants):
            raise ValueError('Saved pair settings or variants differ: '+str(done))
        return [read(done/name/'result.json') for name in read(done/'pair.json')['order']]
    directory.mkdir(parents=True,exist_ok=True)
    attempts=list(directory.glob('attempt-*'))
    for attempt in attempts:
        if (attempt/'pair.json').exists():
            info=read(attempt/'pair.json')
            if info['status']=='running':
                atomic_json(attempt/'pair.json',dict(info,status='interrupted'))
    attempt=directory/f'attempt-{len(attempts)+1:03d}'
    attempt.mkdir()
    order=list(variants)
    if (ordinal+repetition)%2:
        order.reverse()
    info=dict(status='running',order=order,case_id=c['id'],repetition=repetition,settings=s)
    atomic_json(attempt/'pair.json',info)
    results=[]
    for name in order:
        backend='cantera' if name=='cantera' else 'native'
        folder=attempt/name
        folder.mkdir()
        req=dict(case=c,settings=s,backend=backend,variant=name,
                 mechanism=str((out/'inputs'/c['mechanism']).resolve()))
        atomic_json(folder/'request.json',req)
        start=time.perf_counter()
        print(f'{phase}: {c["id"]}, par {repetition+1}, {name}',flush=True)
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(folder/'request.json')],
                                  stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,**THREADS),cwd=ROOT)
            try:
                code=proc.wait(timeout=2*s['max_seconds']+120)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); code=-1
            except KeyboardInterrupt:
                proc.kill(); proc.wait()
                atomic_json(attempt/'pair.json',dict(info,status='interrupted'))
                raise
        path=folder/'result.json'
        result=read(path) if path.exists() else dict(accepted=False)
        if result.get('status')=='running' or not path.exists():
            result.update(status='timeout' if code==-1 else 'failed',accepted=False)
        result.update(process_s=time.perf_counter()-start,returncode=code,
                      phase=phase,case_id=c['id'],condition=c,variant=name,backend=backend,
                      repetition=repetition,order=order,folder=str(folder.relative_to(out)))
        if not result.get('diagnostics_complete'):
            result['usable']=False
        else:
            result['usable']=bool(result.get('accepted'))
        atomic_json(path,result)
        results.append(result)
        print(f'  {result["status"]}; tiempo={result.get("time_s")}; diagnósticos={result.get("diagnostics_complete",False)}',flush=True)
    atomic_json(attempt/'pair.json',dict(info,status='complete'))
    refresh_index(out)
    return results


def refresh_index(out):
    import csv
    records=[]
    for p in out.glob('*/**/pair-*/attempt-*/pair.json'):
        if read(p).get('status')=='complete' and completed_attempt(p.parent.parent)==p.parent:
            records.extend(read(p.parent/name/'result.json') for name in read(p)['order'])
    atomic_json(out/'index.json',records)
    cols=['phase','case_id','repetition','variant','nonlinear_strategy','status','accepted','usable','time_s','process_s','Su','nodes','width','folder']
    with (out/'index.csv.tmp').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); writer.writeheader(); writer.writerows(records)
    replace_file(out/'index.csv.tmp',out/'index.csv')


def relative_change(a,b):
    return abs(a-b)/max(abs(b),1.e-30)


def verify(out,args):
    """Certify the user-selected L4 against L5 and a doubled domain."""
    selection={}
    target=SPATIAL_PROTOCOL['target_relative_change']
    for fuel in ('CH4','H2'):
        checks=[]
        anchors=[case(fuel),case(fuel,pressure=10),case(fuel,temperature=500)]
        for i,c in enumerate(anchors):
            levels={}
            for level in (4,5):
                rows=pair_run(out,f'verify-L{level}',c,0,settings(level,max_seconds=args.max_seconds),['native','cantera'],i)
                levels[level]={r['backend']:r for r in rows}
            usable=all(levels[l][b].get('usable') for l in (4,5) for b in ('native','cantera'))
            domain={}
            if usable:
                width=2*max(r['width'] for r in levels[4].values())
                rows=pair_run(out,'domain-L4',c,0,settings(4,width,args.max_seconds),['native','cantera'],i)
                domain={r['backend']:r for r in rows}
            for backend in ('native','cantera'):
                base,ref=levels[4][backend],levels[5][backend]
                mesh=relative_change(ref['Su'],base['Su']) if base.get('usable') and ref.get('usable') else None
                dom=domain.get(backend,{})
                change=relative_change(dom['Su'],base['Su']) if base.get('usable') and dom.get('usable') else None
                check=dict(case_id=c['id'],backend=backend,mesh_change=mesh,domain_change=change,
                           passed=mesh is not None and mesh<target and change is not None and change<target)
                checks.append(check)
                print(f"{c['id']} / {backend}: mesh={mesh}, domain={change}, passed={check['passed']}",flush=True)
            selection[fuel]=dict(passed=len(checks)==6 and all(v['passed'] for v in checks),
                settings=settings(4,max_seconds=args.max_seconds),target_relative_change=target,
                level_reached=5,selected_level=4,checks=checks,spatial_protocol=SPATIAL_PROTOCOL)
            atomic_json(out/'verification.json',selection)
    if not all(v['passed'] for v in selection.values()):
        raise RuntimeError('L4 sensitivity target 0.5% not met or a solve failed. Inspect verification.json; main remains blocked.')


@contextlib.contextmanager
def campaign_lock(out):
    # OS lock releases on process death, including an interrupted Windows terminal.
    path=out/'.campaign.lock'
    stream=path.open('a+b')
    stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Another campaign process is using this output directory.')
    try:
        yield
    finally:
        stream.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['smoke','verify','main','ablation'],default='main')
    parser.add_argument('--pairs',type=int,default=5)
    parser.add_argument('--spatial-policy',choices=['verified-L4','fixed-L3'],default='verified-L4',
                        help='fixed-L3 starts production directly, without a spatial verification prerequisite')
    parser.add_argument('--output',type=Path,default=Path('runs/thesis_flames_L4'))
    parser.add_argument('--max-seconds',type=float,default=600.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.phase=='verify' and args.spatial_policy=='fixed-L3':
        parser.error('fixed-L3 has no verification phase; use --phase main')
    if args.pairs<1 or args.max_seconds<=0:
        parser.error('pairs and max-seconds must be positive')
    env=environment()
    print(f'Python: {sys.executable}; Cantera {env["packages"]["cantera"]}')
    print(f'56 condiciones; {args.pairs} pares; {56*args.pairs*2} resoluciones principales. Fases adicionales y calentamientos aparte.')
    if args.dry_run:
        if (args.output/'manifest.json').exists():
            args.resume=True
            ensure_manifest(args.output,args)
        print(json.dumps(dict(phase=args.phase,output=str(args.output.resolve()),native_strategies=STRATEGIES,
                             spatial_protocol=spatial_protocol(args),verification_solves=0 if args.spatial_policy=='fixed-L3' else 36,cases=cases()),indent=2))
        return 0
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    # A lock file is infrastructure, not scientific data.
    if not (out/'manifest.json').exists() and any(p.name!='.campaign.lock' for p in out.iterdir()):
        raise ValueError('Nonempty directory without manifest')
    with campaign_lock(out):
        # ensure_manifest ignores the OS lock itself.
        ensure_manifest(out,args)
        if args.phase=='smoke':
            c=case('H2',transport='mixture-averaged',soret=False)
            rows=pair_run(out,'smoke',c,0,settings(0,max_seconds=args.max_seconds),['native','cantera'],0)
            if not all(r.get('usable') for r in rows):
                raise RuntimeError('Smoke comparison failed; inspect the saved logs.')
        elif args.phase=='verify':
            verify(out,args)
        else:
            selected=production_selection(out,args)
            planned=cases() if args.phase=='main' else [case('CH4'),case('H2')]
            for i,c in enumerate(planned):
                variants=['native','cantera'] if args.phase=='main' else ['native','blocks-off' if c['fuel']=='CH4' else 'fits-off']
                for pair in range(args.pairs):
                    pair_run(out,args.phase,c,pair,selected[c['fuel']]['settings'],variants,i)
        refresh_index(out)
    print(f'Guardado: {out}',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrumpido. Use el mismo comando con --resume.',file=sys.stderr)
        raise SystemExit(130)
