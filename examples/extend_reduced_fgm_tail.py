"""Append a constant-pressure chemical relaxation tail, without new flames.

Original resolved vertices are preserved exactly. Extra states come from a
native homogeneous adiabatic reactor starting at each finite-domain endpoint.
The relaxation path retains chemistry, elements and total enthalpy. It is a
post-flame closure, not a new spatial flamelet; validate it a posteriori.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from kflame.chemistry.initialization import hp_equilibrium
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.transport import NativeTransport
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM, _connected_cells


def reactor_tail(model, mech, kinetics, T0, Y0, h0, equilibrium, points, *, endpoint_policy='hp_equilibrium',monotonicity_tolerance=1e-10):
    """Sample a native chemical path; the branch policy never inserts HP Y."""
    weights = model.table['progress_weights']
    initial = np.maximum(Y0, 0.)
    initial /= initial.sum()
    C0, Ceq = float(weights@initial), float(weights@equilibrium)
    def rhs(time, state):
        T, Y = state[0], state[1:]
        rho = model.thermo.density(T, model.metadata['pressure_Pa'], Y)
        if Y.ndim == 2:
            concentrations = rho[None, :]*Y*mech.inv_molecular_weights[:, None]
        else:
            concentrations = rho*Y*mech.inv_molecular_weights
        omega = kinetics.net_production_rates(T, concentrations, model.thermo.g_RT(T))
        heat = -np.sum(model.thermo.partial_molar_enthalpies(T)*omega, axis=0)
        mass = omega*mech.molecular_weights[:, None] if Y.ndim == 2 else omega*mech.molecular_weights
        temperature_rate = heat/rho/model.thermo.cp_mass(T, Y)
        return np.vstack([temperature_rate, mass/rho]) if Y.ndim == 2 else np.r_[temperature_rate, mass/rho]
    # Tiny progress gains are limited by absolute integration accuracy.
    # The final HP endpoint is retained and its remaining gap is recorded.
    gap_limit = max(1e-3*(Ceq-C0), 1e-9)
    target = Ceq-gap_limit
    def close_to_equilibrium(time, state):
        return float(weights@state[1:]-target)
    close_to_equilibrium.terminal = True
    close_to_equilibrium.direction = 1
    def turning_point(time,state):
        return float(weights@rhs(time,state)[1:])
    turning_point.terminal=True
    turning_point.direction=-1
    def species_equilibrium(time,state):
        return float(np.max(abs(state[1:]-equilibrium))-1e-9)
    species_equilibrium.terminal=True
    species_equilibrium.direction=-1
    if endpoint_policy not in ('hp_equilibrium','monotone_branch'):
        raise ValueError('Unknown reactor endpoint policy')
    if endpoint_policy=='monotone_branch' and turning_point(0.,np.r_[T0,initial])<=0.:
        raise ValueError('Physical endpoint has no increasing chemical branch for C')
    events=([close_to_equilibrium] if Ceq-C0>gap_limit else [species_equilibrium]) if endpoint_policy=='monotone_branch' else close_to_equilibrium
    solved = solve_ivp(rhs, (0., 3e6), np.r_[T0, initial], method='BDF',
                       vectorized=True, rtol=1e-9, atol=np.r_[1e-8, np.full(len(Y0), 1e-15)],
                       dense_output=True, events=events)
    if not solved.success or (endpoint_policy=='hp_equilibrium' and not any(len(v) for v in solved.t_events)):
        raise RuntimeError('Native reactor did not approach the HP progress endpoint')
    C = weights@solved.y[1:]
    end_time=float(solved.t[-1]);reason='native_progress_threshold' if any(len(v) for v in solved.t_events) else 'integration_time_limit'
    if endpoint_policy=='monotone_branch':
        # Ignore changes below the declared numerical C resolution. Find the
        # first resolved decrease, then use the preceding actual maximum.
        # Small trace-species transients must not create a zero-length tail.
        drop=np.flatnonzero(np.maximum.accumulate(C)-C>monotonicity_tolerance)
        if len(drop):
            index=int(np.argmax(C[:drop[0]]))
            if index==0:raise ValueError('No resolved increasing chemical branch')
            lo,hi=float(solved.t[index-1]),float(solved.t[min(index+1,len(C)-1)])
            if turning_point(lo,solved.sol(lo))>0. and turning_point(hi,solved.sol(hi))<0.:
                end_time=brentq(lambda time:turning_point(time,solved.sol(time)),lo,hi,xtol=1e-12)
            else:end_time=float(solved.t[index])
            reason='first_resolved_progress_turn'
        mask=solved.t<end_time
        curve_times=np.r_[solved.t[mask],end_time]
        curve_C=np.r_[C[mask],float(weights@solved.sol(end_time)[1:])]
        if np.max(np.maximum.accumulate(curve_C)-curve_C)>monotonicity_tolerance:
            raise ValueError('Selected chemical branch is not monotone at its declared resolution')
        endpoint=solved.sol(end_time)[1:]
    else:
        if np.min(np.diff(C)) < -1e-10:raise ValueError('Chemical relaxation is not monotone for the progress definition')
        endpoint=equilibrium
    endpoint_C=float(weights@endpoint)
    if endpoint_C<=C0:
        raise ValueError('Resolved chemical branch has no positive progress extent')
    levels = (C0+np.linspace(1./points,1.,points)*(endpoint_C-C0) if endpoint_policy=='monotone_branch'
              else np.r_[C0+np.linspace(1./points,(points-1.)/points,points-1)*(target-C0),Ceq])
    times=[]
    for level in levels[:-1]:
        lo,hi=0.,end_time
        if endpoint_policy=='monotone_branch':
            crossing=int(np.flatnonzero(curve_C>=level)[0])
            lo,hi=float(curve_times[max(0,crossing-1)]),float(curve_times[crossing])
        times.append(brentq(lambda time:float(weights@solved.sol(time)[1:]-level),lo,hi,xtol=1e-12))
    Y = np.vstack([solved.sol(times)[1:].T, endpoint])
    raw_sum_error = float(np.max(abs(Y.sum(axis=1)-1.)))
    removed = float(np.max(np.maximum(-Y, 0.).sum(axis=1)))
    if np.min(Y) < -1e-12 or raw_sum_error > 1e-9:
        raise ValueError('Reactor violated nonnegative fractions or mass conservation')
    Y = np.maximum(Y, 0.)
    Y /= Y.sum(axis=1, keepdims=True)
    elements = mech.atom_matrix*mech.inv_molecular_weights[None, :]
    element_error = float(np.max(abs((Y-initial)@elements.T)))
    if element_error > 1e-9:
        raise ValueError('Reactor violated element conservation')
    return Y, dict(chemical_time_s=end_time, evaluations=solved.nfev,
                   roundoff_mass_removed=removed, raw_sum_Y_error=raw_sum_error,
                   element_error=element_error,
                   relative_progress_gap=float(abs(Ceq-weights@endpoint)/max(abs(Ceq-C0),1e-12)),
                   endpoint_species_gap=float(np.max(abs(equilibrium-endpoint))),
                   endpoint_policy=endpoint_policy,
                   monotonicity_tolerance=monotonicity_tolerance,endpoint_reason=reason)


def extend(source, output, *, tail_points=16, endpoints=None, reactor_cache=None):
    if not isinstance(tail_points, int) or tail_points < 3:
        raise ValueError('At least three tail samples are required')
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    model = NonAdiabaticFGM(source)
    mech = load_mechanism(model.metadata['mechanism'])
    kinetics, transport = NativeKinetics(mech), NativeTransport(mech)
    t, shape = model.table, tuple(int(v) for v in model.table['structured_shape'])
    old_Y = t['Y'].reshape(*shape, mech.n_species)
    old_T, old_h = t['T'].reshape(shape), t['h'].reshape(shape)
    end_Y, end_h = old_Y[:, :, -1].reshape(-1, mech.n_species), old_h[:, :, -1].ravel()
    rounding = np.maximum(-end_Y, 0.).sum(axis=1)
    if np.min(end_Y) < -1e-12:
        raise ValueError('Negative endpoint fractions exceed the roundoff allowance')
    cached = None
    source_digest = hashlib.sha256((source/'nonadiabatic_fgm.npz').read_bytes()).hexdigest()
    if reactor_cache is not None:
        with np.load(reactor_cache, allow_pickle=False) as data:
            if str(data['source_table_sha256']) != source_digest:
                raise ValueError('Reactor cache identifies a different source table')
            cached = data['Y']
            audits = json.loads(str(data['audit_json']))
            cache_equilibrium = data['HP_Y'] if 'HP_Y' in data else cached[:,-1]
        if cached.shape != (len(end_Y), tail_points, mech.n_species):
            raise ValueError('Reactor cache has incompatible shape or tail sampling')
        if (not np.isfinite(cached).all() or np.min(cached) < 0.
                or np.max(abs(cached.sum(axis=2)-1.)) > 1e-9):
            raise ValueError('Invalid reactor-cache mass fractions')
        equilibrium = cache_equilibrium
    elif endpoints is None:
        equilibrium = []
        for n, y in enumerate(end_Y):
            seed = np.maximum(y, 0.)
            seed /= seed.sum()
            _, ye, _ = hp_equilibrium(mech, float(old_T[:, :, -1].ravel()[n]),
                                      model.metadata['pressure_Pa'], seed)
            equilibrium.append(ye)
            if n % 80 == 0:
                print(f'Equilibrium endpoint {n}/{len(end_Y)}', flush=True)
        equilibrium = np.asarray(equilibrium)
    else:
        with np.load(endpoints, allow_pickle=False) as data:
            equilibrium = data['Y']
    if (equilibrium.shape != end_Y.shape or not np.isfinite(equilibrium).all()
            or np.min(equilibrium) < 0. or np.max(abs(equilibrium.sum(axis=1)-1.)) > 1e-9):
        raise ValueError('Equilibrium endpoint cache is incompatible with this table')
    elements = mech.atom_matrix*mech.inv_molecular_weights[None, :]
    if np.max(abs((equilibrium-end_Y)@elements.T)) > 1e-9:
        raise ValueError('Equilibrium endpoints do not conserve the source-table elements')
    growth = ((cached[:,-1] if cached is not None else equilibrium)-end_Y) @ t['progress_weights']
    if np.any(growth <= 0.):
        raise ValueError('Equilibrium extension is nonmonotone for this progress definition')
    fractions = np.linspace(1./tail_points, 1., tail_points)
    if cached is None:
        tails, audits = [], []
        for n, y in enumerate(end_Y):
            tail, audit = reactor_tail(model, mech, kinetics, float(old_T[:, :, -1].ravel()[n]),
                                       y, end_h[n], equilibrium[n], tail_points)
            tails.append(tail)
            audits.append(audit)
            if n % 60 == 0:
                print(f'Native reactor tail {n}/{len(end_Y)}', flush=True)
        Y = np.asarray(tails)
    else:
        Y = cached
    if np.max(abs((Y-end_Y[:, None, :])@(mech.atom_matrix*mech.inv_molecular_weights[None, :]).T)) > 1e-9:
        raise ValueError('Tail does not conserve the original flamelet elements')
    h = np.broadcast_to(end_h[:, None], Y.shape[:2]).ravel()
    flat_Y = Y.reshape(-1, mech.n_species).T
    guess = np.repeat(old_T[:, :, -1].ravel(), tail_points)
    T = model._recover_temperature(h, flat_Y, guess)
    rho = model.thermo.density(T, model.metadata['pressure_Pa'], flat_Y)
    omega = kinetics.net_production_rates(T, rho[None, :]*flat_Y*mech.inv_molecular_weights[:, None], model.thermo.g_RT(T))
    X = model.thermo.Y_to_X(flat_Y)
    values = dict(Y=flat_Y.T, T=T, h=h, rho=rho, cp_mass=model.thermo.cp_mass(T, flat_Y),
                  Z=t['bilger_weights'] @ flat_Y+float(t['bilger_offset']), C=t['progress_weights'] @ flat_Y,
                  omega_C=t['progress_weights'] @ (omega*mech.molecular_weights[:, None]),
                  qdot=-np.sum(model.thermo.partial_molar_enthalpies(T)*omega, axis=0),
                  conductivity=transport.thermal_conductivity(T, X))
    if 'omega_Y' in t:
        values['omega_Y']=(omega*mech.molecular_weights[:,None]).T
    payload = {k:v.copy() for k,v in t.items()}
    new_shape = (shape[0], shape[1], shape[2]+tail_points)
    for name, value in values.items():
        vector = name in ('Y','omega_Y')
        tail_shape = (shape[0], shape[1], tail_points, mech.n_species) if vector else (shape[0], shape[1], tail_points)
        original = t[name].reshape((*shape, mech.n_species) if vector else shape)
        joined = np.concatenate([original, np.asarray(value).reshape(tail_shape)], axis=2)
        payload[name] = joined.reshape(-1, mech.n_species) if vector else joined.ravel()
    controls = np.column_stack([payload[k] for k in ('Z', 'C', 'h')])
    cells, triangles, statistics = _connected_cells(new_shape, controls)
    if statistics['excluded_folded_cells'] or statistics['excluded_degenerate_cells']:
        raise ValueError(f'Extension created unusable connected cells: {statistics}')
    payload.update(controls=controls, cells=cells, structured_shape=np.asarray(new_shape),
                   reference_points=np.column_stack([payload[k].reshape(new_shape)[:, 0].ravel() for k in ('Z', 'C')]),
                   reference_cells=triangles, reference_h=payload['h'].reshape(new_shape)[:, 0].ravel(),
                   sampling_coordinate=np.r_[t['sampling_coordinate'], 1.+fractions])
    metadata = dict(model.metadata, mesh=statistics, progress_points=new_shape[2],
                    source_table_sha256=hashlib.sha256((source/'nonadiabatic_fgm.npz').read_bytes()).hexdigest(),
                    tail_extension=dict(method=('native_constant_pressure_adiabatic_reactor_on_monotone_C_branch'
                        if any(a.get('endpoint_policy')=='monotone_branch' for a in audits) else 'native_constant_pressure_adiabatic_reactor_to_HP_equilibrium'),
                        added_states_per_flame=tail_points, additional_flames=0,
                        original_vertices_preserved=True, minimum_progress_growth=float(growth.min()),
                        maximum_progress_growth=float(growth.max()),
                        maximum_roundoff_mass_removed=float(rounding.max()),
                        maximum_reactor_roundoff_mass_removed=max(a['roundoff_mass_removed'] for a in audits),
                        maximum_reactor_element_error=max(a['element_error'] for a in audits),
                        maximum_reactor_raw_sum_Y_error=max(a['raw_sum_Y_error'] for a in audits),
                        maximum_equilibrium_progress_relative_gap=max(a['relative_progress_gap'] for a in audits),
                        maximum_endpoint_species_gap=max(a['endpoint_species_gap'] for a in audits),
                        equilibrium_gap_interpretation='Distance to full HP equilibrium is reference-only; a monotone branch need not reach it.',
                        solver='native_BDF_rtol_1e-9_atol_Y_1e-15',
                        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        limitation='Homogeneous post-flame relaxation, not a solved spatial flamelet; validate a posteriori.'))
    if any(a.get('endpoint_policy')=='monotone_branch' for a in audits):
        extension=metadata['tail_extension']
        extension['maximum_reference_HP_progress_relative_gap']=extension.pop('maximum_equilibrium_progress_relative_gap')
        extension['maximum_reference_HP_species_gap']=extension.pop('maximum_endpoint_species_gap')
    output.mkdir(parents=True)
    np.savez_compressed(output/'nonadiabatic_fgm.npz', **payload)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8', newline='\n')
    np.savez_compressed(output/'equilibrium_endpoints.npz', Y=equilibrium)
    np.savez_compressed(output/'reactor_tail.npz', Y=Y,HP_Y=equilibrium,
                        source_table_sha256=metadata['source_table_sha256'],
                        audit_json=json.dumps(audits),generator_sha256=metadata['tail_extension']['builder_sha256'])
    (output/'reactor_audit.json').write_text(json.dumps(audits, indent=2)+'\n', encoding='utf-8')
    print(statistics, metadata['tail_extension'], flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoints', type=Path)
    parser.add_argument('--reactor-cache', type=Path)
    args = parser.parse_args()
    extend(args.source, args.output, endpoints=args.endpoints, reactor_cache=args.reactor_cache)


if __name__ == '__main__':
    main()
