"""Refining Su must re-solve the whole family at coherent burner fluxes."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from examples import refine_library_enthalpy as refiner
from kflame.chemistry.initialization import fresh_mixture, hp_equilibrium
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo


def prepare(tmp_path, monkeypatch, *, reject_second_burner=False):
    mech = load_mechanism('h2o2.yaml')
    thermo = NativeThermo(mech)
    feed = fresh_mixture(mech, .8, 'H2', 'O2:1,N2:3.76')
    _, burned, _ = hp_equilibrium(mech, 300., 101325., feed)
    hfeed = thermo.enthalpy_mass(300., feed)
    rho = float(thermo.density(300., 101325., feed))
    source = tmp_path/'source'
    source.mkdir()
    def save(path, *, free, deficit, Su=2., flux=None, accepted=True):
        path.mkdir()
        T = brentq(lambda t: thermo.enthalpy_mass(t,burned)-hfeed+deficit, 300., 3500.)
        np.savez(path/'flame.npz', T=[300.,1000.,T], z=[0.,.015,.03],
                 Y=np.column_stack([feed,.5*(feed+burned),burned]))
        meta = dict(accepted=accepted, backend='native_cpu', report={'grid_converged':accepted},
                    flow_type='free_flame' if free else 'isothermal_burner', mechanism='h2o2.yaml',
                    temperature=300., pressure=101325., inlet_Y=feed.tolist(), transport='mixture-averaged',
                    soret=False, final_width=.03, Su=Su, mass_flux=flux, inlet_velocity=0.1,
                    refinement=dict(ratio=2.5,slope=.04,curve=.08,max_points=1600),
                    tolerances=dict(rtol=1e-4,atol=1e-9), runtime_s=.01, nodes=3,
                    heat_loss={'relative_energy_closure_error':0.})
        (path/'metadata.json').write_text(json.dumps(meta))
    rows = []
    for j, fraction in enumerate([None,.95,.5]):
        folder = source/f'flame_{j}'
        save(folder, free=(j==0), deficit=300. if j==0 else 20000.*j,
             flux=None if j==0 else fraction*2.*rho)
        rows.append(dict(composition_index=0,loss_index=j,phi=.8,fraction=fraction,
                         inlet_Y=feed.tolist(),output=folder.name))
    generation = dict(mechanism='h2o2.yaml', temperature_K=300., pressure_Pa=101325.,
                      transport='mixture-averaged', soret=False, phis=[.8], mass_flux_fractions=[.95,.5],
                      max_energy_error=.02, rows=rows)
    calls = []
    def free(**kwargs):
        calls.append(('free',kwargs))
        save(kwargs['output'], free=True, deficit=100., Su=1.9)
    def burner(**kwargs):
        calls.append(('burner',kwargs))
        attempts = sum(kind == 'burner' for kind,k in calls)
        save(kwargs['output'], free=False, deficit=20000., flux=kwargs['mass_flux'],
             accepted=not (reject_second_burner and attempts==2))
    monkeypatch.setattr(refiner,'solve_flame',free)
    monkeypatch.setattr(refiner,'solve_burner_flame',burner)
    return source, generation, rho, calls


def test_refining_free_speed_recomputes_every_burner_mass_flux(tmp_path, monkeypatch):
    source, generation, rho, calls = prepare(tmp_path,monkeypatch)
    result = refiner.family(source,tmp_path/'family_000',generation,0,.5,3,2000)
    assert result['quality']['passed']
    assert [kind for kind,k in calls] == ['free','burner','burner']
    for row, fraction in zip(result['rows'][1:],[.95,.5]):
        assert row['mass_flux'] == pytest.approx(1.9*rho*fraction,rel=1e-12)
        assert row['adiabatic_mass_flux'] == pytest.approx(1.9*rho,rel=1e-12)
    assert result['rows'][0]['Su'] == 1.9
    assert calls[0][1]['Y'].keys() == dict.fromkeys(load_mechanism('h2o2.yaml').species_names).keys()


def test_failed_burner_retry_uses_an_accepted_neighbor(tmp_path, monkeypatch):
    source,generation,rho,calls = prepare(tmp_path,monkeypatch,reject_second_burner=True)
    result = refiner.family(source,tmp_path/'family_000',generation,0,.5,3,2000)
    assert result['quality']['passed']
    first_burner = calls[1][1]['output']
    assert calls[-1][1]['initial_solution'] == first_burner
    assert calls[-1][1]['mass_flux'] == pytest.approx(.5*1.9*rho)
    assert any(a.get('accepted') is False for a in result['attempts'])


def test_reused_family_rechecks_fluxes_before_accepting(tmp_path, monkeypatch):
    source,generation,rho,calls = prepare(tmp_path,monkeypatch)
    cache = tmp_path/'cache'
    cache.mkdir()
    result = refiner.family(source,cache/'family_000',generation,0,.5,3,2000)
    count = len(calls)
    imported = refiner.family(source,tmp_path/'new/family_000',generation,0,.5,3,2000,reuse_from=cache)
    assert imported['quality']['passed'] and len(calls)==count
    meta_path = cache/'family_000'/result['rows'][1]['output']/'metadata.json'
    meta = json.loads(meta_path.read_text())
    meta['mass_flux'] *= 1.01
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='mass flux or energy'):
        refiner.family(source,tmp_path/'invalid/family_000',generation,0,.5,3,2000,reuse_from=cache)
