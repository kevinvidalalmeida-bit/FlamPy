"""Continuation and family policies of the premixed API (fast H2-O2 cases)."""
import json

import numpy as np
import pytest

import kflame
import kflame.api as api


@pytest.fixture(scope='module')
def burner(tmp_path_factory):
    folder = tmp_path_factory.mktemp('continuation')
    free = kflame.solve_flame(mechanism='h2o2.yaml', fuel='H2', phi=.8, output=folder/'free')
    flux = .5*float(free['rho'][0]*free['Su'])
    first = kflame.solve_burner_flame(mechanism='h2o2.yaml', fuel='H2', phi=.8, mass_flux=flux, output=folder/'burner')
    return folder, flux, first


def test_default_output_species_follow_the_mechanism(burner):
    folder, _, first = burner
    header = (first['output']/'profiles.csv').read_text(encoding='utf-8').splitlines()[0]
    assert 'Y_H2O' in header and 'Y_CO2' not in header and 'Y_CH4' not in header
    with pytest.raises(ValueError, match='absent'):
        kflame.solve_flame(mechanism='h2o2.yaml', fuel='H2', phi=.8, species=('CH4',), output=folder/'bad')


def test_flux_change_restarts_on_a_bounded_seed_subset(burner, monkeypatch):
    folder, flux, first = burner
    assert len(first['z']) > 20
    monkeypatch.setattr(api, 'CONTINUATION_POINTS', 20)
    moved = kflame.solve_burner_flame(mechanism='h2o2.yaml', fuel='H2', phi=.8, mass_flux=.7*flux,
                                      initial_solution=first['output'], output=folder/'moved', convection='upwind')
    assert moved['accepted'] and moved['report']['grid_converged']
    assert moved['report']['passes'][0]['grid_points'] == 20
    # With the exponential scheme a flame-moving continuation goes through upwind first.
    exponential = kflame.solve_burner_flame(mechanism='h2o2.yaml', fuel='H2', phi=.8, mass_flux=.7*flux,
                                            initial_solution=first['output'], output=folder/'moved_exp',
                                            convection='exponential')
    assert exponential['accepted'] and exponential['report']['startup_convection']['scheme'] == 'upwind'
    # Same flux: nothing moves, the seed grid is kept as it is.
    same = kflame.solve_burner_flame(mechanism='h2o2.yaml', fuel='H2', phi=.8, mass_flux=flux,
                                     initial_solution=first['output'], output=folder/'same')
    assert same['report']['passes'][0]['grid_points'] == len(first['z'])
    assert abs(same['T'].max()-first['T'].max()) < 1e-2      # re-solved to the Newton tolerance


def test_closure_refinement_only_when_needed(monkeypatch, tmp_path):
    calls = []

    def fake(output, initial_solution, **settings):
        closure = .03 if len(calls) == 0 else .01
        calls.append(dict(output=output, slope=settings.get('slope'), curve=settings.get('curve'),
                          seed=initial_solution))
        return dict(output=output, heat_loss=dict(relative_energy_closure_error=closure))
    monkeypatch.setattr(api, 'solve_burner_flame', fake)
    result = api._closed_burner(tmp_path/'b', 'seed', .02, dict(mass_flux=.1))
    assert len(calls) == 2 and result['output'].name == 'b_refined1'
    assert calls[0]['slope'] is None and calls[1]['slope'] == pytest.approx(.02) and calls[1]['curve'] == pytest.approx(.04)
    assert calls[1]['seed'] == tmp_path/'b'
    calls.clear()
    monkeypatch.setattr(api, 'solve_burner_flame', lambda output, initial_solution, **s: calls.append(1) or
                        dict(output=output, heat_loss=dict(relative_energy_closure_error=.001)))
    api._closed_burner(tmp_path/'c', None, .02, {})
    assert len(calls) == 1
