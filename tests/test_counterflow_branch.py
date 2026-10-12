"""Counterflow flames on the stable burning branch of the S-curve."""
import numpy as np

from kflame import solve_diffusion_flame


def test_a_flame_reports_the_stable_branch(tmp_path):
    flame = solve_diffusion_flame(mechanism='h2o2.yaml', fuel='H2', fuel_mass_flux=.05, oxidizer_mass_flux=.15,
                                  output=tmp_path/'h2')
    branch = flame['report']['branch']
    assert branch['stable'] and branch['steady_sign'] == branch['transient_sign']
    assert not any(h.get('stage') == 'cold_start_retry' for h in flame['report']['history'])


def test_a_strained_cold_start_reaches_the_burning_branch(tmp_path):
    # Ten times the default fluxes: the 31-node cold start is quenched by the
    # coarse grid; continuation from lower strain gives Tmax 1917.1 K.
    flame = solve_diffusion_flame(mechanism='gri30.yaml', fuel='CH4', fuel_mass_flux=1.2, oxidizer_mass_flux=2.4,
                                  output=tmp_path/'strained')
    report = flame['report']
    assert flame['accepted'] and report['branch']['stable']
    assert any(h.get('stage') == 'cold_start_retry' for h in report['history'])
    assert abs(float(np.max(flame['T']))-1917.1) < 2.
