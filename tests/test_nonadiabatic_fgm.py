"""Test table coordinates, validity boundaries and enthalpy recovery."""
import json

import numpy as np
import pytest

from kflame import generate_burner_fgm
from kflame.chemistry.initialization import fresh_mixture, hp_equilibrium
from kflame.chemistry.mechanism import load_mechanism
from kflame.fgm.nonadiabatic import BurnerFGM, tabulate_burner_family


@pytest.fixture
def manifold(tmp_path):
    mech = load_mechanism('h2o2.yaml')
    yin = fresh_mixture(mech, 1., 'H2', 'O2:1,N2:3.76')
    _, yeq, _ = hp_equilibrium(mech, 300., 101325., yin)
    weights = np.array([float(name == 'H2O') for name in mech.species_names])
    # Synthetic, element-conserving trajectories for coordinate/lookup tests;
    # these temperatures are not claimed to solve the flame equations.
    results = []
    for start, end, rise in [(0., .8, 1800.), (.16, .9, 1400.), (.24, 1., 1000.)]:
        conversion = np.linspace(start, end, 40)
        Y = yin[:, None] + conversion[None, :] * (yeq - yin)[:, None]
        results.append(dict(Y=Y, T=300. + rise * conversion,
                            rho=np.ones(40), conductivity=np.full(40, .05),
                            qdot=np.zeros(40), pressure=101325.))
    table = tabulate_burner_family(results, mech, yin, weights, 81)
    np.savez_compressed(tmp_path / 'burner_fgm.npz', **table)
    (tmp_path / 'metadata.json').write_text(json.dumps(dict(mechanism='h2o2.yaml', pressure_Pa=101325.)))
    return BurnerFGM(tmp_path)


def test_lookup_preserves_progress_mass_and_total_enthalpy(manifold):
    t = manifold.table
    c = .6
    h = .5 * sum(np.interp(c, t['c'], t['h'][row]) for row in [0, 1])
    result = manifold.lookup(c=c, h=h)
    recovered = float(manifold.thermo.enthalpy_mass(result['T'], result['Y']))
    assert recovered == pytest.approx(h, abs=1e-5)
    progress = (t['progress_weights'] @ result['Y'] - t['beta_unburned']) / t['beta_span']
    assert progress == pytest.approx(c, abs=1e-12)
    assert result['Y'].sum() == pytest.approx(1., abs=1e-12)
    assert result['delta_h'] > 0.


def test_global_progress_is_not_stretched_or_clipped(manifold):
    t = manifold.table
    assert t['c'][-1] > 1.
    assert t['c'][t['valid'][0]][-1] == 1.
    np.testing.assert_allclose(t['delta_h'][0, t['valid'][0]], 0., atol=1e-8)
    c = 1.05
    h = .5 * sum(np.interp(c, t['c'], t['h'][row]) for row in [1, 2])
    assert manifold.lookup(c=c, h=h)['delta_h'] is None  # No adiabatic reference at this c.


def test_lookup_rejects_unresolved_and_missing_rows(manifold):
    # A known reference state remains accessible without inventing a cooled row.
    fresh_h = manifold.table['h'][0, 0]
    assert manifold.lookup(c=0., h=fresh_h)['T'] == pytest.approx(300., abs=1e-7)
    with pytest.raises(ValueError, match='outside'):
        manifold.lookup(c=.05, h=0.)  # Only the adiabatic row exists here.
    with pytest.raises(ValueError, match='outside'):
        manifold.lookup(c=2., h=0.)
    with pytest.raises(ValueError, match='finite'):
        manifold.lookup(c=np.nan, h=0.)
    t = manifold.table
    c = .6
    h = .5 * sum(np.interp(c, t['c'], t['h'][row]) for row in [0, 2])
    t['valid'][1] = False
    with pytest.raises(ValueError, match='outside'):
        manifold.lookup(c=c, h=h)


@pytest.mark.parametrize('fluxes', [[], [0.], [.1, .1], [np.inf], [[.1]]])
def test_invalid_family_does_not_create_output(tmp_path, fluxes):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='mass_fluxes'):
        generate_burner_fgm(mass_fluxes=fluxes, output=output)
    assert not output.exists()
