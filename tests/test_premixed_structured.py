"""Structured premixed Jacobian: same matrix, same solves and same flame as the dense path."""
import itertools

import numpy as np
import pytest

from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.flame import premixed_structured as PS
from kflame.flame.analytic_jacobian import build_analytic_blocks
from kflame.flame.config import FlameCase
from kflame.flame.equations import factorize, solve_linear, update_transient
from kflame.flame.problem import FreeFlameProblem
from kflame.flame.solver import _apply_fixed_temperature_anchor


def _problem(mech, fuel, burner, convection, soret, basis, outlet):
    kw = dict(mech=mech, fuel=fuel, transport_model='mixture-averaged', soret_enabled=soret,
              cantera_seed_grid=False, convection_scheme=convection, flux_gradient_basis=basis,
              outlet_species_bc=outlet)
    if burner:
        kw['inlet_mass_flux'] = .08
    p = FreeFlameProblem(FlameCase(**kw), n_points=12)
    p.backend = NativeSpeciesBackend(p)
    x = p.make_initial_guess()
    if burner:
        p.setup_fixed_temperature(T_profile=x.reshape(p.n_points, -1)[:, 1])
    else:
        x = _apply_fixed_temperature_anchor(p, x)
    s = x.reshape(p.n_points, -1)
    s[:, 2:] *= 1+.05*np.random.default_rng(1).standard_normal(s[:, 2:].shape)
    s[:, 2:] = np.abs(s[:, 2:])+1e-12
    s[:, 2:] /= s[:, 2:].sum(axis=1, keepdims=True)
    return p, x


CASES = [(burner, convection, soret, basis, outlet, energy)
         for burner, convection, soret, basis, outlet, energy in itertools.product(
             (False, True), ('upwind', 'exponential'), (False, True), ('molar', 'mass'),
             ('zero_gradient', 'flux'), (True, False))
         if (soret, basis, outlet) in ((False, 'molar', 'zero_gradient'), (True, 'mass', 'flux'))]


@pytest.mark.parametrize('burner,convection,soret,basis,outlet,energy', CASES)
def test_structured_blocks_equal_the_dense_analytic_assembly(burner, convection, soret, basis, outlet, energy):
    p, x = _problem('gri30.yaml', 'CH4', burner, convection, soret, basis, outlet)
    p.solve_energy = energy
    if not energy and p.T_profile_fixed is None:
        p.T_profile_fixed = x.reshape(p.n_points, -1)[:, 1].copy()
    p.structured_jacobian = False
    dense = build_analytic_blocks(x, p)
    p.structured_jacobian = True
    structured = build_analytic_blocks(x, p)
    assert isinstance(structured, PS.PremixedStructuredJacobian)
    expanded = structured.dense()
    scale = np.abs(dense.diag).max()
    for name in ('lower', 'diag', 'upper'):
        np.testing.assert_allclose(getattr(expanded, name), getattr(dense, name), rtol=0, atol=1e-13*scale)
    np.testing.assert_allclose(structured.diagonal(), dense.diagonal(), rtol=0, atol=1e-13*scale)
    rhs = np.random.default_rng(2).standard_normal(x.size)
    np.testing.assert_allclose(structured.matvec(rhs), dense.matvec(rhs), rtol=0,
                               atol=1e-12*np.abs(dense.matvec(rhs)).max())
    # Transient diagonal update and factorization as the solver applies them.
    mask = np.ones(x.size)
    for jmat in (structured, dense):
        update_transient(jmat, mask, 1e4, inplace=True)
    expected = solve_linear(factorize(dense), rhs)
    actual = solve_linear(factorize(structured), rhs)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9*np.abs(expected).max())


def test_small_mechanisms_keep_the_dense_path():
    p, x = _problem('h2o2.yaml', 'H2', False, 'upwind', False, 'molar', 'zero_gradient')
    assert not isinstance(build_analytic_blocks(x, p), PS.PremixedStructuredJacobian)


def test_structured_flame_reproduces_the_dense_flame(tmp_path, monkeypatch):
    from kflame import solve_flame
    flames = []
    for minimum in (10**9, PS.STRUCTURED_MIN_SPECIES):
        monkeypatch.setattr(PS, 'STRUCTURED_MIN_SPECIES', minimum)
        flames.append(solve_flame(mechanism='gri30.yaml', fuel='CH4', phi=1.0, max_time=120.,
                                  output=tmp_path/str(minimum)))
    dense, structured = flames
    assert dense['accepted'] and structured['accepted']
    np.testing.assert_array_equal(structured['z'], dense['z'])
    assert abs(structured['Su']-dense['Su']) <= 1e-9*dense['Su']
    np.testing.assert_allclose(structured['T'], dense['T'], rtol=0, atol=1e-6)
