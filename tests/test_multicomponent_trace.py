"""Pruned multicomponent transport (no trace--trace collisions) for large mechanisms."""
import numpy as np
import pytest

from kflame.chemistry.mechanism import R_UNIV, load_mechanism, resolve_mechanism
from kflame.chemistry.multicomponent import NativeMulticomponentTransport
from kflame.chemistry.multicomponent_kernels import (_MIN_C_INTERNAL, composition_terms_ws,
                                                     composition_workspace, temperature_terms)
from kflame.chemistry.multicomponent_trace import (TRACE_MIN_SPECIES, TRACE_TAIL, composition_terms_trace,
                                                   evaluate_faces_trace, resolve_trace_tail, trace_face)
from kflame.chemistry.thermo import NativeThermo

P = 101325.


@pytest.fixture(scope='module')
def gri():
    mech = load_mechanism(resolve_mechanism('gri30.yaml'))
    return mech, NativeMulticomponentTransport(mech, ''), NativeThermo(mech)


def _state(mech, seed, t=1600., smallest=-18.):
    """Flame-like composition: majors plus a log-uniform spectrum of traces."""
    names = list(mech.species_names)
    rng = np.random.default_rng(seed)
    y = 10.**rng.uniform(smallest, -5., mech.n_species)
    for name, value in (('N2', .72), ('O2', .05), ('H2O', .11), ('CO2', .08), ('CH4', .01), ('CO', .02),
                        ('H2', 2e-3), ('OH', 4e-3), ('H', 3e-4), ('O', 8e-4)):
        y[names.index(name)] = value
    return t, y/y.sum()


def _temperature_data(mc, t):
    n = mc.n_species
    mu, rr = np.empty(n), np.empty(n)
    bdiff, a, b, c = (np.empty((n, n)) for _ in range(4))
    temperature_terms(t, mc.mw, mc._log_eps_pair, mc.epsilon_k, mc.zrot, mc.frot_298, mc._viscosity_fit,
                      mc._diffusion_fit, mc._astar_poly, mc._bstar_poly, mc._cstar_poly, mu, rr, bdiff, a, b, c)
    return mu, rr, bdiff, a, b, c


def _pruned_reference(t, y, cp_r, mc, mu, rr, bdiff, a, b, c, tail):
    """Dense 3N system of the pruned model.

    The rank-one regularization of the singular L00 block sits on the most
    abundant species, as in the kernel: D and the thermal solution do not
    depend on it, and Cantera's choice (on every species) leaves the rows of
    small species parallel up to O(x), numerically singular near 1e-20.
    """
    n, mw, crot = mc.n_species, mc.mw, mc.crot
    wmix = 1./np.sum(y/mw)
    x = np.maximum(y*wmix/mw, 1e-20)
    order = np.argsort(-x, kind='mergesort')
    m = max(int((np.cumsum(x[order][::-1])[::-1] > tail).sum()), 2)
    major = np.zeros(n, bool)
    major[order[:m]] = True
    kept = major[:, None] | major[None, :] | np.eye(n, dtype=bool)
    off = kept & ~np.eye(n, dtype=bool)
    cst = 16.*t/25.
    G = np.where(off, x[None, :]/bdiff, 0.).sum(1)
    u = np.zeros(n)
    u[np.argmax(x)] = G[np.argmax(x)]/mw[np.argmax(x)]
    l00 = cst*(np.outer(u, x*mw)+np.where(off, np.outer(x, x)/bdiff, 0.))
    l00[np.diag_indices(n)] = cst*(u*x*mw-x*G)
    cint = cp_r-2.5
    internal = cint > _MIN_C_INTERNAL
    wi, wj, xi, xj = mw[:, None], mw[None, :], x[:, None], x[None, :]
    l01 = np.where(off, -1.6*t*xi*xj*wi*(1.2*c.T-1.)/((wj+wi)*bdiff.T), 0.)
    l01[np.diag_indices(n)] = -l01.sum(0)
    term1 = bdiff*(wi+wj)**2
    ratio = crot/rr
    term2 = 4.*wj*a*(1.+5./(3.*np.pi)*(ratio[None, :]+ratio[:, None]))
    l11 = np.where(kept, cst*xj*xi*wi/(wj*term1)*(13.75*wj**2-3.*wj**2*b-term2*wj), 0.)
    l11[np.diag_indices(n)] -= cst*x*np.sum(np.where(kept, xi/term1*(7.5*wj**2+wi*wi*(6.25-3.*b)+term2*wi), 0.), 0)
    ic = np.where(internal, 32.*t*mw*x*crot/(5.*np.pi*np.where(internal, cint, 1.)*rr), 0.)
    l12 = np.where(kept, xi*a.T*ic[None, :]/((wi+wj)*bdiff.T), 0.)
    l12[np.diag_indices(n)] += l12.sum(0)
    c1 = np.where(internal, 4.*t*x/np.where(internal, cint, 1.), 0.)
    c2 = np.where(internal, 12.*mw*crot/(5.*np.pi*np.where(internal, cint, 1.)*rr), 0.)
    first = np.where(kept, x[None, :]/bdiff, 0.).sum(1)
    second = np.where(off, a/bdiff*(x/mw)[None, :], 0.).sum(1)
    l22 = np.where(internal, -8./np.pi*mw*x**2*crot/(np.where(internal, cint, 1.)**2*R_UNIV*mu*rr)
                   - c1*(first+c2*second), 1.)
    L = np.zeros((3*n, 3*n))
    L[:n, :n], L[:n, n:2*n], L[n:2*n, :n] = l00, l01, l01.T
    L[n:2*n, n:2*n], L[n:2*n, 2*n:], L[2*n:, n:2*n] = l11, l12, l12.T
    L[2*n:, 2*n:] = np.diag(l22)
    sol = np.linalg.solve(L, np.concatenate((np.zeros(n), x, np.where(internal, x, 0.))))
    inv = np.linalg.inv(l00)
    multi = 16.*t*wmix/(25.*P)*x[:, None]/mw[None, :]*(inv-np.diag(inv)[:, None])
    return multi, 1.6/R_UNIV*mw*x*sol[:n], -4.*(x@sol[n:2*n]+(x*internal)@sol[2*n:]), m


def _face(mc, thermo, t, y, tail):
    n = mc.n_species
    cp_r = np.asarray(thermo.cp_R(np.array([t])), dtype=float)[:, 0]
    data = _temperature_data(mc, t)
    multi, thermal = np.empty((n, n)), np.empty(n)
    rho, lam, wmix, m = composition_terms_trace(t, P, y, cp_r, mc.mw, mc.inv_w, mc.crot, *data,
                                                multi, thermal, tail)
    return cp_r, data, multi, thermal, lam, m


@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('tail', [1e-10, 1e-8, 1e-5])
def test_kernel_solves_the_pruned_model(gri, seed, tail):
    mech, mc, thermo = gri
    t, y = _state(mech, seed)
    cp_r, data, multi, thermal, lam, m = _face(mc, thermo, t, y, tail)
    ref_multi, ref_thermal, ref_lam, ref_m = _pruned_reference(t, y, cp_r, mc, *data, tail)
    assert m == ref_m and 2 < m < mech.n_species
    np.testing.assert_allclose(multi, ref_multi, rtol=0., atol=1e-12*np.abs(ref_multi).max())
    np.testing.assert_allclose(thermal, ref_thermal, rtol=0., atol=1e-11*np.abs(ref_thermal).max())
    assert lam == pytest.approx(ref_lam, rel=1e-12)


def test_vanishing_tail_is_the_complete_kernel(gri):
    mech, mc, thermo = gri
    # Every species is a major here: with traces down to 1e-18 the major block
    # has a condition number of order 1e18 and the kernel defers to the
    # complete one (m = -1). Flame traces this small are pruned in practice.
    t, y = _state(mech, 2, smallest=-12.)
    cp_r, data, multi, thermal, lam, m = _face(mc, thermo, t, y, 1e-300)
    assert m == mech.n_species
    n = mech.n_species
    full_multi, full_thermal = np.empty((n, n)), np.empty(n)
    _, full_lam, _ = composition_terms_ws(t, P, y, cp_r, mc.mw, mc.inv_w, mc.crot, *data, full_multi,
                                          full_thermal, composition_workspace(n))
    np.testing.assert_allclose(multi, full_multi, rtol=0., atol=1e-11*np.abs(full_multi).max())
    np.testing.assert_allclose(thermal, full_thermal, rtol=0., atol=1e-10*np.abs(full_thermal).max())
    assert lam == pytest.approx(full_lam, rel=1e-12)


def test_pruning_changes_fluxes_by_the_trace_order_and_conserves_mass(gri):
    mech, mc, thermo = gri
    n, mw = mech.n_species, mc.mw
    t, y = _state(mech, 3)
    _, y2 = _state(mech, 4)
    cp_r, data, multi, thermal, lam, m = _face(mc, thermo, t, y, TRACE_TAIL)
    full_multi, full_thermal = np.empty((n, n)), np.empty(n)
    _, full_lam, wmix = composition_terms_ws(t, P, y, cp_r, mw, mc.inv_w, mc.crot, *data, full_multi,
                                             full_thermal, composition_workspace(n))
    x = lambda Y: Y/mw/np.sum(Y/mw)
    gradient = (x(y2)-x(y))/1e-3                      # sums to zero
    flux = (multi*mw[None, :])@gradient
    full = (full_multi*mw[None, :])@gradient
    assert np.abs(flux-full).sum() <= 1e-7*np.abs(full).sum()
    # sum_k j_k = 0, i.e. sum_k W_k F_k = 0: the pruned D keeps the mass constraint
    # (to the round-off of the face-system inverse, as the complete kernel).
    assert abs(mw@flux) <= 1e-9*np.abs(mw*flux).sum()
    assert abs(mw@full) <= 1e-9*np.abs(mw*full).sum()
    assert lam == pytest.approx(full_lam, rel=1e-12)
    np.testing.assert_allclose(thermal, full_thermal, rtol=0., atol=1e-10*np.abs(full_thermal).max())


def test_fused_face_and_batch_driver_agree(gri):
    mech, mc, thermo = gri
    n = mech.n_species
    states = [_state(mech, seed, t) for seed, t in ((5, 900.), (6, 1500.), (7, 2100.))]
    T = np.array([t for t, _ in states])
    Y = np.ascontiguousarray(np.array([y for _, y in states]).T)
    cp_r = np.ascontiguousarray(np.asarray(thermo.cp_R(T), dtype=float))
    inputs, certified = mc.export_temperature_snapshot()
    assert certified
    mw, log_eps, epsilon, zrot, frot, visc, diff, astar, bstar, cstar = inputs
    rho, lam, wmix, multi, thermal, majors = evaluate_faces_trace(
        T, P, Y, cp_r, mw, mc.inv_w, log_eps, epsilon, mc.crot, zrot, frot, visc, diff, astar, bstar, cstar, 1e-8)
    for f, (t, y) in enumerate(states):
        cp, data, ref_multi, ref_thermal, ref_lam, m = _face(mc, thermo, t, y, 1e-8)
        assert majors[f] == m
        np.testing.assert_allclose(multi[:, :, f], ref_multi, rtol=0., atol=1e-13*np.abs(ref_multi).max())
        np.testing.assert_allclose(thermal[:, f], ref_thermal, rtol=0., atol=1e-12*np.abs(ref_thermal).max())
        assert lam[f] == pytest.approx(ref_lam, rel=1e-13)
        out_multi, out_thermal = np.empty((n, n)), np.empty(n)
        fused = trace_face(t, P, y, np.ascontiguousarray(cp_r[:, f]), mw, mc.inv_w, mc.crot, log_eps, epsilon,
                           zrot, frot, visc, diff, astar, bstar, cstar, 1e-8, out_multi, out_thermal)
        assert fused[3] == m
        np.testing.assert_allclose(out_multi, ref_multi, rtol=0., atol=1e-13*np.abs(ref_multi).max())


def test_tail_setting():
    assert resolve_trace_tail('auto', TRACE_MIN_SPECIES-1) is None
    assert resolve_trace_tail('auto', TRACE_MIN_SPECIES) == TRACE_TAIL
    assert resolve_trace_tail(None, 600) is None and resolve_trace_tail(False, 600) is None
    assert resolve_trace_tail(True, 10) == TRACE_TAIL
    assert resolve_trace_tail(1e-6, 10) == 1e-6
    for bad in ('fast', 0., -1e-8, 1e-1, float('nan')):
        with pytest.raises(ValueError):
            resolve_trace_tail(bad, 600)


def test_backend_selects_the_pruned_model_for_large_mechanisms():
    from types import SimpleNamespace
    from kflame.chemistry.backend import NativeSpeciesBackend
    for name, expected in (('gri30.yaml', None), ('jetsurf2.yaml', TRACE_TAIL)):
        mech = load_mechanism(resolve_mechanism(name))
        case = SimpleNamespace(mech=resolve_mechanism(name), transport_model='multicomponent',
                               soret_enabled=False, flux_gradient_basis='molar', multicomponent_trace_tail='auto')
        problem = SimpleNamespace(case=case, P=P, transport_model='multicomponent', soret_enabled=False,
                                  flux_gradient_basis='molar', mech_data=mech)
        assert NativeSpeciesBackend(problem, mech).multicomponent_transport.trace_tail == expected


def test_counterflow_flame_with_the_pruned_model(tmp_path):
    """Same fixed-grid multicomponent flame with the complete and the pruned kernel."""
    from unittest import mock
    from kflame import solve_diffusion_flame
    flames = {}
    for label, setting in (('complete', None), ('pruned', 1e-8)):
        with mock.patch('kflame.chemistry.backend.resolve_trace_tail',
                        lambda value, n, s=setting: resolve_trace_tail(s, n)):
            flames[label] = solve_diffusion_flame(mechanism='gri30.yaml', fuel='CH4', transport='multicomponent',
                                                  initial_points=41, refine=False, output=tmp_path/label)
    complete, pruned = flames['complete'], flames['pruned']
    assert np.array_equal(complete['z'], pruned['z'])
    assert np.max(abs(complete['T']-pruned['T'])) < 1e-4
    assert np.max(abs(complete['Y']-pruned['Y'])) < 1e-8
    assert pruned['report']['history'][0]['stage'] == 'mixture_averaged_bootstrap'


def test_premixed_newton_uses_gmres_with_the_mixture_preconditioner():
    from kflame.chemistry.backend import NativeSpeciesBackend
    from kflame.flame.analytic_jacobian import build_analytic_blocks, mixture_preconditioned
    from kflame.flame.config import FlameCase
    from kflame.flame.equations import factorize, residual
    from kflame.flame.krylov import KrylovFactor
    from kflame.flame.premixed_structured import PremixedStructuredJacobian
    from kflame.flame.problem import FreeFlameProblem
    from kflame.flame.solver import _apply_fixed_temperature_anchor
    p = FreeFlameProblem(FlameCase(mech='gri30.yaml', fuel='CH4', transport_model='multicomponent',
                                   cantera_seed_grid=False, multicomponent_trace_tail=1e-8), n_points=12)
    p.backend = NativeSpeciesBackend(p)
    assert not mixture_preconditioned(p)          # automatic only from 400 species (AramcoMech)
    p.mixture_preconditioner = True
    assert p.backend.multicomponent_transport.trace_tail == 1e-8 and mixture_preconditioned(p)
    x = _apply_fixed_temperature_anchor(p, p.make_initial_guess())
    jmat = build_analytic_blocks(x, p)
    assert isinstance(jmat, PremixedStructuredJacobian) and jmat.approximate and jmat.copy().approximate
    assert p.backend.uses_multicomponent_flux      # restored after the mixture-form assembly
    factor = factorize(jmat)
    assert isinstance(factor, KrylovFactor)
    F = residual(x, p)
    weights = np.tile(1e-4*np.mean(np.abs(x.reshape(p.n_points, -1)), axis=0)+1e-9, p.n_points)
    factor.set_anchor(lambda v: residual(v, p), x, F, weights)
    step = factor.solve(-F)
    # Complete multicomponent Jacobian by central differences.
    J = np.empty((len(x), len(x)))
    for column in range(len(x)):
        h = 1e-7*max(abs(x[column]), 1e-6)
        plus, minus = x.copy(), x.copy()
        plus[column] += h
        minus[column] -= h
        J[:, column] = (residual(plus, p)-residual(minus, p))/(2*h)
    inner = factor.inner.solve(-F)
    assert np.linalg.norm(J@step+F) <= 3e-2*np.linalg.norm(F)
    assert np.linalg.norm(J@step+F) < np.linalg.norm(J@inner+F)
