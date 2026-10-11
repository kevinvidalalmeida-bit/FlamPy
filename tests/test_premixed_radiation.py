"""Optically thin radiation in premixed (free and burner) flames."""
import numpy as np
import pytest

from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.flame.analytic_jacobian import build_analytic_blocks
from kflame.flame.config import FlameCase
from kflame.flame.diffusion_kernels import radiation_profile
from kflame.flame.equations import radiative_heat_loss, residual
from kflame.flame.premixed_structured import PremixedStructuredJacobian
from kflame.flame.problem import FreeFlameProblem
from kflame.flame.solver import _apply_fixed_temperature_anchor


def _problem(scale, burner=False):
    kw = dict(mech='gri30.yaml', fuel='CH4', transport_model='mixture-averaged', cantera_seed_grid=False,
              radiation_enabled=scale is not None, radiation_scale=1. if scale is None else scale)
    if burner:
        kw['inlet_mass_flux'] = .08
    p = FreeFlameProblem(FlameCase(**kw), n_points=12)
    p.backend = NativeSpeciesBackend(p)
    x = p.make_initial_guess()
    if burner:
        p.setup_fixed_temperature(T_profile=x.reshape(p.n_points, -1)[:, 1])
    else:
        x = _apply_fixed_temperature_anchor(p, x)
    return p, x


def test_radiative_loss_is_the_counterflow_model():
    p, x = _problem(3.)
    s = x.reshape(p.n_points, -1)
    q = radiative_heat_loss(p, s[:, 1], s[:, 2:].T)
    np.testing.assert_array_equal(q, radiation_profile(np.ascontiguousarray(s[:, 1]),
                                                       np.ascontiguousarray(s[:, 2:].T), p.radiation))
    assert q.max() > 0.


def test_zero_scale_reproduces_the_adiabatic_residual():
    p0, x = _problem(None)
    p1, _ = _problem(0.)
    np.testing.assert_array_equal(residual(x, p1), residual(x, p0))


@pytest.mark.parametrize('burner', [False, True])
@pytest.mark.parametrize('structured', [False, True])
def test_analytic_jacobians_include_the_radiative_derivatives(burner, structured):
    # The radiation contribution of the Jacobian against centered differences
    # of the residual difference (radiation on - off), which isolates it.
    on, x = _problem(25., burner)
    off, _ = _problem(None, burner)
    for p in (on, off):
        p.structured_jacobian = structured
    J_on, J_off = build_analytic_blocks(x, on), build_analytic_blocks(x, off)
    assert isinstance(J_on, PremixedStructuredJacobian) == structured
    nv = on.n_vars_per_point
    names = on.species_names
    for node in (3, 6, 9):
        for var in (1, 2+names.index('H2O'), 2+names.index('CO2'), 2+names.index('N2')):
            col = node*nv+var
            step = 1e-6*max(abs(x[col]), 1e-3)
            plus, minus = x.copy(), x.copy()
            plus[col] += step
            minus[col] -= step
            fd = ((residual(plus, on)-residual(plus, off))-(residual(minus, on)-residual(minus, off)))/(2*step)
            e = np.zeros_like(x)
            e[col] = 1.
            analytic = J_on.matvec(e)-J_off.matvec(e)
            np.testing.assert_allclose(analytic[1::nv], fd[1::nv], rtol=1e-5, atol=1e-6*np.max(abs(fd[1::nv])))


def test_radiating_flames(tmp_path):
    from kflame import solve_burner_flame, solve_flame
    adiabatic = solve_flame(mechanism='gri30.yaml', fuel='CH4', phi=1., output=tmp_path/'a', max_time=180.)
    radiating = solve_flame(mechanism='gri30.yaml', fuel='CH4', phi=1., radiation=True, output=tmp_path/'r',
                            max_time=180.)
    # Radiation lowers the burning velocity slightly; the domain stays bounded
    # (the outlet check uses the finished chemistry, not the cooling slope).
    assert .99*adiabatic['Su'] < radiating['Su'] < adiabatic['Su']
    assert radiating['z'][-1] <= .5 and radiating['T'][-1] > 1500.
    assert radiating['radiative_heat_loss'].max() > 1e6
    burner = solve_burner_flame(mechanism='gri30.yaml', fuel='CH4', phi=1., mass_flux=.08, radiation=True,
                                output=tmp_path/'b', max_time=180.)
    loss = burner['heat_loss']
    assert loss['radiative_heat_loss_W_m2'] > 1e4
    assert abs(loss['total_heat_loss_W_m2']-loss['burner_heat_loss_W_m2']-loss['radiative_heat_loss_W_m2']) < 1e-6
    assert loss['relative_energy_closure_error'] < .02
