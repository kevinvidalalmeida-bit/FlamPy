"""A small stiff Newton correction must also satisfy the residual guard."""
from types import SimpleNamespace

import numpy as np

from kflame.flame import solver, equations


def test_stiff_small_correction_keeps_iterating_to_the_residual_guard(monkeypatch):
    problem = SimpleNamespace(n_points=1, n_species=0, solve_energy=True)
    scale = 1e9
    fun = lambda x, *args, **kwargs: scale*np.expm1(x)
    monkeypatch.setattr(equations, 'residual', fun)
    def linear_model(steady_fun, x, *args):
        matrix = np.diag(scale*np.exp(x))
        return matrix, matrix, matrix.diagonal()
    monkeypatch.setattr(solver, '_build_linear_model', linear_model)
    monkeypatch.setattr(solver, 'solve_linear', np.linalg.solve)
    monkeypatch.setattr(solver, 'weighted_norm', lambda step, *args, **kwargs: np.max(abs(step))/.01)
    monkeypatch.setattr(solver, 'bound_step_limit', lambda *args: (1., None))
    x0 = np.full(2, .1)
    options = dict(max_jac_age=1, max_iter=10)
    old, old_ok, old_history, _ = solver.newton_solve(fun, x0, problem, **options)
    assert old_ok and old_history[-1]['s1'] < 1.
    assert np.max(abs(fun(old))) > 1e6
    actual, ok, history, _ = solver.newton_solve(fun, x0, problem,
                                               residual_guard_inf=100., **options)
    assert ok and len(history) > len(old_history)
    assert np.max(abs(fun(actual))) <= 100.
    assert not history[0]['residual_guard_accepted']
    assert history[-1]['residual_guard_accepted']
    assert history[-1]['normF_trial'] == np.max(abs(fun(actual)))
