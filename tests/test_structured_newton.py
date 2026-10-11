"""Analytic structured Jacobian, its factorization, fast faces and Newton-Krylov."""
import numpy as np
import pytest

from kflame.chemistry import analytic
from kflame.chemistry.initialization import _composition, hp_equilibrium
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion as D
from kflame.flame import diffusion_kernels as K
from kflame.flame import fast_transport as F
from kflame.flame import structured_jacobian as S
from kflame.flame import structured_linear as L


def problem(mechanism='h2o2.yaml', fuel='H2', points=9, *, soret=False, radiation=False):
    mech = load_mechanism(resolve_mechanism(mechanism))
    feeds = []
    for text in (fuel, 'O2:1,N2:3.76'):
        y = _composition(text, mech.species_names)*mech.molecular_weights
        feeds.append(y/y.sum())
    case = D.DiffusionCase(resolve_mechanism(mechanism), *feeds, fuel_mass_flux=.05, oxidizer_mass_flux=.15,
                           soret_enabled=soret, radiation_enabled=radiation,
                           radiation_scale=2. if radiation else 1.,
                           boundary_emissivities=(.4, .8) if radiation else (0., 0.))
    return D.DiffusionFlameProblem(case, mech, np.linspace(0, .02, points))


def generic_state(p, seed=0):
    """Every species present and a perturbed temperature: no exact zeros."""
    x = p.pack(p.equilibrated_guess())
    rng = np.random.default_rng(seed)
    b = x[:-1].reshape(p.n, p.nv)
    b[:, 3:] = np.abs(b[:, 3:])+1e-6*rng.random(b[:, 3:].shape)
    b[:, 3:] /= b[:, 3:].sum(axis=1, keepdims=True)
    b[1:-1, 2] *= 1+.02*rng.standard_normal(p.n-2)
    return x


def frozen_residual(p, x0):
    """Residual whose face coefficients keep the composition of x0 (any T)."""
    data = p._compiled_data()
    Y0 = p.unpack(x0)['Y']
    Yf0 = np.ascontiguousarray(.5*(Y0[:, :-1]+Y0[:, 1:]))
    cond_poly, diff_poly, visc_poly, quarter, wilke, eps, cstar, tiny = data['mixture']
    ns, n = p.ns, p.n
    tdata = K._tdata(ns, False)
    work = np.empty((5 if p.soret_enabled else 2, ns))

    def residual(x):
        f = p.unpack(x)
        T, Y = np.ascontiguousarray(f['T']), np.ascontiguousarray(f['Y'])
        omega, hk = np.empty_like(Y), np.empty_like(Y)
        rho, cp = p.backend.eval_grid_thermo_kinetics_into(T, Y, omega, hk)
        hk *= p.mech_data.inv_molecular_weights[:, None]
        lam, mu, J = np.empty(n-1), np.empty(n-1), np.empty((ns, n-1))
        X, Dk, DT, flux = np.empty(ns), np.empty(ns), np.zeros(ns), np.empty(ns)
        for face in range(n-1):
            Tf = .5*(T[face]+T[face+1])
            K.face_temperature_data(Tf, float(p.P), data['mw'], cond_poly, diff_poly, visc_poly, quarter, wilke,
                                    p.soret_enabled, eps, cstar, *tdata)
            rf, wf, lam[face], mu[face] = K._face_mixture_ws(Tf, np.ascontiguousarray(Yf0[:, face]), float(p.P),
                                                             data['invW'], data['mw'], *tdata[:5],
                                                             p.soret_enabled, tdata[5], tiny, X, Dk, DT, work)
            K.mixture_flux(np.ascontiguousarray(Y[:, face]), np.ascontiguousarray(Y[:, face+1]), T[face],
                           T[face+1], p.dz[face], Tf, rf, wf, Dk, DT, p.soret_enabled, data['mw'], flux)
            J[:, face] = flux
        return p.residual(x, cached_properties=(rho, cp, omega, hk, lam, J, mu))
    return residual


@pytest.mark.parametrize('soret,radiation', [(False, False), (True, True)])
def test_structured_jacobian_is_the_frozen_transport_derivative(soret, radiation):
    p = problem(soret=soret, radiation=radiation)
    x = generic_state(p)
    p.compile_kernels(x)
    residual = frozen_residual(p, x)
    np.testing.assert_allclose(residual(x), p.residual(x), rtol=1e-11, atol=1e-13)
    analytic_matrix = S.structured_blocks(p, x).dense().tocsc().toarray()
    for column in range(len(x)):
        h = 1e-7*max(abs(x[column]), 1.) if column % p.nv < 3 or column == len(x)-1 else 1e-7*max(abs(x[column]), 1e-2)
        plus, minus = x.copy(), x.copy()
        plus[column] += h
        minus[column] -= h
        expected = (residual(plus)-residual(minus))/(2*h)
        scale = np.abs(expected).max()+np.abs(analytic_matrix[:, column]).max()
        np.testing.assert_allclose(analytic_matrix[:, column], expected, rtol=0, atol=2e-6*scale+1e-7)


def test_structured_factor_matches_dense_bordered_factor_and_in_place():
    p = problem(points=21)
    x = generic_state(p, 1)
    p.compile_kernels(x)
    blocks = S.structured_blocks(p, x)
    rhs = np.random.default_rng(2).standard_normal(len(x))
    for rdt in (0., 1e3):
        expected = D.BorderedFactor(blocks.dense(), rdt).solve(rhs)
        actual = L.StructuredFactor(blocks, rdt).solve(rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12*np.abs(expected).max())
    copy = S.structured_blocks(p, x)
    in_place = L.StructuredFactor(copy, 0., inplace=True)
    assert copy.consumed
    np.testing.assert_allclose(in_place.solve(rhs), D.BorderedFactor(blocks.dense()).solve(rhs), rtol=1e-9,
                               atol=1e-12)
    with pytest.raises(ValueError):
        L.StructuredFactor(copy, 1.)


def test_gmres_reaches_tolerance_and_reports_failure():
    rng = np.random.default_rng(3)
    A = np.eye(40)+.3*rng.standard_normal((40, 40))/np.sqrt(40)
    b = rng.standard_normal(40)
    inverse = np.linalg.inv(np.diag(np.diag(A)))
    x, ok = D._gmres(lambda v: A@v, lambda v: inverse@v, b, inverse@b, 1e-10, 40)
    assert ok and np.linalg.norm(A@x-b) <= 1e-9*np.linalg.norm(b)
    x, ok = D._gmres(lambda v: A@v, lambda v: inverse@v, b, inverse@b, 1e-12, 2)
    assert not ok and np.linalg.norm(A@x-b) < np.linalg.norm(b)


@pytest.mark.parametrize('soret', [False, True])
@pytest.mark.parametrize('mechanism,fuel', [('h2o2.yaml', 'H2'), ('gri30.yaml', 'CH4')])
def test_fast_faces_match_scalar_face_kernels(mechanism, fuel, soret):
    p = problem(mechanism, fuel, points=40, soret=soret)
    p.fast_transport = True
    data = p._compiled_data()
    rng = np.random.default_rng(4)
    Y = rng.random((p.ns, p.n))**4
    Y[rng.random(Y.shape) < .3] = 0.
    Y /= Y.sum(axis=0)
    T = np.linspace(300., 2400., p.n)
    n = p.n
    lam, mu, J = np.empty(n-1), np.empty(n-1), np.empty((p.ns, n-1))
    K.face_range(0, n-1, T, Y, p.dz, float(p.P), data['invW'], data['mw'], data['mixture'], data['multi'],
                 False, soret, lam, mu, J)
    lam2, mu2, J2 = D.face_properties(p, T, Y)
    np.testing.assert_allclose(lam2, lam, rtol=1e-12)
    np.testing.assert_allclose(mu2, mu, rtol=1e-12)
    np.testing.assert_allclose(J2, J, rtol=0, atol=1e-12*np.abs(J).max())


def test_sparse_partials_match_the_dense_reference():
    from _analytic_partials_reference import species_partials as reference
    p = problem('gri30.yaml', 'CH4')
    rng = np.random.default_rng(5)
    Y = rng.random((p.ns, 6))**6
    Y[rng.random(Y.shape) < .3] = 0.
    Y /= Y.sum(axis=0)
    T = np.linspace(500., 2500., 6)
    jy, jt = analytic.species_partials(p.backend, T, Y, _temperature=True)
    ry, rt = reference(p.backend, T, Y, _temperature=True)
    np.testing.assert_allclose(jy, ry, rtol=0, atol=1e-13*np.abs(ry).max())
    np.testing.assert_allclose(jt, rt, rtol=0, atol=1e-13*np.abs(rt).max())


def test_structured_newton_converges_to_the_finite_difference_newton_root():
    solutions = []
    for structured in (False, True):
        p = problem(points=15)
        p.structured_jacobian = structured
        p.newton_krylov = structured
        fields, accepted, report = D.solve_diffusion_problem(p, max_time=300, refine=False)
        assert report['converged'] and report['maximum_scaled_residual'] < 1e-7
        solutions.append(p.pack(fields))
    scale = np.maximum(np.abs(solutions[0]), 1e-6)
    assert np.max(np.abs(solutions[1]-solutions[0])/scale) < 1e-5


def test_equilibrium_extrapolates_beyond_a_2000_K_common_nasa_limit():
    mech = load_mechanism(resolve_mechanism('uscmech2.yaml'))
    assert mech.max_temperature == 2000.
    Y = _composition('CH4:1,O2:2,N2:7.52', mech.species_names)*mech.molecular_weights
    T, Yeq, rho = hp_equilibrium(mech, 300., 101325., Y/Y.sum())
    assert 2000. < T < 2400. and np.isfinite(Yeq).all()


def test_refinement_ratios_flag_the_same_intervals_as_the_refiner():
    from kflame.flame.solver import AdaptiveRefiner
    z = np.linspace(0., 1., 41)
    profiles = {'a': np.tanh((z-.4)/.03), 'b': np.exp(-((z-.6)/.05)**2), 'c': z}
    refiner = AdaptiveRefiner(ratio=3., slope=.05, curve=.1, prune=-1., max_points=1000)
    insert, _ = refiner.analyze(z, profiles)
    ratios = D._refine_ratios(z, profiles, refiner)
    # Every interval violating slope/curvature is flagged; flags beyond those
    # come only from the adjacent-interval size ratio.
    assert insert and set(np.flatnonzero(ratios > 1.)) <= insert


@pytest.mark.parametrize('rdt', [0., 1e3])
def test_lazy_chunked_factor_matches_the_full_structured_factor(rdt, monkeypatch):
    monkeypatch.setattr(L, 'CHUNK_BYTES', 3*8*13*13)  # three-node chunks on H2
    p = problem(points=23)
    x = generic_state(p, 6)
    p.compile_kernels(x)
    full = S.structured_blocks(p, x)
    lazy = S.structured_blocks(p, x, lazy=True)
    assert lazy.lazy and lazy.diag is None
    rhs = np.random.default_rng(7).standard_normal(len(x))
    expected = L.StructuredFactor(full, rdt).solve(rhs)
    actual = L.StructuredFactor(lazy, rdt).solve(rhs)
    assert actual.dtype == np.float64
    # Same float64 operations in the same order as the full factorization.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12*np.abs(expected).max())
    for name in ('lowD', 'lowP', 'lowQ', 'upD', 'upP', 'upQ', 'border'):
        np.testing.assert_array_equal(getattr(lazy, name), getattr(full, name))
    np.testing.assert_array_equal(lazy.dense().diag, full.diag)


def test_pivot_inverse_pivots_the_rows_of_the_block():
    # LAPACK sees the C-ordered block as its transpose; pivoting must still
    # follow the rows of the block itself (as numpy), which is invariant to
    # the column (variable) scaling of the stiff chemistry blocks.
    import scipy.linalg
    from kflame.flame import structured_linear as L
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 30))*np.logspace(-12, 4, 30)[None, :]
    G = A.copy()
    ipiv = np.empty(30, dtype=L.F_INT_nptype)
    assert L._invert_pivot(G, ipiv, np.empty((30, 0)), np.empty((0, 0)))
    np.testing.assert_array_equal(ipiv-1, scipy.linalg.lu_factor(A)[1])
    np.testing.assert_allclose(G, np.linalg.inv(A), rtol=1e-9)


def test_buffer_pool_reuses_idle_buffers_and_bounds_idle_memory(monkeypatch):
    from kflame.flame import buffer_pool
    pool = buffer_pool.Pool()
    a = pool.take((4, 5))
    b = pool.take((4, 5))
    assert not np.shares_memory(a, b)          # both in use: distinct storage
    base = id(a.base)
    del a
    c = pool.take((4, 5))
    assert id(c.base) == base                   # the idle buffer is reused
    del b, c
    for _ in range(50):                         # repeated same-shape requests never grow the pool
        view = pool.take((4, 5)).T
        del view
    assert pool.stats()[0] == 2
    monkeypatch.setattr(buffer_pool, 'MAX_FREE_BYTES', 1000)
    for size in range(1, 40):                   # many idle shapes: idle bytes stay bounded
        pool.take((size, 10))
    count, total, idle = pool.stats()
    assert idle <= 1000+8*39*10
