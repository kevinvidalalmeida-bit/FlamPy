"""Independent finite-difference checks of frozen face composition derivatives.

The oracle evaluates the existing primal flux and Wilke kernels. It never
calls a derivative helper or normalizes perturbed endpoint mass fractions.
Only composition changes: D, thermal diffusion, temperature data and lambda
remain frozen. At zero species, the oracle approaches from positive Y.
"""

from decimal import Decimal, localcontext
import numpy as np
import pytest

from kflame.chemistry.mechanism import R_UNIV
from kflame.flame import diffusion_kernels as kernels
from kflame.flame.frozen_transport_derivatives import frozen_face_derivatives


_RELATIVE_STEPS = (2e-3, 5e-4, 1.25e-4, 3.125e-5)


def _state(ns, seed=137):
    rng = np.random.default_rng(seed + ns)
    mw = rng.uniform(2., 180., ns)
    yl = rng.uniform(.2, 1., ns)
    yr = rng.uniform(.2, 1., ns)
    # Deliberately distinct, nonunit sums: Newton changes independent Y_k.
    yl *= .73 / yl.sum()
    yr *= 1.19 / yr.sum()
    matrix = rng.uniform(.2e-5, 3.e-5, (ns, ns))
    np.fill_diagonal(matrix, 0.)
    return dict(YL=yl, YR=yr, TL=875., TR=1465., dz=.0027, P=182000.,
                invW=1. / mw, mw=mw, D=matrix,
                root=np.sqrt(rng.uniform(1.e-5, 6.e-5, ns)),
                phi=rng.uniform(.35, 2.5, (ns, ns)),
                thermal=rng.uniform(-2.e-7, 2.e-7, ns))


def _arrays(state):
    return [value for value in state.values() if isinstance(value, np.ndarray)]


def _readonly(state):
    for value in _arrays(state):
        value.flags.writeable = False
    return state


def _derivatives(state, left_changed):
    result = frozen_face_derivatives(
        state['YL'], state['YR'], state['TL'], state['TR'], state['dz'],
        state['P'], state['invW'], state['mw'], state['D'], state['root'],
        state['phi'], left_changed)
    valid, dflux, dmu = result
    ns = state['mw'].size
    assert isinstance(valid, (bool, np.bool_))
    assert dflux.shape == (ns, ns)
    assert dmu.shape == (ns,)
    if valid:
        assert np.isfinite(dflux).all()
        assert np.isfinite(dmu).all()
    return valid, dflux, dmu


def _primal(state, yl, yr):
    """Evaluate production face operations, independently of new derivatives."""
    tf = .5 * (state['TL'] + state['TR'])
    yf = .5 * (yl + yr)
    wm = 1. / max(float(np.dot(yf, state['invW'])), 1e-300)
    rho = state['P'] * wm / (R_UNIV * tf)
    ns = yl.size
    flux = np.empty(ns)
    scratch = np.empty(ns)
    kernels.multicomponent_flux(
        yl, yr, state['TL'], state['TR'], state['dz'], tf, rho, wm,
        state['D'], state['thermal'], state['invW'], state['mw'], flux, scratch)
    x = yf * wm * state['invW']
    mu = kernels.face_viscosity(x, state['root'], state['phi'])
    return flux, float(mu)


def _along(state, left_changed, direction, step):
    yl = state['YL'].copy()
    yr = state['YR'].copy()
    changed = yl if left_changed else yr
    changed += step * direction
    assert np.all(changed >= 0.), 'FD must stay on the physical side of Y=0'
    return _primal(state, yl, yr)


def _scaled_error(actual, expected):
    expected = np.asarray(expected)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    return float(np.max(np.abs(np.asarray(actual) - expected))) / scale


def _wilke_extended(state, yl, yr):
    """Same primal equation at higher precision for tiny zero-species slopes.

    A 2e-10 viscosity slope obtained by subtracting viscosities near 1e-5
    loses useful digits in double precision. This independent primal oracle
    reduces that cancellation without changing the derivative error gate.
    """
    # Decimal is portable: on Windows np.longdouble can equal float64.
    with localcontext() as ctx:
        ctx.prec=50
        number=lambda value:Decimal.from_float(float(value))
        yf=[(number(a)+number(b))/2 for a,b in zip(yl,yr)]
        weights=[number(w) for w in state['invW']]
        denominator=sum(y*w for y,w in zip(yf,weights))
        x=[y*w/denominator for y,w in zip(yf,weights)]
        root=[number(r) for r in state['root']]
        result=Decimal(0);floor=Decimal('1e-300')
        for k in range(len(x)):
            denom=sum(number(p)*xx for p,xx in zip(state['phi'][k],x))
            result+=max(x[k],floor)*root[k]**2/max(denom,floor)
        return result


def _check_sweep(estimates, expected, label):
    """Require accuracy at two fine steps and refinement up to the noise floor."""
    expected_flux, expected_mu = expected
    flux_errors = [_scaled_error(flux, expected_flux) for flux, _ in estimates]
    mu_errors = [_scaled_error(mu, expected_mu) for _, mu in estimates]
    for errors in (flux_errors, mu_errors):
        assert np.isfinite(errors).all(), (label, errors)
        assert max(errors[-2:]) < 2e-7, (label, errors)
        # Exact/nearlinear directions need not decrease below FD roundoff.
        assert min(errors[-2:]) <= max(.2 * errors[0], 5e-8), (label, errors)


def _central_sweep(state, left_changed, direction):
    estimates = []
    for step in _RELATIVE_STEPS:
        plus_flux, plus_mu = _along(state, left_changed, direction, step)
        minus_flux, minus_mu = _along(state, left_changed, direction, -step)
        estimates.append(((plus_flux - minus_flux) / (2. * step),
                          (plus_mu - minus_mu) / (2. * step)))
    return estimates


@pytest.mark.parametrize('ns', [3, 23, 24, 25, 31, 33])
@pytest.mark.parametrize('left_changed', [False, True])
def test_frozen_face_directional_derivatives_match_primal_fd(ns, left_changed):
    state = _readonly(_state(ns))
    witnesses = _arrays(state)
    before = [value.tobytes() for value in witnesses]
    valid, dflux, dmu = _derivatives(state, left_changed)
    assert valid
    endpoint = state['YL'] if left_changed else state['YR']
    rng = np.random.default_rng(811 + ns)
    signed = rng.uniform(-1., 1., ns) * endpoint
    directions = [('dense-signed', signed), ('endpoint-scale', endpoint.copy())]
    # The final index exercises scalar tails beyond any block of eight.
    for q in sorted({0, ns // 2, ns - 1}):
        coordinate = np.zeros(ns)
        coordinate[q] = endpoint[q]
        directions.append((f'column-{q}', coordinate))
    for label, direction in directions:
        expected = dflux @ direction, float(dmu @ direction)
        _check_sweep(_central_sweep(state, left_changed, direction), expected,
                     (ns, left_changed, label))
    assert [value.tobytes() for value in witnesses] == before


@pytest.mark.parametrize('left_changed', [False, True])
def test_each_column_matches_independent_fd(left_changed):
    state = _state(7, seed=941)
    valid, dflux, dmu = _derivatives(state, left_changed)
    assert valid
    endpoint = state['YL'] if left_changed else state['YR']
    for q in range(endpoint.size):
        direction = np.zeros(endpoint.size)
        direction[q] = endpoint[q]
        expected = dflux[:, q] * endpoint[q], dmu[q] * endpoint[q]
        _check_sweep(_central_sweep(state, left_changed, direction), expected,
                     ('every-column', left_changed, q))


@pytest.mark.parametrize('left_changed', [False, True])
def test_readonly_strided_endpoints_and_fortran_coefficients(left_changed):
    state = _state(25, seed=101)
    for key in ('YL', 'YR', 'mw', 'invW', 'root'):
        storage = np.empty(2 * state[key].size)
        storage[::2] = state[key]
        storage[1::2] = -987.
        state[key] = storage[::2]
    for key in ('D', 'phi'):
        state[key] = np.asfortranarray(state[key])
    _readonly(state)
    witnesses = _arrays(state)
    before = [value.tobytes() for value in witnesses]
    valid, dflux, dmu = _derivatives(state, left_changed)
    assert valid
    endpoint = state['YL'] if left_changed else state['YR']
    direction = endpoint * np.linspace(-.8, 1., endpoint.size)
    _check_sweep(_central_sweep(state, left_changed, direction),
                 (dflux @ direction, float(dmu @ direction)),
                 ('strided-readonly', left_changed))
    assert [value.tobytes() for value in witnesses] == before


@pytest.mark.parametrize('ns', [7, 25])
@pytest.mark.parametrize('left_changed', [False, True])
@pytest.mark.parametrize('zero_side', ['changed', 'both'])
def test_zero_species_derivatives_match_positive_right_limit(ns, left_changed, zero_side):
    state = _state(ns, seed=307)
    endpoint = state['YL'] if left_changed else state['YR']
    zero_indices = sorted({0, ns // 2, ns - 1})
    endpoint[zero_indices] = 0.
    if zero_side == 'both':
        state['YL'][zero_indices] = 0.
        state['YR'][zero_indices] = 0.
    # Other species stay positive, so endpoint and face molecular sums exist.
    scale = float(np.min(endpoint[endpoint > 0.]))
    _readonly(state)
    witnesses = _arrays(state)
    before = [value.tobytes() for value in witnesses]
    valid, dflux, dmu = _derivatives(state, left_changed)
    assert valid, 'Zero fractions with positive sums need the positive-side tangent'
    base_flux, base_mu = _primal(state, state['YL'], state['YR'])
    base_mu = _wilke_extended(state,state['YL'],state['YR'])
    for q in zero_indices:
        direction = np.zeros(ns)
        direction[q] = scale
        estimates = []
        for step in _RELATIVE_STEPS:
            f1, m1 = _along(state, left_changed, direction, step)
            f2, m2 = _along(state, left_changed, direction, 2. * step)
            yl,yr=state['YL'].copy(),state['YR'].copy()
            endpoint1=yl if left_changed else yr
            endpoint1+=step*direction
            m1=_wilke_extended(state,yl,yr)
            endpoint1+=step*direction
            m2=_wilke_extended(state,yl,yr)
            # Extrapolated forward FD; never evaluates a negative fraction.
            with localcontext() as ctx:
                ctx.prec=50
                mu_estimate=float((-3*base_mu+4*m1-m2)/Decimal.from_float(2.*step))
            estimates.append(((-3. * base_flux + 4. * f1 - f2) / (2. * step),mu_estimate))
        _check_sweep(estimates, (dflux[:, q] * scale, dmu[q] * scale),
                     ('zero-right-limit', ns, left_changed, zero_side, q))
    assert [value.tobytes() for value in witnesses] == before


@pytest.mark.parametrize('bad_endpoint', ['YL', 'YR'])
@pytest.mark.parametrize('kind', ['zero-sum', 'negative-sum', 'negative-species',
                                  'nan', 'infinity'])
@pytest.mark.parametrize('left_changed', [False, True])
def test_invalid_compositions_request_fallback(bad_endpoint, kind, left_changed):
    state = _state(7)
    y = state[bad_endpoint]
    if kind == 'zero-sum':
        y[:] = 0.
    elif kind == 'negative-sum':
        y[:] *= -1.
        assert np.dot(y, state['invW']) < 0.
    elif kind == 'negative-species':
        # A positive sum alone does not make a negative Y physically valid.
        q = int(np.argmin(y * state['invW']))
        y[q] *= -.25
        assert np.dot(y, state['invW']) > 0.
    elif kind == 'nan':
        y[2] = np.nan
    else:
        y[2] = np.inf
    _readonly(state)
    witnesses = _arrays(state)
    before = [value.tobytes() for value in witnesses]
    valid, _, _ = _derivatives(state, left_changed)
    assert not valid, (bad_endpoint, kind, left_changed)
    # Invalid output payloads are deliberately not consumed by this contract.
    assert [value.tobytes() for value in witnesses] == before


@pytest.mark.parametrize('left_changed', [False, True])
def test_results_are_owned_and_survive_a_b_a_calls(left_changed):
    first = _readonly(_state(25, seed=59))
    second = _readonly(_state(25, seed=71))
    valid, dflux, dmu = _derivatives(first, left_changed)
    assert valid
    saved_flux, saved_mu = dflux.copy(), dmu.copy()
    for output in (dflux, dmu):
        assert all(not np.shares_memory(output, value) for value in _arrays(first))
    assert _derivatives(second, left_changed)[0]
    again_valid, again_flux, again_mu = _derivatives(first, left_changed)
    assert again_valid
    np.testing.assert_array_equal(dflux, saved_flux)
    np.testing.assert_array_equal(dmu, saved_mu)
    np.testing.assert_allclose(again_flux, saved_flux, rtol=2e-14, atol=0.)
    np.testing.assert_allclose(again_mu, saved_mu, rtol=2e-14, atol=0.)


def test_helper_compiles_in_nopython_mode():
    assert _derivatives(_state(3), True)[0]
    assert frozen_face_derivatives.nopython_signatures
