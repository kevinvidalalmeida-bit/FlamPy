"""Independent primal/dispatch gates for algebraic frozen face trials.

This file imports no CLI, benchmark or fixture from the candidate. Each trial
is rebuilt as endpoint arrays and evaluated by existing scalar primal kernels.
The tolerances concern frozen face VALUES, not a differenced Jacobian, Newton
convergence, Cantera agreement or performance. Inactive values are unspecified.
"""

import numpy as np
import pytest

from kflame.flame import algebraic_frozen_transport as subject
from kflame.chemistry.mechanism import R_UNIV
from kflame.flame.diffusion_kernels import multicomponent_flux, face_viscosity
from kflame.flame.frozen_transport_columns import (
    frozen_composition_columns as legacy,
)


def _fixture(ns=9, seed=73, sparse=False, soret=True, steps="ordinary"):
    rng = np.random.default_rng(seed + ns)
    mw = rng.uniform(2., 180., ns)
    y = rng.uniform(.1, 1., (ns, 5))
    if sparse:
        y[0] = 0.                         # Both endpoints have exact zeros.
        y[1] = 1e-310                     # Below the artificial Wilke floor.
    y /= y.sum(axis=0)
    y *= np.array([.9, .73, 1.19, 1.07, .8])[None, :]
    trials = np.repeat(y[:, 2, None], ns + 2, axis=1)
    for q in range(ns):
        base = y[q, 2]
        if steps == "mixed" and base > 1e-100:
            # Exercise actual representable increments, including no change.
            options = (base, np.nextafter(base, np.inf), 1.2 * base, 1.4 * base)
            trials[q, q + 2] = options[q % len(options)]
        else:
            trials[q, q + 2] = base + 1e-10 + 1e-5 * abs(base)
    roots = [np.sqrt(rng.uniform(1e-5, 5e-5, ns)) for _ in range(2)]
    phis = [rng.uniform(.35, 2., (ns, ns)) for _ in range(2)]
    td = [(np.empty(0), np.empty((0, 0)), np.empty(0), roots[s], phis[s],
           np.empty((0, 0))) for s in range(2)]
    return dict(ns=ns, P=182000., YW=y,
                TW=np.array([700., 875., 1165., 1465., 1700.]),
                dzW=np.array([.001, .0027, .0041, .006]),
                mw=mw, invW=1. / mw, Yb=trials, td=td,
                multi=[rng.uniform(-3e-5, 3e-5, (ns, ns)) for _ in range(2)],
                thermal=[rng.uniform(-2e-7, 2e-7, ns) if soret else np.zeros(ns)
                         for _ in range(2)])


def _arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _arrays(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _arrays(child)


def _protect(f):
    arrays = list(_arrays(f))
    for array in arrays:
        array.flags.writeable = False
    return [(a, a.dtype.str, a.shape, a.strides, a.tobytes(order="C"))
            for a in arrays]


def _assert_unchanged(witnesses):
    for array, dtype, shape, strides, content in witnesses:
        assert (array.dtype.str, array.shape, array.strides) == (dtype, shape, strides)
        assert not array.flags.writeable
        assert array.tobytes(order="C") == content


def _args(f, active):
    return (f["ns"], f["YW"], f["TW"], f["dzW"], f["P"], f["invW"], f["mw"],
            f["Yb"], *active, *f["td"], f["multi"], f["thermal"])


def _primal_oracle(f, active):
    """Recompute every complete finite endpoint trial; no rank-update algebra."""
    ns = f["ns"]
    outputs = (np.empty((ns, ns)), np.empty((ns, ns)), np.empty(ns), np.empty(ns))
    for side, enabled in enumerate(active):
        if not enabled:
            continue
        a = side + 1
        tl, tr = f["TW"][a:a + 2]
        tf = .5 * (tl + tr)
        for q in range(ns):
            yl, yr = f["YW"][:, a].copy(), f["YW"][:, a + 1].copy()
            (yr if side == 0 else yl)[q] = f["Yb"][q, q + 2]
            yf = .5 * (yl + yr)
            total = 0.
            for k in range(ns):
                total += yf[k] * f["invW"][k]
            wm = 1. / max(total, 1e-300)
            rho = f["P"] * wm / (R_UNIV * tf)
            flux, scratch = np.empty(ns), np.empty(ns)
            multicomponent_flux(yl, yr, tl, tr, f["dzW"][a], tf, rho, wm,
                                f["multi"][side], f["thermal"][side], f["invW"],
                                f["mw"], flux, scratch)
            outputs[side][:, q] = flux
            x = np.empty(ns)
            for k in range(ns):
                x[k] = yf[k] * wm * f["invW"][k]
            outputs[side + 2][q] = face_viscosity(x, *f["td"][side][3:5])
    return outputs


def _assert_values(actual, expected, active):
    for side, enabled in enumerate(active):
        if not enabled:
            continue
        for index, atol in ((side, 1e-11), (side + 2, 2e-16)):
            assert np.isfinite(actual[index]).all()
            assert np.isfinite(expected[index]).all()
            np.testing.assert_allclose(actual[index], expected[index], rtol=3e-12, atol=atol)


def _assert_private(outputs, f):
    ns = f["ns"]
    assert tuple(a.shape for a in outputs) == ((ns, ns), (ns, ns), (ns,), (ns,))
    for i, array in enumerate(outputs):
        assert array.flags.c_contiguous
        assert all(not np.shares_memory(array, a) for a in _arrays(f))
        assert all(not np.shares_memory(array, other) for other in outputs[i + 1:])


def _probe(f, side):
    ns, a = f["ns"], side + 1
    flux, mu = np.empty((ns, ns)), np.empty(ns)
    valid = subject._rational_face_trials(
        f["YW"][:, a], f["YW"][:, a + 1], f["TW"][a], f["TW"][a + 1],
        f["dzW"][a], f["P"], f["invW"], f["mw"], f["Yb"], side == 1,
        f["multi"][side], f["thermal"][side], *f["td"][side][3:5], flux, mu)
    return valid, flux, mu


@pytest.mark.parametrize("ns", [3, 8, 9, 24, 25, 53])
@pytest.mark.parametrize("soret", [False, True])
def test_finite_trials_against_independent_scalar_primal(ns, soret):
    f = _fixture(ns, soret=soret, steps="mixed")
    witnesses = _protect(f)
    active = (True, True)
    actual = subject.frozen_composition_columns(*_args(f, active))
    _assert_values(actual, _primal_oracle(f, active), active)
    for side in range(2):
        valid, flux, mu = _probe(f, side)
        assert valid, "An unconditional fallback would not validate the new path"
        assert actual[side].tobytes() == flux.tobytes()
        assert actual[side + 2].tobytes() == mu.tobytes()
    _assert_private(actual, f)
    _assert_unchanged(witnesses)


@pytest.mark.parametrize("active", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("zero_denominator", [False, True])
def test_zeros_subfloor_species_and_wilke_clamps(active, zero_denominator):
    f = _fixture(sparse=True)
    if zero_denominator:
        # A zero-species row makes the numerator/denominator floors measurable
        # in non-q=0 columns, without an invalid zero total composition.
        for td in f["td"]:
            td[4][0, :] = 0.
    witnesses = _protect(f)
    actual = subject.frozen_composition_columns(*_args(f, active))
    _assert_values(actual, _primal_oracle(f, active), active)
    for side, enabled in enumerate(active):
        if enabled:
            valid, flux, mu = _probe(f, side)
            assert valid
            assert actual[side].tobytes() == flux.tobytes()
            assert actual[side + 2].tobytes() == mu.tobytes()
    _assert_private(actual, f)
    _assert_unchanged(witnesses)


@pytest.mark.parametrize("bad_side", [0, 1])
@pytest.mark.parametrize("kind", ["negative-endpoint", "zero-endpoint",
                                  "negative-root", "negative-phi", "floor-underflow"])
def test_native_unsafe_path_preserves_legacy_whole_call(bad_side, kind):
    f = _fixture()
    # Mutate the exterior endpoint, keeping the central node/trials intact.
    exterior = 1 if bad_side == 0 else 3
    if kind == "negative-endpoint":
        f["YW"][0, exterior] *= -.1
    elif kind == "zero-endpoint":
        f["YW"][:, exterior] = 0.
    elif kind == "negative-root":
        f["td"][bad_side][3][0] *= -1.
    elif kind == "negative-phi":
        f["td"][bad_side][4][0, 0] = -.01
    else:
        # Normalized primal remains finite, but 1e-300*sF underflows.
        f["invW"] *= 1e-40
        f["mw"][:] = 1. / f["invW"]
    witnesses = _protect(f)
    active = (True, True)
    assert not _probe(f, bad_side)[0]
    if kind != "floor-underflow":
        assert _probe(f, 1 - bad_side)[0]
    args = _args(f, active)
    actual, original = subject.frozen_composition_columns(*args), legacy(*args)
    _assert_values(actual, _primal_oracle(f, active), active)
    for a, b in zip(actual, original):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.isfinite(b).all()
        assert a.tobytes() == b.tobytes(), "Both faces must revert to legacy"
    _assert_private(actual, f)
    _assert_unchanged(witnesses)


@pytest.mark.parametrize("failed_side", [0, 1])
def test_wrapper_discards_all_provisional_outputs(monkeypatch, failed_side):
    """Dispatch contract only: spy Python body plus separate native gates above."""
    f = _fixture()
    witnesses = _protect(f)
    visits, fallback_calls = [], []
    ns = f["ns"]
    sentinel = (np.full((ns, ns), 91.), np.full((ns, ns), 92.),
                np.full(ns, 93.), np.full(ns, 94.))

    def face_spy(*args):
        side = 1 if args[9] else 0  # left_changed for the central node
        visits.append(side)
        args[-2][:] = -10. - side
        args[-1][:] = -20. - side
        return side != failed_side

    def fallback_spy(*args):
        fallback_calls.append(args)
        return sentinel

    monkeypatch.setattr(subject, "_rational_face_trials", face_spy)
    monkeypatch.setattr(subject, "_original_frozen_composition_columns", fallback_spy)
    args = _args(f, (True, True))
    actual = subject.frozen_composition_columns.py_func(*args)
    assert actual is sentinel
    assert visits == ([0] if failed_side == 0 else [0, 1])
    assert len(fallback_calls) == 1
    assert all(a is b for a, b in zip(fallback_calls[0], args))
    _assert_unchanged(witnesses)


@pytest.mark.parametrize("active", [(True, False), (False, True), (False, False)])
def test_inactive_faces_do_not_consume_poisoned_inputs(active):
    f = _fixture()
    for side, enabled in enumerate(active):
        if enabled:
            continue
        exterior = 1 if side == 0 else 3
        f["YW"][:, exterior] = np.nan
        f["TW"][exterior] = np.nan
        f["dzW"][side + 1] = 0.
        for array in (*f["td"][side], f["multi"][side], f["thermal"][side]):
            array[:] = np.nan
    if not any(active):
        f["P"] = np.nan
        f["Yb"][:] = np.nan
    witnesses = _protect(f)
    actual = subject.frozen_composition_columns(*_args(f, active))
    _assert_values(actual, _primal_oracle(f, active), active)
    for side, enabled in enumerate(active):
        if enabled:
            valid, flux, mu = _probe(f, side)
            assert valid
            assert actual[side].tobytes() == flux.tobytes()
            assert actual[side + 2].tobytes() == mu.tobytes()
    _assert_private(actual, f)
    _assert_unchanged(witnesses)
    # Deliberately no equality/finiteness assertions on inactive output values.


def test_readonly_strided_endpoints_and_fortran_coefficients():
    f = _fixture(25, sparse=True)
    for key in ("YW", "Yb"):
        value = f[key]
        storage = np.full((value.shape[0] * 2, value.shape[1] * 2), -917.)
        storage[::2, ::2] = value
        f[key] = storage[::2, ::2]
    for key in ("mw", "invW", "TW", "dzW"):
        value = f[key]
        storage = np.full(value.size * 2, -919.)
        storage[::2] = value
        f[key] = storage[::2]
    for side in range(2):
        f["multi"][side] = np.asfortranarray(f["multi"][side])
        td = list(f["td"][side])
        td[4] = np.asfortranarray(td[4])
        f["td"][side] = tuple(td)
    witnesses = _protect(f)
    actual = subject.frozen_composition_columns(*_args(f, (True, True)))
    _assert_values(actual, _primal_oracle(f, (True, True)), (True, True))
    for side in range(2):
        valid, flux, mu = _probe(f, side)
        assert valid
        assert actual[side].tobytes() == flux.tobytes()
        assert actual[side + 2].tobytes() == mu.tobytes()
    _assert_private(actual, f)
    _assert_unchanged(witnesses)


def test_owned_outputs_survive_a_b_a_calls():
    first, second = _fixture(seed=103), _fixture(seed=211)
    witnesses = _protect(first) + _protect(second)
    active = (True, True)
    a = subject.frozen_composition_columns(*_args(first, active))
    saved = tuple(out.copy() for out in a)
    b = subject.frozen_composition_columns(*_args(second, active))
    again = subject.frozen_composition_columns(*_args(first, active))
    for outputs, fixture in ((a, first), (b, second), (again, first)):
        _assert_private(outputs, fixture)
        _assert_values(outputs, _primal_oracle(fixture, active), active)
    for out, original, repeat in zip(a, saved, again):
        assert out.tobytes() == original.tobytes() == repeat.tobytes()
    assert all(not np.shares_memory(x, y) for x in a for y in (*b, *again))
    _assert_unchanged(witnesses)


@pytest.mark.parametrize('dominant', [False, True])
def test_negative_increment_reverts_whole_call(dominant):
    f = _fixture(3)
    if dominant:
        # Subtracting a dominant species loses the small remaining mole sum.
        f['mw'][:] = 1.
        f['invW'][:] = 1.
        f['YW'][:] = np.array([1., 3 * 2. ** -54, 0.])[:, None]
        f['Yb'][:] = f['YW'][:, 2, None]
        f['Yb'][0, 2] = 0.
    else:
        f['Yb'][0, 2] = .5 * f['YW'][0, 2]
    witnesses = _protect(f)
    assert not _probe(f, 0)[0] and not _probe(f, 1)[0]
    actual = subject.frozen_composition_columns(*_args(f, (True, True)))
    original = legacy(*_args(f, (True, True)))
    for a, b in zip(actual, original):
        assert np.isfinite(b).all()
        assert a.tobytes() == b.tobytes()
    _assert_values(actual, _primal_oracle(f, (True, True)), (True, True))
    _assert_unchanged(witnesses)


def test_subnormal_scaled_floor_preserves_primal_wilke_contribution():
    f = _fixture(sparse=True)
    f['YW'] *= 1e-20
    f['Yb'][:] = f['YW'][:, 2, None]  # Zero increments, tiny face mole sum.
    for td in f['td']:
        td[4][0, :] = 0.
    witnesses = _protect(f)
    actual = subject.frozen_composition_columns(*_args(f, (True, True)))
    _assert_values(actual, _primal_oracle(f, (True, True)), (True, True))
    for side in range(2):
        assert _probe(f, side)[0]
    _assert_unchanged(witnesses)
