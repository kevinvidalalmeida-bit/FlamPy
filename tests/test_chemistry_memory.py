"""Column locality and bitwise regression against the previous C-order route."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import numba
import pytest

from kflame.chemistry import backend as backend_module
from kflame.chemistry import transport as transport_module
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism


@pytest.fixture(scope="module", autouse=True)
def bounded_numba_threads():
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    yield
    numba.set_num_threads(previous)


@contextmanager
def previous_c_route(monkeypatch):
    """Run unchanged kernels with the old C-order preparation, locally only."""
    def force_c(kernel, argument_index):
        def wrapped(*args):
            args = list(args)
            args[argument_index] = np.ascontiguousarray(args[argument_index])
            return kernel(*args)
        return wrapped

    with monkeypatch.context() as patch:
        for module, name, index in (
            (transport_module, "_eval_faces_poly_numba_core", 1),
            (transport_module, "_mix_soret_fused", 2),
            (backend_module, "_eval_thermo_kinetics_sparse_numba_core", 1),
            (backend_module, "_eval_jacobian_grouped_core", 1),
        ):
            patch.setattr(module, name, force_c(getattr(module, name), index))
        yield


def make_backend(mechanism="h2o2.yaml"):
    path = resolve_mechanism(mechanism)
    problem = SimpleNamespace(P=101325.0, case=SimpleNamespace(mech=path))
    return NativeSpeciesBackend(problem, load_mechanism(path))


def state(n_species, *, signed=False):
    temperature = np.array([300.0, 1000.0, 1000.01, 2100.0])
    y = np.random.default_rng(42).uniform(0.01, 1.0, (n_species, 4))
    y /= y.sum(axis=0)
    y[:, 2] *= 1.01  # Unnormalized Newton state must retain its scale.
    y[2, 3] = 0.0
    if signed:
        y[1, 1] *= -0.001
    return temperature, y


def with_layout(array, layout):
    if layout == "F_readonly":
        result = np.array(array, order="F", copy=True)
        result.setflags(write=False)
        return result
    if layout == "strided":
        storage = np.empty((array.shape[0], 2 * array.shape[1]))
        view = storage[:, ::2]
        view[:] = array
        return view
    return np.array(array, order=layout, copy=True)


def assert_bitwise(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("mechanism", ["h2o2.yaml", "gri30.yaml"])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "F_readonly"])
def test_transport_bitwise_c_route_and_independent_outputs(monkeypatch, mechanism, layout):
    current = make_backend(mechanism)
    original = make_backend(mechanism)
    temperature, y = state(current.n_species)
    y = with_layout(y, layout)
    before = y.copy()
    actual = current.eval_faces(temperature, y)
    with previous_c_route(monkeypatch):
        expected = original.eval_faces(temperature, y)
    for result, reference in zip(actual, expected):
        assert_bitwise(result, reference)
        assert not np.shares_memory(result, y)
    assert actual[1].flags.c_contiguous == expected[1].flags.c_contiguous
    assert actual[1].strides == expected[1].strides

    soret = current.transport.thermal_diff_coeffs(temperature, current.problem.P, y)
    with previous_c_route(monkeypatch):
        reference = original.transport.thermal_diff_coeffs(temperature, original.problem.P, y)
    assert_bitwise(soret, reference)
    assert soret.flags.c_contiguous == reference.flags.c_contiguous
    assert soret.strides == reference.strides
    assert not np.shares_memory(soret, y)
    # Later calls and writes to other outputs cannot overwrite retained arrays.
    held = actual[1].copy()
    held_soret = soret.copy()
    current.eval_faces(temperature + 1.0, y)
    current.transport.thermal_diff_coeffs(temperature + 1.0, current.problem.P, y)
    assert_bitwise(soret, held_soret)
    for other in (actual[0], actual[2], actual[3], soret):
        assert not np.shares_memory(actual[1], other)
        other.fill(0.0)
    assert_bitwise(actual[1], held)
    assert_bitwise(y, before)


@pytest.mark.parametrize("mechanism", ["h2o2.yaml", "gri30.yaml"])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "F_readonly"])
def test_fused_thermo_kinetics_bitwise_c_route(monkeypatch, mechanism, layout):
    current = make_backend(mechanism)
    original = make_backend(mechanism)
    temperature, y = state(current.n_species, signed=True)
    y = with_layout(y, layout)
    before = y.copy()
    output_layout = "F" if layout == "F_readonly" else layout
    omega = with_layout(np.zeros_like(y), output_layout)
    hk = with_layout(np.zeros_like(y), output_layout)
    expected_omega, expected_hk = np.empty_like(y), np.empty_like(y)
    actual = current.eval_grid_thermo_kinetics_into(temperature, y, omega, hk)
    with previous_c_route(monkeypatch):
        expected = original.eval_grid_thermo_kinetics_into(
            temperature, y, expected_omega, expected_hk)
    for result, reference in zip(actual, expected):
        assert_bitwise(result, reference)
    assert_bitwise(omega, expected_omega)
    assert_bitwise(hk, expected_hk)
    assert_bitwise(y, before)


def test_single_node_and_grouped_jacobian_bitwise_c_route(monkeypatch):
    current = make_backend()
    original = make_backend()
    _, grid = state(current.n_species, signed=True)
    y = grid[:, 1]
    # A 1D input represents one column, including noncontiguous input views.
    for temperature, composition, method in (
        (np.array([1000.0]), y, "eval_grid_thermo_kinetics_into"),
        (np.full(current.n_species + 2, 1000.0),
         np.tile(y[:, None], (1, current.n_species + 2)),
         "eval_jacobian_thermo_kinetics_into"),
    ):
        if temperature.size > 1:
            temperature[1] += 0.01
            composition[0, 2] += 1e-5
        shape = (current.n_species, temperature.size)
        omega, hk = np.empty(shape, order="F"), np.empty(shape, order="C")
        ref_omega, ref_hk = np.empty(shape), np.empty(shape)
        actual = getattr(current, method)(temperature, composition, omega, hk)
        with previous_c_route(monkeypatch):
            expected = getattr(original, method)(temperature, composition, ref_omega, ref_hk)
        for result, reference in zip(actual, expected):
            assert_bitwise(result, reference)
        assert_bitwise(omega, ref_omega)
        assert_bitwise(hk, ref_hk)


@pytest.mark.parametrize("order", ["C", "F"])
def test_contiguous_inputs_reach_kernels_without_a_data_copy(monkeypatch, order):
    backend = make_backend()
    temperature, y = state(backend.n_species)
    y = np.array(y, order=order)
    observed = []

    def capture(argument_index):
        def kernel(*args):
            composition = args[argument_index]
            assert composition.flags.f_contiguous if order == "F" else composition.flags.c_contiguous
            assert np.shares_memory(composition, y)
            observed.append(composition)
            return None
        return kernel

    monkeypatch.setattr(transport_module, "_eval_faces_poly_numba_core", capture(1))
    monkeypatch.setattr(transport_module, "_mix_soret_fused", capture(2))
    monkeypatch.setattr(backend_module, "_eval_thermo_kinetics_sparse_numba_core", capture(1))
    backend.transport.eval_faces_poly_fast(temperature, backend.problem.P, y, backend.invW)
    backend.transport.thermal_diff_coeffs(temperature, backend.problem.P, y)
    backend.eval_grid_thermo_kinetics_into(temperature, y, np.empty_like(y), np.empty_like(y))
    assert len(observed) == 3
