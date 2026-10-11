"""Bitwise analytic tiles, global validation, and external-pool execution."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from kflame.chemistry import analytic
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion
from _analytic_partials_reference import species_partials as original_partials


@pytest.fixture(scope="module", autouse=True)
def bounded_numba_threads():
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    yield
    numba.set_num_threads(previous)


@pytest.fixture(scope="module", params=["h2o2.yaml", "gri30.yaml"])
def backend(request):
    path = resolve_mechanism(request.param)
    problem = SimpleNamespace(P=101325.0, case=SimpleNamespace(mech=path))
    return NativeSpeciesBackend(problem, load_mechanism(path))


def state(backend, *, signed=False, layout="C", nodes=35):
    mids = np.unique(backend._nb_nasa_tmid)
    boundaries = np.column_stack((np.nextafter(mids, -np.inf), mids,
                                  np.nextafter(mids, np.inf))).ravel()
    temperature = np.resize(np.r_[300.0, boundaries, 2300.0], nodes)
    y = np.random.default_rng(23).uniform(0.01, 1.0, (backend.n_species, nodes))
    y /= y.sum(axis=0)
    y[:, ::3] *= 1.01  # Independent, unnormalized Newton mass fractions.
    if nodes:
        y[2, 0] = -0.0
        y[3, -1] = 0.0
    if signed and nodes > 1:
        y[0, 1::3] *= -0.001
        y[1, 2::3] *= -0.001
        y[0, 2::3] *= -0.001  # More than one negative reactant can occur.
    if layout == "strided":
        storage = np.empty((backend.n_species, 2 * nodes))
        storage[:, ::2] = y
        y = storage[:, ::2]
        t_storage = np.empty(2 * nodes)
        t_storage[::2] = temperature
        temperature = t_storage[::2]
    else:
        y = np.array(y, order=layout[0], copy=True)
    if layout.endswith("readonly"):
        y.setflags(write=False)
        temperature.setflags(write=False)
    return temperature, y


def assert_bitbytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize("layout", ["C", "F_readonly", "strided"])
@pytest.mark.parametrize("signed", [False, True])
def test_derivatives_match_independent_pre_tiling_parallel_kernel(backend, layout, signed):
    temperature, y = state(backend, signed=signed, layout=layout)
    expected = original_partials(backend, temperature, y, _temperature=True)
    actual = analytic.species_partials(backend, temperature, y, _temperature=True)
    # The sparse accumulation (monomial scatter, rank-one default third-body
    # efficiency) reorders sums of the dense reference: equal to rounding.
    for value, reference in zip(actual, expected):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        np.testing.assert_allclose(value, reference, rtol=0, atol=1e-13*np.abs(reference).max())
    tiled = [analytic.species_partials_range(backend, temperature, y, i,
                min(i + 7, temperature.size)) for i in range(0, temperature.size, 7)]
    for component in (0, 1):
        joined = np.concatenate([tile[component] for tile in tiled], axis=1)
        assert_bitbytes(joined, actual[component])


@pytest.mark.parametrize("layout", ["C", "F", "C_readonly", "F_readonly", "strided"])
@pytest.mark.parametrize("signed", [False, True])
def test_tiles_match_full_api_bitbytes(backend, layout, signed):
    temperature, y = state(backend, signed=signed, layout=layout)
    before_t, before_y = temperature.copy(), y.copy()
    prepared_t, prepared_y = analytic.prepare_species_partials(backend, temperature, y)
    for thermal in (False, True):
        full = analytic.species_partials(backend, temperature, y, _temperature=thermal)
        expected_y, expected_t = full if thermal else (full, np.zeros(y.shape))
        for width in (1, 2, 7, 32, 64):
            tiles = [analytic.species_partials_range(
                backend, prepared_t, prepared_y, start, min(start + width, temperature.size),
                _temperature=thermal, _validated=True)
                for start in range(0, temperature.size, width)]
            assert_bitbytes(np.concatenate([tile[0] for tile in tiles], axis=1), expected_y)
            assert_bitbytes(np.concatenate([tile[1] for tile in tiles], axis=1), expected_t)
            for jy, jt in tiles:
                assert all(jy[:, node, :].flags.c_contiguous for node in range(jy.shape[1]))
                assert jt.flags.c_contiguous
            outputs = [output for tile in tiles for output in tile]
            for index, output in enumerate(outputs):
                for input_array in (temperature, y, prepared_t, prepared_y):
                    assert not np.shares_memory(output, input_array)
                for other in outputs[index + 1:]:
                    assert not np.shares_memory(output, other)
        # Default range calls validate the global state and return both outputs.
        jy, jt = analytic.species_partials_range(backend, temperature, y, 1, 4,
                                                 _temperature=thermal)
        assert_bitbytes(jy, expected_y[:, 1:4, :])
        assert_bitbytes(jt, expected_t[:, 1:4])
    assert_bitbytes(temperature, before_t)
    assert_bitbytes(y, before_y)


@pytest.mark.parametrize("nodes", [0, 1])
def test_empty_and_single_node_ranges(backend, nodes):
    temperature, y = state(backend, nodes=nodes, layout="C_readonly")
    temperature, y = analytic.prepare_species_partials(backend, temperature, y)
    for position in range(nodes + 1):
        jy, jt = analytic.species_partials_range(backend, temperature, y, position, position,
                                                 _validated=True)
        assert jy.shape == (backend.n_species, 0, backend.n_species)
        assert jt.shape == (backend.n_species, 0)
    if nodes:
        expected = analytic.species_partials(backend, temperature, y, _temperature=True)
        actual = analytic.species_partials_range(backend, temperature, y, 0, 1, _validated=True)
        for output, reference in zip(actual, expected):
            assert_bitbytes(output, reference)
    assert analytic.species_partials_range_core.targetoptions["nogil"]
    assert not analytic.species_partials_range_core.targetoptions.get("parallel", False)
    assert not analytic._species_partials_node.targetoptions.get("parallel", False)


def test_external_pool_tiles_are_independent(backend):
    temperature, y = state(backend, signed=True, layout="C_readonly")
    temperature, y = analytic.prepare_species_partials(backend, temperature, y)
    expected = analytic.species_partials(backend, temperature, y, _temperature=True)
    # Load the serial specialization before entering the pool, as compile_only does.
    analytic.species_partials_range(backend, temperature, y, 0, 0, _validated=True)
    bounds = [(0, 7), (7, 32), (32, 35)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(analytic.species_partials_range, backend, temperature, y,
                               start, stop, _validated=True) for start, stop in bounds]
        tiles = [future.result() for future in futures]
    for component in (0, 1):
        assert_bitbytes(np.concatenate([tile[component] for tile in tiles], axis=1),
                        expected[component])
    held = tiles[0][0].copy()
    tiles[1][0].fill(0.0)
    analytic.species_partials_range(backend, temperature, y, 0, 7, _validated=True)
    assert_bitbytes(tiles[0][0], held)


def test_global_density_validation_once_and_global_kernel_inputs(backend, monkeypatch):
    temperature, y = state(backend, layout="F")
    invw = backend.invW
    observed = []

    class MatmulProbe:
        def __matmul__(self, global_y):
            observed.append(global_y)
            return invw @ global_y

    with monkeypatch.context() as patch:
        patch.setattr(backend, "invW", MatmulProbe())
        prepared_t, prepared_y = analytic.prepare_species_partials(backend, temperature, y)
    assert len(observed) == 1
    assert observed[0] is prepared_y and prepared_y.shape == y.shape

    def no_revalidation(*args):
        pytest.fail("Prepared tiles must not repeat global validation")

    def capture_core(global_t, global_y, *args):
        assert global_t is prepared_t and global_y is prepared_y
        start, stop = args[-2:]
        return (np.empty((backend.n_species, stop - start, backend.n_species)),
                np.empty((backend.n_species, stop - start)))

    monkeypatch.setattr(analytic, "prepare_species_partials", no_revalidation)
    monkeypatch.setattr(analytic, "species_partials_range_core", capture_core)
    for start, stop in ((0, 0), (0, 32), (32, 35)):
        analytic.species_partials_range(backend, prepared_t, prepared_y, start, stop,
                                        _validated=True)


@pytest.mark.parametrize("bad", ["shape", "nan_y", "density", "zero_t", "nan_t", "orders"])
def test_range_preserves_global_validation(backend, monkeypatch, bad):
    temperature, y = state(backend)
    if bad == "shape":
        y = y[:, :-1]
    elif bad == "nan_y":
        y[0, -1] = np.nan
    elif bad == "density":
        y[:, -1] = 0.0
    elif bad == "zero_t":
        temperature[-1] = 0.0
    elif bad == "nan_t":
        temperature[-1] = np.nan
    elif bad == "orders":
        plan = backend.kinetics._sp_r_plan.copy()
        plan[0, 0] = -1
        monkeypatch.setattr(backend.kinetics, "_sp_r_plan", plan)
    # Invalid nodes outside [0,1) must still reject, just like the full API.
    with pytest.raises(ValueError) as original:
        analytic.species_partials(backend, temperature, y)
    with pytest.raises(ValueError) as tiled:
        analytic.species_partials_range(backend, temperature, y, 0, 1)
    assert str(tiled.value) == str(original.value)


@pytest.mark.parametrize("start,stop", [(-1, 1), (2, 1), (0, 36), (36, 36)])
def test_invalid_range_bounds(backend, start, stop):
    temperature, y = state(backend)
    with pytest.raises(ValueError, match="0 <= start <= stop <= N"):
        analytic.species_partials_range(backend, temperature, y, start, stop)


def test_noninteger_bounds_and_unprepared_inputs(backend):
    temperature, y = state(backend)
    with pytest.raises(TypeError):
        analytic.species_partials_range(backend, temperature, y, 0.5, 2)
    with pytest.raises(ValueError, match="globally prepared"):
        analytic.species_partials_range(backend, temperature, np.asfortranarray(y),
                                        0, 2, _validated=True)


@pytest.fixture(scope="module", params=["mixture-averaged", "multicomponent"])
def gri_problem(request):
    path = resolve_mechanism("gri30.yaml")
    mech = load_mechanism(path)
    feeds = []
    for text in ("CH4", "O2:1,N2:3.76"):
        y = _composition(text, mech.species_names) * mech.molecular_weights
        feeds.append(y / y.sum())
    case = diffusion.DiffusionCase(
        path, *feeds, fuel_mass_flux=0.05, oxidizer_mass_flux=0.15,
        transport_model=request.param, soret_enabled=True,
        radiation_enabled=True, radiation_scale=2.0)
    problem = diffusion.DiffusionFlameProblem(case, mech, np.linspace(0.0, 0.02, 35))
    problem.multicomponent_jacobian = "frozen"
    assert problem.ns > 20
    return problem


def assert_blocks_bitbytes(actual, expected):
    for name in ("lower", "diag", "upper", "border", "mass"):
        assert_bitbytes(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("fixed", [False, True])
def test_gri_jacobian_tiles_match_full_blocks(gri_problem, monkeypatch, threads, fixed):
    problem = gri_problem
    fields = problem.initial_fields()
    fields["T"], fields["Y"] = state(problem.backend, signed=True)
    x = problem.pack(fields)
    temperature = fields["T"].copy() if fixed else None
    previous_threads = numba.get_num_threads()
    # Respect a lower startup ceiling without making this portable test skip.
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    try:
        monkeypatch.setattr(problem, "analytic_chemistry", True)
        monkeypatch.setattr(problem, "chemistry_tile_nodes", 0)
        expected = problem.jacobian_blocks(x, fixed_temperature=temperature)
        calls = []
        original_range = diffusion.species_partials_range

        def record_range(backend, global_t, global_y, start, stop, **kwargs):
            result = original_range(backend, global_t, global_y, start, stop, **kwargs)
            assert global_t.shape == (35,) and global_y.shape == (problem.ns, 35)
            assert kwargs["_validated"] is True
            assert result[0].shape == (problem.ns, stop - start, problem.ns)
            assert result[1].shape == (problem.ns, stop - start)
            calls.append((start, stop))
            return result

        monkeypatch.setattr(diffusion, "species_partials_range", record_range)
        for width in (1, 7, 32):
            monkeypatch.setattr(problem, "chemistry_tile_nodes", width)
            calls.clear()
            actual = problem.jacobian_blocks(x, fixed_temperature=temperature)
            assert_blocks_bitbytes(actual, expected)
            covered = np.zeros(problem.n, dtype=int)
            for start, stop in calls:
                assert 0 < stop - start <= width
                covered[start:stop] += 1
            np.testing.assert_array_equal(covered, np.ones(problem.n, dtype=int))

        # A smaller storage budget must split even a requested wide tile,
        # without dropping nodes or changing the assembled matrix.
        monkeypatch.setattr(diffusion, "_CHEMISTRY_TILE_BUDGET_BYTES", 2 * 8 * problem.ns * (problem.ns + 1))
        monkeypatch.setattr(problem, "chemistry_tile_nodes", 32)
        calls.clear()
        limited = problem.jacobian_blocks(x, fixed_temperature=temperature)
        assert_blocks_bitbytes(limited, expected)
        assert all(0 < stop - start <= 2 for start, stop in calls)
        covered = np.zeros(problem.n, dtype=int)
        for start, stop in calls:
            covered[start:stop] += 1
        np.testing.assert_array_equal(covered, np.ones(problem.n, dtype=int))

        # The integrated compile_only route must load both empty-range kernels.
        calls.clear()
        empty = problem.jacobian_blocks(x, fixed_temperature=temperature, compile_only=True)
        assert calls == [(0, 0)]
        for name in ("lower", "diag", "upper", "border"):
            value = getattr(empty, name)
            assert_bitbytes(value, np.zeros_like(value))
        assert_bitbytes(empty.mass, expected.mass)
    finally:
        numba.set_num_threads(previous_threads)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("bad", ["nan_y", "density", "zero_t"])
def test_gri_invalid_global_validation_falls_back(gri_problem, monkeypatch, threads, bad):
    problem = gri_problem
    x = problem.pack(problem.initial_fields())
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    try:
        monkeypatch.setattr(problem, "chemistry_tile_nodes", 32)
        monkeypatch.setattr(problem, "analytic_chemistry", False)
        expected = problem.jacobian_blocks(x)
        monkeypatch.setattr(problem, "analytic_chemistry", True)
        original_prepare = diffusion.prepare_species_partials
        validations = []

        def reject_global(backend, global_t, global_y):
            assert global_t.shape == (35,) and global_y.shape == (problem.ns, 35)
            validations.append(global_y.shape)
            # Fault injection isolates fallback routing from invalid-state
            # transport arithmetic: the real validator rejects a bad final node
            # outside the first 32-node tile, while the assembled state stays valid.
            invalid_t, invalid_y = global_t.copy(), global_y.copy()
            if bad == "nan_y":
                invalid_y[0, -1] = np.nan
            elif bad == "density":
                invalid_y[:, -1] = 0.0
            else:
                invalid_t[-1] = 0.0
            return original_prepare(backend, invalid_t, invalid_y)

        def no_derivative_worker(*args, **kwargs):
            pytest.fail("Global validation failure must precede all derivative workers")

        monkeypatch.setattr(diffusion, "prepare_species_partials", reject_global)
        monkeypatch.setattr(diffusion, "species_partials_range", no_derivative_worker)
        monkeypatch.setattr(diffusion, "species_partials", no_derivative_worker)
        actual = problem.jacobian_blocks(x)
        assert validations == [(problem.ns, problem.n)]
        assert_blocks_bitbytes(actual, expected)
    finally:
        numba.set_num_threads(previous_threads)
