"""Exact face rotation checks against the independent baseline-current package.

The local frozen snapshot is intentionally never imported as live ``kflame``.
Only import names and Numba's disk-cache flag are rewritten by the loader;
the reference arithmetic and all its dependencies come from the snapshot.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kflame.flame import diffusion_kernels as candidate


_ROOT = Path(__file__).resolve().parents[1]
_BASELINE = _ROOT / '.local/hpc-pending-2026-10-09/baseline-current/src/kflame'
_NAMESPACE = '_face_rotation_baseline'
_CUTS = ((0, 7), (0, 1, 3, 6, 7), (0, 2, 4, 5, 7), tuple(range(8)))
_CASES = (
    ('h2o2.yaml', 'mixture-averaged', False, 'exact'),
    ('h2o2.yaml', 'mixture-averaged', True, 'exact'),
    ('h2o2.yaml', 'multicomponent', False, 'exact'),
    ('h2o2.yaml', 'multicomponent', True, 'exact'),
    ('h2o2.yaml', 'multicomponent', False, 'frozen'),
    ('h2o2.yaml', 'multicomponent', True, 'frozen'),
    ('gri30.yaml', 'multicomponent', True, 'exact'),
    ('gri30.yaml', 'multicomponent', True, 'frozen'),
)


class _FrozenLoader(importlib.abc.SourceLoader):
    def __init__(self, name, path, manifest):
        self.name, self.path, self.manifest = name, path, manifest

    def get_filename(self, fullname):
        return str(self.path)

    def get_data(self, path):
        return Path(path).read_bytes()

    def get_code(self, fullname):
        original = self.path.read_bytes()
        key = 'src\\kflame\\' + str(self.path.relative_to(_BASELINE)).replace('/', '\\')
        assert hashlib.sha256(original).hexdigest() == self.manifest[key], key
        source = original.decode('utf-8').replace('from kflame', f'from {_NAMESPACE}')
        source = source.replace('import kflame', f'import {_NAMESPACE}')
        # No cached production dependencies can enter the frozen namespace.
        source = source.replace('cache=True', 'cache=False')
        return compile(source, str(self.path), 'exec')


class _FrozenFinder(importlib.abc.MetaPathFinder):
    def __init__(self, manifest):
        self.manifest = manifest

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _NAMESPACE and not fullname.startswith(_NAMESPACE + '.'):
            return None
        relative = fullname.split('.')[1:]
        location = _BASELINE.joinpath(*relative)
        is_package = location.is_dir()
        location = location / '__init__.py' if is_package else location.with_suffix('.py')
        if not location.is_file():
            return None
        loader = _FrozenLoader(fullname, location, self.manifest)
        return importlib.util.spec_from_file_location(
            fullname, location, loader=loader,
            submodule_search_locations=[str(location.parent)] if is_package else None,
        )


@pytest.fixture(scope='module')
def frozen_modules():
    if not _BASELINE.is_dir():
        pytest.skip('independent local baseline-current snapshot is unavailable')
    manifest = json.loads((_BASELINE.parents[2] / 'baseline-manifest.json').read_text())
    finder = _FrozenFinder(manifest)
    sys.meta_path.insert(0, finder)
    try:
        modules = SimpleNamespace(**{
            name: importlib.import_module(f'{_NAMESPACE}.{suffix}')
            for name, suffix in (
                ('diffusion', 'flame.diffusion'), ('kernels', 'flame.diffusion_kernels'),
                ('mechanism', 'chemistry.mechanism'), ('initialization', 'chemistry.initialization'),
            )
        })
        assert modules.kernels is not candidate
        assert modules.kernels.composition_terms_ws.__module__.startswith(_NAMESPACE)
        assert modules.kernels.grouped_node.__module__.startswith(_NAMESPACE)
        yield modules
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == _NAMESPACE or name.startswith(_NAMESPACE + '.'):
                del sys.modules[name]


@pytest.fixture(scope='module', params=_CASES, ids=lambda c: '-'.join(map(str, c)))
def reference_problem(request, frozen_modules):
    mechanism, transport, soret, mode = request.param
    path = frozen_modules.mechanism.resolve_mechanism(mechanism)
    mech = frozen_modules.mechanism.load_mechanism(path)
    feeds = []
    for text in ('CH4' if mechanism == 'gri30.yaml' else 'H2', 'O2:1,N2:3.76'):
        y = frozen_modules.initialization._composition(text, mech.species_names) * mech.molecular_weights
        feeds.append(y / y.sum())
    case = frozen_modules.diffusion.DiffusionCase(
        path, *feeds, fuel_mass_flux=.05, oxidizer_mass_flux=.15,
        transport_model=transport, soret_enabled=soret,
        radiation_enabled=True, radiation_scale=2., boundary_emissivities=(.4, .8),
    )
    p = frozen_modules.diffusion.DiffusionFlameProblem(
        case, mech, np.array([0., .001, .004, .009, .013, .018, .02]),
    )
    p.chemistry_tile_nodes = 0
    p.multicomponent_jacobian = mode
    fields = p.initial_fields()
    fields['T'] = np.array([310., 620., 999.999, 1000., 1000.001, 1450., 450.])
    fields['Y'] = fields['Y'].copy()
    trace = mech.species_names.index('OH')
    fields['Y'][trace, 1:-1] = [-1e-12, 0., -2e-12, 3e-13, -1e-13]
    fields['m'][3] = 1e-8 * p.mref  # Exercise the enthalpy blend as well as both flow signs.
    return p, p.pack(fields)


def _arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)


def _bytes_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    assert np.isfinite(actual).all()
    assert actual.tobytes(order='C') == expected.tobytes(order='C')


def _layout(array, layout):
    if layout == 'strided':
        storage = np.full((*array.shape[:-1], 2 * array.shape[-1]), 1234567.125)
        storage[..., ::2] = array
        return storage[..., ::2]
    result = np.array(array, order='F' if layout.startswith('F') else 'C', copy=True)
    if layout.endswith('readonly'):
        result.setflags(write=False)
    return result


def _reference_inputs(monkeypatch, modules, problem, x, analytic, fixed):
    p = problem
    p.analytic_chemistry = analytic
    captures = []
    dispatcher = modules.kernels.jacobian_range

    def capture(start, stop, *arguments):
        captures.append(arguments)
        return dispatcher(start, stop, *arguments)

    def serial_split(function, count, *arguments, minimum=24):
        function(0, count, *arguments)

    with monkeypatch.context() as patch:
        patch.setattr(modules.kernels, 'jacobian_range', capture)
        patch.setattr(modules.diffusion, '_split', serial_split)
        p.jacobian_blocks(x, fixed_temperature=p.unpack(x)['T'].copy() if fixed else None)
    assert len(captures) == 1
    # Everything before the four output blocks is independently baseline-built.
    return captures[0][:-4], tuple(a.copy() for a in captures[0][-4:])


def _run(inputs, expected, cuts, reuse=True, *, threaded=False):
    outputs = tuple(np.zeros_like(a) for a in expected)
    ranges = tuple(zip(cuts, cuts[1:]))
    if threaded:
        # Compile the exact signature before workers use shared read-only input.
        candidate.jacobian_range(0, 0, *inputs, *outputs, reuse_faces=reuse)
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [pool.submit(candidate.jacobian_range, a, b, *inputs, *outputs,
                                 reuse_faces=reuse) for a, b in ranges]
            for task in tasks:
                task.result()
    else:
        for start, stop in ranges:
            candidate.jacobian_range(start, stop, *inputs, *outputs, reuse_faces=reuse)
    for actual, reference in zip(outputs, expected):
        _bytes_equal(actual, reference)
        assert all(not np.shares_memory(actual, a) for a in _arrays(inputs))
    return outputs


@pytest.mark.parametrize('analytic', [False, True], ids=['fd-chemistry', 'analytic-chemistry'])
@pytest.mark.parametrize('fixed', [False, True], ids=['energy', 'fixed-temperature'])
def test_range_rotation_matches_independent_baseline(
        monkeypatch, frozen_modules, reference_problem, analytic, fixed):
    p, x = reference_problem
    inputs, expected = _reference_inputs(monkeypatch, frozen_modules, p, x, analytic, fixed)
    snapshots = [(a, a.tobytes(order='C')) for a in _arrays(inputs)]
    for cuts, layout in zip(_CUTS, ('C', 'F', 'strided', 'F_readonly')):
        changed = list(inputs)
        for index in (3, 4, 7, 8, 11):  # T, Y, omega, hk and base J.
            changed[index] = _layout(changed[index], layout)
        for reuse in (False, True):
            before = [(a, a.tobytes(order='C')) for a in _arrays(changed)]
            outputs = _run(tuple(changed), expected, cuts, reuse)
            saved = [a.copy() for a in outputs]
            # Empty ranges at either end cannot rotate or write outputs.
            candidate.jacobian_range(0, 0, *changed, *outputs, reuse_faces=reuse)
            candidate.jacobian_range(p.n, p.n, *changed, *outputs, reuse_faces=reuse)
            for actual, previous in zip(outputs, saved):
                _bytes_equal(actual, previous)
            for array, original in before:
                assert array.tobytes(order='C') == original
    for array, original in snapshots:
        assert array.tobytes(order='C') == original


def test_threaded_ranges_and_repeated_perturbed_calls_are_private(
        monkeypatch, frozen_modules, reference_problem):
    p, x = reference_problem
    inputs, expected = _reference_inputs(monkeypatch, frozen_modules, p, x, True, False)
    readonly = list(inputs)
    for index in (0, 1, 2, 3, 4, 7, 8, 11):
        readonly[index] = _layout(readonly[index], 'F_readonly')
    outputs = _run(tuple(readonly), expected, _CUTS[1], threaded=True)
    saved = tuple(a.copy() for a in outputs)
    shifted = x.copy()
    shifted[2 * p.nv + 2] *= 1.0003
    shifted[4 * p.nv + 3] += 2e-8
    other_inputs, other_expected = _reference_inputs(monkeypatch, frozen_modules, p, shifted, True, False)
    other = _run(other_inputs, other_expected, _CUTS[2], threaded=True)
    for array, previous in zip(outputs, saved):
        _bytes_equal(array, previous)
        assert all(not np.shares_memory(array, a) for a in other)
    # Independent full builds can also run simultaneously without global scratch.
    candidate.jacobian_range(0, 0, *readonly, *(np.zeros_like(a) for a in expected))
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(_run, tuple(readonly), expected, _CUTS[0]) for _ in range(2)]
        first, second = [task.result() for task in tasks]
    assert all(not np.shares_memory(a, b) for a in first for b in second)


def test_compact_face_ranges_match_baseline_with_soret_and_layouts(frozen_modules, reference_problem):
    p, x = reference_problem
    fields = p.unpack(x)
    data = p._compiled_data()
    inputs = (p.dz, float(p.P), data['invW'], data['mw'], data['mixture'], data['multi'],
              data['multicomponent'], bool(p.soret_enabled))
    shape = (p.ns, p.n - 1)
    expected = (np.empty(p.n - 1), np.empty(p.n - 1), np.empty(shape))
    frozen_modules.kernels.face_range(0, p.n - 1, fields['T'], fields['Y'], *inputs, *expected)
    for layout in ('C', 'F_readonly', 'strided'):
        T, Y = _layout(fields['T'], layout), _layout(fields['Y'], layout)
        before = (T.tobytes(), Y.tobytes())
        outputs = (np.empty(p.n - 1), np.empty(p.n - 1),
                   _layout(np.zeros(shape), layout.replace('_readonly', '')))
        for start, stop in ((0, 1), (1, 3), (3, p.n - 1)):
            candidate.face_range(start, stop, T, Y, *inputs, *outputs)
        for actual, reference in zip(outputs, expected):
            _bytes_equal(actual, reference)
        assert (T.tobytes(), Y.tobytes()) == before


def test_standalone_face_reuses_owned_scratch_and_dead_aliases(frozen_modules, reference_problem):
    p, x = reference_problem
    fields = p.unpack(x)
    data = p._compiled_data()
    multicomponent = data['multicomponent']
    ns = p.ns

    def scratch(module, compact):
        td = module._tdata(ns, multicomponent) if compact else module._tdata(ns)
        md = module._mdata(ns if multicomponent else 1)
        cws = module.composition_workspace(ns if multicomponent else 1)
        # Distinct live slices may share a backing allocation. D and DT have
        # no consumers in multicomponent mode and may also alias live scratch.
        backing = np.full(6 * ns, np.nan)
        X, D, DT, Yf, thermal, flux = backing.reshape(6, ns)
        if multicomponent:
            D, DT = X, flux
        return td, md, X, D, DT, Yf, np.empty((ns, ns)), thermal, flux, cws

    actual_scratch = scratch(candidate, True)
    reference_scratch = scratch(frozen_modules.kernels, False)
    compositions = (
        (fields['Y'][:, 2], fields['Y'][:, 3]),
        (p.case.fuel_Y, p.case.oxidizer_Y),
        (fields['Y'][:, 3], fields['Y'][:, 4]),
    )
    saved = []
    for number, (left, right) in enumerate(compositions):
        common = (999.999, 1000.001, left, right, float(p.dz[2]), float(p.P),
                  data['invW'], data['mw'], data['mixture'], data['multi'],
                  multicomponent, bool(p.soret_enabled))
        a = candidate.face_state(*common, *actual_scratch[:2], number > 0, *actual_scratch[2:])
        b = frozen_modules.kernels.face_state(
            *common, *reference_scratch[:2], number > 0, *reference_scratch[2:])
        _bytes_equal(np.array(a), np.array(b))
        for index in (2, 5, 8):  # final X, Yf and flux, including X's dual scratch role
            _bytes_equal(actual_scratch[index], reference_scratch[index])
        saved.append(actual_scratch[8].copy())
    assert all(not np.shares_memory(a, actual_scratch[8]) for a in saved)


@pytest.mark.parametrize('multicomponent', [False, True])
@pytest.mark.parametrize('ns', [10, 53])
def test_complete_slots_rotate_by_reference_without_scratch_aliases(ns, multicomponent):
    ws = candidate._node_workspace(ns, multicomponent)
    other = candidate._node_workspace(ns, multicomponent)
    arrays = list(_arrays(ws))
    for i, array in enumerate(arrays):
        array.fill(i + .125)
        assert all(not np.shares_memory(array, a) for a in _arrays(other))
        assert all(not np.shares_memory(array, a) for a in arrays[:i])
    before = [a.tobytes() for a in arrays]
    rotated = candidate._rotate_face_workspace(ws)
    mapping = {14: 15, 15: 14, 17: 18, 18: 17}
    for index, value in enumerate(rotated):
        original = ws[mapping.get(index, index)]
        if index in (35, 36, 37):
            original = (ws[index][1], ws[index][0])
        for actual, reference in zip(_arrays(value), _arrays(original)):
            assert actual.shape == reference.shape
            assert actual.ctypes.data == reference.ctypes.data
    assert [a.tobytes() for a in arrays] == before
    twice = candidate._rotate_face_workspace(rotated)
    assert [a.ctypes.data for a in _arrays(twice)] == [a.ctypes.data for a in arrays]
    if multicomponent:
        for index in (14, 15, 16):
            assert [a.size for a in ws[index]] == [0, 0, 0, ns, ns * ns, 0]
        for index in (21, 22, 27, 28, 29, 30, 31, 32, 38):
            assert ws[index].size == 0
    else:
        assert [a.size for a in ws[14]] == [ns, ns * ns, ns, ns, ns * ns, ns * ns]
    # Temperature perturbations own the third slot, unchanged by either rotation.
    assert ws[16][3].ctypes.data == rotated[16][3].ctypes.data
    assert ws[19][0].ctypes.data == rotated[19][0].ctypes.data
