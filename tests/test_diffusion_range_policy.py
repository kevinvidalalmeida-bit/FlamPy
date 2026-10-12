"""Changing range ownership must preserve every assembled value across A-B-A."""
import hashlib
import numpy as np
import pytest

from kflame.chemistry.initialization import _composition
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.flame import diffusion


@pytest.mark.parametrize('name', ['jetsurf2.yaml', 'aramcomech2.yaml', 'aramcomech3.yaml'])
@pytest.mark.parametrize('nodes,soret,fixed', [(9, True, False), (13, False, True)])
def test_small_large_ranges_preserve_coarse_ownership(monkeypatch, name, nodes, soret, fixed):
    try:
        path = resolve_mechanism(name)
    except FileNotFoundError:
        pytest.skip('External mechanism is not installed')
    mech = load_mechanism(path)
    feeds = []
    for text in ('CH4:1,N2:4', 'O2:1,N2:3.76'):
        y = _composition(text, mech.species_names)*mech.molecular_weights
        feeds.append(y/y.sum())
    case = diffusion.DiffusionCase(path, *feeds, transport_model='multicomponent',
        soret_enabled=soret, radiation_enabled=True, boundary_emissivities=(.2, .7))
    problem = diffusion.DiffusionFlameProblem(case, mech, np.linspace(0., .02, nodes))
    problem.multicomponent_jacobian = 'frozen'
    fields = problem.initial_fields()
    fields['T'][:] = np.linspace(600., 1900., nodes)
    y = np.arange(1., problem.ns+1.)
    y /= y.sum()
    fields['Y'][:] = y[:, None]
    first = problem.pack(fields)
    fields['T'][:] *= .999
    fields['Y'][:] = y[::-1, None]
    second = problem.pack(fields)
    real_split = diffusion._split
    monkeypatch.setattr(diffusion.numba, 'get_num_threads', lambda: 4)

    def coarse_split(function, count, *args, minimum=24, threads=None):
        if function.__name__ == 'tiled_range':
            minimum = 4
        return real_split(function, count, *args, minimum=minimum, threads=threads)

    def cycle(x):
        props = problem.properties(problem.unpack(x))
        temperature = problem.unpack(x)['T'].copy() if fixed else None
        residual = problem.residual(x, cached_properties=props, fixed_temperature=temperature)
        blocks = problem.jacobian_blocks(x, cached_properties=props, fixed_temperature=temperature)
        return (*props, residual, *(getattr(blocks, key) for key in
            ('lower', 'diag', 'upper', 'border', 'mass')), blocks.column())

    original_first = None
    for x in (first, second, first):
        monkeypatch.setattr(diffusion, '_split', coarse_split)
        expected = cycle(x)
        monkeypatch.setattr(diffusion, '_split', real_split)
        actual = cycle(x)
        for a, b in zip(expected, actual):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.isfinite(a).all() and a.tobytes() == b.tobytes()
        if x is first:
            hashes = [hashlib.sha256(a.tobytes()).digest() for a in actual]
            if original_first is None:
                original_first = hashes
            else:
                assert hashes == original_first
