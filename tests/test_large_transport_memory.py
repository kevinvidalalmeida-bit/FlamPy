"""Independent pre-HPC oracle for large transport and reused face scratch."""
import numpy as np
import pytest

import _multicomponent_reference as reference
from kflame.chemistry import multicomponent_kernels as kernels
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.multicomponent import NativeMulticomponentTransport
from kflame.chemistry.thermo import NativeThermo


def assert_bits(first, second):
    assert first.shape == second.shape
    assert first.dtype == second.dtype
    assert first.tobytes() == second.tobytes()


@pytest.fixture(scope='module', params=('gri30', 'jetsurf2', 'aramcomech2', 'aramcomech3'))
def transport(request):
    try:
        path = resolve_mechanism(request.param)
    except FileNotFoundError:
        pytest.skip('External mechanism is not installed')
    mechanism = load_mechanism(path)
    return NativeMulticomponentTransport(mechanism, str(path)), NativeThermo(mechanism)


@pytest.mark.parametrize('pressure', (10132.5, 101325., 1013250.))
@pytest.mark.parametrize('composition', ('sparse', 'dense'))
def test_large_schur_against_independent_original(transport, pressure, composition):
    multi, thermo = transport
    n = multi.n_species
    t = 1450.
    y = np.zeros(n)
    if composition == 'dense':
        y[:] = np.arange(1., n+1.)
    else:
        for name, amount in (('CH4', .08), ('O2', .2), ('N2', .69), ('H2O', .03)):
            y[multi.mech.species_names.index(name)] = amount
    y /= y.sum()
    cp = np.ascontiguousarray(thermo.cp_R(t))
    temp = (np.empty(n), np.empty(n), *(np.empty((n, n)) for _ in range(4)))
    multi._temperature_snapshot.evaluate(t, temp)
    mu, rot, binary, a, b, c = temp
    args = (t, pressure, y, cp, multi.mw, multi.inv_w, multi.crot, mu, rot, binary, a, b, c)
    old_multi, old_thermal = np.empty((n, n)), np.empty(n)
    old = reference.composition_terms_ws(*args, old_multi, old_thermal, reference.composition_workspace(n))
    new_multi, new_thermal = np.empty((n, n)), np.empty(n)
    new = kernels.composition_terms_ws(*args, new_multi, new_thermal, kernels.composition_workspace(n))
    assert_bits(np.asarray(new), np.asarray(old))
    assert_bits(new_multi, old_multi)
    assert_bits(new_thermal, old_thermal)


@pytest.mark.parametrize('faces', (0, 7, 33))
@pytest.mark.parametrize('layout', ('C', 'F', 'strided'))
def test_blocked_faces_preserve_outputs_and_layout(transport, faces, layout):
    multi, thermo = transport
    n = multi.n_species
    T = np.linspace(600., 2200., faces)
    y = np.arange(1., n+1.)
    y /= y.sum()
    Y = np.repeat(y[:, None], faces, axis=1)
    cp = np.asarray(thermo.cp_R(T))
    if layout == 'F':
        Y, cp = np.asfortranarray(Y), np.asfortranarray(cp)
    elif layout == 'strided':
        ybuffer, cpbuffer = np.zeros((n, 2*faces)), np.zeros((n, 2*faces))
        ybuffer[:, ::2], cpbuffer[:, ::2] = Y, cp
        Y, cp = ybuffer[:, ::2], cpbuffer[:, ::2]
    mw, logeps, eps, zrot, frot, visc, diff, a, b, c = multi.export_temperature_snapshot()[0]
    values = (T, 101325., Y, cp, mw, multi.inv_w, logeps, eps, multi.crot,
              zrot, frot, visc, diff, a, b, c, False)
    expected = kernels._evaluate_faces_symmetric(*values)
    actual = kernels._evaluate_faces_blocked(*values, True, 4)
    for first, second in zip(actual, expected):
        assert_bits(first, second)
        assert first.strides == second.strides
    # Scratch ownership extends across calls: a changed input must be evaluated.
    if faces:
        Y = Y.copy()
        Y[0, -1] += 1e-8
        changed = (*values[:2], Y, *values[3:])
        expected = kernels._evaluate_faces_symmetric(*changed)
        actual = kernels._evaluate_faces_blocked(*changed, True, 4)
        for first, second in zip(actual, expected):
            assert_bits(first, second)


@pytest.mark.parametrize('dense', (False, True))
@pytest.mark.parametrize('symmetric', (False, True))
def test_blocked_dense_and_ordered_fallbacks(dense, symmetric):
    path = resolve_mechanism('gri30')
    mech = load_mechanism(path)
    multi = NativeMulticomponentTransport(mech, str(path))
    T = np.linspace(650., 2200., 7)
    y = np.arange(1., mech.n_species+1.)
    y /= y.sum()
    Y = np.repeat(y[:, None], T.size, axis=1)
    cp = NativeThermo(mech).cp_R(T)
    mw, logeps, eps, zrot, frot, visc, diff, a, b, c = multi.export_temperature_snapshot()[0]
    values = (T, 101325., Y, cp, mw, multi.inv_w, logeps, eps, multi.crot,
              zrot, frot, visc, diff, a, b, c, dense)
    original = kernels._evaluate_faces_symmetric if symmetric else kernels.evaluate_faces
    for actual, expected in zip(kernels._evaluate_faces_blocked(*values, symmetric, 4), original(*values)):
        assert_bits(actual, expected)


def test_large_fortran_workspace_keeps_original_route():
    try:
        path = resolve_mechanism('aramcomech2')
        mech = load_mechanism(path)
    except FileNotFoundError:
        pytest.skip('External mechanism is not installed')
    multi = NativeMulticomponentTransport(mech, str(path))
    n, t = mech.n_species, 1100.
    y = np.arange(1., n+1.); y /= y.sum()
    cp = np.ascontiguousarray(NativeThermo(mech).cp_R(t))
    temp = (np.empty(n), np.empty(n), *(np.empty((n, n)) for _ in range(4)))
    multi._temperature_snapshot.evaluate(t, temp)
    mu, rot, binary, a, b, c = temp
    args = (t, 101325., y, cp, multi.mw, multi.inv_w, multi.crot, mu, rot, binary, a, b, c)
    workspaces = [tuple(np.asfortranarray(array) for array in fn(n))
                  for fn in (reference.composition_workspace, kernels.composition_workspace)]
    old_multi, old_thermal = np.empty((n, n)), np.empty(n)
    old = reference.composition_terms_ws(*args, old_multi, old_thermal, workspaces[0])
    new_multi, new_thermal = np.empty((n, n)), np.empty(n)
    new = kernels.composition_terms_ws(*args, new_multi, new_thermal, workspaces[1])
    assert_bits(np.asarray(new), np.asarray(old))
    assert_bits(new_multi, old_multi)
    assert_bits(new_thermal, old_thermal)
