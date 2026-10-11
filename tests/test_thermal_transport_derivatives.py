"""Thermal transport derivatives against independent native primal FD."""
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
RELATIVE_H=(1e-3,1e-4,1e-5,1e-6,1e-7)

def _bytes(arrays):
    return tuple(((a.shape, a.dtype.str, a.tobytes()) for a in arrays))

def _readonly(arrays):
    for array in arrays:
        array.flags.writeable = False
    return arrays

def _outputs(n):
    return (np.full(n, np.nan), np.full(n, np.nan), *(np.full((n, n), np.nan) for _ in range(4)))

def _gate(analytic, estimates, primal, coordinate_scale, *, rtol=3e-05, scale_atol=1e-08, label='tangent'):
    analytic = np.asarray(analytic)
    assert np.isfinite(analytic).all(), label
    primal = np.asarray(primal)
    scale = max(float(np.max(np.abs(analytic))), float(np.max(np.abs(primal))) / coordinate_scale, np.finfo(float).tiny)
    errors = []
    for h, fd in estimates:
        fd = np.asarray(fd)
        assert fd.shape == analytic.shape and np.isfinite(fd).all(), (label, h)
        denominator = rtol * np.maximum(np.abs(analytic), np.abs(fd)) + scale_atol * scale
        errors.append((h, float(np.max(np.abs(analytic - fd) / denominator))))
    passing = [h for h, error in errors if error <= 1.0]
    assert len(set(passing)) >= 2, (label, 'need >=2 passing h', errors)

def _primal_terms(T, inputs):
    from kflame.chemistry.multicomponent_kernels import temperature_terms
    out = _outputs(inputs[0].size)
    temperature_terms(T, *inputs, *out)
    return out

def _primitive_tangents(T, inputs, primal):
    from kflame.chemistry.transport_temperature_derivatives import temperature_term_tangents
    derivative = _outputs(inputs[0].size)
    temperature_term_tangents(T, *inputs, *primal, *derivative)
    return derivative

def _synthetic_inputs(layout='C'):
    """Asymmetric positive ordered-pair fits; no symmetry certificate assumed."""
    n = 4
    mw = np.array([2.0, 16.0, 28.0, 32.0])
    log_eps = np.log(np.full((n, n), 200.0))
    epsilon = np.array([40.0, 80.0, 150.0, 200.0])
    zrot = np.array([0.2, 1.0, 2.0, 5.0])
    frot = np.array([2.0, 3.0, 4.0, 5.0])
    visc = np.zeros((n, 5))
    visc[:, 0] = np.linspace(0.0005, 0.001, n)
    visc[:, 1] = 1e-05
    diff = np.zeros((n, n, 5))
    diff[:, :, 0] = 8e-06
    diff[:, :, 1] = 1e-07
    stars = [np.zeros((n, n, 9)) for _ in range(3)]
    for value, coefficients in zip((1.1, 1.2, 1.05), stars):
        coefficients[:, :, 0] = value
        coefficients[:, :, 1] = 0.001
    log_eps[0, 1] += 0.125
    diff[0, 1, 0] *= 1.1
    for coefficients in stars:
        coefficients[0, 1, 1] *= 1.7
    inputs = (mw, log_eps, epsilon, zrot, frot, visc, diff, *stars)
    if layout == 'strided':
        copied = []
        for array in inputs:
            owner = np.empty((*array.shape[:-1], 2 * array.shape[-1]))
            view = owner[..., ::2]
            view[:] = array
            copied.append(view)
        inputs = tuple(copied)
    elif layout == 'F':
        inputs = tuple((np.array(array, order='F') for array in inputs))
    return _readonly(inputs)

@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_primitives_asymmetric_readonly_inputs_and_independent_fd(layout):
    inputs = _synthetic_inputs(layout)
    before = _bytes(inputs)
    T = 1321.8750924889964
    primal = _readonly(_primal_terms(T, inputs))
    primal_bytes = _bytes(primal)
    analytic = _primitive_tangents(T, inputs, primal)
    assert _bytes(inputs) == before and _bytes(primal) == primal_bytes
    estimates = [[] for _ in analytic]
    for relative_h in RELATIVE_H:
        h = T * relative_h
        lower, upper = (_primal_terms(T - h, inputs), _primal_terms(T + h, inputs))
        for i in range(len(analytic)):
            estimates[i].append((h, (upper[i] - lower[i]) / (2 * h)))
    for name, actual, fd, base in zip(('mu', 'rot', 'binary', 'A', 'B', 'C'), analytic, estimates, primal):
        _gate(actual, fd, base, T, label=name)
    assert all((a.flags.c_contiguous and a.flags.owndata for a in analytic))
    assert all((not np.shares_memory(a, b) for i, a in enumerate(analytic) for b in (*inputs, *primal, *analytic[:i])))
    assert _bytes(inputs) == before and _bytes(primal) == primal_bytes

def test_zero_fit_no_log_division_and_A_B_A_lifetime():
    inputs = list(_synthetic_inputs())
    viscosity = inputs[5].copy()
    viscosity[0] = 0.0
    inputs[5] = viscosity
    inputs = _readonly(tuple(inputs))
    before = _bytes(inputs)
    a = _primitive_tangents(900.0, inputs, _readonly(_primal_terms(900.0, inputs)))
    saved = _bytes(a)
    b = _primitive_tangents(1600.0, inputs, _readonly(_primal_terms(1600.0, inputs)))
    again = _primitive_tangents(900.0, inputs, _readonly(_primal_terms(900.0, inputs)))
    assert a[0][0] == 0.0 and a[2][0, 0] == 0.0
    assert all((np.isfinite(x).all() for x in (*a, *b, *again)))
    assert _bytes(a) == saved == _bytes(again) and _bytes(inputs) == before

def test_nasa_low_at_equality_with_a_branch_local_backward_fd():
    from kflame.chemistry.transport_temperature_derivatives import nasa_cp_r_with_tangent
    from kflame.flame.diffusion_kernels import _cp_r
    low = np.zeros((2, 7))
    high = np.zeros((2, 7))
    low[:, 0] = (3.0, 4.0)
    low[:, 1] = (0.0001, 0.0002)
    low[:, 2] = (1e-08, 2e-08)
    high[:] = low
    high[:, 0] += 0.3
    mid = np.array([1000.0, 1000.0])
    _readonly((low, high, mid))
    cp, derivative = (np.full(2, np.nan), np.full(2, np.nan))
    nasa_cp_r_with_tangent(1000.0, low, high, mid, cp, derivative)
    expected = np.empty(2)
    _cp_r(1000.0, low, high, mid, expected)
    assert cp.tobytes() == expected.tobytes()
    fd = []
    for h in (1.0, 0.1, 0.01, 0.001):
        minus, minus2 = (np.empty(2), np.empty(2))
        _cp_r(1000.0 - h, low, high, mid, minus)
        _cp_r(1000.0 - 2 * h, low, high, mid, minus2)
        fd.append((h, (3 * cp - 4 * minus + minus2) / (2 * h)))
    _gate(derivative, fd, cp, 1000.0, rtol=1e-06, label='low branch at equality')

@pytest.fixture(scope='module')
def mechanism():
    from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
    from kflame.chemistry.multicomponent import NativeMulticomponentTransport
    from kflame.chemistry.thermo import NativeThermo
    path = Path(resolve_mechanism('h2o2.yaml'))
    mech = load_mechanism(path)
    multi = NativeMulticomponentTransport(mech, str(path))
    thermo = NativeThermo(mech)
    assert mech.n_species < 24
    return dict(path=path, mech=mech, multi=multi, thermo=thermo)

def _composition(mechanism, kind):
    mech = mechanism['mech']
    y = np.zeros(mech.n_species)
    if kind == 'dense':
        y[:] = np.arange(1.0, mech.n_species + 1)
    elif kind == 'pure':
        y[mech.species_names.index('N2')] = 1.0
    else:
        for name, mass in (('H2', 0.04), ('O2', 0.22), ('N2', 0.72), ('H2O', 0.02)):
            y[mech.species_names.index(name)] = mass
        if kind == 'trace':
            y[mech.species_names.index('OH')] = 1e-25
    y /= y.sum()
    return y

def _state(mechanism, T, y, *, inputs=None, tangent=True):
    """Actual native primal; retained blocks are copied before scratch reuse."""
    from kflame.chemistry.multicomponent_kernels import composition_terms_ws, composition_workspace
    from kflame.chemistry.transport_temperature_derivatives import nasa_cp_r_with_tangent
    multi, mech = (mechanism['multi'], mechanism['mech'])
    if inputs is None:
        inputs, _ = multi.export_temperature_snapshot()
    primal = _primal_terms(T, inputs)
    cp, dcp = (np.empty(mech.n_species), np.empty(mech.n_species))
    nasa = (mech.nasa_low, mech.nasa_high, mech.nasa_Tmid)
    nasa_cp_r_with_tangent(T, *nasa, cp, dcp)
    workspace = composition_workspace(mech.n_species)
    D = np.empty((mech.n_species, mech.n_species))
    DT = np.empty(mech.n_species)
    rho, lam, wm = composition_terms_ws(T, 101325.0, y, cp, inputs[0], multi.inv_w, multi.crot, *primal, D, DT, workspace)
    vectors, matrices, _ = workspace
    blocks = tuple((matrices[i].copy() for i in (0, 3, 4, 5))) + (vectors[9].copy(),)
    return dict(T=T, P=101325.0, y=y.copy(), inputs=inputs, cp=cp, dcp=dcp, mw=inputs[0], inv_w=multi.inv_w, crot=multi.crot, primal=primal, tangent=_primitive_tangents(T, inputs, primal) if tangent else None, rho=rho, lambda_=lam, wm=wm, multi=D, thermal=DT, workspace=workspace, x=vectors[0].copy(), blocks=blocks, nasa=nasa, mechanism=mechanism)

@pytest.mark.parametrize('kind', ['dense', 'sparse', 'trace', 'pure'])
def test_h2o2_primal_against_public_cantera(mechanism, kind):
    import cantera as ct
    state = _state(mechanism, 1321.8750924889964, _composition(mechanism, kind))
    gas = ct.Solution(str(mechanism['path']), transport_model='multicomponent')
    assert tuple(gas.species_names) == tuple(mechanism['mech'].species_names)
    gas.TPY = (state['T'], state['P'], state['y'])
    np.testing.assert_allclose(state['cp'], gas.standard_cp_R, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(state['rho'], gas.density, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(state['wm'], gas.mean_molecular_weight, rtol=2e-12, atol=1e-12)
    for actual, expected in ((state['multi'], gas.multi_diff_coeffs), (state['thermal'], gas.thermal_diff_coeffs), (state['lambda_'], gas.thermal_conductivity)):
        np.testing.assert_allclose(actual, expected, rtol=3e-07, atol=1e-12)

@pytest.mark.parametrize('kind', ['dense', 'trace', 'pure'])
def test_blocks_and_full_closure_tangents_against_native_primal_fd(mechanism, adapter, kind):
    T = 1321.8750924889964
    y = _readonly((_composition(mechanism, kind),))[0]
    state = _state(mechanism, T, y)
    before = _bytes((*state['inputs'], y, state['cp'], *state['primal'], state['multi'], state['thermal']))
    blocks = tuple(adapter.block_tangents(state))
    derivatives = tuple((np.asarray(x) for x in adapter.coefficient_tangent(state)))
    assert len(blocks) == 5 and len(derivatives) == 3
    block_fd, closure_fd = ([[] for _ in blocks], [[] for _ in derivatives])
    for relative_h in RELATIVE_H:
        h = T * relative_h
        lower = _state(mechanism, T - h, y, tangent=False)
        upper = _state(mechanism, T + h, y, tangent=False)
        for i, (a, b) in enumerate(zip(lower['blocks'], upper['blocks'])):
            block_fd[i].append((h, (b - a) / (2 * h)))
        for i, name in enumerate(('multi', 'thermal', 'lambda_')):
            closure_fd[i].append((h, (np.asarray(upper[name]) - lower[name]) / (2 * h)))
    for i, (a, fd, primal) in enumerate(zip(blocks, block_fd, state['blocks'])):
        _gate(a, fd, primal, T, label=f'block {i}')
    for name, a, fd in zip(('multi', 'thermal', 'lambda_'), derivatives, closure_fd):
        _gate(a, fd, state[name], T, rtol=5e-05, scale_atol=2e-08, label=name)
    assert _bytes((*state['inputs'], y, state['cp'], *state['primal'], state['multi'], state['thermal'])) == before

def test_block_tangent_does_not_assume_pair_symmetry(mechanism, adapter):
    inputs, _ = mechanism['multi'].export_temperature_snapshot()
    altered = [array.copy() for array in inputs]
    altered[1][0, 1] += 0.02
    for position in (6, 7, 8, 9):
        altered[position][0, 1, 0] *= 1.01
    inputs = _readonly(tuple(altered))
    y = _composition(mechanism, 'dense')
    T = 1321.8750924889964
    state = _state(mechanism, T, y, inputs=inputs)
    actual = adapter.block_tangents(state)
    estimates = [[] for _ in range(5)]
    for relative_h in RELATIVE_H:
        h = T * relative_h
        a = _state(mechanism, T - h, y, inputs=inputs, tangent=False)
        b = _state(mechanism, T + h, y, inputs=inputs, tangent=False)
        for i in range(5):
            estimates[i].append((h, (b['blocks'][i] - a['blocks'][i]) / (2 * h)))
    for i in range(5):
        _gate(actual[i], estimates[i], state['blocks'][i], T, label=f'asym block {i}')

def _block(state):
    from kflame.chemistry.thermal_block_derivatives import block_temperature_tangents
    s=state
    return block_temperature_tangents(s['T'],s['x'],s['cp'],s['dcp'],s['mw'],s['crot'],*s['primal'],*s['tangent'])

def _coefficient(state):
    from kflame.chemistry.thermal_implicit_derivatives import implicit_temperature_tangent
    s=state
    return implicit_temperature_tangent(s['T'],s['P'],s['wm'],s['mw'],s['workspace'],*_block(s))

@pytest.fixture(scope="module")
def adapter():
    return SimpleNamespace(block_tangents=_block,coefficient_tangent=_coefficient)

@pytest.mark.parametrize('slope',[1e-6,-1e-6])
def test_actual_internal_branch_crossing(slope):
    from kflame.flame.thermal_face_derivatives import temperature_branch_safe
    t=900.; tref=2000.; h=.5e-7*tref
    low=np.zeros((1,7));low[0,1]=slope
    low[0,0]=2.501-slope*(t+.5*h)
    high=low.copy();mid=np.array([1000.])
    assert not temperature_branch_safe(t,t,tref,low,high,mid)
    assert temperature_branch_safe(t-1.,t-1.,tref,low,high,mid)

@pytest.mark.parametrize('t,expected',[(999.999999,False),(1000.,False),(999.,True),(1001.,True)])
def test_actual_nasa_branch_crossing(t,expected):
    from kflame.flame.thermal_face_derivatives import temperature_branch_safe
    low=np.zeros((1,7));low[0,0]=3.; high=low.copy();high[0,0]=4.
    assert temperature_branch_safe(t,t,2000.,low,high,np.array([1000.])) == expected

def test_extra_tangent_vectors_count_against_bank_budget():
    from kflame.flame.diffusion import _face_bank_available
    p=SimpleNamespace(ns=511,n=49,compiled_faces=True,analytic_chemistry=True,
        chemistry_tile_nodes=1,thermal_transport_tangents=False,
        _frozen_multicomponent=lambda:True,
        _compiled_data=lambda:{'multi':(None,)*10+(False,)})
    assert _face_bank_available(p)
    p.thermal_transport_tangents=True
    assert not _face_bank_available(p)
    p.ns=581;p.n=65
    assert not _face_bank_available(p)
    p.n=33
    assert _face_bank_available(p)

def test_implicit_coefficients_preserve_entire_workspace_and_ownership(mechanism):
    state=_state(mechanism,1321.8750924889964,_composition(mechanism,'dense'))
    original=_bytes(state['workspace'])
    first=_coefficient(state);saved=tuple(np.asarray(a).tobytes() for a in first)
    other=_state(mechanism,1600.,_composition(mechanism,'trace'))
    _coefficient(other)
    repeated=_coefficient(state)
    assert _bytes(state['workspace'])==original
    assert tuple(np.asarray(a).tobytes() for a in first)==saved
    assert tuple(np.asarray(a).tobytes() for a in repeated)==saved
    for a in first[:2]:
        # Numba's MemInfo owns returned allocations; NumPy OWNDATA is False.
        # Disjoint storage and survival across another face prove ownership.
        assert a.flags.c_contiguous
        assert all(not np.shares_memory(a,b) for b in (*state['workspace'],*state['primal'],state['cp']))
    assert not np.shares_memory(first[0],first[1])

@pytest.mark.parametrize('kind',['binary_cross','pair_cross','equality','invalid_mu','stable_active','physical'])
def test_floor_guard_checks_actual_trial_without_mutating_inputs(kind):
    from kflame.chemistry.thermal_block_derivatives import block_trial_floor_safe
    mw=np.array([2.,3.]);base=np.full((2,2),1e-6);trial=base.copy()
    mu=np.full(2,1e-5);mut=mu.copy()
    if kind=='binary_cross':base[0,1]=.1e-300;trial[0,1]=2e-300
    elif kind=='pair_cross':base[0,1]=.15e-300;trial[0,1]=.3e-300
    elif kind=='equality':base[0,1]=1e-300
    elif kind=='invalid_mu':mut[0]=0.
    elif kind=='stable_active':base[0,1]=.01e-300;trial[0,1]=.02e-300
    inputs=_readonly((mw,base,trial,mu,mut));saved=_bytes(inputs)
    assert block_trial_floor_safe(*inputs)==(kind in ('physical','stable_active'))
    assert _bytes(inputs)==saved

def test_bank_key_changes_when_temperature_coordinate_scale_changes():
    from kflame.flame.diffusion import _face_bank_key
    data={};p=SimpleNamespace(backend=object(),P=101325.,soret_enabled=True,
        transport_model='multicomponent',flux_gradient_basis='molar',compiled_faces=True,
        analytic_chemistry=True,chemistry_tile_nodes=1,temperature_pair_symmetry=True,
        thermal_transport_tangents=True,Tref=2000.,_compiled_data=lambda:data,
        _frozen_multicomponent=lambda:True)
    before=_face_bank_key(p);p.Tref*=2.
    assert _face_bank_key(p)!=before

def test_face_bank_declines_nasa_join_and_keeps_original_primal(mechanism):
    import cantera as ct
    from kflame.flame.diffusion import DiffusionCase,DiffusionFlameProblem
    from kflame.flame import diffusion_kernels as kernels
    gas=ct.Solution(str(mechanism['path']))
    gas.TPX=300.,101325.,'H2:1';fuel=gas.Y.copy()
    gas.TPX=300.,101325.,'O2:1,N2:3.76'
    p=DiffusionFlameProblem(DiffusionCase(str(mechanism['path']),fuel,gas.Y.copy(),
        transport_model='multicomponent',soret_enabled=True),mechanism['mech'],np.linspace(0,.02,3))
    data=p._compiled_data();ns=p.ns
    T=np.array([900.,1100.,1200.]);Y=np.ascontiguousarray(np.tile(_composition(mechanism,'dense')[:,None],(1,3)))
    def outputs():return (np.empty(2),np.empty(2),np.empty((ns,2)),np.empty((2,ns,ns)),np.empty((2,ns)),np.empty(2))
    ordinary,actual=outputs(),outputs()
    inputs=(0,2,T,Y,p.dz,p.P,data['invW'],data['mw'],data['mixture'],
        data.get('multi_certified',data['multi']),True,True)
    kernels.face_range_bank(*inputs,*ordinary)
    left,right,scalars=np.zeros((2,ns)),np.zeros((2,ns)),np.zeros((2,3))
    kernels.face_range_bank_thermal(*inputs,*actual,p.Tref,left,right,scalars)
    assert _bytes(ordinary)==_bytes(actual)
    assert scalars[0,2]==0. and not np.any(left[0]) and not np.any(right[0])
    assert scalars[1,2]==1. and np.isfinite(left).all() and np.isfinite(right).all()
