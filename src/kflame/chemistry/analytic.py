"""Native analytic chemical and thermal derivatives at constant pressure.

All Y are independent (no normalization or eliminated species). The chain
rule includes mixture density, third bodies, Lindemann/Troe/SRI falloff
and PLOG rates at constant pressure.
The hybrid integration linearizes species chemistry only. The direct spatial
integration also uses thermal derivatives and retains frozen transport.
"""
import math
import operator
import numpy as np
from numba import njit, prange
from kflame.chemistry.mechanism import R_UNIV
from kflame.chemistry.pressure_rates import plog_rate, sri_falloff, troe_temperature_scale


@njit(cache=True)
def _product_partials(C, plan, derivative):
    """Polynomial limit at zero; signed continuation matches native kinetics."""
    derivative[:] = 0.0
    degree = plan[0]
    negative = 0
    product = 1.0
    for i in range(degree):
        c = C[plan[i + 1]]
        negative += c < 0.0
        product *= c
    if negative > 1 or (degree > 3 and negative > 0):
        return 0.0
    for i in range(degree):
        value = 1.0
        for j in range(degree):
            if j != i:
                value *= C[plan[j + 1]]
        derivative[plan[i + 1]] += value
    return product


@njit(cache=True, nogil=True, error_model='numpy')
def _species_partials_node(node, tile_node, out, out_t, Ts, Ys, P, W, invW, lo, hi, tmid,
                          Ah, bh, Eh, Al, bl, El, three, fall, reversible,
                          troe, ta, t3, t1, t2, rp, pp,
                          ni, nn, nc, ei, ed, ec, dn, temperature_partials, plog_data=None, sri_data=None):
    ns = Ys.shape[0]
    T = Ts[node]
    logT = math.log(T)
    s = 0.0
    for k in range(ns):
        s += Ys[k, node] * invW[k]
    rho = P / (R_UNIV * T * s)
    C = rho * Ys[:, node] * invW
    g = np.empty(ns)
    hrt = np.empty(ns)
    for k in range(ns):
        a = hi[k] if T > tmid[k] else lo[k]
        h = a[0] + T*(a[1]/2 + T*(a[2]/3 + T*(a[3]/4 + T*a[4]/5))) + a[5]/T
        entropy = a[0]*logT + T*(a[1] + T*(a[2]/2 + T*(a[3]/3 + T*a[4]/4))) + a[6]
        g[k] = h - entropy
        hrt[k] = h
    # Sparse accumulation: reactant/product monomials touch only their own
    # species; the default (unit) third-body efficiency is a rank-one term
    # u[k] * 1^T added once at the end, and only nondefault efficiencies are
    # scattered. Mathematically identical to the dense per-reaction sweep.
    jc = np.zeros((ns, ns))
    jt = np.zeros(ns)
    u = np.zeros(ns)
    Mdef = 0.0
    for species in range(ns):
        Mdef += C[species]
    for r in range(Ah.size):
        kf = Ah[r] * math.exp(bh[r]*logT - Eh[r]/(R_UNIV*T))
        dg = 0.0
        dh = 0.0
        for i in range(nc[r]):
            dg += nn[r, i] * g[ni[r, i]]
            dh += nn[r, i] * hrt[ni[r, i]]
        kc = math.exp(-min(500.0, max(-500.0, dg))) * (101325.0/(R_UNIV*T))**dn[r]
        kr = kf/max(kc, 1e-300) if reversible[r] else 0.0
        log_kf_t = bh[r]/T + Eh[r]/(R_UNIV*T*T)
        if plog_data is not None and plog_data[0][r] >= 0:
            kf, log_kf_t = plog_rate(T, P, r, plog_data)
            kr = kf/max(kc, 1e-300) if reversible[r] else 0.0
        log_kc_t = ((dh if -500.0 < dg < 500.0 else 0.0)-dn[r])/T
        if kc <= 1e-300:
            log_kc_t = 0.0
        pf, fok = _monomial(C, rp[r])
        pr, rok = _monomial(C, pp[r]) if reversible[r] else (0.0, False)
        factor, factor_m = 1.0, 0.0
        factor_t = 0.0
        if three[r] or fall[r]:
            M = Mdef
            for i in range(ec[r]):
                M += ed[r, i]*C[ei[r, i]]
            if three[r]:
                factor, factor_m = M, 1.0
        if fall[r]:
            k0 = Al[r] * math.exp(bl[r]*logT - El[r]/(R_UNIV*T))
            alpha = k0/max(kf, 1e-300)
            reduced = alpha*M
            lind = reduced/(1.0 + reduced)
            F, dF = 1.0, 0.0
            log_alpha_t = bl[r]/T + El[r]/(R_UNIV*T*T)-log_kf_t
            reduced_t = reduced*log_alpha_t
            F_t = 0.0
            if troe[r]:
                fc = ((1-ta[r])*math.exp(-T/troe_temperature_scale(t3[r]))
                      + ta[r]*math.exp(-T/troe_temperature_scale(t1[r]))
                      + math.exp(-t2[r]/T))
                L = math.log10(max(fc, 1e-300))
                v = math.log10(max(reduced, 1e-300)) - .4 - .67*L
                n = .75 - 1.27*L
                den = n - .14*v
                f = v/den
                F = 10.0**(L/(1+f*f))
                if reduced > 1e-300:
                    dF = F * L * (-2*f)/(1+f*f)**2 * n/(den*den) / reduced
                if temperature_partials:
                    fc_t = (-(1-ta[r])*math.exp(-T/troe_temperature_scale(t3[r]))/troe_temperature_scale(t3[r])
                            -ta[r]*math.exp(-T/troe_temperature_scale(t1[r]))/troe_temperature_scale(t1[r])
                            +math.exp(-t2[r]/T)*t2[r]/(T*T))
                    L_t = fc_t/(fc*math.log(10.0)) if fc > 1e-300 else 0.0
                    v_t = (log_alpha_t/math.log(10.0) if reduced > 1e-300 else 0.0)-.67*L_t
                    den_t = -1.27*L_t-.14*v_t
                    f_t = (v_t*den-v*den_t)/(den*den)
                    F_t = F*math.log(10.0)*(L_t/(1+f*f)-L*2*f*f_t/(1+f*f)**2)
            if sri_data is not None and sri_data[r, 0] != 0.:
                F, log_f_pr, log_f_t = sri_falloff(T, reduced, sri_data[r])
                dF = F*log_f_pr
                if temperature_partials:
                    F_t = F*(log_f_t + log_f_pr*reduced_t)
            factor = lind*F
            factor_m = alpha*(F/(1+reduced)**2 + lind*dF)
            factor_t = reduced_t*F/(1+reduced)**2 + lind*F_t
        q = kf*pf - kr*pr
        if temperature_partials:
            qt = factor*(kf*log_kf_t*pf-kr*(log_kf_t-log_kc_t)*pr)+factor_t*q
            for i in range(nc[r]):
                jt[ni[r, i]] += nn[r, i]*qt
        if fok:
            _scatter_monomial(jc, C, rp[r], factor*kf, ni[r], nn[r], nc[r])
        if rok:
            _scatter_monomial(jc, C, pp[r], -factor*kr, ni[r], nn[r], nc[r])
        if factor_m != 0.0:
            fq = factor_m*q
            if fq != 0.0:
                for i in range(nc[r]):
                    u[ni[r, i]] += nn[r, i]*fq
                for e in range(ec[r]):
                    dq = fq*ed[r, e]
                    column = ei[r, e]
                    for i in range(nc[r]):
                        jc[ni[r, i], column] += nn[r, i]*dq
    for k in range(ns):
        row = jc[k]
        contraction = u[k]*Mdef
        for species in range(ns):
            contraction += row[species]*C[species]
        if temperature_partials:
            out_t[k, tile_node] = W[k]*(jt[k]-contraction/T)
        base = rho*u[k]
        shift = contraction/s
        wk = W[k]
        target = out[k, tile_node]
        for j in range(ns):
            target[j] = wk*invW[j]*(rho*row[j] + base - shift)


@njit(cache=True)
def _monomial(C, plan):
    """Product of a reaction-order plan; False when its derivative is zero.

    Same limit as :func:`_product_partials`: with more than one negative
    factor, or any negative factor beyond degree three, product and
    derivative vanish.
    """
    degree = plan[0]
    negative = 0
    product = 1.0
    for i in range(degree):
        c = C[plan[i + 1]]
        negative += c < 0.0
        product *= c
    if negative > 1 or (degree > 3 and negative > 0):
        return 0.0, False
    return product, degree > 0


@njit(cache=True)
def _scatter_monomial(jc, C, plan, scale, participants, stoichiometry, count):
    """jc[participant, j] += nu * scale * d(prod C)/dC_j for the plan's species j."""
    degree = plan[0]
    for a in range(degree):
        value = scale
        for b in range(degree):
            if b != a:
                value *= C[plan[b + 1]]
        if value != 0.0:
            column = plan[a + 1]
            for i in range(count):
                jc[participants[i], column] += stoichiometry[i]*value


@njit(cache=True, parallel=True)
def species_partials_core(Ts, Ys, P, W, invW, lo, hi, tmid,
                          Ah, bh, Eh, Al, bl, El, three, fall, reversible,
                          troe, ta, t3, t1, t2, rp, pp,
                          ni, nn, nc, ei, ed, ec, dn, temperature_partials=False, plog_data=None, sri_data=None):
    ns, nodes = Ys.shape
    out = np.empty((ns, nodes, ns))
    out_t = np.zeros((ns, nodes))
    for node in prange(nodes):
        _species_partials_node(node, node, out, out_t, Ts, Ys, P, W, invW, lo, hi, tmid,
            Ah, bh, Eh, Al, bl, El, three, fall, reversible,
            troe, ta, t3, t1, t2, rp, pp,
            ni, nn, nc, ei, ed, ec, dn, temperature_partials, plog_data, sri_data)
    return out, out_t


@njit(cache=True, nogil=True)
def species_partials_range_core(Ts, Ys, P, W, invW, lo, hi, tmid,
                          Ah, bh, Eh, Al, bl, El, three, fall, reversible,
                          troe, ta, t3, t1, t2, rp, pp,
                          ni, nn, nc, ei, ed, ec, dn, temperature_partials=False,
                                start=0, stop=-1, plog_data=None, sri_data=None):
    ns = Ys.shape[0]
    nodes = (Ys.shape[1] if stop == -1 else stop) - start
    # Keep one node's KxK derivative matrix together for its immediate consumer;
    # the public logical axes remain (species, node, differentiated species).
    out = np.empty((nodes, ns, ns)).transpose(1, 0, 2)
    out_t = np.zeros((ns, nodes))
    for tile_node in range(nodes):
        _species_partials_node(start + tile_node, tile_node, out, out_t, Ts, Ys, P, W, invW, lo, hi, tmid,
            Ah, bh, Eh, Al, bl, El, three, fall, reversible,
            troe, ta, t3, t1, t2, rp, pp,
            ni, nn, nc, ei, ed, ec, dn, temperature_partials, plog_data, sri_data)
    return out, out_t


@njit(cache=True, nogil=True)
def species_partials_range_into_core(Ts, Ys, P, W, invW, lo, hi, tmid,
                                     Ah, bh, Eh, Al, bl, El, three, fall, reversible,
                                     troe, ta, t3, t1, t2, rp, pp,
                                     ni, nn, nc, ei, ed, ec, dn, start, stop, out, out_t,
                                     plog_data=None, sri_data=None):
    """Like :func:`species_partials_range_core` (with temperature partials)
    into caller-owned ``out[K,B,K]`` and ``out_t[K,B]``, B >= stop-start."""
    for tile_node in range(stop - start):
        _species_partials_node(start + tile_node, tile_node, out, out_t, Ts, Ys, P, W, invW, lo, hi, tmid,
            Ah, bh, Eh, Al, bl, El, three, fall, reversible,
            troe, ta, t3, t1, t2, rp, pp,
            ni, nn, nc, ei, ed, ec, dn, True, plog_data, sri_data)


def species_partials_range_into(backend, T, Y, start, stop, out, out_t):
    """Reusable-buffer variant for prepared T, Y (see ``prepare_species_partials``).

    ``out`` has logical axes (species, node, species) and ``out_t`` (species,
    node); both are fully overwritten for nodes [0, stop-start).
    """
    pressure_rates = {}
    if backend.kinetics._plog_data is not None or backend.kinetics._sri_data is not None:
        pressure_rates = dict(plog_data=backend.kinetics._plog_data,
                              sri_data=backend.kinetics._sri_data)
    species_partials_range_into_core(T, Y, *_species_partials_arguments(backend), start, stop, out, out_t,
                                     **pressure_rates)


def prepare_species_partials(backend, T, Y):
    """Prepare and validate the full T[N], Y[K,N] once before tiled calls.

    Density validation deliberately uses the original global ``invW @ Y``.
    Returned arrays may borrow the inputs; do not mutate them or the backend
    while range calls are running. Read-only inputs are supported.
    """
    b, k = backend, backend.kinetics
    for side, relevant in (('r', np.ones(k._nb_is_reversible.size, dtype=bool)),
                           ('p', k._nb_is_reversible)):
        rate_plan = getattr(k, f'_sp_{side}_plan')
        analytic_plan = getattr(k, f'_analytic_{side}_plan', rate_plan)
        # A negative rate sentinel is allowed only for a genuine polynomial
        # above degree three. Invalid orders and corrupted plans still reject.
        larger = getattr(k, f'_sp_{side}_nu').sum(axis=1) > 3
        invalid = (analytic_plan[:, 0] < 0) | ((rate_plan[:, 0] < 0) & ~larger)
        if np.any(invalid & relevant):
            raise ValueError('Analytic chemistry requires nonnegative integer reaction orders')
    Y = np.ascontiguousarray(Y, dtype=float)
    T = np.ascontiguousarray(T, dtype=float)
    if T.ndim != 1 or Y.shape != (b.n_species, T.size):
        raise ValueError('Expected T[N] and Y[K,N] for analytic chemistry')
    if not np.all(np.isfinite(Y)) or np.any(b.invW @ Y <= 1e-300):
        raise ValueError('Analytic chemistry requires finite, unclipped mixture density')
    if np.any(T <= 0) or not np.all(np.isfinite(T)):
        raise ValueError('Analytic chemistry requires positive finite temperatures')
    return T, Y


def _species_partials_arguments(b):
    k = b.kinetics
    return (b._P_float, b.W, b.invW,
        b._nb_nasa_lo, b._nb_nasa_hi, b._nb_nasa_tmid,
        k._nb_A_hi, k._nb_b_hi, k._nb_Ea_hi, k._nb_A_lo, k._nb_b_lo, k._nb_Ea_lo,
        k._nb_is_three_body, k._nb_is_falloff, k._nb_is_reversible,
        k._nb_has_troe, k._nb_troe_A, k._nb_troe_T3, k._nb_troe_T1, k._nb_troe_T2,
        getattr(k, '_analytic_r_plan', k._sp_r_plan),
        getattr(k, '_analytic_p_plan', k._sp_p_plan), k._sp_net_idx, k._sp_net_nu, k._sp_net_count,
        k._sp_eff_idx, k._sp_eff_delta, k._sp_eff_count, k._nb_delta_nu)


def species_partials(backend, T, Y, *, _temperature=False):
    """Return d(omega_mass[k,node])/dY[j,node], shape (K,N,K)."""
    T, Y = prepare_species_partials(backend, T, Y)
    pressure_rates = {}
    if backend.kinetics._plog_data is not None or backend.kinetics._sri_data is not None:
        pressure_rates = dict(plog_data=backend.kinetics._plog_data,
                              sri_data=backend.kinetics._sri_data)
    result = species_partials_core(T, Y, *_species_partials_arguments(backend),
                                  _temperature, **pressure_rates)
    return result if _temperature else result[0]


def species_partials_range(backend, T, Y, start, stop, *, _temperature=True,
                           _validated=False):
    """Return (jy[K,B,K], jt[K,B]) for global nodes [start, stop), B=stop-start.

    Uses a serial nogil kernel, suitable for an external thread pool. Outputs
    own only the tile's storage; each jy[:, node, :] is contiguous. The kernel
    reads global T/Y without slicing.
    By default the whole state is prepared and validated. To do that once,
    call ``T, Y = prepare_species_partials(backend, T, Y)`` before submitting
    ranges with ``_validated=True``. That flag requires the same unchanged
    backend and prepared arrays; it skips chemistry/density/finite checks.
    ``_temperature=False`` returns a zero jt while still returning the pair.
    """
    if not _validated:
        T, Y = prepare_species_partials(backend, T, Y)
    else:
        if (not isinstance(T, np.ndarray) or not isinstance(Y, np.ndarray)
                or T.dtype != np.float64 or Y.dtype != np.float64
                or not T.flags.c_contiguous or not Y.flags.c_contiguous
                or T.ndim != 1 or Y.shape != (backend.n_species, T.size)):
            raise ValueError('Expected globally prepared T[N] and Y[K,N]')
    start, stop = operator.index(start), operator.index(stop)
    if not 0 <= start <= stop <= T.size:
        raise ValueError('Require 0 <= start <= stop <= N for analytic chemistry')
    pressure_rates = {}
    if backend.kinetics._plog_data is not None or backend.kinetics._sri_data is not None:
        pressure_rates = dict(plog_data=backend.kinetics._plog_data,
                              sri_data=backend.kinetics._sri_data)
    return species_partials_range_core(
        T, Y, *_species_partials_arguments(backend), _temperature, start, stop,
        **pressure_rates)


def thermochemical_partials(backend, T, Y):
    """Analytic nodal derivatives in [u,T,Y...] for direct block assembly."""
    jy, jt = species_partials(backend, T, Y, _temperature=True)
    ns, nodes = Y.shape
    nv = ns+2
    invW = backend.invW
    s = invW @ Y
    rho = backend._P_float/(R_UNIV*T*s)
    drho = np.zeros((nodes, nv))
    drho[:, 1] = -rho/T
    drho[:, 2:] = -rho[:, None]*invW[None, :]/s[:, None]
    dcp = np.zeros((nodes, nv))
    cp_molar = backend.thermo.partial_molar_cp(T)
    dcp[:, 2:] = (cp_molar*invW[:, None]).T
    a = np.where((T[None, :] > backend._nb_nasa_tmid[:, None])[:, :, None],
                 backend._nb_nasa_hi[:, None, :], backend._nb_nasa_lo[:, None, :])
    cp_t = R_UNIV*(a[:, :, 1]+T*(2*a[:, :, 2]+T*(3*a[:, :, 3]+4*T*a[:, :, 4])))
    dcp[:, 1] = np.sum(cp_t*Y*invW[:, None], axis=0)
    domega = np.zeros((nodes, ns, nv))
    domega[:, :, 1] = jt.T
    domega[:, :, 2:] = jy.transpose(1, 0, 2)
    return drho, dcp, domega, np.ascontiguousarray(cp_molar)


def hybrid_center_properties(backend, x, T_all, Y_all):
    """Linearize chemistry in species columns; retain exact perturbed thermo.

The native residual assembler applies the remaining spatial finite
differences with its usual frozen transport. No species is eliminated.
"""
    nodes, nv = x.shape
    ns = nv - 2
    T, Y = x[:, 1], x[:, 2:].T
    jac = species_partials(backend, T, Y)
    # Only base and temperature-perturbed nodes require reaction evaluation.
    ts = np.column_stack((T, T_all.reshape(nodes, nv)[:, 1])).ravel()
    ys = np.repeat(Y, 2, axis=1)
    om, h = np.empty_like(ys), np.empty_like(ys)
    _, cp2 = backend.eval_grid_thermo_kinetics_into(ts, ys, om, h)
    delta = (Y_all.reshape(ns, nodes, nv)[np.arange(ns), :, np.arange(ns)+2].T
             - x[:, 2:])
    omega = np.repeat(om[:, ::2, None], nv, axis=2)
    omega[:, :, 1] = om[:, 1::2]
    omega[:, :, 2:] += jac * delta[None, :, :]
    # Exact ideal-gas density and NASA cp in the perturbed states are cheap.
    rho = backend._P_float / (R_UNIV * T_all * (backend.invW @ Y_all))
    cp_species = backend.thermo.partial_molar_cp(T) * backend.invW[:, None]
    cp = np.repeat(cp2[::2, None], nv, axis=1)
    cp[:, 1] = cp2[1::2]
    cp[:, 2:] += cp_species.T * delta
    hk = np.repeat(h[:, ::2, None], nv, axis=2)
    hk[:, :, 1] = h[:, 1::2]
    return rho.reshape(nodes, nv), cp, omega, hk
