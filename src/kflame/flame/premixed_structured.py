"""Premixed (free and burner) flame Jacobian in structured block form.

Same derivative as :mod:`kflame.flame.analytic_jacobian` (frozen face
transport, analytic chemistry, the residual's own upwind/closure branches),
but stored as in :mod:`kflame.flame.structured_jacobian`: dense diagonal
blocks and off-diagonal blocks ``A[j, j+-1] = diag(D) + P Q^T`` of rank
``RANK``. With frozen coefficients a face flux depends on either node
through a diagonal plus two rank-one terms (molar gradient and correction
velocity) and a temperature column (Soret); the continuity and energy rows
add one dense row each. The dense path stores three nv x nv blocks per node
plus ns x nv face partials and factors them with two dense products per
node; here a node costs one LAPACK inverse and O(RANK nv^2).

Chemistry partials are computed tile by tile, so the full nodes x ns x nv
chemistry Jacobian is never held. Mixture-averaged transport only (with or
without Soret); multicomponent fluxes and thresholded blocks keep the dense
path.
"""
import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV
from kflame.flame.diffusion_kernels import radiative_loss
from kflame.flame.structured_jacobian import _radiation_derivatives

RANK = 5
# Columns of P/Q: continuity row, energy row, two face rank-one terms, T column.
_ROW0, _ROW1, _RANK1, _RANK2, _TCOL = 0, 1, 2, 3, 4
# Rows of a face record (one value per species).
_DL, _P1L, _Q1L, _Q2L, _DR, _P1R, _Q1R, _Q2R, _P2, _TL, _TR = range(11)
_PARTIALS_TILE_BYTES = 16*1024**2


@njit(cache=True, nogil=True)
def premixed_faces(start, stop, state, invW, z, coeff, thermal, molar, face, flux):
    """Frozen-coefficient fluxes and their structured partials, faces [start, stop).

    dflux_k/dY_vL = dL_k delta_kv + p1L_k q1L_v + p2_k q2L_v, dflux_k/dT_L = tL_k
    and likewise on the right node (as in ``analytic_jacobian.face_partials``).
    """
    ns = state.shape[1]-2
    for f in range(start, stop):
        dz = z[f+1]-z[f]
        sl = 0.
        sr = 0.
        for k in range(ns):
            sl += state[f, k+2]*invW[k]
            sr += state[f+1, k+2]*invW[k]
        total = 0.
        cxl = 0.
        cxr = 0.
        for k in range(ns):
            c = coeff[k, f]
            if molar:
                xl = state[f, k+2]*invW[k]/sl
                xr = state[f+1, k+2]*invW[k]/sr
                raw = -c*(xr-xl)/dz
                cxl += c*xl
                cxr += c*xr
            else:
                raw = -c*(state[f+1, k+2]-state[f, k+2])/dz
            flux[f, k] = raw
            total += raw
        dT = state[f+1, 1]-state[f, 1]
        w = face[f]
        for k in range(ns):
            c = coeff[k, f]
            yl = state[f, k+2]
            flux[f, k] -= yl*total+thermal[k, f]*dT/dz
            if molar:
                w[_DL, k] = c*invW[k]/(sl*dz)-total
                w[_P1L, k] = -c*(yl*invW[k]/sl)/dz
                w[_Q1L, k] = invW[k]/sl
                w[_Q2L, k] = invW[k]*(c-cxl)/(sl*dz)
                w[_DR, k] = -c*invW[k]/(sr*dz)
                w[_P1R, k] = c*(state[f+1, k+2]*invW[k]/sr)/dz
                w[_Q1R, k] = invW[k]/sr
                w[_Q2R, k] = -invW[k]*(c-cxr)/(sr*dz)
            else:
                w[_DL, k] = c/dz-total
                w[_P1L, k] = 0.
                w[_Q1L, k] = 0.
                w[_Q2L, k] = c/dz
                w[_DR, k] = -c/dz
                w[_P1R, k] = 0.
                w[_Q1R, k] = 0.
                w[_Q2R, k] = -c/dz
            w[_P2, k] = -yl
            w[_TL, k] = thermal[k, f]/dz
            w[_TR, k] = -thermal[k, f]/dz


@njit(cache=True)
def _dense_face(w, left, out):
    """out[k, v] = dflux_k/dx_v at one node of a face (columns U, T, Y...)."""
    ns = w.shape[1]
    d, p1, q1, q2, t = (_DL, _P1L, _Q1L, _Q2L, _TL) if left else (_DR, _P1R, _Q1R, _Q2R, _TR)
    for k in range(ns):
        row = out[k]
        row[0] = 0.
        row[1] = w[t, k]
        a = w[p1, k]
        b = w[_P2, k]
        for v in range(ns):
            row[v+2] = a*w[q1, v]+b*w[q2, v]
        row[k+2] += w[d, k]


@njit(cache=True)
def _offdiag(j, col, x, z, rho, cp, h, drho, cp_molar, flux, face, lam, invW,
             energy, has_profile, fixed, outlet_flux, inlet_mass_flux, convection,
             conv_w, conv_wy, D, P, Q):
    """Block A[j, col] (col = j-1 or j+1) as diag(D) + P Q^T."""
    nodes, nv = x.shape
    ns = nv-2
    D[:] = 0.
    P[:, :] = 0.
    Q[:, :] = 0.
    u = x[j, 0]
    lower = col == j-1
    # Face whose partials enter this block and the side of node col on it.
    f = j-1 if lower else j
    left = lower
    d, p1, q1, q2, t = (_DL, _P1L, _Q1L, _Q2L, _TL) if left else (_DR, _P1R, _Q1R, _Q2R, _TR)
    w = face[f]
    boundary = j == 0 or j == nodes-1
    # Continuity row: dmo = d(rho u)[col]/dx_col.
    other = -1
    scale0 = 0.
    if boundary:
        other = 1 if j == 0 else j-1
        if j == 0:
            scale0 = 0. if inlet_mass_flux > 0. else -1./(z[1]-z[0])
        else:
            scale0 = -1.
    elif not (fixed >= 0 and j == fixed):
        other = j-1 if inlet_mass_flux > 0. or (fixed >= 0 and j > fixed) else j+1
        scale0 = 1./(z[j]-z[j-1]) if other == j-1 else -1./(z[j+1]-z[j])
    if other == col and scale0 != 0.:
        P[0, _ROW0] = 1.
        Q[0, _ROW0] = scale0*rho[col]
        for v in range(1, nv):
            Q[v, _ROW0] = scale0*x[col, 0]*drho[col, v]
    if boundary:
        close = np.argmax(x[j, 2:])
        if j == nodes-1 and (energy or not has_profile):
            P[1, _ROW1] = 1.
            Q[1, _ROW1] = -1.
        # Species rows: cs * face partials + diagonal convection terms.
        cs = 0.
        conv = 0.
        if j == 0:
            cs = -1.
            if convection != 0:
                conv = -rho[j]*u*0.5
        elif outlet_flux:
            cs = 1.
        else:
            conv = -1.
        for k in range(ns):
            if k == close:
                continue
            D[k+2] = cs*w[d, k]+conv
            P[k+2, _RANK1] = cs*w[p1, k]
            P[k+2, _RANK2] = cs*w[_P2, k]
            P[k+2, _TCOL] = cs*w[t, k]
        if cs != 0.:
            for v in range(ns):
                Q[v+2, _RANK1] = w[q1, v]
                Q[v+2, _RANK2] = w[q2, v]
            Q[1, _TCOL] = 1.
        return
    dzm, dzp, dz2 = z[j]-z[j-1], z[j+1]-z[j], z[j+1]-z[j-1]
    up = j if u > 0. else j+1
    down = up-1
    dzup = z[up]-z[down]
    wm, wp, hm, hp = conv_w[j, 0], conv_w[j, 1], conv_w[j, 2], conv_w[j, 3]
    cs = 2./(dz2*rho[j]) if lower else -2./(dz2*rho[j])
    for k in range(ns):
        if convection == 0:
            dgrad = ((1. if up == col else 0.)-(1. if down == col else 0.))/dzup
        else:
            dgrad = -conv_wy[k, j, 0] if lower else conv_wy[k, j, 1]
        D[k+2] = cs*w[d, k]-u*dgrad
        P[k+2, _RANK1] = cs*w[p1, k]
        P[k+2, _RANK2] = cs*w[_P2, k]
        P[k+2, _TCOL] = cs*w[t, k]
    for v in range(ns):
        Q[v+2, _RANK1] = w[q1, v]
        Q[v+2, _RANK2] = w[q2, v]
    Q[1, _TCOL] = 1.
    if energy:
        rc = 1./(rho[j]*cp[j])
        if convection == 0:
            dgradT = ((1. if up == col else 0.)-(1. if down == col else 0.))/dzup
        else:
            dgradT = -wm if lower else wp
        dcond = -2.*lam[j-1]/(dzm*dz2) if lower else -2.*lam[j]/(dzp*dz2)
        # sum_k g_k dflux_k/dx_v, g_k = gradh_k invW_k / 2 (dflux from one face).
        sp1 = 0.
        sp2 = 0.
        st = 0.
        tsum = 0.
        P[1, _ROW1] = 1.
        for k in range(ns):
            if convection == 0:
                gradh = (h[k, up]-h[k, down])/dzup
                dgradh = (cp_molar[k, up]*(1. if up == col else 0.)
                          - cp_molar[k, down]*(1. if down == col else 0.))/dzup
            else:
                gradh = hp*(h[k, j+1]-h[k, j])+hm*(h[k, j]-h[k, j-1])
                dgradh = -hm*cp_molar[k, j-1] if lower else hp*cp_molar[k, j+1]
            g = .5*gradh*invW[k]
            sp1 += g*w[p1, k]
            sp2 += g*w[_P2, k]
            st += g*w[t, k]
            tsum += .5*(flux[j-1, k]+flux[j, k])*dgradh*invW[k]
            Q[k+2, _ROW1] = -rc*g*w[d, k]
        for v in range(ns):
            Q[v+2, _ROW1] -= rc*(sp1*w[q1, v]+sp2*w[q2, v])
        Q[1, _ROW1] = -u*dgradT-rc*(dcond+st+tsum)


@njit(cache=True)
def _diag_block(j, x, z, rho, cp, omega, h, drho, dcp, dy, dt, b, cp_molar, flux, FM, FP,
                lam, invW, yin, energy, fixed, outlet_flux, inlet_mass_flux, convection,
                conv_w, conv_wy, B, a, g, T_all, Y_all, radiation, dq_dY):
    """Dense block A[j, j] (as ``analytic_jacobian.assemble_blocks``, column j).

    dy[k, b, v] and dt[k, b] are the chemistry partials of node j.
    """
    nodes, nv = x.shape
    ns = nv-2
    u = x[j, 0]
    boundary = j == 0 or j == nodes-1
    if boundary:
        close = np.argmax(x[j, 2:])
        for v in range(nv):
            dm = rho[j]*(1. if v == 0 else 0.)+u*drho[j, v]
            if j == 0:
                B[0, v] = dm if inlet_mass_flux > 0. else dm/(z[1]-z[0])
            else:
                B[0, v] = dm
            B[1, v] = 1. if v == 1 else 0.
        for k in range(ns):
            row = B[k+2]
            if k == close:
                row[0] = 0.
                row[1] = 0.
                for v in range(2, nv):
                    row[v] = -1.
                continue
            for v in range(nv):
                dm = rho[j]*(1. if v == 0 else 0.)+u*drho[j, v]
                if j == 0 and convection == 0:
                    val = -FP[k, v]+dm*(yin[k]-x[j, k+2])
                elif j == 0:
                    dw0 = dt[k, b] if v == 1 else (dy[k, b, v-2] if v >= 2 else 0.)
                    val = (-FP[k, v]+dm*(yin[k]-.5*(x[0, k+2]+x[1, k+2]))
                           + dw0*.5*(z[1]-z[0]))
                elif outlet_flux:
                    val = FM[k, v]+dm*x[j, k+2]
                else:
                    val = 0.
                row[v] = val
            if j == 0 and convection == 0:
                row[k+2] -= rho[j]*u
            elif j == 0:
                row[k+2] -= rho[j]*u*.5
            elif outlet_flux:
                row[k+2] += rho[j]*u
            else:
                row[k+2] += 1.
        return
    dzm, dzp, dz2 = z[j]-z[j-1], z[j+1]-z[j], z[j+1]-z[j-1]
    # Continuity row.
    if fixed >= 0 and j == fixed:
        for v in range(nv):
            if energy:
                B[0, v] = 1. if v == 1 else 0.
            else:
                B[0, v] = rho[j]*(1. if v == 0 else 0.)+u*drho[j, v]
    else:
        other = j-1 if inlet_mass_flux > 0. or (fixed >= 0 and j > fixed) else j+1
        scale = -1./dzm if other == j-1 else 1./dzp
        for v in range(nv):
            B[0, v] = scale*(rho[j]*(1. if v == 0 else 0.)+u*drho[j, v])
    up = j if u > 0. else j+1
    down = up-1
    dzup = z[up]-z[down]
    wm, wp, hm, hp = conv_w[j, 0], conv_w[j, 1], conv_w[j, 2], conv_w[j, 3]
    if convection == 0:
        gradT = (x[up, 1]-x[down, 1])/dzup
        cT = ((1. if up == j else 0.)-(1. if down == j else 0.))/dzup
    else:
        gradT = wp*(x[j+1, 1]-x[j, 1])+wm*(x[j, 1]-x[j-1, 1])
        cT = wm-wp
    cond = -2.*(lam[j]*(x[j+1, 1]-x[j, 1])/dzp-lam[j-1]*(x[j, 1]-x[j-1, 1])/dzm)/dz2
    dcond = 2.*(lam[j]/dzp+lam[j-1]/dzm)/dz2
    en = 0.
    den_t = 0.
    rr2 = 1./(rho[j]*rho[j])
    # Energy row accumulates den_v = sum_k a_k domega_kv + g_k (FM+FP)_kv.
    for v in range(nv):
        a[v] = 0.
    for k in range(ns):
        fm, fp = flux[j-1, k], flux[j, k]
        if convection == 0:
            gradh = (h[k, up]-h[k, down])/dzup
            cgh = (cp_molar[k, up]*(1. if up == j else 0.)-cp_molar[k, down]*(1. if down == j else 0.))/dzup
            gradY = (x[up, k+2]-x[down, k+2])/dzup
            cY = ((1. if up == j else 0.)-(1. if down == j else 0.))/dzup
        else:
            gradh = hp*(h[k, j+1]-h[k, j])+hm*(h[k, j]-h[k, j-1])
            cgh = (hm-hp)*cp_molar[k, j]
            ym, yp = conv_wy[k, j, 0], conv_wy[k, j, 1]
            gradY = yp*(x[j+1, k+2]-x[j, k+2])+ym*(x[j, k+2]-x[j-1, k+2])
            cY = ym-yp
        en += (h[k, j]*omega[k, j]+.5*(fm+fp)*gradh)*invW[k]
        den_t += (cp_molar[k, j]*omega[k, j]+.5*(fm+fp)*cgh)*invW[k]
        diff = 2.*(fp-fm)/dz2
        hk = h[k, j]*invW[k]
        gk = .5*gradh*invW[k]
        row = B[k+2]
        fm_row, fp_row = FM[k], FP[k]
        source = (omega[k, j]-diff)*rr2
        # U column: the chemistry, face fluxes and density do not depend on u.
        row[0] = -gradY
        for v in range(1, nv):
            dw = dt[k, b] if v == 1 else dy[k, b, v-2]
            ddiff = 2.*(fp_row[v]-fm_row[v])/dz2
            row[v] = (dw-ddiff)/rho[j]-source*drho[j, v]
            a[v] += hk*dw+gk*(fm_row[v]+fp_row[v])
        row[k+2] -= u*cY
        g[k] = gk
    # Optically thin radiative loss: local to the node.
    en += radiative_loss(T_all, Y_all, j, radiation)
    den_t += _radiation_derivatives(T_all, Y_all, j, radiation, dq_dY)
    for k in range(ns):
        a[k+2] += dq_dY[k]
    if energy:
        rc = 1./(rho[j]*cp[j])
        base = (cond+en)*rc
        for v in range(nv):
            du = 1. if v == 0 else 0.
            dtv = 1. if v == 1 else 0.
            dgradT = cT*dtv
            den = a[v]+dtv*den_t
            B[1, v] = (-du*gradT-u*dgradT-(dcond*dtv+den)*rc
                       + base*(drho[j, v]/rho[j]+dcp[j, v]/cp[j]))
    else:
        for v in range(nv):
            B[1, v] = 1. if v == 1 else 0.


@njit(cache=True, nogil=True)
def premixed_rows_range(start, stop, x, z, rho, cp, omega, h, drho, dcp, dy, dt, first,
                        cp_molar, flux, face, lam, invW, yin, energy, has_profile, fixed,
                        outlet_flux, inlet_mass_flux, convection, conv_w, conv_wy,
                        diag, diag_offset, lowD, lowP, lowQ, upD, upP, upQ, radiation):
    """Rows of nodes [start, stop); dy/dt hold the chemistry tile from ``first``."""
    nodes, nv = x.shape
    ns = nv-2
    FM = np.zeros((ns, nv))
    FP = np.zeros((ns, nv))
    a = np.empty(nv)
    g = np.empty(ns)
    dq_dY = np.empty(ns)
    T_all = np.ascontiguousarray(x[:, 1])
    Y_all = np.ascontiguousarray(x[:, 2:].T)
    for j in range(start, stop):
        if j > 0:
            _dense_face(face[j-1], False, FM)
        else:
            FM[:, :] = 0.
        if j < nodes-1:
            _dense_face(face[j], True, FP)
        else:
            FP[:, :] = 0.
        _diag_block(j, x, z, rho, cp, omega, h, drho, dcp, dy, dt, j-first, cp_molar, flux,
                    FM, FP, lam, invW, yin, energy, fixed, outlet_flux, inlet_mass_flux,
                    convection, conv_w, conv_wy, diag[j-diag_offset], a, g, T_all, Y_all, radiation, dq_dY)
        if j > 0:
            _offdiag(j, j-1, x, z, rho, cp, h, drho, cp_molar, flux, face, lam, invW,
                     energy, has_profile, fixed, outlet_flux, inlet_mass_flux, convection,
                     conv_w, conv_wy, lowD[j-1], lowP[j-1], lowQ[j-1])
        if j < nodes-1:
            _offdiag(j, j+1, x, z, rho, cp, h, drho, cp_molar, flux, face, lam, invW,
                     energy, has_profile, fixed, outlet_flux, inlet_mass_flux, convection,
                     conv_w, conv_wy, upD[j], upP[j], upQ[j])


# Below this size the dense two-ended factorization is already cheap.
STRUCTURED_MIN_SPECIES = 20


def eligible(problem, cache):
    """Structured form applies to mixture-averaged faces without thresholding."""
    setting = getattr(problem, 'structured_jacobian', 'auto')
    if setting is False or cache['flux_model'] != 'mixture-averaged':
        return False
    if float(getattr(problem, 'jacobian_threshold', 0.0)) != 0.:
        return False
    return setting is True or int(problem.n_species) >= STRUCTURED_MIN_SPECIES


class PremixedStructuredJacobian:
    """Block-tridiagonal premixed Jacobian with ``diag + P Q^T`` coupling blocks.

    Offers the interface the premixed solver uses on its Jacobian
    (``diagonal``, ``setdiag``, ``copy``, ``matvec``); ``factorize`` turns it
    into a :class:`PremixedStructuredFactor`.
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.n_blocks = int(blocks.n)
        self.block_size = int(blocks.nv)
        n = self.n_blocks*self.block_size
        self.shape = (n, n)

    def copy(self):
        from kflame.flame.structured_jacobian import StructuredBlocks
        b = self.blocks
        out = PremixedStructuredJacobian(StructuredBlocks(
            b.diag.copy(), b.lowD.copy(), b.lowP.copy(), b.lowQ.copy(),
            b.upD.copy(), b.upP.copy(), b.upQ.copy(), None, None))
        for name in ('_profile_problem', 'use_compiled_substitution'):
            if hasattr(self, name):
                setattr(out, name, getattr(self, name))
        return out

    def diagonal(self):
        return np.diagonal(self.blocks.diag, axis1=1, axis2=2).reshape(-1).copy()

    def setdiag(self, diag):
        d = np.asarray(diag, dtype=float).reshape(self.n_blocks, self.block_size)
        index = np.arange(self.block_size)
        self.blocks.diag[:, index, index] = d

    def matvec(self, vector):
        b = self.blocks
        x = np.asarray(vector, dtype=float).reshape(self.n_blocks, self.block_size)
        out = np.einsum('nij,nj->ni', b.diag, x, optimize=True)
        if self.n_blocks > 1:
            out[1:] += b.lowD*x[:-1]+np.einsum('nar,nr->na', b.lowP, np.einsum('nbr,nb->nr', b.lowQ, x[:-1]))
            out[:-1] += b.upD*x[1:]+np.einsum('nar,nr->na', b.upP, np.einsum('nbr,nb->nr', b.upQ, x[1:]))
        return out.ravel()

    def abs_row_sums(self):
        """Absolute row sums. The part without the main diagonal is cached:
        transient updates (setdiag) change only that diagonal."""
        b = self.blocks
        diagonal = abs(np.diagonal(b.diag, axis1=1, axis2=2))
        if getattr(self, '_rows_off', None) is None:
            rows = abs(b.diag).sum(axis=2)-diagonal
            if self.n_blocks > 1:
                rows[1:] += abs(b.lowD)+np.einsum('nar,nr->na', abs(b.lowP), abs(b.lowQ).sum(axis=1))
                rows[:-1] += abs(b.upD)+np.einsum('nar,nr->na', abs(b.upP), abs(b.upQ).sum(axis=1))
            self._rows_off = rows
        return (self._rows_off+diagonal).ravel()

    def dense(self):
        """Equivalent :class:`BlockTridiagJacobian` (tests and fallback)."""
        from kflame.flame.equations import BlockTridiagJacobian
        from kflame.flame.structured_jacobian import StructuredBlocks
        b = self.blocks
        out = BlockTridiagJacobian(StructuredBlocks._expand(b.lowD, b.lowP, b.lowQ), b.diag.copy(),
                                   StructuredBlocks._expand(b.upD, b.upP, b.upQ))
        out.use_compiled_substitution = True
        return out


# Largest normwise backward error accepted from a block elimination.
BACKWARD_ERROR_LIMIT = 1e-8


def block_backward_error(jmat, x, b):
    """Normwise backward error of x as a solution of ``jmat x = b``.

    The infinity norm of the matrix is bounded by its absolute row sums
    (``jmat`` offers ``matvec`` and ``abs_row_sums``).
    """
    if not np.isfinite(x).all():
        return np.inf
    residual = float(np.max(abs(jmat.matvec(x)-b)))
    scale = float(np.max(jmat.abs_row_sums()))*float(np.max(abs(x)))+float(np.max(abs(b)))
    return residual/scale if scale > 0. else (0. if residual == 0. else np.inf)


class PremixedStructuredFactor:
    """Two-ended structured factorization; pivoted dense fallback if singular."""

    def __init__(self, jmat):
        from kflame.flame.structured_linear import factor_structured
        self.jmat = jmat
        self.fallback = None
        b = jmat.blocks
        # Block elimination pivots only inside blocks: the first solve is
        # verified against the matrix and falls back to pivoted LU if needed.
        self.backward_error = None
        try:
            self.G, self.GP, self.k = factor_structured(b, np.zeros((b.n, b.nv)))
        except np.linalg.LinAlgError:
            from kflame.flame.equations import factorize
            self.G = self.GP = None
            self.fallback = factorize(jmat.dense())

    def solve(self, rhs):
        from kflame.flame.structured_linear import parallel_solve
        if self.fallback is not None:
            from kflame.flame.equations import solve_linear
            return solve_linear(self.fallback, rhs)
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.ndim == 2:
            return np.column_stack([self.solve(column) for column in rhs.T])
        out = parallel_solve(self.jmat.blocks, self.G, self.GP, self.k, rhs)
        if self.backward_error is None:
            from kflame.flame.equations import factorize, solve_linear
            self.backward_error = block_backward_error(self.jmat, out, rhs)
            if not self.backward_error <= BACKWARD_ERROR_LIMIT:
                self.G = self.GP = None
                self.fallback = factorize(self.jmat.dense())
                return solve_linear(self.fallback, rhs)
        return out


def _radiation_of(problem):
    radiation = getattr(problem, 'radiation', None)
    if radiation is None:
        return (False, 1.0, 101325.0, -1, -1, 0.0, np.ones(int(problem.n_species)))
    return radiation


def build_structured(x, problem, cache):
    """Structured analytic Jacobian of the premixed residual at ``x``."""
    from kflame.chemistry.analytic import prepare_species_partials, species_partials_range_into
    from kflame.flame.diffusion import _split, heavy_threads
    from kflame.flame.equations import _jacobian_backend, _outlet_species_flux_bc
    from kflame.flame.structured_jacobian import StructuredBlocks
    b = _jacobian_backend(problem)
    nodes, ns = int(problem.n_points), int(problem.n_species)
    nv = ns+2
    state = np.ascontiguousarray(np.asarray(x, dtype=float).reshape(nodes, nv))
    T, Y = prepare_species_partials(b, state[:, 1].copy(), np.ascontiguousarray(state[:, 2:].T))
    invW = np.asarray(b.invW, dtype=float)
    # Ideal-gas density and mixture cp partials (as thermochemical_partials).
    s = invW@Y
    rho_ig = b._P_float/(R_UNIV*T*s)
    drho = np.zeros((nodes, nv))
    drho[:, 1] = -rho_ig/T
    drho[:, 2:] = -rho_ig[:, None]*invW[None, :]/s[:, None]
    cp_molar = np.ascontiguousarray(b.thermo.partial_molar_cp(T))
    dcp = np.zeros((nodes, nv))
    dcp[:, 2:] = (cp_molar*invW[:, None]).T
    a = np.where((T[None, :] > b._nb_nasa_tmid[:, None])[:, :, None],
                 b._nb_nasa_hi[:, None, :], b._nb_nasa_lo[:, None, :])
    cp_t = R_UNIV*(a[:, :, 1]+T*(2*a[:, :, 2]+T*(3*a[:, :, 3]+4*T*a[:, :, 4])))
    dcp[:, 1] = np.sum(cp_t*Y*invW[:, None], axis=0)
    del a, cp_t
    thermal = cache['soret_face_coeff']
    thermal = (np.zeros((ns, max(nodes-1, 1))) if thermal is None
               else np.ascontiguousarray(thermal, dtype=float))
    z = np.asarray(cache['z'], dtype=float)
    coeff = np.ascontiguousarray(cache['face_coeff'], dtype=float)
    face = np.empty((nodes-1, 11, ns))
    flux = np.empty((nodes-1, ns))
    threads = heavy_threads() if ns >= 100 else None
    minimum = 2 if ns >= 100 else 32
    _split(premixed_faces, nodes-1, state, invW, z, coeff, thermal, bool(cache['basis_molar']), face, flux,
           minimum=minimum, threads=threads)
    diag = np.empty((nodes, nv, nv))
    lowD, lowP, lowQ = np.zeros((nodes-1, nv)), np.zeros((nodes-1, nv, RANK)), np.zeros((nodes-1, nv, RANK))
    upD, upP, upQ = np.zeros((nodes-1, nv)), np.zeros((nodes-1, nv, RANK)), np.zeros((nodes-1, nv, RANK))
    tile = max(1, min(nodes, _PARTIALS_TILE_BYTES//(8*ns*ns)))
    common = (state, z, np.asarray(cache['rho'], dtype=float), np.asarray(cache['cp_n'], dtype=float),
              np.ascontiguousarray(cache['omega'], dtype=float), np.ascontiguousarray(cache['hk_n'], dtype=float),
              drho, dcp)
    tail = (cp_molar, flux, face, np.asarray(cache['lam_face'], dtype=float), invW,
            np.asarray(cache['Y_in'], dtype=float), bool(cache['solve_energy']), bool(cache['has_T_prof']),
            int(cache['j_fixed']), bool(_outlet_species_flux_bc(problem)),
            float(getattr(problem, 'inlet_mass_flux', -1.0)), int(cache['convection']),
            np.ascontiguousarray(cache['conv_w'], dtype=float), np.ascontiguousarray(cache['conv_wy'], dtype=float),
            diag, 0, lowD, lowP, lowQ, upD, upP, upQ, _radiation_of(problem))

    def rows(start, stop):
        dy = np.empty((tile, ns, ns)).transpose(1, 0, 2)
        dt = np.empty((ns, tile))
        for first in range(start, stop, tile):
            last = min(first+tile, stop)
            species_partials_range_into(b, T, Y, first, last, dy, dt)
            premixed_rows_range(first, last, *common, dy, dt, first, *tail)

    _split(rows, nodes, minimum=minimum, threads=threads)
    jmat = PremixedStructuredJacobian(StructuredBlocks(diag, lowD, lowP, lowQ, upD, upP, upQ, None, None))
    jmat.use_compiled_substitution = True
    return jmat
