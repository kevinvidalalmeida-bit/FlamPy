"""Composition tangents of the existing steady, frozen-transport rows.

The residual is unchanged. Temperature and the flow columns still use their
original differences. These columns differentiate the ideal-gas density,
mixture heat capacity, analytic rates, radiation, advection and face tangents
at fixed temperature. No subtraction of nearly equal residuals is needed.
"""
from numba import njit


@njit(cache=True)
def composition_row_derivative(out, w, n, j, q, m, V, T, Y, rho, hk, J, dz,
                               scalars, excess, fixed, radiation, drho, dcp,
                               rates, dleft, dright, muleft, muright):
    ns = Y.shape[0]
    out[:] = 0.
    g = j-2+w
    mref, aref, Tref, lref, species_scale, energy_scale = scalars[:6]
    if g == 0 or g == n-1:
        face = w if g == 0 else w-1
        for k in range(ns):
            dj = 0.
            if face == 1:
                dj = dleft[k, q]
            elif face == 2:
                dj = dright[k, q]
            out[3+k] = (dj + (m[w] if w == 2 and k == q else 0.))/mref
        out[3+excess[g]] = 1. if w == 2 else 0.
        return
    hm, hp = dz[w-1], dz[w]
    volume = .5*(hm+hp)
    if w == 2 or w-1 == 2:
        out[0] = hm*drho*V[2]/mref
    if m[w] >= 0.:
        donor, spacing, direction = w-1, hm, 1.
    else:
        donor, spacing, direction = w+1, hp, -1.
    dml = muleft[q] if w-1 == 1 else (muright[q] if w-1 == 2 else 0.)
    dmr = muleft[q] if w == 1 else (muright[q] if w == 2 else 0.)
    out[1] = (-(dmr*(V[w+1]-V[w])/hp-dml*(V[w]-V[w-1])/hm)/volume
              + (drho*V[w]**2 if w == 2 else 0.))/lref
    backward = .5+m[w]/(2.*1e-4*mref)
    energy = 0.
    for k in range(ns):
        dl = dleft[k, q] if w-1 == 1 else (dright[k, q] if w-1 == 2 else 0.)
        dr = dleft[k, q] if w == 1 else (dright[k, q] if w == 2 else 0.)
        grad = 0.
        if k == q:
            grad = direction*((1. if w == 2 else 0.)-(1. if donor == 2 else 0.))/spacing
        out[3+k] = (m[w]*grad+(dr-dl)/volume-(rates[k] if w == 2 else 0.))/species_scale
        if not fixed:
            if backward >= 1. or backward <= 0.:
                gradH = direction*(hk[k, w]-hk[k, donor])/spacing
            else:
                gradH = (backward*(hk[k, w]-hk[k, w-1])/hm
                         +(1.-backward)*(hk[k, w+1]-hk[k, w])/hp)
            energy += .5*(dl+dr)*gradH
            if w == 2:
                energy += hk[k, w]*rates[k]
    if not fixed:
        if w == 2:
            energy += m[w]*dcp*direction*(T[w]-T[donor])/spacing
            enabled, scale, P, iH2O, iCO2, background, invW = radiation
            if enabled:
                denom, numerator, direct = 0., 0., 0.
                x = 1000./T[w]
                for k in range(ns):
                    denom += Y[k, w]*invW[k]
                if iH2O >= 0:
                    a = -0.23093+x*(-1.12390+x*(9.41530+x*(-2.99880+x*(0.51382+x*-1.86840e-5))))
                    numerator += Y[iH2O, w]*invW[iH2O]*a
                    if q == iH2O:
                        direct += invW[q]*a
                if iCO2 >= 0:
                    a = 18.741+x*(-121.310+x*(273.500+x*(-194.050+x*(56.310+x*-5.8169))))
                    numerator += Y[iCO2, w]*invW[iCO2]*a
                    if q == iCO2:
                        direct += invW[q]*a
                wm = 1./denom
                energy += (scale*2.*P/101325.*(2.*5.670374419e-8*T[w]**4-background)
                           *wm*(direct-numerator*wm*invW[q]))
        out[2] = energy/energy_scale
