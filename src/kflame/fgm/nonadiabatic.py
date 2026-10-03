"""Burner FGM at fixed inlet composition, with physical controls (c, h).

Rows retain their different progress limits. Lookup never fills an unreachable
part of a cooled flamelet or extrapolates across a missing row.
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.thermo import NativeThermo


def tabulate_burner_family(results, mech, inlet_Y, weights, progress_points):
    """Adiabatic reference first; burner rows ordered by increasing cooling."""
    thermo = NativeThermo(mech)
    kinetics = NativeKinetics(mech)
    beta_u = float(weights @ inlet_Y)
    span = float(weights @ results[0]['Y'][:, -1] - beta_u)
    if not np.isfinite(span) or span <= 1e-10:
        raise ValueError('Progress weights must give a positive adiabatic progress span')
    # Cooling changes equilibrium composition: c may exceed its adiabatic end
    # value of one. Preserve those states rather than stretching or clipping.
    max_c = max(float(np.max((weights @ r['Y'] - beta_u) / span)) for r in results)
    c_grid = .5 * max_c * (1. - np.cos(np.linspace(0., np.pi, progress_points)))
    one_index = int(np.argmin(abs(c_grid[:-1] - 1.))) if max_c > 1. else progress_points - 1
    c_grid[one_index] = 1.
    c_grid.sort()
    n_rows = len(results)
    shape = (n_rows, progress_points)
    table = {name: np.full(shape, np.nan) for name in
             ('T', 'h', 'delta_h', 'rho', 'cp_mass', 'conductivity', 'qdot', 'omega_c')}
    table['Y'] = np.full((*shape, mech.n_species), np.nan)
    table['valid'] = np.zeros(shape, dtype=bool)
    for row, result in enumerate(results):
        c = (weights @ result['Y'] - beta_u) / span
        if np.any(np.diff(c) < -1e-6):
            raise ValueError(f'Flamelet {row}: progress is nonmonotone; choose different weights')
        # Keep the last sample of repeated plateaus, including the burned state.
        keep = []
        for j, value in enumerate(c):
            if keep and value <= c[keep[-1]] + 1e-12:
                keep[-1] = j
            else:
                keep.append(j)
        cx = c[keep]
        source_y = result['Y'][:, keep].copy()
        source_T = result['T'][keep].copy()
        if row == 0 and abs(c[0]) < 1e-8:
            # Exact fresh reference, avoiding numerical inlet trace products.
            cx[0], source_y[:, 0], source_T[0] = 0., inlet_Y, result['T'][0]
        if cx.size < 2 or np.any(np.diff(cx) <= 0):
            raise ValueError(f'Flamelet {row}: degenerate progress parameterization')
        valid = (c_grid >= cx[0]) & (c_grid <= cx[-1])
        table['valid'][row] = valid
        target = c_grid[valid]
        # Linear species interpolation preserves elemental mass and the chosen c.
        y = np.array([np.interp(target, cx, species) for species in source_y])
        T = np.interp(target, cx, source_T)
        table['Y'][row, valid] = y.T
        table['T'][row, valid] = T
        table['h'][row, valid] = thermo.enthalpy_mass(T, y)
        table['conductivity'][row, valid] = np.interp(target, cx, result['conductivity'][keep])
        # Evaluate the reduced source from native chemistry at tabulated states.
        pressure = float(result['pressure'])
        rho = thermo.density(T, pressure, y)
        table['rho'][row, valid] = rho
        table['cp_mass'][row, valid] = thermo.cp_mass(T, y)
        concentrations = rho[None, :] * y * mech.inv_molecular_weights[:, None]
        omega = kinetics.net_production_rates(T, concentrations, thermo.g_RT(T))
        table['omega_c'][row, valid] = weights @ (omega * mech.molecular_weights[:, None]) / span
        table['qdot'][row, valid] = -np.sum(thermo.partial_molar_enthalpies(T) * omega, axis=0)
    reference_valid = table['valid'][0]
    h_reference = PchipInterpolator(c_grid[reference_valid], table['h'][0, reference_valid], extrapolate=False)(c_grid)
    for row in range(n_rows):
        table['delta_h'][row] = h_reference - table['h'][row]
    # Two controls must distinguish adjacent trajectories throughout their overlap.
    for row in range(n_rows - 1):
        overlap = table['valid'][row] & table['valid'][row + 1]
        difference = table['h'][row, overlap] - table['h'][row + 1, overlap]
        if difference.size < 2 or np.any(difference <= 0.):
            raise ValueError('Adjacent flamelets cross in (c,h) or have no resolved overlap; refine the family or change controls')
    for name in ('T', 'h', 'Y', 'rho', 'cp_mass', 'conductivity', 'qdot', 'omega_c'):
        if not np.isfinite(table[name][table['valid']]).all():
            raise ValueError(f'Nonfinite {name} in resolved burner states')
    table.update(c=c_grid, h_reference=h_reference, progress_weights=weights,
                 beta_unburned=np.array(beta_u), beta_span=np.array(span),
                 species_names=np.asarray(mech.species_names))
    return table


class BurnerFGM:
    """Load a fixed-composition burner table and look up physical (c, h).

    Interpolation uses adjacent resolved flamelets only. Temperature is
    recovered from h and the interpolated species using NASA thermodynamics.
    Values outside the sampled reacting manifold raise ValueError.
    """
    def __init__(self, path):
        folder = Path(path)
        with np.load(folder / 'burner_fgm.npz', allow_pickle=False) as saved:
            self.table = {name: saved[name] for name in saved.files}
        self.metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        mech = load_mechanism(self.metadata['mechanism'])
        if list(self.table['species_names']) != list(mech.species_names):
            raise ValueError('Table species order differs from its mechanism')
        self.thermo = NativeThermo(mech)
        self.max_temperature = 2. * float(mech.max_temperature)

    def lookup(self, *, c, h):
        """Scalar c and total mass enthalpy h [J/kg]; no clipping or extrapolation."""
        c, h = float(c), float(h)
        if not np.isfinite(c) or not np.isfinite(h):
            raise ValueError('c and h must be finite')
        table, grid = self.table, self.table['c']
        if c < grid[0] or c > grid[-1]:
            raise ValueError('Progress outside the sampled manifold')
        right = min(int(np.searchsorted(grid, c)), grid.size - 1)
        left = right if c == grid[right] else max(0, right - 1)
        fraction = 0. if right == left else (c - grid[left]) / (grid[right] - grid[left])
        valid = table['valid'][:, left] & table['valid'][:, right]
        def along(name):
            return (1. - fraction) * table[name][:, left] + fraction * table[name][:, right]
        hh = along('h')
        candidates = [row for row in range(hh.size - 1)
                      if valid[row] and valid[row + 1] and hh[row + 1] <= h <= hh[row]]
        exact = np.flatnonzero(valid & np.isclose(hh, h, rtol=1e-12, atol=1e-7))
        if exact.size:
            row, other, cooling = int(exact[0]), int(exact[0]), 0.
        elif candidates:
            row = candidates[0]
            other = row + 1
            cooling = (hh[row] - h) / (hh[row] - hh[other])
        else:
            raise ValueError('Enthalpy/progress outside adjacent resolved flamelets')
        def blend(name):
            values = along(name)
            return (1. - cooling) * values[row] + cooling * values[other]
        Y = blend('Y')
        T = brentq(lambda temperature: float(self.thermo.enthalpy_mass(temperature, Y)) - h,
                   200., self.max_temperature, xtol=1e-8)
        h_ad = np.interp(c, grid, table['h_reference'])
        pressure = self.metadata['pressure_Pa']
        return dict(c=c, h=h, delta_h=float(h_ad - h) if np.isfinite(h_ad) else None, T=T, Y=Y,
                    omega_c=float(blend('omega_c')), qdot=float(blend('qdot')),
                    conductivity=float(blend('conductivity')),
                    rho=float(self.thermo.density(T, pressure, Y)),
                    cp_mass=float(self.thermo.cp_mass(T, Y)))
