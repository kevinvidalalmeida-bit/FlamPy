"""Native PLOG rates: grouped pressure knots and analytic d(log k)/dT.

The packed tables are shared read-only; evaluation owns no mutable cache.
Repeated Arrhenius entries at a pressure are summed before interpolation.
"""
import math

import numpy as np
from numba import njit

from kflame.chemistry.mechanism import R_UNIV


@njit(cache=True, inline='always')
def troe_temperature_scale(value):
    """Keep signed Troe fit scales; apply the legacy zero guard only at >=0."""
    return value if value < 0. else max(value, 1e-300)


def pack_sri(mechanism):
    if not any(r.sri_parameters is not None for r in mechanism.reactions):
        return None
    parameters = np.zeros((mechanism.n_reactions, 6))
    for reaction in mechanism.reactions:
        if reaction.sri_parameters is not None:
            parameters[reaction.index] = (1., *reaction.sri_parameters)
            if (not np.isfinite(parameters[reaction.index]).all()
                    or parameters[reaction.index, 3] <= 0.
                    or parameters[reaction.index, 4] <= 0.):
                raise ValueError('SRI requires finite parameters, C > 0 and D > 0')
    parameters.setflags(write=False)
    return parameters


@njit(cache=True)
def sri_falloff(T, reduced, row):
    """Return F, d(log F)/dPr and d(log F)/dT at fixed Pr."""
    log_reduced = math.log10(max(reduced, 1e-300))
    exponent = 1./(1.+log_reduced*log_reduced)
    first = row[1]*math.exp(-row[2]/T)
    second = math.exp(-T/row[3])
    center = first + second
    if center <= 0.:
        raise ValueError('SRI requires a positive centering function')
    log_center = math.log(center)
    F = row[4]*math.exp(exponent*log_center)*T**row[5]
    derivative_pr = 0.
    if reduced > 1e-300:
        derivative_pr = -2.*log_reduced*exponent*exponent*log_center/(reduced*math.log(10.))
    derivative_t = exponent*(first*row[2]/(T*T)-second/row[3])/center + row[5]/T
    return F, derivative_pr, derivative_t


def pack_plog(mechanism):
    """Return contiguous lookup arrays, or None for the legacy rate models."""
    selected = [(r.index, r.plog_rates) for r in mechanism.reactions
                if r.plog_rates is not None]
    if not selected:
        return None
    lookup = np.full(mechanism.n_reactions, -1, dtype=np.int64)
    reaction_offsets, group_offsets = [0], [0]
    pressures, coefficients = [], []
    for slot, (index, rows) in enumerate(selected):
        lookup[index] = slot
        ordered = rows[np.argsort(rows[:, 0], kind='stable')]
        for pressure in np.unique(ordered[:, 0]):
            group = ordered[ordered[:, 0] == pressure]
            pressures.append(pressure)
            coefficients.extend(group[:, 1:])
            group_offsets.append(len(coefficients))
        reaction_offsets.append(len(pressures))
    coefficients = np.asarray(coefficients, dtype=np.float64)
    packed = (lookup, np.asarray(reaction_offsets, dtype=np.int64),
              np.asarray(pressures, dtype=np.float64),
              np.asarray(group_offsets, dtype=np.int64),
              *(np.ascontiguousarray(coefficients[:, i]) for i in range(3)))
    for array in packed:
        array.setflags(write=False)
    return packed


@njit(cache=True)
def _pressure_group(T, group, data):
    rate, derivative = 0.0, 0.0
    logT = math.log(T)
    for i in range(data[3][group], data[3][group + 1]):
        value = data[4][i] * math.exp(data[5][i]*logT - data[6][i]/(R_UNIV*T))
        rate += value
        derivative += value * (data[5][i]/T + data[6][i]/(R_UNIV*T*T))
    if rate <= 0.0 or not math.isfinite(rate):
        raise ValueError('PLOG requires a positive finite sum at each pressure knot')
    return rate, derivative/rate


@njit(cache=True)
def plog_rate(T, pressure, reaction, data):
    """Return (k, d(log k)/dT at fixed P), including endpoint clamping."""
    if T <= 0.0 or pressure <= 0.0 or not math.isfinite(T) or not math.isfinite(pressure):
        raise ValueError('PLOG requires positive finite temperature and pressure')
    slot = data[0][reaction]
    start, stop = data[1][slot], data[1][slot + 1]
    if pressure <= data[2][start]:
        return _pressure_group(T, start, data)
    if pressure >= data[2][stop - 1]:
        return _pressure_group(T, stop - 1, data)
    upper = start + 1
    while pressure > data[2][upper]:
        upper += 1
    if pressure == data[2][upper]:
        return _pressure_group(T, upper, data)
    lower = upper - 1
    k0, derivative0 = _pressure_group(T, lower, data)
    k1, derivative1 = _pressure_group(T, upper, data)
    weight = math.log(pressure/data[2][lower])/math.log(data[2][upper]/data[2][lower])
    rate = math.exp(math.log(k0) + weight*(math.log(k1) - math.log(k0)))
    return rate, derivative0 + weight*(derivative1 - derivative0)
