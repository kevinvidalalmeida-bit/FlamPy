"""CPU NASA-7 mixture enthalpy and derivative, without species temporaries."""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def species_cp_h_RT(temperatures, low, high, midpoints, want_cp, want_h):
    """Per-species cp/R and h/(RT), shaped (nodes, species) like the NumPy path.

    Same Horner order and branch selection as NativeThermo, with no
    (nodes, species, 7) coefficient temporary; IEEE arithmetic is kept.
    """
    nodes, species = len(temperatures), len(midpoints)
    cp = np.empty((nodes if want_cp else 0, species))
    h = np.empty((nodes if want_h else 0, species))
    for node in range(nodes):
        T = temperatures[node]
        for k in range(species):
            c = high[k] if T > midpoints[k] else low[k]
            if want_cp:
                cp[node, k] = c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4])))
            if want_h:
                h[node, k] = c[0]+T*(c[1]/2.+T*(c[2]/3.+T*(c[3]/4.+T*c[4]/5.)))+c[5]/T
    return cp, h


@njit(cache=True, nogil=True)
def enthalpy_cp_mass(temperatures, fractions, low, high, midpoints, inverse_weights, gas_constant):
    """Broadcast one temperature/composition or evaluate matching profiles.

    Each species selects its own polynomial branch. IEEE arithmetic is kept;
    chemical formation enthalpy is included in h and excluded from cp.
    """
    nodes = max(len(temperatures), fractions.shape[1])
    if len(temperatures) not in (1, nodes) or fractions.shape[1] not in (1, nodes):
        raise ValueError('Temperature and composition profiles must broadcast')
    h, cp = np.empty(nodes), np.empty(nodes)
    for node in range(nodes):
        T = temperatures[0 if len(temperatures) == 1 else node]
        j = 0 if fractions.shape[1] == 1 else node
        enthalpy, capacity = 0., 0.
        for k in range(len(inverse_weights)):
            c = high[k] if T > midpoints[k] else low[k]
            cp_R = c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4])))
            h_RT = c[0]+T*(c[1]/2.+T*(c[2]/3.+T*(c[3]/4.+T*c[4]/5.)))+c[5]/T
            enthalpy += fractions[k,j]*(h_RT*gas_constant*T)*inverse_weights[k]
            capacity += fractions[k,j]*(cp_R*gas_constant)*inverse_weights[k]
        h[node], cp[node] = enthalpy, capacity
    return h, cp
