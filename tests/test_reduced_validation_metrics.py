"""Equal-progress state and source errors of the reduced-burner validator."""
import numpy as np

from examples.validate_against_native_references import source_errors, state_errors


class _Model:
    table = {'progress_weights': np.array([1., 0.])}


def _flame(shift):
    z = np.linspace(0., 2e-3, 801)
    C = 0.5*(1+np.tanh((z-1e-3-shift)/5e-5))
    Y = np.vstack([C*0.3, 1-C*0.3])
    return dict(z=z, T=300.+1700.*C, Y=Y, omega_C=np.exp(-((z-1e-3-shift)/5e-5)**2))


def test_a_shifted_front_is_not_a_state_error():
    reduced, reference = _flame(15e-6), _flame(0.)
    same_z = np.max(abs(reduced['T']-np.interp(reduced['z'], reference['z'], reference['T'])))
    dT, dY = state_errors(_Model(), reduced, reference)
    assert same_z > 100.          # the same flame, displaced 15 um
    assert dT < 1e-6 and dY < 1e-9
    errors = source_errors(_Model(), reduced, reference, dict(omega_C=reference['omega_C']))['omega_C']
    assert errors['integral_relative'] < 1e-6
    assert errors['peak_relative'] < 1e-3


def test_a_wrong_state_is_still_detected():
    reduced, reference = _flame(0.), _flame(0.)
    reduced['T'] = reduced['T']+25.
    dT, _ = state_errors(_Model(), reduced, reference)
    assert abs(dT-25.) < 1e-9
