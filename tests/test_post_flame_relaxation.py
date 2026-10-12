"""Progress coordinates with a post-flame relaxation to equilibrium (high pressure)."""
from types import SimpleNamespace

import numpy as np
import pytest

from kflame.fgm.adaptive import _check_progress_coordinate
from kflame.fgm.common import POST_FLAME_RELAXATION, post_flame_relaxation


def test_monotone_progress_has_no_relaxation():
    assert post_flame_relaxation(np.linspace(0., 1., 50)) == 0.


def test_overshoot_that_relaxes_to_the_end_value_is_accepted():
    # Stoichiometric CH4/air at 20 atm: the progress peaks 6.4e-4 above its
    # equilibrium end value and relaxes monotonically to it.
    c = np.r_[np.linspace(0., 1.00064, 40), np.linspace(1.00064, 1., 15)[1:]]
    assert post_flame_relaxation(c) == pytest.approx(6.4e-4)
    _check_progress_coordinate(SimpleNamespace(beta=.1+.2*c, phi=1.))


@pytest.mark.parametrize('c', [
    np.r_[np.linspace(0., .5, 10), .49, np.linspace(.5, 1., 10)],            # decrease inside the flame
    np.r_[np.linspace(0., 1.+2*POST_FLAME_RELAXATION, 20), 1.],              # relaxation too large
    np.r_[np.linspace(0., 1.002, 20), .99, 1.],                              # falls below the end value
])
def test_nonmonotone_progress_is_rejected(c):
    assert post_flame_relaxation(c) is None
    with pytest.raises(RuntimeError):
        _check_progress_coordinate(SimpleNamespace(beta=c, phi=1.))


def test_rounding_noise_is_tolerated():
    c = np.linspace(0., 1., 30)
    c[3] -= 3e-10
    assert post_flame_relaxation(c) is not None
