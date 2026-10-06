"""Common progress must cover slow chemistry without hiding reversals."""
import numpy as np
import pytest
from kflame.fgm.progress_selection import select_progress_weights


def test_selection_covers_a_turning_intermediate_and_the_burned_source():
    shape=np.array([2,2,3])
    curve=np.array([[1.,0.,0.],[.5,.45,.05],[.05,.25,.7]])
    Y=np.tile(curve,(4,1))
    source=np.tile([-.01,-.02,.03],(len(Y),1))
    weights,audit=select_progress_weights(Y,source,shape)
    assert np.all(np.diff(curve@weights)>0.)
    assert source[0]@weights>0.
    assert audit['minimum_normalized_margin']>0.
    assert weights.sum()==pytest.approx(1.)
    # Changing local reaction time scales must not hide a slow reversal.
    scaled=source.copy().reshape(*shape,3)
    scaled[0,0]*=1e-12
    other,_=select_progress_weights(Y,scaled.reshape(-1,3),shape,normalization=8.)
    np.testing.assert_allclose(other,8.*weights,rtol=0.,atol=1e-10)


def test_opposing_physical_directions_are_reported_as_incompatible():
    shape=np.array([2,2,3])
    Y=np.tile([[1.,0.,0.],[.5,.45,.05],[.05,.25,.7]],(4,1))
    source=np.tile([-.01,-.02,.03],(len(Y),1))
    source.reshape(*shape,3)[0,0,-1]*=-1.
    with pytest.raises(ValueError,match='No strictly increasing'):
        select_progress_weights(Y,source,shape)


@pytest.mark.parametrize('reference,expected', [([0.,1.],[0.,1.]),([1.,0.],[.499999,.500001])])
def test_nearest_selection_preserves_feasible_progress_and_minimally_repairs_a_reversal(reference,expected):
    shape=np.array([2,2,3])
    Y=np.tile([[1.,0.],[.5,.5],[0.,1.]],(4,1))
    source=np.tile([-1.,1.],(len(Y),1))
    weights,audit=select_progress_weights(Y,source,shape,reference_weights=reference)
    np.testing.assert_allclose(weights,expected,atol=1e-8,rtol=0.)
    assert audit['minimum_normalized_margin']>0.
    assert audit['method']=='nonnegative_linear_progress_nearest_reference_L1'
