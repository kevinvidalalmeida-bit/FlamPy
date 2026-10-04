"""Source norms must respect units, nonuniform cells and unsupported gaps."""
import numpy as np
import pytest

from examples.audit_nonadiabatic_sources import source_metrics


def test_source_norms_use_peak_not_pointwise_relative_error():
    z=np.array([0.,.1,1.])
    truth=np.array([0.,2.,0.])
    metrics=source_metrics(z,truth,1.05*truth)
    assert metrics['Linf_over_truth_peak']==pytest.approx(.05)
    assert metrics['L1_relative']==pytest.approx(.05)
    assert metrics['integral_error_over_abs_integral']==pytest.approx(.05)
    assert metrics['L2_relative']==pytest.approx(.05)
    assert metrics['absolute_source_coverage']==pytest.approx(1.)


def test_source_integration_does_not_bridge_uncovered_gaps():
    z=np.arange(5.,dtype=float)
    truth=np.ones(5)
    covered=np.array([True,True,False,True,True])
    prediction=np.array([2.,2.,np.nan,2.,2.])
    metrics=source_metrics(z,truth,prediction,covered)
    assert metrics['absolute_source_coverage']==pytest.approx(.5)
    assert metrics['integral_error_over_abs_integral']==pytest.approx(.5)
    assert metrics['L1_relative']==pytest.approx(.5)
    assert metrics['Linf_over_truth_peak']==pytest.approx(1.)


def test_source_integral_cancellation_cannot_hide_local_errors():
    z=np.arange(3.,dtype=float)
    truth=np.ones(3)
    metrics=source_metrics(z,truth,np.array([1.1,1.,.9]))
    assert metrics['integral_error_over_abs_integral']==pytest.approx(0.,abs=1e-15)
    assert metrics['L1_relative']==pytest.approx(.05)
    assert metrics['Linf_over_truth_peak']==pytest.approx(.1)


def test_uncovered_or_invalid_profiles_cannot_report_success():
    with pytest.raises(ValueError):
        source_metrics(np.array([0.,1.]),np.ones(2),np.full(2,np.nan),np.zeros(2,dtype=bool))
    with pytest.raises(ValueError):
        source_metrics(np.array([1.,0.]),np.ones(2),np.ones(2))
