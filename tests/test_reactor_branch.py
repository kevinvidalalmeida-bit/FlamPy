"""An intermediate-species progress reverses before full equilibrium."""
from types import SimpleNamespace
import numpy as np
from examples.extend_reduced_fgm_tail import reactor_tail


def test_native_branch_stops_at_real_turn_without_inserting_equilibrium():
    # Analytic A -> B -> D, rates 1 and 0.5: B peaks at 0.4225.
    zeros=lambda T:np.zeros((3,)+np.shape(T))
    thermo=SimpleNamespace(density=lambda T,P,Y:np.ones(np.shape(T)),
        g_RT=zeros,partial_molar_enthalpies=zeros,cp_mass=lambda T,Y:np.full(np.shape(T),1000.))
    model=SimpleNamespace(table={'progress_weights':np.array([0.,1.,0.])},metadata={'pressure_Pa':101325.},thermo=thermo)
    mech=SimpleNamespace(inv_molecular_weights=np.ones(3),molecular_weights=np.ones(3),atom_matrix=np.ones((1,3)))
    kinetics=SimpleNamespace(net_production_rates=lambda T,c,g:np.array([-c[0],c[0]-.5*c[1],.5*c[1]]))
    initial=np.array([.5,.3,.2]);equilibrium=np.array([0.,0.,1.])
    Y,audit=reactor_tail(model,mech,kinetics,1000.,initial,0.,equilibrium,16,endpoint_policy='monotone_branch')
    assert audit['endpoint_reason']=='first_resolved_progress_turn'
    np.testing.assert_allclose(Y[-1,1],.4225,atol=5e-9)
    np.testing.assert_allclose(audit['chemical_time_s'],-2*np.log(.65),atol=2e-8)
    assert np.all(np.diff(np.r_[initial[1],Y[:,1]])>0.)
    np.testing.assert_allclose(Y.sum(axis=1),1.,atol=1e-12)
    assert np.max(abs(Y[-1]-equilibrium))>.1
