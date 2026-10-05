"""Boundary refinement must resolve weak gradients without losing old nodes."""
from types import SimpleNamespace
import numpy as np
from kflame.flame.burner_grid import refine_burner_boundary_grid


def test_thermal_peclet_bound_retains_nodes_and_is_idempotent():
    z=np.array([0.,4e-5,8e-5,1.2e-4,3e-3])
    thermo=SimpleNamespace(cp_mass=lambda T,Y:1000.)
    transport=SimpleNamespace(mech=SimpleNamespace(molecular_weights=np.array([28.])),
        eval_faces_poly_fast=lambda T,P,Y,invW:(None,None,np.array([.025]),None))
    refined=refine_burner_boundary_grid(z,np.full(5,300.),np.ones((1,5)),.1,101325.,thermo,transport)
    assert len(refined)>len(z) and all(np.any(refined==point) for point in z)
    assert np.all(np.diff(refined)>0.)
    cold=refined[refined<=z[3]]
    assert (.1*1000.*np.diff(cold)/.025).max()<=.02*(1.+1e-14)
    other=refine_burner_boundary_grid(refined,np.full(len(refined),300.),np.ones((1,len(refined))),.1,101325.,thermo,transport)
    np.testing.assert_array_equal(other,refined)
