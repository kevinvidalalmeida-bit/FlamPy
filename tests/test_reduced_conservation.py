from types import SimpleNamespace
import numpy as np
import pytest
from kflame.fgm.conservation import species_conservation_diagnostics


def test_reaction_and_transport_satisfy_each_species_and_element_balance():
    mechanism=SimpleNamespace(n_species=2,species_names=['A','B'],element_names=['X'],
        atom_matrix=np.ones((1,2)),inv_molecular_weights=np.ones(2))
    # A -> B, one kilogram transferred across the two physical volumes.
    source=np.array([[0.,-1.,-2.,0.],[0.,1.,2.,0.]])
    flux=np.array([[2.,1.8,1.],[0.,.2,1.]])
    audit=species_conservation_diagnostics(flux,source,[.2,.4],[1.,0.],2.,
        np.tile([[0.],[10.]],(1,4)),mechanism,bilger_weights=[1.,1.],progress_weights=[0.,1.],energy_scale=1.)
    assert audit['maximum_species_integral_defect_over_mdot'] < 1e-15
    assert audit['maximum_species_residual_L1_over_mdot'] < 1e-15
    assert audit['maximum_relative_element_flux_defect'] < 1e-15
    assert audit['relative_omitted_species_caloric_residual_L1'] < 1e-14


def test_projected_balances_do_not_certify_full_species_or_element_conservation():
    mechanism=SimpleNamespace(n_species=5,species_names=list('ABCDE'),element_names=list('xyz'),
        atom_matrix=np.array([[1.,0.,0.,1.,0.],[0.,1.,0.,0.,1.],[0.,0.,1.,0.,0.]]),
        inv_molecular_weights=np.ones(5))
    feed=np.ones(5)/5.
    # This perturbation has zero total mass, Z and C, but transfers elements.
    defect=np.array([.01,.01,-.02,0.,0.])
    flux=np.tile((feed+defect)[:,None],(1,3))
    audit=species_conservation_diagnostics(flux,np.zeros((5,4)),[.2,.4],feed,1.,
        np.tile([[0.],[0.],[10.],[0.],[0.]],(1,4)),mechanism,
        bilger_weights=[1.,-1.,0.,1.,-1.],progress_weights=[0.,0.,0.,1.,0.],energy_scale=1.)
    assert audit['projected_integral_defects_over_mdot'] == pytest.approx(dict(Z=0.,C=0.),abs=1e-15)
    assert audit['maximum_species_integral_defect_over_mdot'] == pytest.approx(.02)
    assert audit['maximum_relative_element_flux_defect'] == pytest.approx(.1)
    assert audit['inlet_species_face_enthalpy_defect_W_m2'] == pytest.approx(-.2)
