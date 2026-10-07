"""Mechanism admission and kinetics regressions against an independent oracle."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.mechanism import load_mechanism,resolve_mechanism
from kflame.chemistry.thermo import NativeThermo


def mechanism_file(tmp_path,transform):
    data=yaml.safe_load(Path(resolve_mechanism('h2o2.yaml')).read_text(encoding='utf-8'))
    data['reactions']=[dict(equation='H + O <=> OH',**{'rate-constant':dict(A=1e8,b=0.,Ea=1000.)})]
    transform(data)
    path=tmp_path/'mechanism.yaml';path.write_text(yaml.safe_dump(data),encoding='utf-8')
    return path


@pytest.mark.parametrize('variant',['default_units','time_minutes','default_efficiency','implicit_third_body','troe_zero'])
def test_admitted_kinetics_match_cantera(tmp_path,variant):
    ct=pytest.importorskip('cantera')
    def transform(data):
        reaction=data['reactions'][0]
        if variant=='default_units':
            data.pop('units')
        elif variant=='time_minutes':
            data['units']['time']='min'
        elif variant in ('default_efficiency','implicit_third_body'):
            reaction['equation']='H + O + M <=> OH + M'
            if variant=='default_efficiency':
                reaction.update(type='three-body',**{'default-efficiency':0.,'efficiencies':{'H2':2.}})
        else:
            reaction.pop('rate-constant')
            reaction.update(equation='H + O (+M) <=> OH (+M)',type='falloff',
                **{'high-P-rate-constant':dict(A=1e8,b=0.,Ea=0.),
                   'low-P-rate-constant':dict(A=1e12,b=0.,Ea=0.),
                   'Troe':dict(A=.5,T3=100.,T1=1000.,T2=0.)})
    path=mechanism_file(tmp_path,transform)
    gas=ct.Solution(str(path));mech=load_mechanism(path)
    thermo=NativeThermo(mech);kinetics=NativeKinetics(mech)
    for temperature,pressure in ((700.,101325.),(1100.,202650.),(1800.,101325.)):
        gas.TPX=temperature,pressure,'H2:.4,O2:.2,N2:.389,H:.005,O:.005,OH:.001'
        concentrations=thermo.density(temperature,pressure,gas.Y)*gas.Y*mech.inv_molecular_weights
        actual=kinetics.net_production_rates(temperature,concentrations,thermo.g_RT(temperature))
        np.testing.assert_allclose(actual,gas.net_production_rates,rtol=2e-11,atol=1e-12)
        np.testing.assert_allclose(mech.atom_matrix@actual,0.,atol=1e-9)


@pytest.mark.parametrize('unsupported',['orders','SRI','Tsang','nested_units','reference_pressure','NASA9'])
def test_unsupported_physics_is_rejected_instead_of_silently_changed(tmp_path,unsupported):
    def transform(data):
        reaction=data['reactions'][0]
        if unsupported=='orders':
            reaction['orders']={'H':.5}
        elif unsupported in ('SRI','Tsang'):
            reaction[unsupported]={'A':.5,'B':100.,'C':1000.}
        elif unsupported=='nested_units':
            reaction['units']={'quantity':'mol'}
        elif unsupported=='reference_pressure':
            data['species'][0]['thermo']['reference-pressure']='1 bar'
        else:
            data['species'][0]['thermo']['model']='NASA9'
    path=mechanism_file(tmp_path,transform)
    with pytest.raises(ValueError,match='supported|requires|NASA7'):
        load_mechanism(path)


def test_single_nasa7_region_matches_cantera(tmp_path):
    ct=pytest.importorskip('cantera')
    def transform(data):
        for species in data['species']:
            thermo=species['thermo']
            thermo['data']=[copy.deepcopy(thermo['data'][0])]
            thermo['temperature-ranges']=thermo['temperature-ranges'][:2]
    path=mechanism_file(tmp_path,transform);mech=load_mechanism(path)
    thermo=NativeThermo(mech);gas=ct.Solution(str(path))
    for temperature in (400.,800.,1000.):
        gas.TP=temperature,101325.
        np.testing.assert_allclose(thermo.cp_R(temperature),gas.standard_cp_R,rtol=2e-12)
        np.testing.assert_allclose(thermo.partial_molar_enthalpies(temperature),gas.partial_molar_enthalpies,rtol=2e-12)
