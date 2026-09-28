"""Manufactured profiles for construction-only coordinate screening."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import compare_fgm_progress as progress

SETTINGS=dict(c_fine=101,indicator_species='CO2,CO',indicator_weight_grad=1.,
              indicator_weight_conc=.6,indicator_weight_temp=.8,indicator_weight_qdot=.4,refine_bias=5.)


def profiles():
    c=np.linspace(0,1,31);result=[]
    for phi,z in [(.8,.03),(1.,.05),(1.2,.07)]:
        y=np.array([.1*c+z,.15*c,.01*c,.002*c,.001*c])
        beta=y[0]+y[1]+y[2]+.5*y[3]
        result.append(SimpleNamespace(phi=phi,Z=z,Y=y,beta=beta,T=300+1000*c+z,
            qdot=1e6*c+z,species_names=np.array(['CO2','H2O','CO','H2','OH']),
            solve_ok=True,final_accepted=True))
    return result


def test_affine_fields_reconstruct_for_all_candidates():
    result=progress.screen(profiles(),SETTINGS,n_c=31)
    assert len(result['summary'])==4
    assert len(result['folds'])==20
    for r in result['summary']:
        assert r['status']=='screened'
        assert r['loo_rows']==1
        assert r['loo_worst_p95_percent']<1e-10


def test_fold_indicator_never_receives_removed_profile(monkeypatch):
    original=progress.build_global_indicator;calls=[]
    def checked(records,*args):
        calls.append([r.phi for r in records])
        assert all(r.phi!=1. for r in records)
        return original(records,*args)
    monkeypatch.setattr(progress,'build_global_indicator',checked)
    progress.screen(profiles(),SETTINGS,n_c=31)
    assert len(calls)==4


def test_invalid_coordinate_is_not_sorted_into_validity():
    rs=profiles()
    rs[1].Y[0,15]=rs[1].Y[0,14]-.01
    result=progress.screen(rs,SETTINGS,n_c=31)
    c1=result['summary'][0]
    assert c1['inverted_flames']==1
    assert c1['status']=='invalid_coordinate'
    assert c1['loo_worst_p95_percent'] is None
    assert not [r for r in result['folds'] if r['candidate']=='C1']


def test_two_profiles_cannot_supply_internal_row_validation():
    result=progress.screen(profiles()[:2],SETTINGS)
    assert all(r['status']=='insufficient_rows' for r in result['summary'])
    assert not result['folds']


def test_reject_unaccepted_input():
    rs=profiles();rs[0].final_accepted=False
    with pytest.raises(ValueError,match='accepted'):progress.screen(rs,SETTINGS)
