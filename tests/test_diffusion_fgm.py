import numpy as np
import pytest

from kflame.chemistry.mechanism import load_mechanism,resolve_mechanism
from kflame.chemistry.thermo import NativeThermo
from kflame.fgm.diffusion import build_diffusion_manifold,DiffusionFGM
from kflame.fgm.nonadiabatic3d import OutsideManifoldError


def synthetic_family():
    mech=load_mechanism(resolve_mechanism('h2o2.yaml'));thermo=NativeThermo(mech)
    z=np.linspace(0.,1.,9);Z=1.-z
    profiles=[]
    for scale in (.1,.2,.3):
        Y=np.zeros((mech.n_species,len(z)))
        reaction=scale*Z*(1.-Z)
        Y[mech.species_names.index('H2')]=.1*Z-reaction*.1
        Y[mech.species_names.index('O2')]=.2*(1.-Z)-reaction*.2
        Y[mech.species_names.index('H2O')]=reaction*.3
        Y[mech.species_names.index('N2')]=1.-Y.sum(axis=0)
        T=300.+reaction*1000.
        profiles.append(dict(z=z,Z=Z,Y=Y,T=T,h=thermo.enthalpy_mass(T,Y),
            omega_Y=np.zeros_like(Y),qdot=np.zeros(len(z)),species_names=np.array(mech.species_names)))
    return profiles,dict(mechanism=resolve_mechanism('h2o2.yaml'),pressure=101325.)


def test_connected_lookup_conserves_mass_and_total_enthalpy(tmp_path):
    profiles,metadata=synthetic_family()
    report=build_diffusion_manifold(profiles,metadata,tmp_path,mixture_points=9,progress_weights={'H2O':1.})
    f=DiffusionFGM(tmp_path)
    points=f.table['controls'][f.table['cells']].mean(axis=1)
    result=f.lookup_batch(Z=points[:,0],C=points[:,1])
    np.testing.assert_allclose(result['Y'].sum(axis=0),1.,atol=1e-14)
    np.testing.assert_allclose(f.thermo.enthalpy_mass(result['T'],result['Y']),result['h'],atol=1e-6)
    np.testing.assert_allclose(result['Y'][list(f.table['species_names']).index('H2O')],points[:,1],atol=1e-14)
    assert report['geometry']['excluded_folded_cells']==0
    with pytest.raises(OutsideManifoldError):
        f.lookup(Z=.5,C=1.)


def test_progress_selection_is_generic(tmp_path):
    profiles,metadata=synthetic_family()
    report=build_diffusion_manifold(profiles,metadata,tmp_path,mixture_points=9)
    assert report['geometry']['triangles']>0
    assert sum(abs(v) for v in report['progress_weights'].values())==pytest.approx(1.)


def test_differential_diffusion_overshoot_is_excluded_without_clipping(tmp_path):
    profiles,metadata=synthetic_family()
    for profile in profiles:
        profile['Z'][0]=1.001
    report=build_diffusion_manifold(profiles,metadata,tmp_path,mixture_points=9,
                                   progress_weights={'H2O':1.})
    assert report['geometry']['excluded_spatial_points_outside_feed_Z']==[1,1,1]
    fgm=DiffusionFGM(tmp_path)
    assert fgm.table['controls'][:,0].max()==.875
    with pytest.raises(OutsideManifoldError):
        fgm.lookup(Z=1.,C=0.)


def test_folds_in_spatial_mixture_coordinate_are_rejected(tmp_path):
    profiles,metadata=synthetic_family()
    profiles[0]['Z'][3]=profiles[0]['Z'][2]+.05
    with pytest.raises(ValueError,match='folds'):
        build_diffusion_manifold(profiles,metadata,tmp_path,mixture_points=9)


def test_frozen_table_tampering_is_rejected(tmp_path):
    profiles,metadata=synthetic_family()
    build_diffusion_manifold(profiles,metadata,tmp_path,mixture_points=9,progress_weights={'H2O':1.})
    with (tmp_path/'diffusion_fgm.npz').open('ab') as target:
        target.write(b'changed')
    with pytest.raises(ValueError,match='fingerprint'):
        DiffusionFGM(tmp_path)


def test_withheld_validation_reports_coverage_and_interpolated_sources(tmp_path):
    import hashlib,json
    from pathlib import Path
    from kflame.fgm.diffusion import validate_diffusion_fgm
    profiles,metadata=synthetic_family()
    for profile in profiles:
        profile['qdot']=1000.*profile['Z']*(1.-profile['Z'])
    report=build_diffusion_manifold([profiles[0],profiles[2]],metadata,tmp_path/'table',
                                  mixture_points=9,progress_weights={'H2O':1.})
    native=tmp_path/'withheld';native.mkdir()
    np.savez(native/'flame.npz',**profiles[1])
    metadata.update(accepted=True,flow_type='counterflow_diffusion',
                    mechanism_sha256=hashlib.sha256(Path(metadata['mechanism']).read_bytes()).hexdigest())
    (native/'metadata.json').write_text(json.dumps(metadata))
    result=validate_diffusion_fgm(tmp_path/'table',native,tmp_path/'validation')
    assert result['passed'] and result['reaction_coverage']==1.
    assert result['total_points']==len(profiles[1]['Z'])
    assert result['metrics']['species_source_relative']==0.
    metadata['transport']='other'
    (native/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='transport'):
        validate_diffusion_fgm(tmp_path/'table',native,tmp_path/'wrong')
