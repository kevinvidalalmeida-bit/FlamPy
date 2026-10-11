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


def synthetic_heat_loss_family(strains=(.1,.2,.3),losses=(0.,.3,.6)):
    mech=load_mechanism(resolve_mechanism('h2o2.yaml'));thermo=NativeThermo(mech)
    names=mech.species_names
    z=np.linspace(0.,1.,41);Z=1.-z
    rows=[]
    for strain in strains:
        row=[]
        for loss in losses:
            reaction=(.3-strain)*4.*Z*(1.-Z)
            Y=np.zeros((mech.n_species,len(z)))
            Y[names.index('H2')]=.1*Z-reaction*.05
            Y[names.index('O2')]=.2*(1.-Z)-reaction*.1
            Y[names.index('H2O')]=reaction*.15
            Y[names.index('N2')]=1.-Y.sum(axis=0)
            # A loss level cools the hot region only: h decreases with loss.
            T=300.+reaction*1500.*(1.-loss)
            row.append(dict(z=z,Z=Z,Y=Y,T=T,h=thermo.enthalpy_mass(T,Y),omega_Y=np.zeros_like(Y),
                            qdot=1000.*reaction,species_names=np.array(names)))
        rows.append(row)
    return rows,dict(mechanism=resolve_mechanism('h2o2.yaml'),pressure=101325.,
                     fuel_Y=[1. if name=='H2' else 0. for name in names])


def test_heat_loss_table_interpolates_connected_strain_and_loss_cells(tmp_path):
    from kflame.fgm.diffusion import build_diffusion_heat_loss_manifold,DiffusionHeatLossFGM
    rows,metadata=synthetic_heat_loss_family()
    meta=build_diffusion_heat_loss_manifold(rows,metadata,tmp_path,mixture_points=21)
    assert meta['controls']==['Z','C','h'] and meta['geometry']['excluded_folded_cells']==0
    assert meta['progress_rule']=='products:H2O'  # oxidation product absent from the fuel
    f=DiffusionHeatLossFGM(tmp_path)
    points=f.table['controls'][f.table['cells']].mean(axis=1)
    result=f.lookup_batch(Z=points[:,0],C=points[:,1],h=points[:,2])
    assert result['covered'].all()
    np.testing.assert_allclose(result['Y'].sum(axis=0),1.,atol=1e-12)
    np.testing.assert_allclose(f.thermo.enthalpy_mass(result['T'],result['Y']),points[:,2],atol=1e-6)
    # Cooled states lie below the adiabatic reference chart at the same (Z,C).
    lossy=np.isfinite(result['delta_h'])
    assert lossy.any() and np.nanmin(result['delta_h'])>-1e-6*np.ptp(f.table['h'])
    masked=f.lookup_batch(Z=[.5],C=[1.],h=[0.],outside='mask')
    assert not masked['covered'][0] and np.isnan(masked['T'][0])
    with pytest.raises(OutsideManifoldError):
        f.lookup(Z=.5,C=1.,h=0.)


def test_adaptive_mixture_grid_bounds_the_sampling_error_of_every_flamelet():
    from kflame.fgm.diffusion import _mixture_columns
    rows,_=synthetic_heat_loss_family()
    flat=[profile for row in rows for profile in row]
    union,_,_,_,_=_mixture_columns(flat,11)
    grid,fields,_,_,_=_mixture_columns(flat,11,1e-3)
    assert 11<=len(grid)<=len(union)
    for profile,T in zip(flat,fields['T']):
        rebuilt=np.interp(profile['Z'][::-1],grid,T)
        assert abs(rebuilt-profile['T'][::-1]).max()<=1e-3*np.ptp([p['T'] for p in flat])+1e-9


def test_heat_loss_family_rejects_unordered_or_nonadiabatic_levels(tmp_path):
    from kflame.fgm.diffusion import generate_diffusion_heat_loss_fgm
    with pytest.raises(ValueError,match='radiation_scales'):
        generate_diffusion_heat_loss_fgm(radiation_scales=(1.,2.),output=tmp_path/'a')
    with pytest.raises(ValueError,match='radiation_scales'):
        generate_diffusion_heat_loss_fgm(radiation_scales=(0.,2.,2.),output=tmp_path/'b')
    with pytest.raises(ValueError,match='controls'):
        generate_diffusion_heat_loss_fgm(radiation=True,output=tmp_path/'c')
    assert not (tmp_path/'a').exists()


def test_native_hydrogen_heat_loss_table_predicts_a_withheld_strain_and_loss(tmp_path):
    from kflame import solve_diffusion_flame
    from kflame.fgm.diffusion import generate_diffusion_heat_loss_fgm,validate_diffusion_heat_loss_fgm
    base=dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15,max_time=120.)
    family=generate_diffusion_heat_loss_fgm(**base,flow_scales=(.9,1.1),radiation_scales=(0.,15.,30.),
                                            output=tmp_path/'table')
    meta=family['metadata']
    assert meta['geometry']['excluded_folded_cells']==0 and meta['maximum_enthalpy_deficit_J_kg']>1e5
    withheld=solve_diffusion_flame(**{**base,'fuel_mass_flux':.05,'oxidizer_mass_flux':.15},radiation=True,
                                   radiation_scale=22.,initial_solution=family['flamelets'][0][1]['output'],
                                   output=tmp_path/'withheld')
    report=validate_diffusion_heat_loss_fgm(tmp_path/'table',withheld['output'],tmp_path/'validation')
    assert report['passed'],report['metrics']
    assert report['reaction_coverage']>=.99 and report['metrics']['temperature_K']<10.


def test_parallel_heat_loss_rows_build_the_serial_table(tmp_path):
    from kflame.fgm.diffusion import generate_diffusion_heat_loss_fgm
    base=dict(mechanism='h2o2.yaml',fuel='H2',fuel_mass_flux=.05,oxidizer_mass_flux=.15,max_time=120.,
              flow_scales=(.9,1.1),radiation_scales=(0.,15.,30.))
    serial=generate_diffusion_heat_loss_fgm(**base,output=tmp_path/'serial')
    parallel=generate_diffusion_heat_loss_fgm(**base,output=tmp_path/'parallel',jobs=2)
    assert parallel['metadata']['table_sha256']==serial['metadata']['table_sha256']
    with pytest.raises(ValueError,match='jobs'):
        generate_diffusion_heat_loss_fgm(**base,output=tmp_path/'bad',jobs=0)


def _profile(Z,T):
    n=len(Z)
    Y=np.vstack([Z,1.-Z])
    return dict(Z=np.asarray(Z,float),T=np.asarray(T,float),h=np.asarray(T,float)*1e3,qdot=np.zeros(n),
                Y=Y,omega_Y=np.zeros_like(Y))


def test_adaptive_mixture_sampling_bounds_the_stored_table_error():
    # Review P2: one flamelet spans [0,1], the other [0.1,0.9]. The table
    # samples each full profile; the bound must hold for what is stored.
    from kflame.fgm.diffusion import _mixture_columns
    # The wide flamelet changes steeply between its knot at Z=0, outside the
    # shared domain, and its first inside knot at Z=0.3: the table value at
    # Z=0.1 comes from that steep segment, not from the first inside knot.
    wide=np.array([1.,.8,.5,.3,0.])
    narrow=np.linspace(.9,.1,9)
    profiles=[_profile(wide,[300.,300.,300.,300.,2000.]),
              _profile(narrow,300.+1000.*narrow)]  # linear: needs no inserted knots
    for p in profiles:                              # isolate T and h: uniform composition
        p['Y']=np.full_like(p['Y'],.5)
    tolerance=1e-3
    Zgrid,fields,info,lower,upper=_mixture_columns(profiles,3,tolerance)
    assert (lower,upper)==(.1,.9)
    span=max(np.ptp(p['T'][(p['Z']>=lower)&(p['Z']<=upper)]) for p in profiles)
    for row,p in zip(fields['T'],profiles):
        inside=(p['Z']>=lower)&(p['Z']<=upper)
        error=np.max(abs(np.interp(p['Z'][inside],Zgrid,row)-p['T'][inside]))/span
        assert error<=tolerance
    assert info['sampling_error']<=tolerance


def test_mixture_sampling_reports_an_insufficient_point_budget():
    from kflame.fgm import diffusion as D
    Z=np.linspace(1.,0.,401)
    profiles=[_profile(Z,300.+1700.*np.sin(7*np.pi*Z)**2),_profile(Z,300.+1600.*np.sin(7*np.pi*Z)**2)]
    columns=D._mixture_columns(profiles,3)[0]  # only checks the profiles are admissible
    assert len(columns)
    with pytest.raises(ValueError,match='more than 20 points'):
        cols=[(p['Z'][::-1],np.arange(len(Z))[::-1],p) for p in profiles]
        D._adaptive_mixture_grid(cols,0.,1.,3,1e-4,max_points=20)


def test_accumulated_z_reversals_are_a_fold_not_rounding(tmp_path):
    # Review P2: three reversals of 8e-7 accumulate 2.4e-6 > FOLD_TOLERANCE.
    from kflame.fgm.diffusion import FOLD_TOLERANCE,_mixture_columns
    Z=np.linspace(1.,0.,12)
    Z[5:8]=Z[4]+np.array([8e-7,16e-7,24e-7])        # Z rises three times in a row
    assert np.all(np.diff(Z[4:8])<=FOLD_TOLERANCE)  # each single step looks like rounding
    profiles=[_profile(Z,300.+1000.*Z),_profile(np.linspace(1.,0.,12),300.+900.*np.linspace(1.,0.,12))]
    with pytest.raises(ValueError,match='folds'):
        _mixture_columns(profiles,5)


def test_rounding_level_reversals_of_z_are_dropped_and_counted(tmp_path):
    profiles,metadata=synthetic_family()
    for profile in profiles:
        profile['Z'][-2]=profile['Z'][-3]+1e-9  # reversal far below FOLD_TOLERANCE
    report=build_diffusion_manifold(profiles,metadata,tmp_path,mixture_points=9,progress_weights={'H2O':1.})
    assert report['geometry']['dropped_rounding_reversals_of_Z']==[1,1,1]
