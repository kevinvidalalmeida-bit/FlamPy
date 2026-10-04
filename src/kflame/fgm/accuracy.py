"""User-selected, dimensionally explicit FGM interpolation tolerances."""
from dataclasses import asdict,dataclass,field
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FGMTolerances:
    temperature_K: float=20.
    species_absolute: float=.005
    node_coverage_min: float=.85
    source_peak_relative: dict=field(default_factory=lambda:dict(omega_C=.05,qdot=.05))
    source_L1_relative: dict=field(default_factory=lambda:dict(omega_C=.05,qdot=.05))
    source_integral_relative: dict=field(default_factory=lambda:dict(omega_C=.03,qdot=.03))
    source_coverage_min: dict=field(default_factory=lambda:dict(omega_C=.99,qdot=.99))

    def __post_init__(self):
        for key,value in asdict(self).items():
            values=value.values() if isinstance(value,dict) else [value]
            if isinstance(value,dict) and set(value)!= {'omega_C','qdot'}:
                raise ValueError(f'{key} must contain omega_C and qdot')
            try:
                invalid = any(isinstance(v, bool) or not np.isfinite(v) or v<=0. for v in values)
            except TypeError as error:
                raise ValueError(f'{key} must contain numeric tolerances') from error
            if invalid:
                raise ValueError(f'{key} must be finite and positive')
            if 'coverage' in key and any(v>1. for v in values):
                raise ValueError(f'{key} must be at most one')
            object.__setattr__(self,key,{name:float(v) for name,v in value.items()}
                               if isinstance(value,dict) else float(value))

    @classmethod
    def from_dict(cls,settings):
        defaults=asdict(cls())
        if not isinstance(settings,dict) or set(settings)-set(defaults):
            raise ValueError('Unknown FGM tolerance field')
        for key,value in settings.items():
            if isinstance(defaults[key],dict):
                if not isinstance(value,dict) or set(value)-set(defaults[key]):
                    raise ValueError(f'Unknown source tolerance in {key}')
                defaults[key].update(value)
            else:
                defaults[key]=value
        return cls(**defaults)

    @classmethod
    def from_file(cls,path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return asdict(self)


def source_metrics(z,truth,prediction,covered=None):
    """Peak and integrated errors on covered nonuniform intervals.

    Never bridge holes. Integral denominators include the full truth profile;
    source-weighted coverage prevents missing a significant reaction zone.
    """
    z,truth,prediction=(np.asarray(a,dtype=float) for a in (z,truth,prediction))
    covered=np.ones(len(z),dtype=bool) if covered is None else np.asarray(covered,dtype=bool)
    if (z.ndim!=1 or len(z)<2 or truth.shape!=z.shape or prediction.shape!=z.shape
            or covered.shape!=z.shape or not covered.any() or not np.isfinite(z).all()
            or np.any(np.diff(z)<=0.) or not np.isfinite(truth).all()
            or not np.isfinite(prediction[covered]).all()):
        raise ValueError('Source metrics require increasing finite coordinates and finite covered sources')
    error=prediction[covered]-truth[covered]
    actual_peak=float(np.max(abs(truth)));peak=max(actual_peak,1e-300)
    intervals=covered[:-1]&covered[1:];dz=np.diff(z)[intervals]
    full_error=np.where(covered,prediction-truth,0.)
    def integrate(values):
        return float(np.sum(.5*(values[:-1][intervals]+values[1:][intervals])*dz))
    integral_abs=float(np.trapezoid(abs(truth),z));denom=max(integral_abs,1e-300)
    j=int(np.flatnonzero(covered)[np.argmax(abs(error))])
    return dict(Linf_over_truth_peak=float(np.max(abs(error))/peak),
                L2_relative=float(np.sqrt(integrate(full_error**2)/max(integrate(truth**2),1e-300))),
                L1_relative=integrate(abs(full_error))/denom,
                integral_error_over_abs_integral=abs(integrate(full_error))/denom,
                integral_truth=float(np.trapezoid(truth,z)),
                absolute_source_coverage=integrate(abs(truth))/denom if integral_abs else 1.,
                worst_index=j,worst_z_mm=float(1000*z[j]),truth_at_worst=float(truth[j]),
                prediction_at_worst=float(prediction[j]),truth_peak=actual_peak,
                peak_amplitude_error=abs(float(np.max(prediction[covered]))-float(np.max(truth[covered])))/peak)


def assess_profile(truth,prediction,tolerances):
    """Return independent error ratios; one requires refinement above 1."""
    covered=np.asarray(prediction['covered'],dtype=bool)
    if not covered.any():
        return dict(passed=False,score=None,reason='no_covered_states',sources={})
    if (not np.isfinite(prediction['T'][covered]).all()
            or not np.isfinite(prediction['Y'][covered]).all()
            or not np.isfinite(truth['T']).all() or not np.isfinite(truth['Y']).all()):
        raise ValueError('Profile temperatures and mass fractions must be finite')
    T_error=float(np.max(abs(prediction['T'][covered]-truth['T'][covered])))
    Y_error=float(np.max(abs(prediction['Y'][covered]-truth['Y'].T[covered])))
    sources={name:source_metrics(truth['z'],truth[name],prediction[name],covered) for name in ('omega_C','qdot')}
    ratios=dict(temperature=T_error/tolerances.temperature_K,species=Y_error/tolerances.species_absolute,
                node_coverage=tolerances.node_coverage_min/float(covered.mean()))
    for name,metrics in sources.items():
        for label,key,limits in [('peak','Linf_over_truth_peak',tolerances.source_peak_relative),
                                 ('L1','L1_relative',tolerances.source_L1_relative),
                                 ('integral','integral_error_over_abs_integral',tolerances.source_integral_relative)]:
            ratios[f'{name}_{label}']=metrics[key]/limits[name]
        ratios[f'{name}_coverage']=tolerances.source_coverage_min[name]/max(metrics['absolute_source_coverage'],1e-300)
    score=max(ratios.values())
    return dict(passed=bool(score<=1.),score=float(score),ratios=ratios,
                temperature_Linf_K=T_error,all_Y_Linf=Y_error,node_coverage=float(covered.mean()),sources=sources)
