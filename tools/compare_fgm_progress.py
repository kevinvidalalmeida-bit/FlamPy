"""Construction-only progress-coordinate screening, with no flame solves.

Leave-one-row-out errors diagnose the existing fixed family; they are not an
independent validation of its historically adaptive composition sampling.
Chemical sources of alternative coordinates are deliberately not inferred
from the saved source of the original progress variable.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kflame.fgm.common import build_global_indicator, build_adaptive_c_grid, monotonicize_on_c

CANDIDATES={
    'C1':dict(CO2=1.),
    'C2':dict(CO2=1.,H2O=1.),
    'C3':dict(CO2=1.,H2O=1.,CO=1.,H2=1.),
    'C4':dict(CO2=1.,H2O=1.,CO=1.,H2=.5),
}
FIELD_NAMES=('T','CO2','CO','OH','qdot')


def fields(record):
    names=list(record.species_names)
    return np.column_stack([record.T,*[record.Y[names.index(k)] for k in ('CO2','CO','OH')],record.qdot])


def project(record, cgrid):
    values=fields(record)
    unique,vs=monotonicize_on_c(record.c,{k:values[:,i] for i,k in enumerate(FIELD_NAMES)})
    return np.column_stack([np.interp(cgrid,unique,vs[k]) for k in FIELD_NAMES])


def screen(records, settings, n_c=241):
    """No held-out argument: only a single construction family is admissible."""
    records=sorted(records,key=lambda r:r.Z)
    if not records: raise ValueError('No construction profiles.')
    if any(not (r.solve_ok and r.final_accepted) for r in records):
        raise ValueError('Candidate comparison requires accepted construction profiles.')
    z=np.array([r.Z for r in records])
    if np.any(np.diff(z)<=0): raise ValueError('Construction coordinates must increase.')
    raw=[fields(r) for r in records]
    if any(not np.isfinite(q).all() for q in raw): raise ValueError('Nonfinite fields.')
    # Identical physical states for every candidate: sample on original c.
    base_samples=[];reference=[]
    for r,q in zip(records,raw):
        b=(r.beta-r.beta[0])/(r.beta[-1]-r.beta[0])
        if not np.isfinite(b).all() or np.min(np.diff(b)) < -1e-7:
            raise ValueError('Original progress cannot provide common comparison states.')
        base_samples.append(b)
        reference.append(project(SimpleNamespace(**{**vars(r),'c':b}),np.linspace(0,1,1001)))
    summaries=[];audits=[];folds=[]
    for candidate,weights in CANDIDATES.items():
        remapped=[];candidate_audit=[];valid=True
        for r,q in zip(records,raw):
            names=list(r.species_names)
            beta=sum(w*r.Y[names.index(k)] for k,w in weights.items())
            span=float(beta[-1]-beta[0])
            a=dict(candidate=candidate,phi=float(r.phi),span=span,degenerate=abs(span)<=1e-14,
                   inversions=0,outside_nodes=0,maximum_gradient=None,plateau_change_percent=None)
            if a['degenerate']:
                valid=False;candidate_audit.append(a);continue
            c=(beta-beta[0])/span;dc=np.diff(c)
            a['inversions']=int(np.count_nonzero(dc < -1e-7))
            a['outside_nodes']=int(np.count_nonzero((c < -1e-7)|(c > 1+1e-7)))
            scale=np.maximum(np.max(np.abs(q),axis=0),1e-30)
            scale[0]=max(np.ptp(r.T),1e-30)
            scale[1:4]=np.maximum(scale[1:4],1e-5)
            changes=np.abs(np.diff(q,axis=0))/scale
            resolved=dc>1e-10
            a['maximum_gradient']=float(np.max(changes[resolved]/dc[resolved,None])) if resolved.any() else None
            # Consecutive nearly flat segments, including their accumulated field change.
            plateau=0.;i=0
            while i<len(dc):
                if abs(dc[i])>1e-10: i+=1;continue
                start=i
                while i<len(dc) and abs(dc[i])<=1e-10:i+=1
                plateau=max(plateau,float(np.max(np.ptp(q[start:i+1],axis=0)/scale)*100))
            a['plateau_change_percent']=plateau
            valid &= not (a['inversions'] or a['outside_nodes'])
            candidate_audit.append(a)
            # Only the coordinates change. Do not create an alternative omega_c.
            remapped.append(SimpleNamespace(**{**vars(r),'c':np.clip(c,0,1),'beta':beta}))
        audits.extend(candidate_audit)
        s=dict(candidate=candidate,profiles=len(records),
               inverted_flames=sum(a['inversions']>0 for a in candidate_audit),
               degenerate_flames=sum(a['degenerate'] for a in candidate_audit),
               outside_flames=sum(a['outside_nodes']>0 for a in candidate_audit),
               maximum_gradient=max((a['maximum_gradient'] for a in candidate_audit if a['maximum_gradient'] is not None),default=None),
               plateau_change_percent=max((a['plateau_change_percent'] for a in candidate_audit if a['plateau_change_percent'] is not None),default=None),
               loo_rows=0,loo_worst_p95_percent=None,loo_worst_field=None,
               loo_worst_mean_percent=None,loo_worst_maximum_percent=None,
               status='invalid_coordinate' if not valid else 'insufficient_rows')
        if valid and len(records)>=3:
            errors=[]
            for j in range(1,len(records)-1):
                training=remapped[:j]+remapped[j+1:]
                cf=np.linspace(0,1,settings['c_fine'])
                indicator=build_global_indicator(training,list(records[0].species_names),
                    settings['indicator_species'].split(','),cf,settings['indicator_weight_grad'],
                    settings['indicator_weight_conc'],settings['indicator_weight_temp'],settings['indicator_weight_qdot'])
                grid=build_adaptive_c_grid(cf,indicator,n_c,settings['refine_bias'])
                # Linear Z interpolation uses only the two remaining neighbours.
                weight=(z[j]-z[j-1])/(z[j+1]-z[j-1])
                interp=(1-weight)*project(remapped[j-1],grid)+weight*project(remapped[j+1],grid)
                bc,cv=monotonicize_on_c(base_samples[j],dict(candidate=remapped[j].c))
                target=np.interp(np.linspace(0,1,1001),bc,cv['candidate'])
                prediction=np.column_stack([np.interp(target,grid,interp[:,k]) for k in range(len(FIELD_NAMES))])
                # Scales depend only on the rows remaining in this fold.
                training_q=[q for k,q in enumerate(raw) if k!=j]
                scale=np.maximum(np.max([np.max(np.abs(q),axis=0) for q in training_q],axis=0),1e-30)
                scale[0]=max(np.ptp(q[:,0]) for q in training_q)
                scale[1:4]=np.maximum(scale[1:4],1e-5)
                error=100*np.abs(prediction-reference[j])/scale
                errors.append(error)
                for k,field in enumerate(FIELD_NAMES):
                    folds.append(dict(candidate=candidate,phi=float(records[j].phi),field=field,
                                      mean=float(error[:,k].mean()),p95=float(np.quantile(error[:,k],.95)),
                                      maximum=float(error[:,k].max()),scale=float(scale[k])))
            ep=np.array(errors);p95=np.quantile(ep,.95,axis=(0,1));k=int(np.argmax(p95))
            s.update(status='screened',loo_rows=len(errors),loo_worst_p95_percent=float(p95[k]),
                     loo_worst_field=FIELD_NAMES[k],
                     loo_worst_mean_percent=float(ep.mean(axis=(0,1)).max()),
                     loo_worst_maximum_percent=float(ep.max()))
        summaries.append(s)
    return dict(summary=summaries,audits=audits,folds=folds,n_c=n_c,
                candidates=CANDIDATES,fields=FIELD_NAMES,
                scope='Construction-only diagnostic on historically fixed rows. No holdouts, no weight selection.',
                sampling='Same 1001 physical states per row, uniform in original normalized progress.',
                source_policy='Compare qdot; alternative chemical progress sources are not reconstructed.')


def write_report(out, result, provenance):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'progress_candidates.json').write_text(json.dumps(dict(**result,provenance=provenance),indent=2),encoding='utf-8')
    for key in ('summary','audits','folds'):
        if not result[key]: continue
        with (out/f'progress_candidates_{key}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(result[key][0]));writer.writeheader();writer.writerows(result[key])
    def fmt(value):return '--' if value is None else rf'\num{{{value:.3g}}}'
    lines=[r'\begin{table}[!htbp]\centering\small',
           r'\caption{Comparación de coordenadas sobre una familia de construcción: '+provenance['tex_description']+
           r'. Inv.: llamas con retrocesos $\Delta c<-10^{-7}$; Deg.: normalizaciones degeneradas. '
           r'$G_{\max}$ mide el mayor gradiente normalizado de los campos respecto a $c$; '
           r'$V_{\rm plano}$ mide su mayor variación en tramos casi constantes. '
           r'P95 es el mayor percentil 95 entre los cinco campos al retirar filas interiores. '
           r'Una coordenada inválida queda sin error de reconstrucción.}',
           r'\label{tab:fgm-progress-candidates}\begin{tabular}{lrrrrr}\toprule',
           r'Definición & Inv. & Deg. & $G_{\max}$ & $V_{\rm plano}$ [\%] & P95 [\%]\\\midrule']
    for r in result['summary']:
        label=r['candidate']+(' (actual)' if r['candidate']=='C4' else '')
        lines.append(f'{label} & {r["inverted_flames"]} & {r["degenerate_flames"]} & '+
                     ' & '.join(fmt(r[k]) for k in ('maximum_gradient','plateau_change_percent','loo_worst_p95_percent'))+r'\\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    (out/'table_progress_candidates.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    current=next(r for r in result['summary'] if r['candidate']=='C4')
    text=(f'En la familia utilizada para tabular, la definición actual presenta '
          f'{current["inverted_flames"]} llamas con retrocesos y '
          f'{current["degenerate_flames"]} normalizaciones degeneradas. ')
    if current['status']=='screened':
        text+=(rf'Al retirar sucesivamente {current["loo_rows"]} filas interiores, '
               rf'su mayor P95 entre campos es \num{{{current["loo_worst_p95_percent"]:.3g}}}\%. ')
    text+=(r'El apéndice compara cuatro definiciones sobre la misma familia; '
           r'la evaluación independiente conserva la definición fijada en el protocolo.')
    (out/'analysis_progress.tex').write_text(text+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',type=Path,required=True)
    p.add_argument('--settings',type=Path,required=True,help='Campaign manifest or historical metadata')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    meta=json.loads(args.settings.read_text(encoding='utf-8'))
    settings=meta.get('settings',meta.get('args'))
    records=[]
    for file in sorted(args.profiles.glob('*.npz')):
        with np.load(file,allow_pickle=False) as a:
            d={k:(a[k].item() if k in ('phi','Z','solve_ok','final_accepted') else a[k])
               for k in ('phi','Z','solve_ok','final_accepted','T','Y','qdot','beta')}
            # Historical species labels are object arrays; use their JSON copy.
            d['species_names']=np.array(meta['species_names']) if 'species_names' in meta else a['species_names']
        records.append(SimpleNamespace(**d))
    result=screen(records,settings)
    write_report(args.output,result,dict(profiles=str(args.profiles.resolve()),settings=str(args.settings.resolve()),
                 settings_sha256=hashlib.sha256(args.settings.read_bytes()).hexdigest(),
                 analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 profile_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.profiles.glob('*.npz'))},
                 tex_description='estudio separado de los perfiles indicados en el archivo de procedencia'))
    for row in result['summary']:print(json.dumps(row))


if __name__=='__main__':main()
