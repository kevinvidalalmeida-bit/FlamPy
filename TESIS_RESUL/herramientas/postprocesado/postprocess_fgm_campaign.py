"""Offline FGM figures, held-out interpolation errors and lookup microbenchmark.

Reads saved profiles only. Does not resolve flames. Incomplete campaigns are
explicit and never receive completed timing or fidelity conclusions.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key] = '1'
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'TESIS_RESUL/herramientas/campanas'))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from benchmark_fgm_campaign import atomic, completed, latest, verify_artifacts, digest
from kflame.fgm.common import build_global_indicator, build_adaptive_c_grid, _tabulate_profiles, monotonicize_on_c

FIELDS = ('T','CO2','CO','omega_c')
LABELS = (r'$T$',r'$Y_{CO_2}$',r'$Y_{CO}$',r'$\dot\omega_c$')


def load_npz(path):
    # The native builder historically saved its predictor labels as an object
    # array. Read only numerical fields and the unicode species names here.
    with np.load(path, allow_pickle=False) as a:
        return {k:a[k] for k in a.files if k != 'predictor_kind'}


def load_records(attempt):
    return [SimpleNamespace(**{k:v.item() if v.ndim==0 else v for k,v in load_npz(p).items()})
            for p in sorted((attempt/'profiles').glob('*.npz'))]


def coordinate_audit(records, phase, repetition):
    """Inspect original profiles before sorting/merging their progress samples.

    Small dc alone is harmless in a uniform burned tail; report field variation
    across each near-constant-c segment as well. This is a resolution diagnostic,
    not proof of global manifold uniqueness.
    """
    rows=[]
    for r in records:
        beta=np.asarray(r.beta,dtype=float)
        if beta.size<2 or not np.isfinite(beta).all():
            raise ValueError('Invalid raw progress samples.')
        span=float(beta[-1]-beta[0])
        if abs(span)<=1e-14: raise ValueError('Degenerate progress normalization.')
        raw=(beta-beta[0])/span
        dc=np.diff(raw)
        fields=np.vstack([r.T,*r.Y,r.omega_c])
        names=['T',*[str(k) for k in r.species_names],'omega_c']
        if not np.isfinite(fields).all(): raise ValueError('Nonfinite profile fields.')
        scales=np.max(np.abs(fields),axis=1)
        scales[0]=np.ptp(r.T)
        # Negligible species remain bounded by a mass-fraction scale of 1e-5.
        scales[1:-1]=np.maximum(scales[1:-1],1e-5)
        scales=np.maximum(scales,1e-30)
        worst=0.;worst_field='none';segments=0
        i=0
        while i<len(dc):
            if abs(dc[i])>1e-10:
                i+=1;continue
            start=i
            while i<len(dc) and abs(dc[i])<=1e-10: i+=1
            changes=100*np.ptp(fields[:,start:i+1],axis=1)/scales
            k=int(np.argmax(changes));segments+=1
            if changes[k]>worst: worst=float(changes[k]);worst_field=names[k]
        rows.append(dict(phase=phase,repetition=repetition,phi=float(r.phi),Z=float(r.Z),
             beta_span=span,c_min=float(raw.min()),c_max=float(raw.max()),
             min_dc=float(dc.min()),max_dc=float(dc.max()),
             backward_steps=int(np.count_nonzero(dc < -1e-7)),
             outside_nodes=int(np.count_nonzero((raw < -1e-7)|(raw > 1+1e-7))),
             plateau_segments=segments,plateau_max_field_change_percent=worst,
             plateau_worst_field=worst_field))
    return rows


def write_coordinate_audit(out, families, holdout):
    rows=[];spacing=[]
    for rep,folder,_ in families:
        records=load_records(folder)
        rows.extend(coordinate_audit(records,'family',rep))
        z=np.array([r.Z for r in records]);dz=np.diff(z)
        if not np.isfinite(z).all() or np.any(dz<=0):
            raise ValueError('Construction Z coordinates must be finite and strictly increasing.')
        spacing.append(dict(repetition=rep,min_dZ=float(dz.min()),max_dZ=float(dz.max())))
    if holdout: rows.extend(coordinate_audit(load_records(holdout),'holdout',1))
    csv_write(out/'coordinate_audit.csv',rows)
    worst=max(rows,key=lambda r:r['plateau_max_field_change_percent'])
    summary=dict(profiles=len(rows),backward_steps=sum(r['backward_steps'] for r in rows),
                 outside_nodes=sum(r['outside_nodes'] for r in rows),
                 minimum_abs_beta_span=min(abs(r['beta_span']) for r in rows),
                 plateau_tolerance_dc=1e-10,monotonicity_tolerance_dc=1e-7,
                 plateau_max_field_change_percent=worst['plateau_max_field_change_percent'],
                 plateau_worst_profile=worst,spacing=spacing,
                 scope='Raw profiles before sorting. Plateau metric is a diagnostic; heldout fidelity remains necessary.')
    atomic(out/'coordinate_audit.json',summary)
    lines=[r'\begin{table}[!htbp]\centering\small',
        r'\caption{Auditoría de las coordenadas sobre los perfiles originales de todas las ejecuciones completas '
        r'disponibles, incluidas las retenidas. Los conteos reúnen esas ejecuciones; los límites son extremos. '
        r'Se consideran retrocesos con $\Delta c<-10^{-7}$ y tramos casi constantes con '
        r'$|\Delta c|\leq10^{-10}$. La variación en esos tramos se normaliza por el salto térmico '
        r'o el máximo módulo por perfil; para especies se fija una escala mínima de $10^{-5}$.}',
        r'\label{tab:fgm-coordinates}\begin{tabular}{lr}\toprule Diagnóstico & Valor\\\midrule',
        f'Perfiles auditados & {len(rows)}'+r'\\',
        f'Retrocesos de progreso & {summary["backward_steps"]}'+r'\\',
        f'Nodos fuera de $[0,1]$ con margen $10^{{-7}}$ & {summary["outside_nodes"]}'+r'\\',
        r'Mínimo $|\beta_{c,\rm b}-\beta_{c,\rm u}|$ & '+f'{summary["minimum_abs_beta_span"]:.3g}'+r'\\',
        r'Mayor variación en tramos casi constantes [\%] & '+f'{summary["plateau_max_field_change_percent"]:.3g}'+r'\\',
        r'Mínimo / máximo $\Delta Z_{\rm in}$ & '+f'{min(s["min_dZ"] for s in spacing):.3g} / {max(s["max_dZ"] for s in spacing):.3g}'+r'\\',
        r'\bottomrule\end{tabular}\end{table}']
    (out/'table_coordinates.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if summary['backward_steps'] or summary['outside_nodes']:
        raise ValueError('Raw progress audit failed; inspect coordinate_audit.json before interpreting interpolation.')
    unique=len({r['phi'] for r in rows if r['phase']=='family'})
    (out/'analysis_coordinates.tex').write_text(
        f'En los {len(rows)} perfiles disponibles, correspondientes a {unique} composiciones '
        f'de construcción y sus controles de retención disponibles, se registran '
        f'{summary["backward_steps"]} retrocesos de progreso mayores que $10^{{-7}}$ '
        f'y {summary["outside_nodes"]} nodos fuera de $[0,1]$ con ese margen. '
        f'La mayor variación normalizada de un campo en tramos casi constantes es '
        rf'\num{{{summary["plateau_max_field_change_percent"]:.3g}}}\%; '
        r'el apéndice detalla sus escalas. La reconstrucción independiente completa este diagnóstico local.'+'\n',
        encoding='utf-8')


def table_from_records(records, settings, nc):
    names = list(records[0].species_names)
    cf=np.linspace(0,1,settings['c_fine'])
    indicator=build_global_indicator(records,names,settings['indicator_species'].split(','),cf,
                settings['indicator_weight_grad'],settings['indicator_weight_conc'],
                settings['indicator_weight_temp'],settings['indicator_weight_qdot'])
    c=build_adaptive_c_grid(cf,indicator,nc,settings['refine_bias'])
    return dict(Z_grid=np.array([r.Z for r in records]),phi_grid=np.array([r.phi for r in records]),
                c_grid=c,species_names=np.array(names),**_tabulate_profiles(records,len(names),c))


def interpolator(table):
    names=list(table['species_names'])
    values=np.stack([table['T'],table['Y'][:,names.index('CO2')],
                     table['Y'][:,names.index('CO')],table['omega_c']],axis=-1)
    return RegularGridInterpolator((table['Z_grid'],table['c_grid']),values,bounds_error=True)


def fidelity(table, retained):
    """Equal weight per held-out flame and uniform progress; no division near zero."""
    if not retained: raise ValueError('No holdouts.')
    phi=np.array([r.phi for r in retained])
    if np.isclose(phi[:,None],table['phi_grid'][None,:],rtol=0,atol=1e-12).any():
        raise ValueError('Holdout leakage: a retained flame is a construction row.')
    c=np.linspace(0,1,1001)
    truth=[]; estimates=[]
    interp=interpolator(table)
    for r in retained:
        if not (r.solve_ok and r.final_accepted): raise ValueError('Unaccepted holdout.')
        raw_c=(r.beta-r.beta[0])/(r.beta[-1]-r.beta[0])
        if np.min(np.diff(raw_c)) < -1e-7: raise ValueError('Nonmonotone retained progress.')
        names=list(r.species_names)
        if names != list(table['species_names']): raise ValueError('Species mismatch.')
        cu,v=monotonicize_on_c(r.c,dict(T=r.T,CO2=r.Y[names.index('CO2')],
                                      CO=r.Y[names.index('CO')],omega_c=r.omega_c))
        truth.append(np.column_stack([np.interp(c,cu,v[k]) for k in FIELDS]))
        estimates.append(interp(np.column_stack([np.full(c.size,r.Z),c])))
    truth=np.array(truth); estimates=np.array(estimates)
    scale=np.max(np.abs(truth),axis=(0,1))
    scale[0]=np.max(truth[:,:,0].max(axis=1)-truth[:,:,0].min(axis=1))
    if np.any(scale<=1e-14): raise ValueError('An evaluated field has zero reference scale.')
    error=100*np.abs(estimates-truth)/scale
    metrics=[dict(field=k,scale=float(scale[i]),mean=float(error[:,:,i].mean()),
                  p95=float(np.quantile(error[:,:,i],.95)),maximum=float(error[:,:,i].max()))
             for i,k in enumerate(FIELDS)]
    return dict(phi=phi,Z=np.array([r.Z for r in retained]),c=c,truth=truth,prediction=estimates,
                error_percent=error,scale=scale),metrics


def query_benchmark(table, seed):
    rng=np.random.default_rng(seed)
    points=np.column_stack([rng.uniform(table['Z_grid'][0],table['Z_grid'][-1],10000),rng.uniform(0,1,10000)])
    interp=interpolator(table)
    rows=[]
    for batch in (1,10000):
        target=points[:batch]
        interp(target)
        start=time.perf_counter()
        for _ in range(10): interp(target)
        calls=max(1,min(100000,int(.12/max((time.perf_counter()-start)/10,1e-8))))
        for rep in range(5):
            start=time.perf_counter()
            for _ in range(calls): answer=interp(target)
            elapsed=time.perf_counter()-start
            if not np.isfinite(answer).all(): raise ValueError('Nonfinite lookup.')
            rows.append(dict(batch=batch,repetition=rep+1,calls=calls,seconds=elapsed,
                             us_per_state=elapsed*1e6/(calls*batch)))
    return rows,points


def csv_write(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def cost_statistics(results, observations):
    """Aggregate totals per run first; throughput is computed per observation."""
    def stats(values):
        q=np.quantile(values,[.25,.5,.75])
        return dict(observations=list(map(float,values)),q25=float(q[0]),median=float(q[1]),q75=float(q[2]))
    total=[r['setup_s']+r['family_s']+r['tabulation_s'] for r in results]
    queries={str(batch):dict(
        us_per_state=stats([r['us_per_state'] for r in observations if r['batch']==batch]),
        states_per_second=stats([1e6/r['us_per_state'] for r in observations if r['batch']==batch]))
        for batch in (1,10000)}
    return dict(offline_total_s=stats(total),query=queries)


def plot_fidelity(data, table, out):
    """One central holdout selected by composition, alongside all held-out errors."""
    import matplotlib.pyplot as plt
    selected=int(np.argmin(np.abs(np.log(data['phi']))))
    ep=data['error_percent']
    fig,axes=plt.subplots(3,2,figsize=(7.6,7.8),layout='constrained')
    units=[r'$T$ [K]',r'$Y_{CO_2}$ [kg/kg]',r'$Y_{CO}$ [kg/kg]',
           r'$\dot\omega_c$ [kg m$^{-3}$ s$^{-1}$]']
    for i,ax in enumerate(axes[:2].flat):
        ax.plot(data['c'],data['truth'][selected,:,i],color='#27758b',lw=1.8,label='Llama retenida')
        ax.plot(data['c'],data['prediction'][selected,:,i],color='#bd7940',ls='--',lw=1.5,label='FGM')
        ax.set(xlabel=r'Progreso $c$',ylabel=units[i],xlim=(0,1))
        ax.set_title(f'({chr(97+i)})',loc='left',fontsize=10)
        ax.tick_params(top=False,right=False)
    axes[0,0].legend(fontsize=8,frameon=False)
    axes[0,1].text(.03,.92,rf'$\phi={data["phi"][selected]:.4f}$',transform=axes[0,1].transAxes,fontsize=10)
    ax=axes[2]
    ax[0].boxplot([ep[:,:,i].ravel() for i in range(4)],tick_labels=LABELS,showfliers=False,whis=(5,95))
    visible_max=float(np.quantile(ep,.95,axis=(0,1)).max())
    if visible_max>1:
        ax[0].set_yscale('symlog',linthresh=1)
        scale_caption=r'eje lineal hasta 1\% y logarítmico por encima'
    else:
        from matplotlib.ticker import MaxNLocator, StrMethodFormatter
        ax[0].yaxis.set_major_locator(MaxNLocator(5))
        ax[0].yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        scale_caption='eje lineal'
    ax[0].set_ylim(0,max(visible_max*1.15,1e-8))
    ax[0].set(ylabel='Error normalizado [%]',title='(e) Todas las llamas retenidas')
    zstar=(data['Z']-table['Z_grid'][0])/np.ptp(table['Z_grid'])
    xx,yy=np.meshgrid(zstar,data['c'][::10],indexing='ij')
    scatter=ax[1].scatter(xx.ravel(),yy.ravel(),c=ep[:,::10,:].max(axis=2).ravel(),s=5,cmap='magma')
    fig.colorbar(scatter,ax=ax[1],label='Mayor error [%]',fraction=.06,pad=.03)
    ax[1].set(xlabel=r'$Z^\star$',ylabel=r'$c$',xlim=(0,1),ylim=(0,1),title='(f) Localización del error')
    for a in ax: a.tick_params(top=False,right=False)
    save_figure(fig,out,'03_fidelidad',
        rf'Reconstrucción y error en {len(data["phi"])} llamas retenidas. '
        rf'(a--d): perfil independiente más próximo a $\phi=1$ en $\log\phi$ '
        rf'($\phi={data["phi"][selected]:.4f}$); línea continua: referencia, discontinua: tabla. '
        r'(e): error en 1001 posiciones de $c$ por llama; mediana, caja de cuartiles 25--75 '
        'y bigotes P5--P95, con '+scale_caption+'. '+
        r'(f): mayor error de los cuatro campos en cada estado, mostrando una de cada diez posiciones. '
        r'Los errores se normalizan según el cuadro de errores; cada composición pesa igual.')
    return selected


def save_figure(fig, out, name, caption):
    import matplotlib.pyplot as plt
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)
    (out/(name+'.tex')).write_text(
        r'\begin{figure}[!htbp]\centering'+'\n'+
        r'\includegraphics[width=\textwidth,height=.67\textheight,keepaspectratio]{\FGMReportRoot/'+name+'.pdf}\n'+
        r'\caption{'+caption+'}\n'+r'\label{fig:fgm-'+name+r'}\end{figure}'+'\n',encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path,
                   default=Path('TESIS_RESUL/corridas/FGM/thesis_fgm'))
    p.add_argument('--output', type=Path,
                   default=Path('TESIS_RESUL/corridas/reproduccion/thesis_fgm_report'))
    p.add_argument('--benchmark-queries',action='store_true',help='Five lookup batches per batch size; no flames')
    args=p.parse_args(argv)
    root=args.input.resolve(); out=args.output.resolve()
    if out==root or root in out.parents and out.parts[len(root.parts)] in ('family','holdout','inputs'):
        raise ValueError('Report must not overwrite scientific records.')
    if not (root/'manifest.json').exists():
        print('Campaign pending; no results generated.')
        return
    out.mkdir(parents=True,exist_ok=True)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    families=[];pending=[]
    for rep in range(1,m['repeats']+1):
        job=root/'family'/f'rep-{rep:02d}'
        result=completed(job)
        if result and result['status']=='accepted':
            verify_artifacts(latest(job),result)
            families.append((rep,latest(job),result))
        else: pending.append(dict(phase='family',rep=rep,status=(result or {}).get('status','pending')))
    hjob=root/'holdout/rep-01';hr=completed(hjob)
    holdout=latest(hjob) if hr and hr['status']=='accepted' else None
    if holdout: verify_artifacts(holdout,hr)
    else: pending.append(dict(phase='holdout',rep=1,status=(hr or {}).get('status','pending')))
    atomic(out/'status.json',dict(status='complete' if not pending else 'partial',pending=pending,
           manifest_sha256=digest(root/'manifest.json'),smoke=m['smoke'],families_complete=len(families),
           preflight_complete=bool(families and holdout)))
    # Remove only stale generated inclusions; retained PDF/CSV artifacts remain available.
    for name in ('01_familia','02_mapa','03_fidelidad','04_coste','table_errors','table_sensitivity','table_coordinates','table_progress_candidates','analysis_progress','analysis_coordinates','analysis_fidelity','analysis_sensitivity','analysis_cost'):
        (out/(name+'.tex')).unlink(missing_ok=True)
    (out/'table_configuration.tex').write_text(
        r'\begin{table}[!htbp]\centering\small'+ '\n'+
        r'\caption{Diseño FGM. Construcción y retención usan CH$_4$--aire, GRI-Mech 3.0, '
        r'300 K, 1 atm y transporte promediado sin Soret. Retención designa llamas excluidas '
        r'de la construcción y del indicador adaptativo. Se indica la disponibilidad de cada conjunto; '
        r'la revisión de fidelidad utiliza la primera familia completa y las llamas retenidas.}'+'\n'+
        r'\label{tab:fgm-configuration}\begin{tabular}{lrr}\toprule Grupo & Llamas distintas & Conjuntos disponibles\\\midrule'+'\n'+
        f'Construcción & {len(m["schedule"]["phi_resolved"])} & {len(families)}'+r'\\'+'\n'+
        f'Retención independiente & {len(m["holdouts"])} & {int(bool(holdout))}'+r'\\\bottomrule\end{tabular}\end{table}'+'\n',encoding='utf-8')
    if not families:
        print('Family pending; configuration table generated.')
        return
    write_coordinate_audit(out,families,holdout)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.grid':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    rep,reference,result=families[0]
    table=load_npz(reference/'fgm_table.npz')
    records=load_records(reference)
    from compare_fgm_progress import screen, write_report
    print('Comparando coordenadas con perfiles de construcción guardados...',flush=True)
    progress=screen(records,m['settings'])
    write_report(out,progress,dict(reference_family=str(reference),repetition=rep,
        manifest_sha256=digest(root/'manifest.json'),smoke=m['smoke'],
        tex_description=f'{len(records)} composiciones de la repetición {rep}'+
            (' (prueba técnica)' if m['smoke'] else '')))
    traces=json.loads((reference/'trace.json').read_text(encoding='utf-8'))
    per_rep=[json.loads((folder/'trace.json').read_text(encoding='utf-8')) for _,folder,_ in families]
    times=np.array([[t['measured_s'] for t in tt] for tt in per_rep])
    fig,axes=plt.subplots(2,1,figsize=(7.3,5.5),sharex=True,layout='constrained')
    phi=table['phi_grid']
    axes[0].plot(phi,np.median(times,axis=0),'-o',ms=3,color='#27758b')
    if len(times)>1:
        axes[0].errorbar(phi,np.median(times,axis=0),yerr=np.array([np.median(times,axis=0)-np.quantile(times,.25,axis=0),
                                 np.quantile(times,.75,axis=0)-np.median(times,axis=0)]),fmt='none',color='#27758b')
    axes[0].set(ylabel='Tiempo por llama [s]',yscale='log')
    categories={'cold':0,'copy':1,'secant':2}
    kinds=[categories.get(t['predictor_kind'],2 if 'secant' in t['predictor_kind'] else 1) for t in traces]
    axes[1].scatter(phi,kinds,c=['#27758b' if r.requested else '#bd7940' for r in records],s=20)
    axes[1].set(yticks=[0,1,2],yticklabels=['Arranque físico','Copia previa','Secante'],xlabel=r'Razón de equivalencia $\phi$',ylim=(-.4,2.5))
    time_caption=(f'Arriba: tiempo individual de resolución y proyección de propiedades por llama '
                  f'en la repetición {rep}. ' if len(families)==1 else
                  f'Arriba: mediana del tiempo de resolución y proyección de propiedades por llama, '
                  f'con {len(families)} repeticiones completas; barras: cuartiles 25 y 75. ')
    save_figure(fig,out,'01_familia',
        f'Recorrido de la familia previamente adaptada de {len(records)} llamas. '+time_caption+
        f'Abajo: predictor registrado en la repetición {rep}; azul: estados solicitados; ocre: filas puente '
        r'de la familia previamente refinada. El primer estado parte de una estimación física. '
        r'Las composiciones se mantienen fijas durante estas repeticiones.')
    # Preserve the user's approved four-panel design and never overwrite the historical map.
    from kflame.fgm.plot import plot_adaptive_fgm_map
    shutil.copy2(reference/'fgm_table.npz',out/'fgm_table.npz')
    plot_adaptive_fgm_map(out,220,out_name='fgm_mapa_adaptativo.pdf')
    (out/'02_mapa.tex').write_text(r'\begin{figure}[!htbp]\centering'+'\n'+
        r'\includegraphics[width=\textwidth,height=.68\textheight,keepaspectratio]{\FGMReportRoot/fgm_mapa_adaptativo.pdf}'+'\n'+
        r'\caption{Manifold de CH$_4$--aire: perfiles de CO$_2$ y CO, liberación de calor y módulo de la fuente '
        r'de progreso. $Z^\star=(Z_{\rm in}-Z_{\min})/(Z_{\max}-Z_{\min})$ identifica la fila; '
        r'$c$ recorre el progreso desde mezcla fresca hasta quemada. Cada curva procede de una llama '
        f'de la repetición {rep}, con {len(records)} filas y {len(table["c_grid"])} nodos de progreso. '
        r'Las marcas verticales señalan filas reales; los colores entre filas proceden de interpolación lineal.}'+'\n'+
        r'\label{fig:fgm-02-mapa}\end{figure}'+'\n',encoding='utf-8')
    metrics=None
    if holdout:
        retained=load_records(holdout)
        data,metrics=fidelity(table,retained)
        atomic(out/'heldout_errors.npz',data,npz=True)
        csv_write(out/'error_metrics.csv',metrics)
        rows=[r'\begin{table}[!htbp]\centering\small',
              r'\caption{Error de interpolación en llamas de retención independientes. Media aritmética, percentil 95 '
              r'y máximo sobre 1001 posiciones uniformes de $c$ por llama, con igual peso por llama. '
              r'$e_Q=100|Q_{\rm FGM}-Q_{\rm ref}|/s_Q$: $s_T$ es el mayor salto térmico de retención y '
              r'$s_Q$ el máximo módulo de referencia para las otras magnitudes. Los valores se expresan en porcentaje de esa escala.}',
              r'\label{tab:fgm-errors}\begin{tabular}{lrrr}\toprule Magnitud & Media [\%] & P95 [\%] & Máximo [\%]\\\midrule']
        rows += [f'{lab} & {v["mean"]:.3g} & {v["p95"]:.3g} & {v["maximum"]:.3g}'+r'\\' for lab,v in zip(LABELS,metrics)]
        rows += [r'\bottomrule\end{tabular}\end{table}']
        (out/'table_errors.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')
        ep=data['error_percent']
        selected=plot_fidelity(data,table,out)
        atomic(out/'fidelity_selection.json',dict(phi=float(data['phi'][selected]),
               criterion='Minimum absolute log(phi); independent of observed interpolation error.',
               family_repetition=rep,normalization='Reference c uses each retained flame endpoints; error scales only assess fidelity.'))
        worst=max(metrics,key=lambda x:x['maximum'])
        worst_index=FIELDS.index(worst['field'])
        flame_index,c_index=np.unravel_index(np.argmax(ep[:,:,worst_index]),ep.shape[:2])
        worst_label=LABELS[worst_index]
        (out/'analysis_fidelity.tex').write_text(
            f'La mayor desviación normalizada corresponde a {worst_label}, '
            f'con un máximo de {worst["maximum"]:.3g}\\% en '
            f'$\\phi={data["phi"][flame_index]:.4f}$ y $c={data["c"][c_index]:.3f}$. '
            r'Este estado identifica una región concreta para revisar el muestreo. '
            r'Estas discrepancias miden la representación '
            r'de los perfiles detallados retenidos dentro del intervalo estudiado. '+
            f'El mayor P95 entre campos es {max(q["p95"] for q in metrics):.3g}\\%, '
            r'El contraste entre ambos estadísticos distingue las desviaciones extendidas de '
            r'los extremos locales mostrados en el mapa de error.'+'\n',encoding='utf-8')
        sens=[]
        for nc in (121,241,481):
            refined=table_from_records(records,m['settings'],nc)
            _,mm=fidelity(refined,retained)
            sens.append(dict(n_phi=len(records),n_c=nc,**{q['field']+'_max':q['maximum'] for q in mm}))
        csv_write(out/'table_resolution_sensitivity.csv',sens)
        middle,last=sens[1:]
        (out/'analysis_sensitivity.tex').write_text(
            f'Al pasar de 241 a 481 nodos, el máximo de temperatura pasa de '
            f'{middle["T_max"]:.3g}\\% a {last["T_max"]:.3g}\\%, mientras que '
            f'el de la fuente de progreso pasa de {middle["omega_c_max"]:.3g}\\% '
            f'a {last["omega_c_max"]:.3g}\\%. '
            r'La sensibilidad difiere entre campos: el refinamiento de $c$ mejora la '
            r'reconstrucción térmica, y la fuente conserva una discrepancia residual que '
            r'requiere examinar también la interpolación entre composiciones. '
            r'Los 241 nodos definen la resolución evaluada; este control cuantifica la '
            r'sensibilidad restante.'+'\n',encoding='utf-8')
        (out/'table_sensitivity.tex').write_text(
            r'\begin{table}[!htbp]\centering\small\caption{Control de resolución en $c$, reutilizando la misma familia '
            r'y las mismas llamas retenidas. Máximos del error normalizado [\%]; cambia únicamente $N_c$. '
            r'Este control no refina el muestreo entre composiciones.}\label{tab:fgm-resolution}'+'\n'+
            r'\begin{tabular}{rrrrr}\toprule $N_c$ & $T$ & $Y_{CO_2}$ & $Y_{CO}$ & $\dot\omega_c$\\\midrule'+'\n'+
            '\n'.join(' & '.join([str(s['n_c'])]+[f'{s[k+"_max"]:.3g}' for k in FIELDS])+r'\\' for s in sens)+'\n'+
            r'\bottomrule\end{tabular}\end{table}'+'\n',encoding='utf-8')
    queryfile=out/'query_timings.json'
    if args.benchmark_queries:
        observations,points=query_benchmark(table,m['seed'])
        atomic(queryfile,dict(table_sha256=digest(reference/'fgm_table.npz'),observations=observations,
                             python=sys.version,platform=platform.platform(),cpu=platform.processor(),
                             numpy=np.__version__,threads={k:os.environ[k] for k in
                             ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
                             seed=m['seed'],fields=FIELDS))
        atomic(out/'query_points.npz',dict(points=points),npz=True)
    query=json.loads(queryfile.read_text(encoding='utf-8')) if queryfile.exists() else None
    if query and query['table_sha256']!=digest(reference/'fgm_table.npz'):
        raise ValueError('Query benchmark belongs to a different table; rerun --benchmark-queries.')
    if len(families)==m['repeats'] and query:
        fig,ax=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
        stats=cost_statistics([r for _,_,r in families],query['observations'])
        offline=[[r[k] for _,_,r in families] for k in ('setup_s','family_s','tabulation_s')]
        offline.append(stats['offline_total_s']['observations'])
        online=[[r['us_per_state'] for r in query['observations'] if r['batch']==batch] for batch in (1,10000)]
        for panel,sets,labels in ((ax[0],offline,['Preparación','Familia','Tabla','Total']),
                                  (ax[1],online,['Un estado','Lote de 10 000'])):
            for i,values in enumerate(sets):
                q=np.quantile(values,[.25,.5,.75])
                panel.scatter(i+np.linspace(-.07,.07,len(values)),values,s=15,color='#a4afb5')
                panel.errorbar(i,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt='D',color='#27758b',capsize=4)
            panel.set(xticks=range(len(sets)),xticklabels=labels,yscale='log')
            panel.tick_params(which='both',top=False,right=False)
        ax[0].set(ylabel='Tiempo offline [s]');ax[1].set(ylabel='Tiempo de consulta [µs/estado]')
        save_figure(fig,out,'04_coste',
            f'Coste de construcción y consulta. Izquierda: {len(families)} ejecuciones completas de la familia, '
            r'con calentamiento previo, perfiles iniciales independientes entre repeticiones y continuación interna; '
            r'se excluye escritura. Derecha: cinco mediciones repetidas de consultas lineales a cuatro campos, '
            r'en llamadas de un estado o lotes de 10000; el tiempo se divide por los estados consultados. '
            r'Puntos grises: observaciones; rombo: mediana; barras: cuartiles 25--75. '
            r'El total suma las tres fases en cada repetición antes de calcular los estadísticos. '
            r'Las fases usan sus propias unidades y escalas logarítmicas.')
        atomic(out/'timing_summary.json',dict(offline=[dict(repetition=i,**r) for i,_,r in families],query=query,statistics=stats))
        med=np.median(offline[1]);tab=np.median(offline[2])
        (out/'analysis_cost.tex').write_text(
            f'La mediana del coste total previo es {stats["offline_total_s"]["median"]:.3g} s '
            f'(cuartiles {stats["offline_total_s"]["q25"]:.3g}--{stats["offline_total_s"]["q75"]:.3g} s); '
            f'la de generación es {med:.3g} s y la de tabulación {tab:.3g} s. '
            r'La separación de ambas fases distingue el coste de obtener las llamas del de construir su representación. '
            f'La consulta individual requiere una mediana de {stats["query"]["1"]["us_per_state"]["median"]:.3g} '
            r'$\mu$s por estado; en lotes de 10000, '
            f'{stats["query"]["10000"]["us_per_state"]["median"]:.3g} '+r'$\mu$s por estado, '
            rf'con una mediana de \num{{{stats["query"]["10000"]["states_per_second"]["median"]:.3g}}} estados/s. '
            r'La diferencia entre tamaños de lote cuantifica cómo cambia el coste de acceso por estado.'+'\n',encoding='utf-8')
    atomic(out/'provenance.json',dict(reference_family=str(reference),repetition=rep,
           table_sha256=digest(reference/'fgm_table.npz'),holdout=str(holdout) if holdout else None,
           completed_families=len(families),npz_size_bytes=(reference/'fgm_table.npz').stat().st_size,
           arrays_bytes=sum(a.nbytes for a in table.values()),fields=FIELDS))
    review=['# Revisión de la variable de progreso',
        '**Prueba técnica: no usar como resultado científico.**' if m['smoke'] else
        'Caso base: CH4/GRI-Mech 3.0, 300 K, 1 atm, transporte promediado sin Soret.',
        f'Familia evaluada: repetición {rep}, {len(records)} composiciones. Definición fijada: '
        'CO2 + H2O + CO + 0.5 H2, en fracciones másicas.',
        '## Comparación interna sobre construcción',
        '| Candidata | Llamas con retroceso | Degeneradas | P95 máximo entre campos [%] | Máximo puntual [%] |',
        '|---|---:|---:|---:|---:|']
    def number(value):return 'pendiente / no admisible' if value is None else f'{value:.4g}'
    for r in progress['summary']:
        review.append(f'| {r["candidate"]} | {r["inverted_flames"]} | {r["degenerate_flames"]} | '
            f'{number(r["loo_worst_p95_percent"])} | {number(r["loo_worst_maximum_percent"])} |')
    review += ['C1: CO2; C2: CO2+H2O; C3: CO2+H2O+CO+H2; C4: definición actual.',
        'El contraste interno retira filas del conjunto de construcción; las 12 retenidas no intervienen '
        'en la elección de coordenadas. Las escalas y estados comunes figuran en progress_candidates.json.',
        '## Reconstrucción independiente de la definición fijada']
    if metrics:
        review += [f'{len(retained)} llamas independientes; 1001 posiciones de progreso por llama.',
                   '| Campo | Media [%] | P95 [%] | Máximo [%] |', '|---|---:|---:|---:|']
        review += [f'| {r["field"]} | {r["mean"]:.4g} | {r["p95"]:.4g} | {r["maximum"]:.4g} |' for r in metrics]
        review += ['Errores normalizados por las escalas guardadas en error_metrics.csv. '
                   'La figura 03_fidelidad muestra una reconstrucción y la localización de los errores.']
    else:
        review += ['Pendiente de completar las llamas retenidas.']
    review += ['## Alcance de la decisión',
        'Revisar juntos retrocesos, normalizaciones, variación de campos en tramos casi constantes, '
        'errores por campo y sensibilidad Nc=121/241/481. El informe no cambia los pesos automáticamente.',
        'La evidencia corresponde a las condiciones evaluadas. Cada futura familia a otra presión, '
        'temperatura o transporte necesita su propia comprobación de coordenadas antes de tabular.',
        'Las repeticiones adicionales se destinan a medir dispersión temporal; esta revisión de fidelidad '
        'puede realizarse con una familia completa y las llamas independientes.']
    (out/'progress_review.md').write_text('\n\n'.join(review).replace('|\n\n|','|\n|')+'\n',encoding='utf-8')
    print(f'FGM report: {len(families)}/{m["repeats"]} families; holdouts={bool(holdout)}; query={bool(query)}; {out}')


if __name__=='__main__': main()
