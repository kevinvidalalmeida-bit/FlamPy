"""Offline tables for diagnostic replays. Imports no solver and never resolves flames."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import re
import numpy as np
import postprocess_flame_sweeps as pp


def cantera_log_metrics(text):
    """Cantera 3.2 loglevel=2 exposes steady dampStep outcomes, not transient inner Newton."""
    accepted=len(re.findall(r'Damping coefficient found \(solution has (?:not converged yet|converged)\)',text))
    rejected=text.count('No damped step can be taken without violating solution component bounds.')
    rejected+=text.count('No damping coefficient found (max damping iterations reached)')
    return dict(steady_newton_accepted=accepted,steady_newton_rejected=rejected,
                euler_rejected=text.count('Timestep failed-->'),
                damping_trials=len(re.findall(r'^\s+\d+\s+[-+\d.eE]+\s+[-+\d.eE]+\s+[-+\d.eE]+\s+[-+\d.eE]+\s+[-+\d.eE]+\s+\d+\s+\d+/\d+\s*$',text,re.M)),
                scope='Accepted/rejected steady damped Newton outcomes printed at loglevel=2; transient inner iterations unavailable. Iter column enumerates damping trials.')


def profile_totals(report):
    """Each report profile excludes its separate bootstrap child; add those once."""
    totals={name:dict(time_s=v['time_s'],count=v['count']) for name,v in report.get('profile',{}).items()}
    if report.get('transport_bootstrap'):
        for name,v in profile_totals(report['transport_bootstrap']).items():
            dst=totals.setdefault(name,dict(time_s=0.,count=0))
            dst['time_s']+=v['time_s'];dst['count']+=v['count']
    return totals


def metrics(r):
    ins=r['instrumentation'];history=[h for p in ins['phases'] for h in p.get('history',[])]
    counts=Counter((h.get('scheme','steady'),bool(h.get('ok'))) for h in history)
    native=r['backend']=='native';rep=r['report'];calls=ins['function_calls'];times=ins['exclusive_root_seconds']
    def count(kind):return calls.get(kind) if native else sum(rep[kind+'_count_stats'])
    def seconds(kind):return times.get(kind,0.) if native else sum(rep[kind+'_time_stats'])
    row=dict(case_id=r['case_id'],backend=r['backend'],nodes=r['nodes'],time_instrumented_s=r['time_s'],
        jacobian_calls=count('jacobian'),residual_calls=count('residual' if native else 'eval'),
        jacobian_s=seconds('jacobian'),residual_s=seconds('residual' if native else 'eval'),
        factorization_s=times.get('factorize'),backsolve_s=times.get('linear_solve'),
        factorization_calls=calls.get('factorize'),backsolve_calls=calls.get('linear_solve'),
        ptc_accepted=counts['PTC-SER',True] if native else None,
        ptc_rejected=counts['PTC-SER',False] if native else None,
        euler_accepted=sum(counts[s,True] for s in ('BE-direct','BE-fallback')) if native else
                       sum(e['kind']=='transient_accepted' for e in ins['cantera_events']),
        euler_rejected=sum(counts[s,False] for s in ('BE-direct','BE-fallback')) if native else None,
        steady_attempts=sum(h.get('phase')=='steady' for h in history) if native else r['log_counts']['newton_attempts'],
        newton_inner_iterations=sum(len(h.get('iterations',[])) for h in history) if native else None,
        reported_cantera_steps=sum(rep.get('time_step_stats',[])) if not native else None,
        bootstrap_s=rep.get('transport_bootstrap',{}).get('total_time_s') if native else None)
    row['unassigned_s']=r['time_s']-row['jacobian_s']-row['residual_s']-(row['factorization_s'] or 0)-(row['backsolve_s'] or 0)
    steady=[it for h in history if h.get('phase')=='steady' for it in h.get('iterations',[])]
    row['steady_newton_accepted']=sum(it.get('status') in ('step','ok') for it in steady) if native else None
    row['steady_newton_rejected']=sum(it.get('status') not in ('step','ok') for it in steady) if native else None
    return row


def table(caption,heads,rows):
    return '\n'.join([r'\begin{table}[htbp]\centering\footnotesize',r'\caption{'+caption+'}',
        r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{'+'l'+'r'*(len(heads)-1)+'}',
        r'\hline',' & '.join(heads)+r'\\\hline',*[' & '.join(map(str,row))+r'\\' for row in rows],
        r'\hline\end{tabular}\end{table}'])


def fmt(v):return '--' if v is None else str(v) if isinstance(v,int) else f'{v:.3g}'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3_diagnostics'))
    ap.add_argument('--campaign', type=Path,
                    default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3'))
    ap.add_argument('--conditioning', type=Path,
                    default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3_conditioning_v2'))
    ap.add_argument('--output', type=Path,
                    default=Path('TESIS_RESUL/corridas/reproduccion/thesis_flames_L3_report'))
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    records=pp.read(args.input/'index.json');prod=pp.read(args.campaign/'index.json')
    summary=pp.read(args.campaign/'report/summary.json');ss={s['id']:s for s in summary['conditions']}
    cases=pp.read(args.input/'manifest.json')['cases'];labels={c['id']:str(i+1) for i,c in enumerate(cases)}
    rows=[];phases=[]
    for r in records:
        if not r.get('usable'):continue
        row=metrics(r)
        if r['backend']=='cantera':
            logpath=args.input/r['folder']/'console.log'
            if logpath.exists():row.update(cantera_log_metrics(logpath.read_text(encoding='utf-8')))
        ref=next(p for p in prod if p['case_id']==r['case_id'] and p['backend']==r['backend'] and p['phase']=='main' and p.get('usable'))
        row['Su_vs_production_percent']=100*abs(r['Su']-ref['Su'])/abs(ref['Su'])
        a=pp.profile(args.input,r);b=pp.profile(args.campaign,ref)
        row['exact_production_profile']=all(np.array_equal(a[k],b[k]) for k in ('z','T','Y','u'))
        row['production_median_s']=ss[r['case_id']]['statistics']['paired_medians'][r['backend']]
        rows.append(row)
        for i,p in enumerate(r['instrumentation']['phases']):
            phases.append(dict(case_id=r['case_id'],backend=r['backend'],ordinal=i,**{k:v for k,v in p.items() if k!='history'},
                               transient_attempts=sum(h.get('phase')=='transient' for h in p.get('history',[]))))
    cond=[pp.read(p) for p in sorted(args.conditioning.glob('*/*/conditioning.json'))]
    pp.write_csv(out/'desglose_diagnostico.csv',rows);pp.write_csv(out/'etapas_diagnosticas.csv',phases)
    pp.write_csv(out/'condicionamiento.csv',[dict(case_id=r['case_id'],backend=r['backend'],dimension=r['dimension'],
        raw_estimate=r['raw']['kappa1_lower_estimate'],equilibrated_estimate=r['equilibrated']['kappa1_lower_estimate'],
        polish_changes=r.get('polish_changes')) for r in cond])
    pp.json_write(out/'diagnosticos.json',dict(runs=rows,conditioning=cond,
        scope='Single instrumented replay per solver/case. Main medians from five original pairs. Cantera retained statistics can omit interrupted stages; accepted-step callbacks span the entire solve.'))
    lines=[r'\subsection{Desglose del coste y trayectorias de resolución}',
        r'Se instrumentan diez estados, con una ejecución por solver y estado, precedida de un calentamiento independiente. Los ocho estados centrales cubren ambos mecanismos y las cuatro opciones de transporte; se añaden hidrógeno a 10 atm y una mezcla pobre con Soret. Las medianas e intervalos temporales proceden de los cinco pares de la campaña principal. Los tiempos instrumentados describen una trayectoria individual y contienen el coste del registro adicional.',
        r'Las llamadas al residual de \KFLAME incluyen evaluaciones completas de control, mientras que los contadores de \Cantera conservan las estadísticas publicadas al finalizar la resolución. Estos últimos pueden omitir etapas interrumpidas al ampliar el dominio, y excluyen las evaluaciones internas usadas para construir el Jacobiano por diferencias finitas. Los pasos Euler de \Cantera se cuentan mediante callbacks a lo largo de toda la resolución. Así se conserva la diferencia entre los pasos realmente observados y el contador final de la biblioteca. En las tablas, K y C identifican \KFLAME y \Cantera, y un guion indica un contador no disponible o no aplicable.']
    keyrows=[]
    for c in cases:
        s=ss[c['id']]['statistics'];name=('CH$_4$' if c['fuel']=='CH4' else 'H$_2$')+' / '+('Prom.' if c['transport']=='mixture-averaged' else 'Multi.')+(' + S' if c['soret'] else '')
        ci=s.get('ci95')
        interval=f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '--'
        keyrows.append([labels[c['id']],name,c['phi'],c['pressure_atm'],fmt(s['paired_medians']['native']),fmt(s['paired_medians']['cantera']),fmt(s['ratio']),interval])
    lines.append(table(r'Estados del diagnóstico, todos a 300 K. S indica Soret; $p$ se expresa en atm. K y C designan KFLAME y Cantera. Los tiempos son medianas de los cinco pares originales; C/K es su razón y el intervalo bootstrap pareado corresponde al 95\%.',
                       ['Caso','Sistema / transporte',r'$\phi$',r'$p$','K [s]','C [s]','C/K',r'IC 95\%'],keyrows))
    lines.append(table(r'Costes instrumentados [s] de una ejecución individual por caso y solver. $t_J$ y $t_F$ corresponden al Jacobiano y al residual; $t_L$ reúne factorización y sustituciones en K. Resto es la diferencia respecto al tiempo total e incluye tareas sin contador separado. Los contadores de C tienen el alcance descrito en el texto.',
        ['Caso/solver','Nodos',r'$N_J$',r'$N_F$',r'$t_J$',r'$t_F$',r'$t_L$','Resto','Total'],
        [[labels[r['case_id']]+'/'+('K' if r['backend']=='native' else 'C'),r['nodes'],r['jacobian_calls'],r['residual_calls'],fmt(r['jacobian_s']),fmt(r['residual_s']),fmt((r['factorization_s'] or 0)+(r['backsolve_s'] or 0)) if r['backend']=='native' else '--',fmt(r['unassigned_s']),fmt(r['time_instrumented_s'])] for r in rows]))
    lines.append(table(r'Trayectoria no lineal de una ejecución instrumentada por caso y solver. Los contadores acumulan las operaciones de esa resolución. SS designa la resolución estacionaria; A/R son actualizaciones aceptadas/rechazadas. Newton SS cuenta resultados de pasos amortiguados, incluidos los rechazos por límites; los intentos SS son llamadas completas. En C se extraen del log de nivel 2; Euler aceptado procede de callbacks y rechazado de los mensajes de fallo del paso temporal. Las iteraciones Newton internas de Euler no se incluyen en Newton SS. Una corrección PTC y un paso Euler convergido tienen costes distintos.',
        ['Caso/solver','Intentos SS','Newton SS A/R','PTC A/R','Euler A/R','Factoriz.','Sustituc.'],
        [[labels[r['case_id']]+'/'+('K' if r['backend']=='native' else 'C'),r['steady_attempts'],
          '--' if r['steady_newton_accepted'] is None else f"{r['steady_newton_accepted']}/{r['steady_newton_rejected']}",
          '--' if r['ptc_accepted'] is None else f"{r['ptc_accepted']}/{r['ptc_rejected']}",
          f"{r['euler_accepted']}/"+('--' if r['euler_rejected'] is None else str(r['euler_rejected'])),fmt(r['factorization_calls']),fmt(r['backsolve_calls'])] for r in rows]))
    bootstrap=[]
    for r in records:
        if r['backend']!='native' or 'transport_bootstrap' not in r['report']:continue
        boot=r['report']['transport_bootstrap'];t=boot['total_time_s']
        bootstrap.append([labels[r['case_id']],boot['n_points_final'],r['nodes'],fmt(t),fmt(r['time_s']-t),
                          r['report']['profile']['jacobian_build']['count']])
    lines.append(table(r'Inicialización multicomponente de K, medida en una ejecución instrumentada por caso: primero se resuelve con transporte promediado sin Soret y una malla preliminar más laxa. $N_0$ y $N_f$ son los nodos preliminares y finales. El tiempo restante incluye construcción del problema, cambio de transporte y adaptación final. La última columna cuenta los Jacobianos construidos tras la etapa preliminar.',
        ['Caso',r'$N_0$',r'$N_f$','Preliminar [s]','Restante [s]',r'$N_{J,\mathrm{final}}$'],bootstrap))
    if cond:
        lines += [r'\subsection{Condicionamiento de las linealizaciones}',
            r'Se estudian cuatro estados centrales, combinando ambos mecanismos con transporte promediado sin Soret y multicomponente con Soret. Se estima $\kappa_1(J)=\lVert J\rVert_1\lVert J^{-1}\rVert_1$ mediante productos con factores dispersos y \texttt{onenormest}, con semilla fija. El resultado es una estimación inferior, sujeta a aritmética de punto flotante. Se conserva el Jacobiano y se repite la estimación tras equilibrar primero las filas y después las columnas por su máximo absoluto: $\widetilde J=D_rJD_c$. Este escalado permite mostrar cuánto influye la magnitud de las variables y ecuaciones.',
            r'En \KFLAME se reconstruye la linealización estacionaria en la solución final. La interfaz de \Cantera permite extraer la matriz con el backend disperso; se realiza por ello una corrección Newton adicional sobre la malla ya convergida, conservando el tiempo original obtenido con el backend de banda. El cambio del perfil durante esta corrección se registra por separado. Cada matriz conserva las variables, ecuaciones y malla de su solver, de modo que sus magnitudes se interpretan dentro de esa formulación. El condicionamiento final caracteriza la sensibilidad local y por sí solo no establece la causa de una reducción del tiempo.']
        lines.append(table(r'Estimación de condicionamiento en norma 1 de una linealización estacionaria final por caso y solver, evaluada con sus escalas originales y equilibradas. $n$ es el orden de la matriz; la última columna es el cambio relativo de velocidad durante la corrección diagnóstica dispersa de C, en porcentaje.',
            ['Caso/solver',r'$n$',r'$\widehat\kappa_1(J)$',r'$\widehat\kappa_1(\widetilde J)$',r'$\Delta S_u$ [\%]'],
            [[labels[r['case_id']]+'/'+('K' if r['backend']=='native' else 'C'),r['dimension'],fmt(r['raw']['kappa1_lower_estimate']),fmt(r['equilibrated']['kappa1_lower_estimate']),fmt(r.get('polish_changes',{}).get('Su_percent'))] for r in cond]))
    (out/'diagnostics_tables.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    tables=[line for line in lines if line.startswith(r'\begin{table}')]
    for name,content in zip(('cases','cost','iterations','bootstrap','conditioning'),tables):
        content=content.replace(r'\begin{tabular}',r'\label{tab:diagnostic-'+name+'}\n'+r'\begin{tabular}',1)
        (out/f'table_diagnostic_{name}.tex').write_text(content+'\n',encoding='utf-8')
    details=[]
    for r in records:
        if r['backend']!='native' or not r.get('usable'):continue
        for name,value in profile_totals(r['report']).items():
            details.append(dict(case_id=r['case_id'],component=name,**value))
    pp.write_csv(out/'quimica_transporte_subtiempos.csv',details)
    selected=('residual_nodal_thermochemistry','residual_face_transport_and_flux',
              'jacobian_cache','jacobian_analytic_thermochemistry','jacobian_analytic_spatial')
    subrows=[]
    for c in cases:
        values={r['component']:r for r in details if r['case_id']==c['id']}
        if values:subrows.append([labels[c['id']],*[f"{values[k]['time_s']:.3f} / {values[k]['count']}" if k in values else '--' for k in selected]])
    subtable=table(r'Subtiempos de una ejecución instrumentada de KFLAME por caso: cada celda indica segundos / número de llamadas al bloque, sumando la etapa preliminar cuando existe. TC-F: evaluación termoquímica nodal del residual; Tr-F: transporte y flujos en caras; Caché-J: preparación de propiedades para el Jacobiano; TC-J: derivadas termoquímicas; Esp-J: ensamblaje espacial. Estas operaciones están incluidas en los tiempos globales de residual y Jacobiano; sus llamadas procesan varios nodos o caras. Cantera no expone este desglose de química y transporte en los registros de la campaña.',
                   ['Caso','TC-F [s/N]','Tr-F [s/N]','Caché-J [s/N]','TC-J [s/N]','Esp-J [s/N]'],subrows)
    subtable=subtable.replace(r'\begin{tabular}',r'\label{tab:diagnostic-subtimes}'+'\n'+r'\begin{tabular}',1)
    (out/'table_diagnostic_subtimes.tex').write_text(subtable+'\n',encoding='utf-8')
    print(f'{len(rows)} diagnostic records, {len(cond)} matrices; profiles exactly reproduced: {sum(r["exact_production_profile"] for r in rows)}/{len(rows)}')


if __name__=='__main__':main()
