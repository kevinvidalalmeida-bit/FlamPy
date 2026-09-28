"""Offline report for four KFLAME and four Cantera adaptive base-state FGM."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from postprocess_fgm_adaptive_campaign import collect, write_csv, LABELS
from benchmark_fgm_campaign import atomic


def compare_tables(kpath, cpath):
    """Concordance on common normalized coordinates; not independent validation."""
    with np.load(kpath, allow_pickle=False) as d: k = dict(d)
    with np.load(cpath, allow_pickle=False) as d: c = dict(d)
    zs = np.linspace(max(k['Z_grid'][0],c['Z_grid'][0]), min(k['Z_grid'][-1],c['Z_grid'][-1]),101)
    cs = np.linspace(0,1,501)
    fields = ['T', 'CO2', 'CO', 'omega_c']
    def sample(table, field):
        source = table[field] if field in ('T','omega_c') else table['Y'][:,list(table['species_names']).index(field),:]
        by_c = np.array([np.interp(cs,table['c_grid'],row) for row in source])
        return np.array([np.interp(zs,table['Z_grid'],by_c[:,i]) for i in range(len(cs))]).T
    rows = []
    for field in fields:
        a,b = sample(k,field),sample(c,field)
        scale = float(np.max(b)-np.min(b)) if field=='T' else float(np.max(abs(b)))
        errors = 100*abs(a-b)/max(scale,1e-30)
        rows.append(dict(field=field,scale=scale,mean_percent=float(errors.mean()),
                         p95_percent=float(np.percentile(errors,95)),maximum_percent=float(errors.max())))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/thesis_fgm_comparison'))
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv); source=args.input.resolve()
    if not (source/'manifest.json').exists(): print('No campaign yet; no files written.');return 0
    m,rows,attempts=collect(source,expected_kind='fgm-solver-comparison')
    out=args.output or source/'report';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'conditions.csv',rows);atomic(out/'conditions.json',rows);atomic(out/'attempts.json',attempts)
    counts={s:sum(r['status']==s for r in rows) for s in ('accepted','failed','pending','incomplete')}
    atomic(out/'status.json',dict(counts,expected=8,repetitions=1,technical_check=m['technical_check']))
    lines=[r'\begin{table}[!htbp]\centering\small',
           r'\caption{Construcciones FGM del caso base (CH$_4$--aire, GRI-Mech~3.0, 300 K, 1 atm). K: KFLAME; C: Cantera; P: promediado; M: multicomponente; S: Soret. Una ejecución por fila. Los tiempos individuales excluyen calentamiento y escritura e incluyen todas las inserciones. $t_f$: llamas y propiedades; $t_a$: tabulación y selección. El total incluye también preparación y control de progreso. $d_{\max}$ es el indicador interno final, en porcentaje.}',
           r'\label{tab:fgm-comparison-cost}',
           r'\begin{tabular}{llrrrrr}\toprule',
           r'Solver & Transporte & $N_f$ & $t_f$ [s] & $t_a$ [s] & Total [s] & $d_{\max}$\\\midrule']
    for tag in LABELS:
        for solver in ('kflame','cantera'):
            r=next(r for r in rows if r['transport']==tag and r['solver']==solver)
            lead=f'{"K" if solver=="kflame" else "C"} & {dict(P="P",PS="P+S",M="M",MS="M+S")[tag]}'
            if r['status']=='accepted':
                ta=r['tabulation_s']+r['selection_s']
                lines.append(lead+f' & {r["n_flames"]} & {r["family_s"]:.2f} & {ta:.2f} & {r["compute_s"]:.2f} & {100*r["max_defect"]:.3f} '+r'\\')
            else:
                label='Fallido' if r['status']=='failed' else 'Pendiente'
                lines.append(lead+f' & \\multicolumn{{5}}{{c}}{{{label}}} '+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    if m['technical_check']:
        lines[1] = lines[1].replace(r'\caption{', r'\caption{Prueba técnica, fuera de la campaña científica ($N_c=61$, objetivo del 10\%). ')
    (out/'table_cost.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if counts['accepted']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.grid':False,
                             'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
        fig,axs=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
        tags=list(LABELS)
        for index,(solver,label,color) in enumerate([('kflame','KFLAME','#246091'),('cantera','Cantera','#bd682e')]):
            group=[next(r for r in rows if r['solver']==solver and r['transport']==tag) for tag in tags]
            x=np.arange(4)+(index-.5)*.34
            for ax,key in zip(axs,('compute_s','n_flames')):
                vals=[r[key] if r['status']=='accepted' else np.nan for r in group]
                bars=ax.bar(x,vals,width=.31,label=label,color=color)
                for bar,value in zip(bars,vals):
                    if np.isfinite(value):
                        ax.annotate(f'{value:.1f}' if key=='compute_s' else str(int(value)),
                                    (bar.get_x()+bar.get_width()/2,value),xytext=(0,4),
                                    textcoords='offset points',ha='center',fontsize=8)
        for ax in axs:
            ax.set_xticks(range(4),['Prom.','Prom.\n+ Soret','Multi.','Multi.\n+ Soret'])
            ax.margins(y=.20)
        axs[0].set_ylabel('Tiempo individual [s]');axs[1].set_ylabel('Llamas finales, $N_f$')
        axs[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
        if m['technical_check']:fig.suptitle('Prueba técnica; fuera de la campaña científica')
        for ext in ('pdf','png'):fig.savefig(out/f'01_coste_comparado.{ext}',dpi=180)
        plt.close(fig)
        caption=(r'Coste total y tamaño final de ocho FGM adaptativos del caso base, uno por solver y transporte. '
                 r'Cada barra es una ejecución individual; los ausentes quedan sin barra. La selección de filas '
                 r'y la tabulación usan el mismo algoritmo y objetivo del 1\%, con hasta diez inserciones '
                 r'por ronda y $N_c=241$. Cada solver obtiene su propia familia. El coste incluye continuación, '
                 r'recuperación y adaptación, con escritura y calentamiento excluidos. Se compara el proceso completo; '
                 r'los tiempos carecen de intervalos estadísticos porque no hay repeticiones.')
        if m['technical_check']:caption=r'Prueba técnica de ejecución y guardado; objetivo del 10\% y $N_c=61$.'
        (out/'01_coste_comparado.tex').write_text(
            '\\begin{figure}[!htbp]\\centering\n'
            '\\includegraphics[width=\\textwidth,height=.45\\textheight,keepaspectratio]{\\FGMComparisonRoot/01_coste_comparado.pdf}\n'
            f'\\caption{{{caption}}}\\label{{fig:fgm-comparison-cost}}\n\\end{{figure}}\n',encoding='utf-8')
    else:(out/'01_coste_comparado.tex').unlink(missing_ok=True)
    agreement=[]
    for tag in LABELS:
        group={r['solver']:r for r in rows if r['transport']==tag and r['status']=='accepted'}
        if len(group)!=2:continue
        comparison=compare_tables(source/group['kflame']['attempt']/'fgm_table.npz',
                                  source/group['cantera']['attempt']/'fgm_table.npz')
        agreement.extend([dict(transport=tag,**r) for r in comparison])
    atomic(out/'table_concordance.json',agreement)
    if agreement:write_csv(out/'table_concordance.csv',agreement)
    (out/'README.md').write_text(
        '# Cuatro transportes, dos solvers\n\n'+json.dumps(counts)+'\n\n'
        'Los tiempos son individuales. Cada FGM selecciona sus filas; el tamaño puede diferir entre solvers. '
        'conditions.csv contiene el coste por etapa y attempts.json todos los intentos. '
        'table_concordance.json compara los campos tabulados sobre 101×501 coordenadas comunes '
        '(Z_in,c); es concordancia entre representaciones, no validación independiente. '
        'La escala del error es el rango térmico de Cantera para T y el máximo módulo para los otros campos. '
        'Los FGM ausentes o fallidos no participan en esa comparación.\n',encoding='utf-8')
    print(json.dumps(counts));print(out)
    return 0


if __name__=='__main__':raise SystemExit(main())
