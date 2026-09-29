"""Assemble the thesis FGM figures and tables from verified saved campaigns.

No flame solver is called. --benchmark-queries runs only a small in-memory
interpolation benchmark; later assemblies reuse its saved observations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from benchmark_fgm_campaign import atomic, digest, completed, latest, verify_artifacts
from postprocess_fgm_adaptive_campaign import collect, LABELS


def wrapper(out,name,caption,label,height='.58'):
    (out/f'{name}.tex').write_text(
        '\\begin{figure}[H]\\centering\n'
        f'\\includegraphics[width=\\textwidth,height={height}\\textheight,keepaspectratio]'
        f'{{\\FGMChapterRoot/{name}.pdf}}\n\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n\\end{{figure}}\n',encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path,
                   default=ROOT/'TESIS_RESUL/corridas/FGM/thesis_fgm')
    p.add_argument('--comparison', type=Path,
                   default=ROOT/'TESIS_RESUL/corridas/FGM/thesis_fgm_comparison')
    p.add_argument('--output', type=Path,
                   default=ROOT/'TESIS_RESUL/corridas/reproduccion/figuras_FGM_CH4')
    p.add_argument('--benchmark-queries',action='store_true')
    args=p.parse_args(argv); out=args.output.resolve()
    reference=args.reference.resolve(); comparison=args.comparison.resolve()
    manifest,rows,attempts=collect(comparison,expected_kind='fgm-solver-comparison')
    if manifest['technical_check']: raise ValueError('Technical runs cannot be published in the thesis.')
    if len(rows)!=8 or any(r['status']!='accepted' for r in rows):
        raise ValueError('This chapter requires all eight completed FGM; inspect the campaign report.')
    for job in ('family/rep-01','holdout/rep-01'):
        result=completed(reference/job)
        if not result or result['status']!='accepted':raise ValueError(f'Missing reference data: {job}')
        verify_artifacts(latest(reference/job),result)
    out.mkdir(parents=True,exist_ok=True)
    provenance=[]
    def copy(source,name):
        target=out/name
        if source.suffix=='.tex':
            text=source.read_text(encoding='utf-8').replace('\\FGMReportRoot','\\FGMChapterRoot').replace('\\FGMComparisonRoot','\\FGMChapterRoot')
            target.write_text(text,encoding='utf-8')
        else:shutil.copy2(source,target)
        provenance.append(dict(source=str(source),sha256=digest(source),published=name))
    ref=reference/'report'; comp=comparison/'report'
    for base in ('03_fidelidad',):
        for ext in ('pdf','png','tex'):copy(ref/f'{base}.{ext}',f'{base}.{ext}')
    copy(comp/'01_coste_comparado.pdf','05_coste_comparado.pdf')
    copy(comp/'01_coste_comparado.png','05_coste_comparado.png')
    for name in ('table_errors','table_coordinates','table_sensitivity','table_progress_candidates'):
        copy(ref/f'{name}.tex',f'{name}.tex')
    copy(comp/'table_cost.tex','table_cost.tex')
    wrapper(out,'05_coste_comparado',
        r'Construcción completa de los ocho FGM a 300 K y 1 atm: tiempo individual y número final de llamas. '
        r'Cada barra corresponde a una construcción, sin repeticiones. Ambos solvers comparten el algoritmo de '
        r'tabulación, $N_c=241$ y el objetivo interno del 1\%, con hasta diez inserciones por ronda. '
        r'El tiempo incluye preparación, resolución, propiedades, controles y retabulaciones; excluye calentamiento y escritura.',
        'fig:fgm-comparison-cost','.44')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.grid':False,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    from fgm_chapter_figures import current_figures
    for source in current_figures(comparison, rows, out, LABELS):
        provenance.append(dict(source=str(source), sha256=digest(source), generated='01_familia / 02_mapa'))
    fig,axes=plt.subplots(2,2,figsize=(9.3,6),sharex=True,sharey=True,layout='constrained')
    diagnostics=[]
    for ax,(tag,title) in zip(axes.flat,LABELS.items()):
        for solver,label,color,marker in [('kflame','KFLAME','#246091','o'),('cantera','Cantera','#bd682e','s')]:
            row=next(r for r in rows if r['transport']==tag and r['solver']==solver)
            attempt=comparison/row['attempt']
            rounds=json.loads((attempt/'rounds.json').read_text())
            traces=json.loads((attempt/'trace.json').read_text())
            ax.semilogy([r['n_flames'] for r in rounds],[100*r['max_defect'] for r in rounds],
                        marker=marker,color=color,label=label,markersize=4,linewidth=1.2,markerfacecolor='none')
            diagnostics.append(dict(id=row['id'],n_flames=row['n_flames'],
                coordinate_invalid=sum(not t['coordinate']['valid'] for t in traces),
                minimum_c_step=min(t['coordinate']['min_step'] for t in traces),
                initial_five_time_s=sum(t['measured_s'] for t in traces[:5]),
                family_fraction=row['family_s']/row['compute_s'],
                predictors={kind:sum(t['predictor_kind']==kind for t in traces) for kind in {t['predictor_kind'] for t in traces}}))
        ax.axhline(1,color='.4',ls=':',lw=1)
        ax.set_title(title)
    for ax in axes[1]:ax.set_xlabel('Llamas resueltas, $N_f$')
    for ax in axes[:,0]:ax.set_ylabel('Indicador máximo [%]')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    for ext in ('pdf','png'):fig.savefig(out/f'04_refinamiento.{ext}',dpi=180)
    plt.close(fig)
    wrapper(out,'04_refinamiento',
        r'Evolución del indicador interno tras cada ronda de construcción de los ocho FGM. '
        r'Cada símbolo representa una tabla reconstruida, con el número de llamas efectivamente resueltas en el eje horizontal. '
        r'La línea punteada marca el objetivo del 1\%; el eje vertical es logarítmico. '
        r'El indicador combina errores de reconstrucción entre filas vecinas en temperatura, especies, calor liberado y fuente de progreso. '
        r'Las nuevas composiciones y el eje adaptativo de $c$ se actualizan en cada ronda.',
        'fig:fgm-adaptive-convergence','.51')
    concordance=json.loads((comp/'table_concordance.json').read_text())
    lines=[r'\begin{table}[!htbp]\centering\small',
        r'\caption{Concordancia de las tablas KFLAME--Cantera en $101\times501$ posiciones comunes de $(Z_{\rm in},c)$ por transporte. Cada celda presenta P95 (máximo) del error normalizado, en porcentaje. La escala térmica es el rango de $T$ de Cantera en ese muestreo; las otras escalas son los máximos módulos de cada campo de Cantera. P: promediado; M: multicomponente; S: Soret.}',
        r'\label{tab:fgm-table-concordance}',r'\begin{tabular}{lrrrr}\toprule',
        r'Transporte & $T$ & $Y_{\rm CO_2}$ & $Y_{\rm CO}$ & $\dot\omega_c$\\\midrule']
    for tag in LABELS:
        values=[next(r for r in concordance if r['transport']==tag and r['field']==field) for field in ('T','CO2','CO','omega_c')]
        lines.append(dict(P='P',PS='P+S',M='M',MS='M+S')[tag]+' & '+
                     ' & '.join(f'{r["p95_percent"]:.3f} ({r["maximum_percent"]:.3f})' for r in values)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (out/'table_concordance.tex').write_text('\n'.join(lines),encoding='utf-8')
    # Queries are measured on one saved production table; they never call a flame solver.
    native=next(r for r in rows if r['transport']=='P' and r['solver']=='kflame')
    tablepath=comparison/native['attempt']/'fgm_table.npz'
    querypath=comp/'chapter_query_timings.json'
    if args.benchmark_queries:
        from postprocess_fgm_campaign import query_benchmark
        with np.load(tablepath,allow_pickle=False) as d:table=dict(d)
        observations,points=query_benchmark(table,20260927)
        atomic(querypath,dict(table_sha256=digest(tablepath),seed=20260927,observations=observations))
        atomic(comp/'chapter_query_points.npz',dict(points=points),npz=True)
    if querypath.exists():
        queries=json.loads(querypath.read_text())
        if queries['table_sha256']!=digest(tablepath):raise ValueError('Query benchmark table changed.')
        lines=[r'\begin{table}[!htbp]\centering\small',
            r'\caption{Consulta en memoria de $T$, $Y_{\rm CO_2}$, $Y_{\rm CO}$ y $\dot\omega_c$ mediante interpolación lineal de la tabla KFLAME promediada (40 filas, 241 nodos de progreso). Cinco mediciones por tamaño de lote, con puntos reproducibles y calentamiento previo. Se muestran mediana [Q25, Q75]; la preparación del interpolador queda fuera del tiempo.}',
            r'\label{tab:fgm-query}',
            r'\begin{tabular}{rrr}\toprule Estados por llamada & Tiempo por estado [$\mu$s] & Estados por segundo\\\midrule']
        query_summary={}
        for batch in (1,10000):
            values=np.array([r['us_per_state'] for r in queries['observations'] if r['batch']==batch])
            q25,med,q75=np.quantile(values,[.25,.5,.75]);through=float(np.median(1e6/values))
            query_summary[str(batch)]=dict(median_us=med,q25=q25,q75=q75,states_per_second=through)
            lines.append(f'{batch} & {med:.3f} [{q25:.3f}, {q75:.3f}] & '+rf'\num{{{through:.3g}}}'+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        (out/'table_query.tex').write_text('\n'.join(lines),encoding='utf-8')
        atomic(out/'query_summary.json',query_summary)
        copy(querypath,'query_timings.json')
    import pymupdf
    doc=pymupdf.open(out/'02_mapa.pdf');doc[0].get_pixmap(matrix=pymupdf.Matrix(1.8,1.8)).save(out/'02_mapa.png');doc.close()
    atomic(out/'diagnostics.json',diagnostics)
    # Keep explanatory text and its figure together in reading order.
    for name in ('03_fidelidad.tex','table_errors.tex','table_concordance.tex','table_cost.tex','table_query.tex'):
        path=out/name
        if path.exists():path.write_text(path.read_text(encoding='utf-8').replace('[!htbp]','[H]'),encoding='utf-8')
    atomic(out/'provenance.json',dict(reference_manifest=digest(reference/'manifest.json'),
           comparison_manifest=digest(comparison/'manifest.json'),copied=provenance,
           figure_protocol='01,02,04,05: current eight adaptive families; 03: reference 44 rows and 12 held-out flames; no flame solves'))
    (out/'README.md').write_text(
        '# Figuras y cuadros del capítulo FGM\n\n'
        '| Archivo (PDF y PNG) | Contenido | Procedencia |\n|---|---|---|\n'
        '| 01_familia | Ocho recorridos de tiempo e inicialización, en orden real | Ocho FGM actuales, 318 llamas |\n'
        '| 02_mapa | T, u, CO2, CO, densidad, conductividad, calor y fuente | KFLAME promediado actual, 40 filas y 241 nodos de c |\n'
        '| 03_fidelidad | Reconstrucción y error | 12 llamas independientes de esa referencia |\n'
        '| 04_refinamiento | Indicador y número de filas por ronda | Ocho FGM nuevos |\n'
        '| 05_coste_comparado | Tiempo individual y tamaño final | Ocho FGM nuevos |\n\n'
        'Los .tex incluyen pies y etiquetas listos para el capítulo. Los cuadros corresponden '
        'a errores independientes, concordancia entre tablas, costes, consultas y controles del apéndice. '
        'provenance.json identifica los archivos originales. Los tiempos de construcción son individuales; '
        'las consultas se resumen mediante cinco mediciones, sin nuevas resoluciones de llama.\n\n'
        'Regenerar desde la raíz: `python tools/postprocess_fgm_solver_comparison.py` y '
        '`python tools/assemble_fgm_chapter.py`. Usar `--benchmark-queries` solo para repetir '
        'el ensayo breve de consultas en memoria.\n',encoding='utf-8')
    print(f'{len(rows)} accepted FGM; {sum(r["n_flames"] for r in rows)} stored flame profiles. Chapter folder: {out}')


if __name__=='__main__':main()
