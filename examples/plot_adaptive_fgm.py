"""Reproduce tolerance, performance and source figures from published data."""
import argparse
import json
from pathlib import Path

import numpy as np


def figures(bundle, output, pdf_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    read = lambda name: json.loads((bundle/name).read_text(encoding='utf-8'))
    tolerances, summary = read('tolerances.json'), read('summary.json')
    benchmark, adaptive = read('benchmarks.json'), read('adaptive_report.json')
    reports = [read(f'validation_{name}.json') for name in ('legacy','development','confirmation')]
    cases = [case for report in reports for case in report['cases']]
    blue, orange, green = '#0077BB','#EE7733','#009988'
    pdf = None
    if pdf_path:
        pdf_path = Path(pdf_path); pdf_path.parent.mkdir(parents=True,exist_ok=True)
        pdf = PdfPages(pdf_path)
    def save(fig,name,number,note):
        fig.text(.07,.047,note,fontsize=9,va='bottom',linespacing=1.4)
        fig.text(.94,.025,f'{number} / 3',ha='right',fontsize=9,color='.4')
        fig.savefig(output/(name+'.png'),dpi=200)
        if pdf: pdf.savefig(fig)
        plt.close(fig)
    try:
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                             'axes.spines.top':False,'axes.spines.right':False,
                             'xtick.direction':'in','ytick.direction':'in'}):
            fig,(left,right) = plt.subplots(1,2,figsize=(11.7,8.3),gridspec_kw={'width_ratios':[1.05,1.]})
            fig.subplots_adjust(left=.065,right=.95,bottom=.21,top=.79,wspace=.35)
            fig.suptitle('Tolerancias por magnitud y comprobación del FGM',fontsize=17,y=.95)
            fig.text(.5,.885,'CH₄-aire | 300 K | 101 325 Pa | GRI-Mech 3.0 | 17 casos fuera de la tabla',ha='center')
            left.axis('off')
            rows = [['Temperatura',f"{tolerances['temperature_K']:g} K",f"{summary['temperature_max_K']:.2f} K"],
                    ['Fracción másica',f"{tolerances['species_absolute']:g}",f"{summary['species_max_absolute']:.2e}"]]
            for label,key,field in [('Pico','Linf_over_truth_peak','source_peak_relative'),
                                    ('L1','L1_relative','source_L1_relative'),
                                    ('Integral','integral_error_over_abs_integral','source_integral_relative')]:
                for source,symbol in [('omega_C','ΩC'),('qdot','q̇')]:
                    maximum = max(c['sources'][source][key] for c in cases)
                    rows.append([f'{symbol}: {label}',f'{100*tolerances[field][source]:g} %',f'{100*maximum:.2f} %'])
            rows.append(['Cobertura de fuentes','≥ 99 %',f"≥ {100*summary['source_coverage_min']:.2f} %"])
            table = left.table(cellText=rows,colLabels=['Magnitud','Límite','Observado'],cellLoc='center',
                               colWidths=[.45,.27,.28],bbox=[0.,.04,1.,.9])
            table.auto_set_font_size(False);table.set_fontsize(10)
            for (r,c),cell in table.get_celld().items():
                cell.set_edgecolor('.78');cell.set_linewidth(.5)
                if r==0:cell.set_facecolor('#edf2f5');cell.set_text_props(weight='bold')
                else:cell.set_facecolor('white')
            left.set_title('(a) Límites escritos por el usuario',pad=15)
            labels = ['Temperatura','Especies','ΩC: pico','q̇: pico','ΩC: L1','q̇: L1','ΩC: integral','q̇: integral']
            ratios = [summary['temperature_max_K']/tolerances['temperature_K'],
                      summary['species_max_absolute']/tolerances['species_absolute']]
            for key,field in [('Linf_over_truth_peak','source_peak_relative'),('L1_relative','source_L1_relative'),
                              ('integral_error_over_abs_integral','source_integral_relative')]:
                ratios.extend(max(c['sources'][s][key] for c in cases)/tolerances[field][s] for s in ('omega_C','qdot'))
            right.barh(labels,ratios,color=blue,height=.6)
            right.axvline(1.,color='.15',ls='--',lw=1.3,label='Límite de aceptación')
            right.invert_yaxis();right.set(xlim=(0.,1.15),xlabel='Error observado / tolerancia',title='(b) Peor error de los 17 casos')
            right.grid(axis='x',alpha=.2);right.legend(loc='lower right',fontsize=9)
            save(fig,'01_tolerancias',1,
                '12 regresiones, 3 casos de desarrollo y 2 casos reservados; los cinco nuevos incluyen soluciones Cantera independientes.\n'
                'Errores de fuentes frente a perfiles nativos detallados. L1 integra diferencias absolutas y no cancela errores de signo opuesto.')

            fig,axes = plt.subplots(2,2,figsize=(11.7,8.3))
            fig.subplots_adjust(left=.085,right=.95,bottom=.21,top=.81,wspace=.32,hspace=.65)
            fig.suptitle('Coste de consulta y coste de seleccionar la biblioteca',fontsize=17,y=.96)
            query = benchmark['lookup'];n = benchmark['query_states']
            times = [1e6*query[name]['median_s']/n for name in ('baseline_scalar','optimized_scalar','optimized_batch')]
            labels = ['Anterior\nescalar','Nueva\nescalar','Nueva\npor lotes']
            axes[0,0].bar(labels,times,color=['.65',blue,green]);axes[0,0].set_yscale('log')
            axes[0,0].set(ylabel='µs por estado (escala log)',title=f'(a) Consulta: {n} estados idénticos')
            for x,y in enumerate(times):axes[0,0].text(x,y*1.12,f'{y:.1f}',ha='center',fontsize=9)
            axes[0,0].set_ylim(min(times)*.6,max(times)*2.5)
            times_chem = [1000*benchmark['chemistry'][name]['median_s'] for name in ('baseline','optimized')]
            axes[0,1].bar(['Anterior','Nueva'],times_chem,color=['.65',blue])
            axes[0,1].set(ylabel='ms por evaluación',title=f"(b) Tasas químicas: {benchmark['chemistry_profile_nodes']} estados")
            for x,y in enumerate(times_chem):axes[0,1].text(x,y+max(times_chem)*.04,f'{y:.2f}',ha='center',fontsize=9)
            axes[0,1].set_ylim(0.,max(times_chem)*1.25)
            axes[1,0].bar(['Uniforme','Evaluadas\ncon sondas','Retenidas\nen la tabla'],
                          [450,adaptive['evaluated_flames'],adaptive['selected_flames']],color=['.65',orange,blue])
            axes[1,0].set(ylabel='Número de llamas',ylim=(0.,530.),title='(c) Selección: se incluyen las sondas')
            for x,y in enumerate([450,adaptive['evaluated_flames'],adaptive['selected_flames']]):
                axes[1,0].text(x,y+15,str(y),ha='center',fontsize=10)
            loading = [benchmark['load_s'][name] for name in ('baseline','optimized')]
            axes[1,1].bar(['Anterior','Nueva'],loading,color=['.65',blue])
            axes[1,1].set(ylabel='Tiempo de carga [s]',title='(d) Carga inicial: tabla de 450 llamas')
            for x,y in enumerate(loading):axes[1,1].text(x,y+max(loading)*.04,f'{y:.2f}',ha='center')
            axes[1,1].set_ylim(0.,max(loading)*1.25)
            for ax in axes.flat:ax.grid(axis='y',alpha=.15)
            speed_scalar=times[0]/times[1];speed_batch=times[0]/times[2];speed_chem=times_chem[0]/times_chem[1]
            save(fig,'02_costes',2,
                f'Medianas en esta máquina: consulta escalar {speed_scalar:.1f}×; consulta por lotes {speed_batch:.0f}×; tasas químicas {speed_chem:.1f}× más rápidas.\n'
                '5 repeticiones para consulta y 3 para tasas; calentamiento excluido. Carga: una medición. 450 evaluadas en la selección; validación externa aparte.')

            fig,axes = plt.subplots(2,2,figsize=(11.7,8.3))
            fig.subplots_adjust(left=.085,right=.95,bottom=.24,top=.80,wspace=.32,hspace=.6)
            fig.suptitle('Fuentes químicas: comprobación de la región simplificada',fontsize=17,y=.95)
            fig.text(.5,.885,'Caso de desarrollo: φ = 1,127; r = 0,075; comparación de tablas de 414 y 432 llamas',ha='center')
            with np.load(bundle/'development_01_comparison.npz',allow_pickle=False) as d:fixed={k:d[k] for k in d.files}
            with np.load(bundle/'development_414_failed_case.npz',allow_pickle=False) as d:coarse={k:d[k] for k in d.files}
            for ax,key,symbol,units in [(axes[0,0],'omega_C','ΩC','kg/(m³ s)'),(axes[0,1],'qdot','q̇','MW/m³')]:
                factor=1e-6 if key=='qdot' else 1.
                active=np.flatnonzero(abs(fixed[key])>.002*np.max(abs(fixed[key])))
                lo,hi=1000*fixed['z'][active[[0,-1]]];margin=.1*(hi-lo)
                ax.plot(1000*fixed['z'],factor*fixed[key],color='.1',lw=1.8,label='Nativa detallada')
                ax.plot(1000*fixed['reference_z'],factor*fixed['reference_'+key],color='.55',ls='--',lw=1.5,label='Cantera')
                ax.plot(1000*coarse['z'],factor*coarse['fgm_'+key],color=orange,ls=':',lw=2.,label='FGM: 414 llamas')
                ax.plot(1000*fixed['z'],factor*fixed['fgm_'+key],color=blue,ls='-.',lw=1.5,label='FGM: 432 llamas')
                ax.set(xlim=(max(0.,lo-margin),hi+margin),xlabel='Posición [mm]',ylabel=f'{symbol} [{units}]',
                       title='(a) Fuente de progreso' if key=='omega_C' else '(b) Liberación de calor')
                ax.grid(alpha=.15)
            axes[0,0].legend(fontsize=8)
            before=read('development_414_failed.json')['cases']
            dev,reserved=reports[1]['cases'],reports[2]['cases']
            x=np.arange(5);width=.32
            axes[1,0].bar(x[:3]-width/2,[100*c['sources']['qdot']['L1_relative'] for c in before],width,
                          color=orange,label='414: desarrollo')
            axes[1,0].bar(x+width/2,[100*c['sources']['qdot']['L1_relative'] for c in dev+reserved],width,
                          color=blue,label='432: final')
            axes[1,0].axhline(5.,color='.2',ls='--',lw=1.2)
            axes[1,0].set(xticks=x,xticklabels=['D1','D2','D3','R1','R2'],ylabel='Error L1 de q̇ [%]',
                          title='(c) Error absoluto integrado',ylim=(0.,6.6))
            axes[1,0].legend(fontsize=8);axes[1,0].grid(axis='y',alpha=.15)
            coverage=[100*min(c['sources'][s]['absolute_source_coverage'] for s in ('omega_C','qdot')) for c in dev+reserved]
            axes[1,1].bar(x,coverage,color=green,width=.55)
            axes[1,1].axhline(99.,color='.2',ls='--',lw=1.2)
            axes[1,1].set(xticks=x,xticklabels=['D1','D2','D3','R1','R2'],ylabel='Cobertura de fuentes [%]',
                          ylim=(98.5,100.1),title='(d) Zona reactiva cubierta')
            axes[1,1].grid(axis='y',alpha=.15)
            save(fig,'03_fuentes',3,
                'D: desarrollo; R: casos reservados después de fijar el margen. Ventana de reacción arriba; errores integrados en todo el dominio.\n'
                'El ensayo de 414 llamas incumplió L1 de q̇ (5,31 %). La versión final pasa los cinco casos nuevos y las doce regresiones.\n'
                'Comprobación numérica a priori; transporte reducido y experimento pendientes. Método: docs/adaptive-fgm.md.\n'
                'Referencias: Ramaekers et al. (2010), DOI 10.1007/s10494-009-9223-1; Liu y Pope (2005), DOI 10.1080/13647830500307436.\n'
                'Kovaleva et al. (2022): Interpolation Error of FGM Tabulation, póster PROCI, repositorio ORCA de Cardiff.')
    finally:
        if pdf:pdf.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path('docs/assets/adaptive-fgm'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pdf',type=Path)
    args=parser.parse_args()
    figures(args.bundle,args.output,args.pdf)


if __name__=='__main__':
    main()
