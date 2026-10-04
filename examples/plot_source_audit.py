"""Plot source accuracy separately from detailed-flame convergence.

python examples/plot_source_audit.py --output output/figures/source_accuracy \
    --pdf output/pdf/Revision_fuentes_FGM.pdf
Published inputs contain six cases excluded from the training family.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def figures(bundle, output, pdf_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    bundle,output=Path(bundle),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    old=load(bundle/'baseline/source_audit.json')
    new=load(bundle/'refined/source_audit.json')
    lean=load(bundle/'convergence_lean.json')
    rich=load(bundle/'convergence_rich.json')
    reference=load(bundle/'cantera_convergence.json')
    if [c['case'] for c in old['cases']]!=[c['case'] for c in new['cases']]:
        raise ValueError('Before/after audits must use the same retained cases')
    pdf=None
    if pdf_path:
        Path(pdf_path).parent.mkdir(parents=True,exist_ok=True)
        pdf=PdfPages(pdf_path)

    def save(fig,name,page,note):
        fig.subplots_adjust(left=.09,right=.96,bottom=.17,top=.85,hspace=.63,wspace=.30)
        fig.text(.09,.04,note,fontsize=9,va='bottom')
        fig.text(.96,.025,str(page),ha='right',fontsize=9,color='.4')
        fig.savefig(output/f'{name}.png',dpi=200)
        if pdf:
            pdf.savefig(fig)
        plt.close(fig)

    labels=[rf'$\phi={c["phi"]:.3f}$'+'\n'+rf'$r={c["fraction"]:g}$' for c in new['cases']]
    x=np.arange(len(labels))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'dejavusans',
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,2,figsize=(11.7,8.3))
        fig.suptitle('Fuentes FGM: error máximo y error integrado',fontsize=17,y=.97)
        fig.text(.5,.914,f'{old["training_flames"]} frente a {new["training_flames"]} llamas | '
                 'Los seis casos de comprobación permanecen fuera de la tabla',ha='center')
        specs=[('omega_C','Linf_over_truth_peak',r'(a) Fuente de progreso $\Omega_C$',5.),
               ('qdot','Linf_over_truth_peak',r'(b) Liberación de calor $\dot q$',5.),
               ('omega_C','integral_error_over_abs_integral',r'(c) Integral de $\Omega_C$',3.),
               ('qdot','integral_error_over_abs_integral',r'(d) Integral de $\dot q$',3.)]
        for ax,(field,metric,title,limit) in zip(axes.flat,specs):
            a=[100*c['sources'][field][metric] for c in old['cases']]
            b=[100*c['sources'][field][metric] for c in new['cases']]
            ax.bar(x-.18,a,.36,color='#777777',label=f'Anterior: {old["training_flames"]}')
            ax.bar(x+.18,b,.36,color='#0077BB',label=f'Refinada: {new["training_flames"]}')
            ax.axhline(limit,color='#CC3311',ls=':',lw=1.5,label=f'Objetivo: {limit:g} %')
            ax.set(title=title,ylabel='Error [%]')
            ax.set_xticks(x,labels,fontsize=8)
            ax.set_ylim(0.,max(max(a),max(b),limit)*1.22)
            ax.grid(axis='y',alpha=.18)
        axes[0,0].legend(fontsize=8,ncol=3,loc='upper left')
        coverage=min(c['sources'][f]['absolute_source_coverage'] for c in new['cases'] for f in ('omega_C','qdot'))
        save(fig,'05_fuentes_antes_despues',1,
             'Error máximo: max|FGM - detallada| / max|detallada|. Error integrado: |integral(FGM - detallada)| / integral|detallada|.\n'
             f'Se integran solo intervalos cubiertos; se representa al menos el {100*coverage:.3f} % de la fuente. Los objetivos son propios de esta validación.')

        fig,axes=plt.subplots(2,3,figsize=(11.7,8.3))
        fig.suptitle('Comprobación directa de las fuentes químicas',fontsize=17,y=.97)
        for i,(ax,case) in enumerate(zip(axes.flat,new['cases'])):
            with np.load(bundle/'refined'/f'{case["case"]}_sources.npz',allow_pickle=False) as stored:
                data={k:stored[k] for k in stored.files}
            with np.load(bundle/'baseline'/f'{case["case"]}_sources.npz',allow_pickle=False) as stored:
                previous=stored['omega_C_fgm']
            z=data['z']*1000.
            ax.plot(z,data['omega_C_native'],color='#0077BB',lw=2.,label='Nativa detallada')
            ax.plot(z,data['omega_C_cantera_flame'],color='.15',ls=':',lw=1.8,label='Cantera independiente')
            ax.plot(z,previous,color='.65',ls='--',lw=1.2,label='FGM anterior')
            ax.plot(z,data['omega_C_fgm'],color='#CC3311',ls='--',lw=1.5,label='FGM refinada')
            active=np.flatnonzero(abs(data['omega_C_native'])>.01*np.max(abs(data['omega_C_native'])))
            ax.set(title=labels[i].replace('\n','; '),xlabel='Distancia al quemador [mm]',
                   ylabel=r'$\Omega_C$ [kg/(m³ s)]',xlim=(0.,1.1*z[active[-1]]))
            ax.grid(alpha=.18)
            ax.text(.97,.92,f'FGM: {100*case["sources"]["omega_C"]["Linf_over_truth_peak"]:.2f} %',
                    transform=ax.transAxes,ha='right',fontsize=9)
        handles,text=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,text,ncol=4,loc='upper center',bbox_to_anchor=(.5,.93),fontsize=9)
        save(fig,'06_perfiles_fuentes',2,
             'Cada panel amplía la zona de reacción; los errores se evalúan en los 30 mm del dominio, sin alinear ni desplazar las llamas.\n'
             'El FGM se consulta con los controles de una llama detallada. Esta comparación no resuelve las ecuaciones de transporte reducidas.')

        fig,axes=plt.subplots(2,2,figsize=(11.7,8.3))
        fig.suptitle('¿Está convergiendo la llama detallada?',fontsize=17,y=.97)
        fig.text(.5,.914,'Referencia independiente, balances físicos y sensibilidad a tolerancias y malla',ha='center')
        for offset,field,color,label in [(-.18,'detailed_flame_vs_cantera_C','#0077BB',r'$\Omega_C$'),
                                         (.18,'detailed_flame_vs_cantera_q','#CC3311',r'$\dot q$')]:
            axes[0,0].bar(x+offset,[100*c['sources'][field]['Linf_over_truth_peak'] for c in new['cases']],.36,color=color,label=label)
        axes[0,0].set(title='(a) Nativa / Cantera: fuentes del perfil',ylabel='Máximo error / pico [%]')
        axes[0,0].legend(fontsize=9)
        axes[0,1].bar(x,[100*c['conservation']['integrated_progress_balance_relative_error'] for c in new['cases']],color='#009988')
        axes[0,1].axhline(.5,color='.3',ls=':',label='Objetivo: 0,5 %')
        axes[0,1].set(title='(b) Balance integral de progreso',ylabel='Desequilibrio relativo [%]')
        axes[0,1].legend(fontsize=9)
        for ax in axes[0]:
            ax.set_xticks(x,np.arange(1,7))
            ax.set_xlabel('Caso de comprobación (orden de la página 1)')
        for offset,report,color,label in [(-.18,lean,'#0077BB',r'$\phi=0.922$, $r=0.35$'),
                                         (.18,rich,'#CC3311',r'$\phi=1.223$, $r=0.085$')]:
            axes[1,0].bar(np.arange(2)+offset,[100*c['omega_C']['Linf_over_truth_peak'] for c in report['checks']],.36,color=color,label=label)
            axes[1,1].bar(np.arange(2)+offset,[c['temperature_Linf_K'] for c in report['checks']],.36,color=color,label=label)
        axes[1,0].set(title='(c) Cambio de la fuente nativa',ylabel=r'Cambio de $\Omega_C$ / pico [%]',yscale='log')
        axes[1,0].axhline(100*reference['omega_C']['Linf_over_truth_peak'],color='.3',ls=':',label='Malla más fina: Cantera (caso rico)')
        axes[1,1].set(title='(d) Cambio de la temperatura nativa',ylabel='Máxima diferencia [K]')
        for ax in axes[1]:
            ax.set_xticks(np.arange(2),['Tolerancias / 100','Tolerancias / 100\ny malla más fina'])
            ax.legend(fontsize=8,loc='upper left')
        for ax in axes.flat:
            ax.grid(axis='y',alpha=.18)
        chemistry=max(c['same_state_chemistry']['omega_C_Linf_over_peak'] for c in new['cases'])
        save(fig,'07_convergencia_fuentes',3,
             f'Misma T y composición: química nativa / Cantera, error relativo máximo de la fuente de progreso {chemistry:.2g}.\n'
             'El balance usa integral(ΩC dz) = mdot (C_salida - C_entrada), con flujo difusivo nulo a la salida y entrada de Danckwerts.')
        confirmation_path=bundle.parent/'validation_confirmatory.json'
        failed_path=bundle/'development_failed_case400.npz'
        if confirmation_path.exists() and failed_path.exists():
            confirm=load(confirmation_path)
            development=load(bundle.parent/'validation_development.json')
            failed=load(bundle/'development_confirmation400.json')['cases'][2]
            panels=[(c,bundle.parent/f'confirmatory_{i:02d}_comparison.npz') for i,c in enumerate(confirm['cases'])]
            panels.append((development['cases'][2],bundle.parent/'development_02_comparison.npz'))
            fig,axes=plt.subplots(2,2,figsize=(11.7,8.3))
            fig.suptitle('Casos nuevos y comprobación de la corrección',fontsize=17,y=.97)
            fig.text(.5,.914,'Tres condiciones fijadas después de congelar la tabla y el caso que detectó un intervalo insuficiente',ha='center',fontsize=10)
            for i,(ax,(case,path)) in enumerate(zip(axes.flat,panels)):
                with np.load(path,allow_pickle=False) as stored:
                    data={k:stored[k] for k in stored.files}
                z=data['z']*1000.
                ax.plot(z,data['omega_C'],color='#0077BB',lw=2.,label='Nativa detallada')
                ax.plot(1000.*data['reference_z'],data['reference_omega_C'],color='.15',ls=':',lw=1.8,label='Cantera independiente')
                if i==3:
                    with np.load(failed_path,allow_pickle=False) as previous:
                        ax.plot(1000.*previous['z'],previous['fgm_omega_C'],color='.65',ls='--',label='FGM de 400 llamas')
                ax.plot(z,data['fgm_omega_C'],color='#CC3311',ls='--',lw=1.5,label='FGM de 450 llamas')
                active=np.flatnonzero(abs(data['omega_C'])>.01*np.max(abs(data['omega_C'])))
                title=rf'$\phi={case["phi"]:.3f}$; $r={case["fraction"]:g}$'
                ax.set(title=title,xlabel='Distancia al quemador [mm]',ylabel=r'$\Omega_C$ [kg/(m³ s)]',xlim=(0.,1.1*z[active[-1]]))
                ax.text(.97,.90,f'Error / pico: {100*case["sources"]["omega_C"]["Linf_over_truth_peak"]:.2f} %',transform=ax.transAxes,ha='right',fontsize=9)
                ax.grid(alpha=.18)
                if i in (0,3):
                    ax.legend(fontsize=8,loc='center right')
            old_error=100*failed['sources']['qdot']['integral_error_over_abs_integral']
            new_error=100*development['cases'][2]['sources']['qdot']['integral_error_over_abs_integral']
            save(fig,'08_confirmacion_fuentes',4,
                 'Los tres casos nuevos pasan los criterios de pico, integral, cobertura y comparación con la referencia detallada.\n'
                 f'Caso del panel inferior derecho: error integrado de calor {old_error:.2f} % → {new_error:.2f} %; se mantiene fuera del entrenamiento.')
    if pdf:
        pdf.close()
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path('docs/assets/nonadiabatic3d/source_accuracy'))
    parser.add_argument('--output',type=Path,default=Path('output/figures/source_accuracy'))
    parser.add_argument('--pdf',type=Path)
    args=parser.parse_args()
    print(figures(args.bundle,args.output,args.pdf))


if __name__=='__main__':
    main()
