"""Seven scientific diagnostic figures from published reduced-burner arrays.

No flame is solved and no curve is aligned or shifted to improve agreement.
PNG/SVG exports are independent of the optional multipage PDF.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from examples.validate_reduced_burner import distance

STYLES = {'reduced': dict(color='#0072B2', ls='-', lw=1.8, label='FGM reducido'),
          'native': dict(color='#222222', ls='--', lw=1.35, label='Química detallada nativa'),
          'reference': dict(color='#D55E00', ls=':', lw=1.65, label='Cantera independiente')}
COLORS = ['#0072B2', '#D55E00', '#009E73']


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def figures(data, *, output, pdf=None, table=None):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for folder in sorted(data.glob('case_*')):
        if (folder/'validation.json').exists():
            records.append((json.loads((folder/'validation.json').read_text(encoding='utf-8')),
                            read(folder/'comparison.npz')))
    if not records:
        raise ValueError('No reduced-burner comparison arrays found')
    main = [(v,p) for v,p in records if v['phi'] in [.815,.985,1.235]]
    phis = sorted({v['phi'] for v,p in main})
    settings = json.loads((data/'plan.json').read_text(encoding='utf-8'))['settings']
    limits = settings['limits']
    plt.rcParams.update({'font.size':10, 'axes.titlesize':11, 'axes.labelsize':10,
                         'legend.fontsize':9, 'figure.dpi':130, 'savefig.dpi':220,
                         'axes.grid':True, 'grid.alpha':.2, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none'})
    book = PdfPages(pdf) if pdf else None
    saved = []
    if book:
        fig=plt.figure(figsize=(11.69,8.27))
        fig.text(.08,.87,'FGM con pérdidas hacia un quemador',fontsize=23,weight='bold')
        fig.text(.08,.81,'CH₄-aire · 432 llamas · controles físicos (Z, C, h)',fontsize=15,color='#335566')
        passed=sum(v['passed'] for v,p in records);converged=sum(v['accepted'] for v,p in records)
        rows=[('Comprobaciones independientes',f'{converged}/{len(records)} casos convergen; {passed}/{len(records)} cumplen todas las tolerancias.\n'
               'Los cuatro casos reservados después de fijar el algoritmo cumplen los límites.'),
              ('Qué queda pendiente','φ = 0.815, r = 0.575: error del flujo térmico 2.27 % y error L1 de ΩC 5.94 %.\n'
               'φ = 1.235, r = 0.575: error L1 de la liberación química de calor 5.47 %.\n'
               'φ = 1.235, r = 0.095: el residuo se estanca; ese caso se marca sin converger.'),
              ('Tolerancias de comparación','Temperatura: 20 K; fracciones másicas: 0.005; flujo hacia el quemador: 2 %; δ: 20 µm.\n'
               'Fuentes: error máximo / pico ≤ 5 %, error L1 ≤ 5 % y error integral ≤ 3 %.'),
              ('Algoritmo y alcance','Matriz dispersa con nueve colores, caché de temperatura y aproximación convexa suave en progreso.\n'
               'Se conservan los 181 estados originales y se añaden 16 estados de relajación química por llama.\n'
               'Quemador plano isotérmico: estas pruebas no validan apagado transitorio ni una pared multidimensional.')]
        for (heading,paragraph),y in zip(rows,[.71,.55,.37,.23]):
            fig.text(.08,y,heading,fontsize=13,weight='bold')
            fig.text(.08,y-.035,paragraph,fontsize=11,va='top',linespacing=1.65)
        fig.text(.08,.06,'Referencias: van Oijen y de Goey (2000), doi:10.1080/00102200008935814; Gövert et al. (2018),\n'
                 'doi:10.1007/s10494-017-9848-4. Comparaciones numéricas; no se emplearon medidas experimentales.',fontsize=9)
        book.savefig(fig);plt.close(fig)
    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=15, y=.98)
        fig.text(.5,.026,note,ha='center',va='bottom',fontsize=9)
        fig.subplots_adjust(top=.88,bottom=.16 if name=='07_errores_convergencia' else .13,
                            left=.14 if name=='07_errores_convergencia' else .09,right=.95,wspace=.30,hspace=.37)
        for ext in ['png','svg']:
            path=output/(name+'.'+ext)
            fig.savefig(path)
            if ext=='svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',
                                encoding='utf-8',newline='\n')
        if book:book.savefig(fig)
        plt.close(fig)
        saved.append(name)
    def pick(phi,r):
        return next((v,p) for v,p in main if abs(v['phi']-phi)<1e-12 and abs(v['fraction']-r)<1e-12)
    def legend(fig,*,composition_colors=False):
        handles, labels = [], []
        for key,style in STYLES.items():
            drawing={k:v for k,v in style.items() if k!='label'}
            if composition_colors:drawing['color']='#333333'
            handles.append(plt.Line2D([],[],**drawing));labels.append(style['label'])
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.936),ncol=3,frameon=False)
    def curves(ax,p,key,*,scale=1.,species=None):
        for prefix,style in STYLES.items():
            y=p[prefix+'_'+key] if species is None else p[prefix+'_Y'][species]
            ax.plot(p[prefix+'_z']*1000,y*scale,**style)
        ax.set_xlim(0,3.0)
        ax.set_xlabel('Distancia al quemador, x (mm)')

    if table is not None:
        from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
        from kflame.chemistry.initialization import fresh_mixture
        from kflame.chemistry.mechanism import load_mechanism
        model=NonAdiabaticFGM(table)
        feed=fresh_mixture(load_mechanism(model.metadata['mechanism']),1.,model.metadata['fuel'],model.metadata['oxidizer'])
        fixed_Z=float(model.table['bilger_weights']@feed+float(model.table['bilger_offset']))
        C=np.linspace(float(model.table['C'].min()),float(model.table['C'].max()),220)
        h=np.linspace(float(model.table['h'].min()),float(model.table['h'].max()),180)
        xx,yy=np.meshgrid(C,h)
        mapped=model.lookup_batch(Z=np.full(xx.size,fixed_Z),C=xx.ravel(),h=yy.ravel(),outside='mask')
        p=dict(C_grid=C,h_grid_MJ_kg=h/1e6,fixed_Z=fixed_Z,
               T=mapped['T'].reshape(xx.shape),omega_C=mapped['omega_C'].reshape(xx.shape))
        np.savez_compressed(data/'manifold_slice.npz',**p)
    else:
        cached=data/'manifold_slice.npz'
        if not cached.exists():raise ValueError('Provide --table or the published manifold_slice.npz')
        p=read(cached)
    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27))
    for ax,key,label in zip(axes,['T','omega_C'],['Temperatura, T (K)',r'Fuente, $\Omega_C$ (kg m$^{-3}$ s$^{-1}$)']):
        values=np.ma.masked_invalid(p[key])
        artist=ax.pcolormesh(p['C_grid'],p['h_grid_MJ_kg'],values,shading='auto',cmap='cividis',rasterized=True)
        ax.contour(p['C_grid'],p['h_grid_MJ_kg'],values,levels=5,colors='white',linewidths=.45,alpha=.6)
        ax.set_xlabel('Progreso sin normalizar, C');ax.set_ylabel(r'Entalpía total, h (MJ kg$^{-1}$)');ax.set_title(label)
        ax.set_facecolor('#eeeeee');fig.colorbar(artist,ax=ax,shrink=.65,pad=.025)
    save(fig,'01_mapas_C_h','Tabla FGM: sección en progreso y entalpía total',
         f"Z local de Bilger = {float(p['fixed_Z']):.6f}. Consulta baricéntrica; gris: fuera de la tabla. h incluye energía de formación.")

    fig,axes=plt.subplots(3,len(phis),figsize=(11.69,8.27))
    for j,phi in enumerate(phis):
        v,p=pick(phi,.285)
        for k,key,label,scale in [(0,'T','Temperatura, T (K)',1.),(1,'C','Progreso sin normalizar, C',1.),
                                   (2,'h',r'Entalpía total, h (MJ kg$^{-1}$)',1e-6)]:
            curves(axes[k,j],p,key,scale=scale)
            if j==0:axes[k,j].set_ylabel(label)
            if k<2:axes[k,j].set_xlabel('')
        axes[0,j].set_title(f'φ = {phi:.3f}; r = 0.285')
    legend(fig)
    save(fig,'02_perfiles_T_C_h','Transporte reducido frente a dos soluciones detalladas',
         'x = 0: superficie a 300 K. r = ṁ / ṁ adiabático nativo. Comparaciones sin desplazar las curvas.')

    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27))
    for color,phi in zip(COLORS,phis):
        group=sorted([(v,p) for v,p in main if v['phi']==phi],key=lambda item:item[0]['mass_flux_kg_m2_s'])
        for prefix,style in STYLES.items():
            flux=[v['mass_flux_kg_m2_s'] for v,p in group]
            axes[0].plot(flux,[p[prefix+'_T'][-1] if prefix!='reduced' or v['accepted'] else np.nan for v,p in group],color=color,ls=style['ls'],marker='o',ms=4,
                         label=f'φ = {phi:.3f}' if prefix=='reduced' else None)
            axes[1].plot(flux,[distance(p[prefix+'_z'],p[prefix+'_omega_C'])*1e3 if prefix!='reduced' or v['accepted'] else np.nan for v,p in group],
                         color=color,ls=style['ls'],marker='o',ms=4)
            if prefix=='reduced':
                for v,p in group:
                    if not v['accepted']:
                        axes[0].plot(v['mass_flux_kg_m2_s'],p['reduced_T'][-1],marker='x',color='#CC3311',ms=9,ls='none')
                        axes[1].plot(v['mass_flux_kg_m2_s'],distance(p['reduced_z'],p['reduced_omega_C'])*1e3,marker='x',color='#CC3311',ms=9,ls='none')
    axes[0].set_ylabel('Temperatura a x = 30 mm, T (K)');axes[1].set_ylabel('Separación del quemador, δ (mm)')
    for ax in axes:ax.set_xlabel(r'Caudal másico superficial, $\dot{m}$ (kg m$^{-2}$ s$^{-1}$)')
    axes[0].legend(loc='best',frameon=False);legend(fig,composition_colors=True)
    save(fig,'03_temperatura_separacion','Temperatura de salida y posición de la zona reactiva',
         'δ: posición del máximo de |ΩC|. Puntos: r = 0.095, 0.285 y 0.575. Cruz roja: iteración FGM sin converger.')

    species=['H2','H','O','O2','OH','H2O','HO2','H2O2','C','CH','CH2','CH2(S)','CH3','CH4','CO']
    if (data/'manifest.json').exists():species=json.loads((data/'manifest.json').read_text(encoding='utf-8'))['species_names']
    fig,axes=plt.subplots(2,len(phis),figsize=(11.69,8.27))
    for j,phi in enumerate(phis):
        v,p=pick(phi,.285)
        for k,name in enumerate(['CO','OH']):
            curves(axes[k,j],p,'Y',species=species.index(name))
            if j==0:axes[k,j].set_ylabel(f'Fracción másica, Y({name})')
        axes[0,j].set_title(f'φ = {phi:.3f}; r = 0.285')
    legend(fig)
    save(fig,'04_CO_OH','Especies sensibles a las pérdidas de calor',
         'CO y OH se recuperan de la tabla; sus ecuaciones individuales no se transportan en el modelo reducido.')

    fig,axes=plt.subplots(2,2,figsize=(11.69,8.27))
    for color,phi in zip(COLORS,phis):
        group=sorted([(v,p) for v,p in main if v['phi']==phi],key=lambda item:item[0]['mass_flux_kg_m2_s'])
        for prefix,style in STYLES.items():
            flux=[v['mass_flux_kg_m2_s'] for v,p in group]
            q=[(-p['reduced_conductive_heat_flux'][0] if prefix=='reduced' else float(p[prefix+'_wall_heat_flux_W_m2'])) for v,p in group]
            plotted=np.array([heat if prefix!='reduced' or v['accepted'] else np.nan for (v,p),heat in zip(group,q)])
            axes[0,0].plot(flux,plotted/1000,color=color,ls=style['ls'],marker='o',ms=4,
                           label=f'φ = {phi:.3f}' if prefix=='reduced' else None)
            for (v,p),heat in zip(group,q):
                deficit=float(p[prefix+'_outlet_heat_flux_W_m2']) if prefix!='reduced' else float(v['solver']['diagnostics']['outlet_enthalpy_loss_W_m2'])
                axes[0,1].plot(deficit/1000,heat/1000,marker={'reduced':'o','native':'s','reference':'^'}[prefix],
                               color=color,ms=4,linestyle='none',alpha=.8)
                if prefix=='reduced' and not v['accepted']:
                    axes[0,0].plot(v['mass_flux_kg_m2_s'],heat/1000,marker='x',color='#CC3311',ms=9,ls='none')
                    axes[0,1].plot(deficit/1000,heat/1000,marker='x',color='#CC3311',ms=9,ls='none')
        v,p=pick(phi,.285)
        for prefix,style in STYLES.items():
            axes[1,0].plot(p[prefix+'_z_face']*1000,p[prefix+'_total_enthalpy_flux']/1000,
                           color=color,ls=style['ls'])
        axes[1,1].plot(p['reduced_z_face']*1000,-p['reduced_conductive_heat_flux']/1000,color=color,
                       ls='-',label=f'Conducción, φ = {phi:.3f}')
        axes[1,1].plot(p['reduced_z_face']*1000,p['reduced_diffusive_enthalpy_flux']/1000,color=color,ls='--')
    axes[0,0].set_xlabel(r'$\dot{m}$ (kg m$^{-2}$ s$^{-1}$)');axes[0,0].set_ylabel(r'Pérdida hacia quemador, q (kW m$^{-2}$)');axes[0,0].legend(frameon=False)
    lo,hi=axes[0,1].get_xlim();axes[0,1].plot([lo,hi],[lo,hi],color='#555555',ls=':',lw=.9)
    axes[0,1].set_xlabel(r'$\dot{m}(h_{entrada}-h_{salida})$ (kW m$^{-2}$)');axes[0,1].set_ylabel(r'q del gradiente térmico (kW m$^{-2}$)')
    axes[1,0].set_xlabel('Distancia, x (mm)');axes[1,0].set_ylabel(r'Flujo total de entalpía (kW m$^{-2}$)')
    axes[1,1].set_xlabel('Distancia, x (mm)');axes[1,1].set_ylabel(r'Conducción y difusión (kW m$^{-2}$)');axes[1,1].set_xlim(0,3)
    axes[1,1].text(.97,.97,'Continua: −qcond\nDiscontinua: ∑hk Jk',ha='right',va='top',transform=axes[1,1].transAxes,fontsize=9)
    legend(fig,composition_colors=True)
    save(fig,'05_flujos_balance','Flujos de calor y balance de entalpía total',
         'q ≈ ṁΔh comprueba el balance discreto impuesto; la referencia verifica la precisión física. Cruz roja: sin converger.')

    fig,axes=plt.subplots(2,len(phis),figsize=(11.69,8.27))
    for j,phi in enumerate(phis):
        v,p=pick(phi,.575)
        for k,key,label,scale in [(0,'omega_C',r'$\Omega_C$ (kg m$^{-3}$ s$^{-1}$)',1.),
                                  (1,'qdot',r'$\dot{q}_{química}$ (MW m$^{-3}$)',1e-6)]:
            curves(axes[k,j],p,key,scale=scale)
            if j==0:axes[k,j].set_ylabel(label)
        axes[0,j].set_title(f'φ = {phi:.3f}; r = 0.575')
    legend(fig)
    save(fig,'06_fuentes','Fuentes químicas para r = 0.575',
         'Las fuentes del FGM se tabulan. La liberación química de calor no se añade de nuevo a la ecuación de entalpía total.')

    fig=plt.figure(figsize=(11.69,8.27));layout=fig.add_gridspec(2,2,width_ratios=[1.4,1.],height_ratios=[1.,1.])
    ax=fig.add_subplot(layout[:,0]);names=['T\n20 K','Y\n0.005','ΩC\n5/5/3 %','q̇\nquímica\n5/5/3 %','q\nquemador\n2 %','δ\n20 µm']
    values=[];labels=[]
    for v,p in records:
        m=v['metrics'];s=v['sources']
        values.append([m['temperature_K']/limits['temperature_K'],m['species_absolute']/limits['species_absolute'],
                      max(s['omega_C'][k]/limits['source_'+k] for k in s['omega_C']),
                      max(s['qdot'][k]/limits['source_'+k] for k in s['qdot']),
                      m['wall_heat_flux_relative']/limits['wall_heat_flux_relative'],m['stand_off_distance_m']/limits['stand_off_distance_m']])
        labels.append(f"{v['phi']:.3f} / {v['fraction']:.3f}"+(' *' if not v['accepted'] else ''))
    ratios=np.array(values);artist=ax.imshow(ratios,cmap='cividis',vmin=0,vmax=max(1.25,float(ratios.max())),aspect='auto')
    ax.set_xticks(range(len(names)),names);ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.grid(False)
    ax.set_ylabel('φ / r');ax.set_title('Error / tolerancia (aceptable ≤ 1)')
    for i in range(len(labels)):
        for j in range(len(names)):
            ax.text(j,i,f'{ratios[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if ratios[i,j]<.55 else '#222222')
            if ratios[i,j]>1:ax.plot(j,i,marker='s',ms=25,markerfacecolor='none',markeredgecolor='#D55E00',markeredgewidth=1.5)
    ax2=fig.add_subplot(layout[0,1]);ax3=fig.add_subplot(layout[1,1])
    convergence=data/'convergence.json'
    if convergence.exists():
        conv=json.loads(convergence.read_text(encoding='utf-8'))
        for label in dict.fromkeys(e['label'] for e in conv.get('grid',[])):
            group=[e for e in conv['grid'] if e['label']==label]
            ax2.plot([e['nodes'] for e in group],[e['temperature_change_K'] for e in group],marker='o',label=label)
        ax2.axhline(limits['grid_temperature_change_K'],color='#555555',ls=':',label='Límite: 5 K')
        ax2.set_xlabel('Nodos espaciales');ax2.set_ylabel('Cambio máximo de T (K)');ax2.set_title('Refinamiento espacial');ax2.legend(fontsize=8,frameon=False)
        for entry in conv.get('domain',[]):
            ax3.plot(np.asarray(entry['widths_m'])*1000,entry['outlet_temperature_K'],marker='o',label=entry['label'])
        ax3.set_xlabel('Longitud del dominio (mm)');ax3.set_ylabel('Temperatura de salida (K)');ax3.set_title('Extensión del dominio');ax3.legend(fontsize=8,frameon=False)
    else:
        for axis in [ax2,ax3]:axis.axis('off');axis.text(.5,.5,'Prueba pendiente',ha='center',transform=axis.transAxes)
    save(fig,'07_errores_convergencia','Errores, criterios de aceptación y comprobaciones espaciales',
         'Fuentes: máximo de los cocientes de error pico, L1 e integral. Recuadro: excede tolerancia. *: solver sin converger.')
    if book:book.close()
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('docs/assets/reduced-burner'))
    parser.add_argument('--output',type=Path,default=Path('output/figures/reduced-burner'))
    parser.add_argument('--pdf',type=Path)
    parser.add_argument('--table',type=Path)
    args=parser.parse_args()
    if args.pdf:args.pdf.parent.mkdir(parents=True,exist_ok=True)
    print(figures(args.data,output=args.output,pdf=args.pdf,table=args.table))


if __name__=='__main__':main()
