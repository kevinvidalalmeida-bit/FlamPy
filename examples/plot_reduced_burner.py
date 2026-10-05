"""Twelve scientific diagnostic figures from published reduced-burner arrays.

No flame is solved and no curve is aligned or shifted to improve agreement.
PNG/SVG exports are independent of the optional multipage PDF.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
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
        fig.text(.08,.81,'CH₄-aire · 432 llamas · estado experimental en pérdidas pequeñas',fontsize=15,color='#335566')
        passed=sum(v['passed'] for v,p in records);converged=sum(v['accepted'] for v,p in records)
        worst_T=max(v['metrics']['temperature_K'] for v,p in records)
        worst_q=max(v['metrics']['wall_heat_flux_relative'] for v,p in records)*100
        worst_source=max(max(s.values()) for v,p in records for s in v['sources'].values())*100
        audit=json.loads((data/'parabolicity_audit.json').read_text(encoding='utf-8'))
        summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
        benchmark=json.loads((data/'benchmark.json').read_text(encoding='utf-8'))
        rows=[('Comparaciones físicas',f'{converged}/{len(records)} convergen; {passed}/{len(records)} cumplen todos los límites.\n'
               f'Máximos: temperatura {worst_T:.2f} K; calor hacia quemador {worst_q:.2f} %; fuentes {worst_source:.2f} %.\n'
               f"{summary['development_case_count']} casos de desarrollo y {summary['reserved_case_count']} condiciones nuevas de confirmación."),
              ('Tolerancias y referencias','Temperatura: 20 K; especies: 0.005; calor hacia quemador: 2 %; separación: 20 µm.\n'
               'Fuentes: error / pico ≤ 5 %, L1 ≤ 5 % e integral ≤ 3 %. Perfiles sin desplazamiento.\n'
               'Referencias nativa y Cantera verificadas; balances y refinamiento espacial separados.'),
              ('Estabilidad e interpolación',f"{audit['samples']:,} consultas: {audit['negative_samples']} con difusión negativa y "
               f"{audit['nonpositive_chart_samples']} con orientación no positiva.\n"
               'Cierre C1 limitado; proyección adaptativa de fuente y transporte. Extensión propia.\n'
               'Un muestreo finito no demuestra validez en cada punto de la tabla.'),
              ('Rendimiento y alcance',f"Jacobiano idéntico: mejora conjunta {benchmark['combined_speedup']:.2f}×; "
               f"lote grande, 4 hilos: {benchmark['large_batch_parallel_speedup']:.2f}×.\n"
               'Quemador plano isotérmico, CH₄-aire; cola química restringida al tramo monótono.\n'
               'Sin validación experimental, apagado transitorio, sólido conjugado ni H₂ con pérdidas.')]
        for (heading,paragraph),y in zip(rows,[.71,.55,.39,.23]):
            fig.text(.08,y,heading,fontsize=13,weight='bold')
            fig.text(.08,y-.035,paragraph,fontsize=11,va='top',linespacing=1.55)
        fig.text(.08,.04,'Referencias: Gövert et al. (2018), doi:10.1007/s10494-017-9848-4; Gupta et al. (2021),\n'
                 'doi:10.1080/13647830.2021.1926544; Luo et al. (2023), arXiv:2308.07833.\n'
                 'Los artículos motivan controles y diagnósticos; no reproducimos sus geometrías ni atribuimos garantías a nuestra extensión.',fontsize=9)
        book.savefig(fig);plt.close(fig)
    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=15, y=.98)
        fig.text(.5,.026,textwrap.fill(note,width=88 if fig.get_figwidth()<9 else 135),ha='center',va='bottom',fontsize=9)
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
         'δ: posición del máximo de |ΩC|. r: caudal impuesto / caudal de la llama libre.')

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
         'q ≈ ṁΔh comprueba el balance discreto impuesto; la referencia verifica la precisión física.')

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

    fig=plt.figure(figsize=(8.27,11.69));layout=fig.add_gridspec(2,2,width_ratios=[2.,1.],height_ratios=[1.,1.])
    ax=fig.add_subplot(layout[:,0]);names=['T\n20 K','Y\n0.005',r'$\Omega_C$'+'\n5/5/3 %',r'$\dot{q}$'+'\n5/5/3 %','q pared\n2 %','δ\n20 µm']
    values=[];labels=[]
    for v,p in records:
        m=v['metrics'];s=v['sources']
        values.append([m['temperature_K']/limits['temperature_K'],m['species_absolute']/limits['species_absolute'],
                      max(s['omega_C'][k]/limits['source_'+k] for k in s['omega_C']),
                      max(s['qdot'][k]/limits['source_'+k] for k in s['qdot']),
                      m['wall_heat_flux_relative']/limits['wall_heat_flux_relative'],m['stand_off_distance_m']/limits['stand_off_distance_m']])
        labels.append(f"{v['phi']:.3f} / {v['fraction']:.3f}"+(' *' if not v['accepted'] else ''))
    ratios=np.array(values);artist=ax.imshow(ratios,cmap='cividis',vmin=0,vmax=max(1.25,float(ratios.max())),aspect='auto')
    ax.set_xticks(range(len(names)),names,fontsize=8);ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.grid(False)
    ax.set_ylabel('φ / r');ax.set_title('Error / tolerancia (aceptable ≤ 1)')
    for i in range(len(labels)):
        for j in range(len(names)):
            rgba=artist.cmap(artist.norm(ratios[i,j]));luminance=np.dot(rgba[:3],[.2126,.7152,.0722])
            ax.text(j,i,f'{ratios[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if luminance<.5 else '#222222')
            if ratios[i,j]>1:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='#D55E00',linewidth=1.5))
    ax2=fig.add_subplot(layout[0,1]);ax3=fig.add_subplot(layout[1,1])
    convergence=data/'convergence.json'
    if convergence.exists():
        conv=json.loads(convergence.read_text(encoding='utf-8'))
        for label in dict.fromkeys(e['label'] for e in conv.get('grid',[])):
            group=[e for e in conv['grid'] if e['label']==label]
            ax2.plot([e['nodes'] for e in group],[e['temperature_change_K'] for e in group],marker='o',label=label.replace('\u00cf\u2020','φ'))
        ax2.axhline(limits['grid_temperature_change_K'],color='#555555',ls=':',label='Límite: 5 K')
        ax2.set_xlabel('Nodos espaciales');ax2.set_ylabel('Cambio máximo de T (K)');ax2.set_title('Refinamiento espacial');ax2.legend(fontsize=8,frameon=False)
        for entry in conv.get('domain',[]):
            x=np.asarray(entry['widths_m'])*1000;y=np.array(entry['outlet_temperature_K']);accepted=np.array(entry['accepted'])
            ax3.plot(x,np.where(accepted,y,np.nan),marker='o',label=entry['label'])
            ax3.plot(x[~accepted],y[~accepted],marker='x',color='#D55E00',ms=8,ls='none')
        ax3.set_xlabel('Longitud del dominio (mm)');ax3.set_ylabel('Temperatura de salida (K)');ax3.set_title('Extensión del dominio');ax3.legend(fontsize=8,frameon=False)
    else:
        for axis in [ax2,ax3]:axis.axis('off');axis.text(.5,.5,'Prueba pendiente',ha='center',transform=axis.transAxes)
    save(fig,'07_errores_convergencia','Errores, criterios de aceptación y comprobaciones espaciales',
         'Fuentes: máximo de errores pico, L1 e integral / límite. Recuadro: excede tolerancia. Cruz en dominio: FGM rechazado a 90 mm.')
    additional_figures(data, records, save)
    if book:book.close()
    return saved


def additional_figures(data,records,save):
    audit=json.loads((data/'parabolicity_audit.json').read_text(encoding='utf-8'))
    guard=audit['chart_guard']['chart_guard']
    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27))
    axes[0].plot([e['iteration'] for e in guard['history']],
        [e['bad_samples'] for e in guard['history']],marker='o',color=COLORS[0])
    axes[0].set_xlabel('Iteración de preparación del cierre')
    axes[0].set_ylabel('Consultas con orientación no admisible')
    axes[0].set_title('Control de geometría antes de resolver')
    total=audit['samples'];active=audit['projected_samples']
    axes[1].bar(['Cierre conservativo','Proyección activada'],[total-active,active],color=[COLORS[0],COLORS[1]])
    axes[1].set_yscale('log');axes[1].set_ylabel('Número de consultas (escala logarítmica)')
    axes[1].set_title(f"{audit['negative_samples']} difusiones negativas tras la corrección")
    axes[1].text(.5,.93,f'{active/total*100:.3f} % requieren proyección',ha='center',transform=axes[1].transAxes)
    save(fig,'08_estabilidad','Geometría del cierre y auditoría de difusión',
        f"{total:,} puntos: centro y ocho puntos de Gauss por celda. Auditoría finita; controles adicionales en nodos y caras.")

    bench=json.loads((data/'benchmark.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27))
    keys=['numpy_full','full','temperature_cache','cached']
    labels=['NumPy','Numba','Numba +\ncaché T','Numba +\ncachés T y P']
    times=[bench['timings'][k]['median_seconds']*1000 for k in keys]
    axes[0].bar(labels,times,color=[COLORS[1],COLORS[0],COLORS[2],'#56B4E9'])
    for i,v in enumerate(times):axes[0].text(i,v,f'{v:.1f}',ha='center',va='bottom',fontsize=10)
    axes[0].set_ylabel('Nueve residuos del Jacobiano (ms)');axes[0].set_ylim(0,max(times)*1.2)
    axes[0].set_title(f"Mismo problema: mejora {bench['combined_speedup']:.2f}×")
    times=[bench['large_batch'][k]['median_seconds']*1000 for k in ['serial','parallel']]
    axes[1].bar(['1 hilo','4 hilos'],times,color=[COLORS[1],COLORS[0]])
    for i,v in enumerate(times):axes[1].text(i,v,f'{v:.1f}',ha='center',va='bottom',fontsize=10)
    axes[1].set_ylabel('Consulta de 76 636 estados (ms)');axes[1].set_ylim(0,max(times)*1.2)
    axes[1].set_title(f"Lote grande: mejora {bench['large_batch_parallel_speedup']:.2f}×")
    save(fig,'09_rendimiento','Rendimiento medido sobre entradas idénticas',
        'Medianas de 9 repeticiones del Jacobiano y 7 del lote. P: multiplicadores de proyección; carga y preparación excluidas.')

    near=sorted([(v,p) for v,p in records if v['phi']==.985],key=lambda vp:vp[0]['fraction'])
    fig,axes=plt.subplots(1,3,figsize=(11.69,8.27))
    for prefix,style in STYLES.items():
        fractions=[v['fraction'] for v,p in near]
        axes[0].plot(fractions,[p[prefix+'_T'][-1] for v,p in near],marker='o',**style)
        axes[1].plot(fractions,[(v['solver']['diagnostics']['burner_heat_loss_W_m2'] if prefix=='reduced'
            else float(p[prefix+'_wall_heat_flux_W_m2']))/1000 for v,p in near],marker='o',**style)
        axes[2].plot(fractions,[(v['solver']['diagnostics']['outlet_enthalpy_loss_W_m2'] if prefix=='reduced'
            else float(p[prefix+'_outlet_heat_flux_W_m2']))/1000 for v,p in near],marker='o',**style)
    for ax,label in zip(axes,['T de salida (K)','Calor hacia quemador (kW m⁻²)','ṁ Δh (kW m⁻²)']):
        ax.set_xlabel('r = ṁ / ṁ de la llama libre');ax.set_ylabel(label)
    axes[0].legend(frameon=False,fontsize=8)
    save(fig,'10_limite_adiabatico','Reducción de pérdidas al aproximarse al caudal adiabático',
        'φ = 0.985; r llega a 0.95. Tendencia en ese intervalo; no demuestra convergencia matemática cuando r → 1.')

    tail=read(data/'reactor_tail.npz');meta=json.loads((data/'extended_metadata.json').read_text(encoding='utf-8'))
    species=json.loads((data/'manifest.json').read_text(encoding='utf-8'))['species_names']
    weights=np.array([meta['progress_species'].get(k,0.) for k in species])
    audits=json.loads(str(tail['audit_json']));index=max(range(len(audits)),key=lambda n:audits[n]['endpoint_species_gap'])
    row=sorted(meta['rows'],key=lambda r:(r['composition_index'],r['loss_index']))[index]
    C=tail['Y'][index]@weights;HP_C=float(tail['HP_Y'][index]@weights)
    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27));steps=np.arange(1,len(C)+1)
    axes[0].plot(steps,C,marker='o',color=COLORS[0],label='Estados del reactor tabulados')
    axes[0].plot(len(C)+2,HP_C,marker='x',ms=9,color=COLORS[1],ls='none',label='Equilibrio HP de referencia')
    axes[0].set_ylabel('Progreso físico C');axes[0].set_xlabel('Estado químico posterior a la llama');axes[0].legend(frameon=False,fontsize=8)
    n=species.index('NO');axes[1].plot(steps,tail['Y'][index,:,n],marker='o',color=COLORS[0])
    axes[1].plot(len(C)+2,tail['HP_Y'][index,n],marker='x',ms=9,color=COLORS[1],ls='none')
    axes[1].set_ylabel('Fracción másica Y(NO)');axes[1].set_xlabel('Estado químico posterior a la llama')
    save(fig,'11_cola_quimica','Alcance de la continuación química monótona',
        f"φ = {row['phi']:.3f}; r = {row.get('fraction',1.):.3f}. HP es una referencia separada; no se inserta como último estado.")

    sensitivity=json.loads((data/'sensitivity.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27))
    for color,(phi,r) in zip(COLORS,[(.735,.925),(1.295,.925),(.985,.95)]):
        label=f'case_{phi:.6f}_{r:.6f}';baseline=next(v for v,p in records if v['case_label']==label)
        refinements=sorted([d for d in sensitivity['weak_grid'] if d['case'].startswith(label)],key=lambda d:d['case'])
        levels=[0]+[int(d['case'][-1]) for d in refinements]
        metrics=[baseline['metrics']]+[d['metrics'] for d in refinements]
        axes[0].plot(levels,[d['wall_heat_flux_relative']*100 for d in metrics],marker='o',color=color,label=f'φ={phi:.3f}, r={r:.3f}')
        axes[1].plot(levels,[d['temperature_K'] for d in metrics],marker='o',color=color)
    axes[0].axhline(2.,color='#555555',ls=':',label='Límite: 2 %')
    axes[1].axhline(20.,color='#555555',ls=':')
    axes[0].set_ylabel('Error de calor hacia quemador (%)');axes[1].set_ylabel('Error máximo de temperatura (K)')
    for ax in axes:ax.set_xlabel('Refinamientos sucesivos de la malla');ax.set_xticks([0,1,2])
    axes[0].legend(frameon=False,fontsize=9)
    save(fig,'12_sensibilidad','El refinamiento espacial no elimina los fallos de calor',
        'Cada nivel biseca los intervalos anteriores. El fallo físico permanece aunque el residuo y la estabilidad numérica pasen.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('docs/assets/reduced-burner-hpc'))
    parser.add_argument('--output',type=Path,default=Path('output/figures/reduced-burner'))
    parser.add_argument('--pdf',type=Path)
    parser.add_argument('--table',type=Path)
    args=parser.parse_args()
    if args.pdf:args.pdf.parent.mkdir(parents=True,exist_ok=True)
    print(figures(args.data,output=args.output,pdf=args.pdf,table=args.table))


if __name__=='__main__':main()
