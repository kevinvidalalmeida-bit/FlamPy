"""Offline thesis report: reads archived JSON/NPZ only, never imports solvers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = {'native':'KFLAME', 'cantera':'Cantera', 'blocks-off':'Sin sustitución compilada',
         'fits-off':'Sin reutilización molecular'}
COLORS = {'native':'#1368a6', 'cantera':'#b84b32', 'blocks-off':'#6d4b8c', 'fits-off':'#6d4b8c'}
TITLES = ['Convergencia espacial', 'Perfiles de referencia', 'Barrido de composición',
          'Barrido de presión', 'Barrido de temperatura', 'Efecto del transporte',
          'Tiempos de resolución', 'Ablaciones pareadas', 'Tiempos por modelo de transporte',
          'Mallas finales de las llamas de referencia']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_manifest(root):
    """Combine the immutable base design and its separately recorded extension."""
    root = Path(root)
    manifest = read(root/'manifest.json')
    path = root/'transport_extension.json'
    if path.exists():
        extension = read(path)
        if extension['base_manifest_sha256'] != hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Extension refers to a different base manifest')
        if extension['pairs'] != manifest['pairs'] or extension['environment'] != manifest['environment']:
            raise ValueError('Extension protocol differs from base campaign')
        ids = {c['id'] for c in manifest['cases']}
        extra = extension['cases']
        if len({c['id'] for c in extra}) != len(extra) or any(c['id'] in ids for c in extra):
            raise ValueError('Extension contains duplicate/shared conditions')
        manifest = dict(manifest, cases=manifest['cases'] + extra, transport_extension=extension)
    return manifest


def json_write(path,value):
    with Path(path).open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def load_records(root):
    """Read committed attempts directly, so stale indexes cannot hide results."""
    result=[]
    for directory in root.glob('*/**/pair-*'):
        if not directory.is_dir():
            continue
        committed=[]
        for path in directory.glob('attempt-*/pair.json'):
            info=read(path)
            if info['status']=='complete':
                committed.append(path)
        if committed:
            p=sorted(committed)[-1]
            result.extend(read(p.parent/name/'result.json') for name in read(p)['order'])
    return result


def paired_statistics(records, left='native', right='cantera', seed=20260927):
    pairs={}
    for row in records:
        if row.get('usable') and row['variant'] in (left,right):
            pairs.setdefault(row['repetition'],{})[row['variant']]=row['time_s']
    values=np.array([[v[left],v[right]] for _,v in sorted(pairs.items()) if left in v and right in v],dtype=float)
    n=len(values)
    if n==0:
        return dict(n_pairs=0,ratio=None,ci95=None)
    ratio=float(np.median(values[:,1])/np.median(values[:,0]))
    ci=None
    if n>=3:
        indices=np.random.default_rng(seed).integers(0,n,size=(20000,n))
        sample=values[indices]
        ratios=np.median(sample[:,:,1],axis=1)/np.median(sample[:,:,0],axis=1)
        ci=np.quantile(ratios,[.025,.975]).tolist()
    return dict(n_pairs=n,ratio=ratio,ci95=ci,
                numerator=right,denominator=left,
                paired_medians={left:float(np.median(values[:,0])),right:float(np.median(values[:,1]))})


def profile(root,row):
    with np.load(root/row['folder']/'profile.npz',allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def aligned(fields):
    target=fields['T'][0]+.5*(fields['T'][-1]-fields['T'][0])
    candidates=np.flatnonzero((fields['T'][:-1]-target)*(fields['T'][1:]-target)<=0)
    if not len(candidates):
        raise ValueError('No thermal midpoint crossing')
    j=int(candidates[0]); dt=fields['T'][j+1]-fields['T'][j]
    center=fields['z'][j]+(target-fields['T'][j])*(fields['z'][j+1]-fields['z'][j])/dt
    return fields['z']-center


def comparison(root, a, b):
    fa,fb=profile(root,a),profile(root,b)
    za,zb=aligned(fa),aligned(fb)
    grid=np.linspace(max(za.min(),zb.min()),min(za.max(),zb.max()),2001)
    result={'Su_percent':100*abs(a['Su']-b['Su'])/max(abs(b['Su']),1.e-30)}
    for key in ('T','qdot'):
        av,bv=np.interp(grid,za,fa[key]),np.interp(grid,zb,fb[key])
        norm=np.linalg.norm(bv)
        result[key+'_L2_percent']=100*float(np.linalg.norm(av-bv))/max(float(norm),1.e-30)
        result[key+'_max_abs']=float(np.max(abs(av-bv)))
    species=[]
    for i,name in enumerate(fa['species_names']):
        j=list(fb['species_names']).index(name)
        av,bv=np.interp(grid,za,fa['Y'][i]),np.interp(grid,zb,fb['Y'][j])
        if max(av.max(),bv.max())>=1.e-5:
            species.append(dict(name=str(name),L2_percent=100*float(np.linalg.norm(av-bv))/max(float(np.linalg.norm(bv)),1.e-30),max_abs=float(np.max(abs(av-bv)))))
    result['active_species']=species
    result['max_species_L2_percent']=max((v['L2_percent'] for v in species),default=0.)
    return result


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def summarize(root,records,manifest):
    summaries=[]
    for c in manifest['cases']:
        rows=[r for r in records if r['phase']=='main' and r['case_id']==c['id']]
        item=dict(c,statistics=paired_statistics(rows,seed=manifest['bootstrap_seed']),solvers={})
        for backend in ('native','cantera'):
            rr=[r for r in rows if r['variant']==backend]
            good=[r for r in rr if r.get('usable')]
            times=[r['time_s'] for r in good]
            summary=dict(recorded=len(rr),accepted=sum(bool(r.get('accepted')) for r in rr),usable=len(good),
                         failed=sum(not r.get('accepted',False) for r in rr),times=times,
                         median_s=float(np.median(times)) if times else None,
                         quartiles_s=np.quantile(times,[.25,.75]).tolist() if times else None)
            for field in ('Su','Tb','thickness','qdot_peak','nodes','width','mass_error','species_sum_error',
                          'elemental_error_mass_scaled','energy_error_sensible_scaled'):
                values=[r[field] for r in good if field in r]
                summary[field]=dict(min=min(values),max=max(values),median=float(np.median(values))) if values else None
            item['solvers'][backend]=summary
        errors=[]
        for rep in range(manifest['pairs']):
            pair={r['variant']:r for r in rows if r['repetition']==rep and r.get('usable')}
            if 'native' in pair and 'cantera' in pair:
                errors.append(dict(repetition=rep,**comparison(root,pair['native'],pair['cantera'])))
        item['profile_errors']=errors
        summaries.append(item)
    return summaries


def representative(c):
    return (c['transport']=='mixture-averaged' and not c['soret']) if c['fuel']=='CH4' else (c['transport']=='multicomponent' and c['soret'])


def reference_record(records,fuel,backend,smoke=False):
    """One shared selection for the profile and original-mesh figures."""
    candidates=[r for r in records if r.get('usable') and r['phase'] in ('main','smoke') and
        r['backend']==backend and r['condition']['fuel']==fuel and r['condition']['phi']==1 and
        r['condition']['temperature']==300 and r['condition']['pressure_atm']==1 and
        (representative(r['condition']) or smoke)]
    return min(candidates,key=lambda r:r['repetition']) if candidates else None


def in_sweep(c,kind):
    if kind=='composition':
        return c['pressure_atm']==1 and c['temperature']==300
    if kind=='pressure':
        return c['phi']==1 and c['temperature']==300 and representative(c)
    return c['phi']==1 and c['pressure_atm']==1 and representative(c)


def plots(root,out,records,manifest,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False})
    outputs=[]
    smoke=manifest.get('smoke',False)
    label='PRUEBA TÉCNICA: no pertenece a la campaña' if smoke else 'Campaña parcial' if any(s['statistics']['n_pairs']<manifest['pairs'] for s in summary) else 'Campaña principal'
    good=[r for r in records if r.get('usable')]
    level=manifest.get('spatial_protocol',{}).get('selected_level',4)
    unverified=manifest.get('spatial_protocol',{}).get('verification_required') is False
    policy=('GRI30: Newton/PTC-SER/BE; h2o2: Newton/BE; '+('prueba L0' if smoke else f'malla L{level}') if manifest.get('schema',1)>=2 else '')
    def finish(fig,num,caption):
        # Unpopulated axes are explicitly pending, never synthetic observations.
        for ax in fig.axes:
            if not ax.has_data():
                ax.text(.5,.5,'Verificación espacial no realizada' if num==1 and unverified else 'Pendiente de datos',transform=ax.transAxes,ha='center',va='center',color='.45',fontsize=8)
            handles,labels=ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8,loc='best')
            ax.grid(alpha=.15)
        fig.suptitle(f'{TITLES[num-1]}\n{label}'+('\n'+policy if policy else ''),fontsize=11)
        stem=f'{num:02d}_'+['convergencia','perfiles','composicion','presion','temperatura','transporte','tiempos','ablaciones','tiempos_transportes','mallas_finales'][num-1]
        fig.savefig(out/(stem+'.pdf'),bbox_inches='tight')
        fig.savefig(out/(stem+'.png'),dpi=160,bbox_inches='tight')
        plt.close(fig)
        available=not ((num==1 and not any(r['phase'].startswith('verify-') for r in good)) or
                       (num==8 and not any(r['phase']=='ablation' for r in good)))
        outputs.append(dict(file=stem+'.pdf',caption=caption+(' '+policy+'.' if policy else ''),available=available))

    fig,axs=plt.subplots(2,3,figsize=(8,6),layout='constrained')
    for i,fuel in enumerate(('CH4','H2')):
        for j,(p,t) in enumerate(((1,300),(10,300),(1,500))):
            ax=axs[i,j]; ax.set_title(f'{fuel}, {p} atm, {t} K'); ax.set_xlabel('Nodos'); ax.set_ylabel('Velocidad [m/s]')
            for backend in ('native','cantera'):
                rr=sorted([r for r in good if r['phase'].startswith('verify-L') and r['condition']['fuel']==fuel and r['condition']['pressure_atm']==p and r['condition']['temperature']==t and r['backend']==backend],key=lambda r:r['settings']['level'])
                if rr:
                    ax.plot([r['nodes'] for r in rr],[r['Su'] for r in rr],'-o',label=NAMES[backend],color=COLORS[backend])
                    if policy:
                        for r in rr:ax.annotate('L'+str(r['settings']['level']),(r['nodes'],r['Su']),xytext=(4,5 if backend=='native' else -12),textcoords='offset points',fontsize=7)
                dom=[r for r in good if r['phase']=='domain-L4' and r['condition']['fuel']==fuel and r['condition']['pressure_atm']==p and r['condition']['temperature']==t and r['backend']==backend]
                if dom:ax.scatter([r['nodes'] for r in dom],[r['Su'] for r in dom],marker='x',color=COLORS[backend],label=NAMES[backend]+' dominio ampliado')
    finish(fig,1,'Se utiliza L3 por elección del usuario, sin estudio de sensibilidad espacial previo en esta campaña.' if unverified else 'Velocidad de propagación frente al número de nodos en los seis estados de control. L4 es el nivel seleccionado y L5 la referencia más fina en el protocolo revisado; las cruces corresponden a L4 con dominio ampliado. Cada punto es una resolución independiente.')

    fig,axs=plt.subplots(2,3,figsize=(8,6),layout='constrained')
    for i,fuel in enumerate(('CH4','H2')):
        limits=[]
        for backend in ('native','cantera'):
            r=reference_record(good,fuel,backend,smoke)
            if r is None:continue
            f=profile(root,r); z=aligned(f)*1000
            progress=(f['T']-f['T'][0])/(f['T'][-1]-f['T'][0])
            active=np.flatnonzero((progress>=.05)&(progress<=.95))
            if len(active)>1:
                limits.append((z[active[0]],z[active[-1]]))
            axs[i,0].plot(z,f['T'],'-' if backend=='native' else '--',label=NAMES[backend],color=COLORS[backend])
            axs[i,1].plot(z,f['qdot']/1.e6,'-' if backend=='native' else '--',label=NAMES[backend],color=COLORS[backend])
            for species,style in zip((fuel,'O2','OH'),('-','--',':')):
                k=list(f['species_names']).index(species)
                axs[i,2].plot(z,f['Y'][k],style,label=f'{NAMES[backend]} {species}',color=COLORS[backend])
        for j,unit in enumerate(('Temperatura [K]','Calor liberado [MW/m³]','Fracción másica [kg/kg]')):
            axs[i,j].set_title(fuel); axs[i,j].set_ylabel(unit); axs[i,j].set_xlabel('Posición alineada [mm]')
            if limits:
                lo=min(v[0] for v in limits); hi=max(v[1] for v in limits); pad=.3*(hi-lo)
                axs[i,j].set_xlim(lo-pad,hi+pad)
    finish(fig,2,'Perfiles estequiométricos a 300 K y 1 atm: caso 1, CH4/GRI30 promediado sin Soret, y caso 9, H2/h2o2 multicomponente con Soret. Se alinean por el punto medio del salto térmico y se amplía la zona entre el 5 y el 95 por ciento del salto, con margen lateral. Se representa la primera repetición utilizable, también utilizada en la figura de mallas; todas se conservan para calcular errores.')

    modes=[('mixture-averaged',False),('mixture-averaged',True),('multicomponent',False),('multicomponent',True)]
    mode_names=['Promediado','Promediado + Soret','Multicomponente','Multicomponente + Soret']
    fig,axs=plt.subplots(4,2,figsize=(8,10),layout='constrained')
    for i,fuel in enumerate(('CH4','H2')):
        for j,mode in enumerate(modes):
            ss=sorted([s for s in summary if s['fuel']==fuel and in_sweep(s,'composition') and (s['transport'],s['soret'])==mode],key=lambda s:s['phi'])
            ax=axs[j,i]; ax.set_title(f'{fuel}: {mode_names[j]}'); ax.set_xlabel('Relación de equivalencia'); ax.set_ylabel('Velocidad [m/s]')
            for backend in ('native','cantera'):
                vv=[s for s in ss if s['solvers'][backend]['Su']]
                if vv:ax.plot([s['phi'] for s in vv],[s['solvers'][backend]['Su']['median'] for s in vv],'-o' if backend=='native' else '--s',label=NAMES[backend],color=COLORS[backend])
    finish(fig,3,'Velocidad de propagación en el barrido de composición a 300 K y 1 atm para los cuatro modelos de transporte. Cada solver se evalúa con el mismo mecanismo y condiciones.')

    for num,kind,key,xlabel in ((4,'pressure','pressure_atm','Presión [atm]'),(5,'temperature','temperature','Temperatura de entrada [K]')):
        fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained')
        for i,fuel in enumerate(('CH4','H2')):
            ss=sorted([s for s in summary if s['fuel']==fuel and in_sweep(s,kind)],key=lambda s:s[key])
            for backend in ('native','cantera'):
                vv=[s for s in ss if s['solvers'][backend]['Su']]
                if vv:axs[i,0].plot([s[key] for s in vv],[s['solvers'][backend]['Su']['median'] for s in vv],'-o' if backend=='native' else '--s',label=NAMES[backend],color=COLORS[backend])
            vv=[s for s in ss if s['profile_errors']]
            if vv:axs[i,1].plot([s[key] for s in vv],[max(e['Su_percent'] for e in s['profile_errors']) for s in vv],'-o',color='#555555')
            for ax in axs[i]:ax.set_xlabel(xlabel); ax.set_title(fuel)
            axs[i,0].set_ylabel('Velocidad [m/s]'); axs[i,1].set_ylabel('Discrepancia máxima de velocidad [%]')
            axs[i,1].set_ylim(bottom=0)
        finish(fig,num,f'Barrido de {"presión" if num==4 else "temperatura"} a composición estequiométrica, con transporte promediado en metano y multicomponente con Soret en hidrógeno. La discrepancia se normaliza por la velocidad de Cantera.')

    fig,axs=plt.subplots(2,3,figsize=(8,6),layout='constrained')
    contrasts=[(0,1,'Soret en promediado'),(2,3,'Soret en multicomponente'),(0,2,'Multicomponente frente a promediado')]
    for i,fuel in enumerate(('CH4','H2')):
        ss=[s for s in summary if s['fuel']==fuel and in_sweep(s,'composition')]
        for j,(a,b,title) in enumerate(contrasts):
            ax=axs[i,j]; ax.set_title(f'{fuel}\n{title}',fontsize=9); ax.set_xlabel('Relación de equivalencia'); ax.set_ylabel('Cambio de velocidad [%]')
            for backend in ('native','cantera'):
                points=[]
                for phi in sorted(set(s['phi'] for s in ss)):
                    lookup={(s['transport'],s['soret']):s['solvers'][backend]['Su'] for s in ss if s['phi']==phi}
                    if lookup.get(modes[a]) and lookup.get(modes[b]):
                        va,vb=lookup[modes[a]]['median'],lookup[modes[b]]['median']; points.append((phi,100*(vb-va)/va))
                if points:ax.plot(*zip(*points),'-o',label=NAMES[backend],color=COLORS[backend])
    finish(fig,6,'Cambios de velocidad debidos al modelo de transporte y a la activación de Soret, calculados dentro de cada solver respecto de la configuración indicada como base.')

    fig,axs=plt.subplots(2,3,figsize=(8,6),layout='constrained')
    for i,fuel in enumerate(('CH4','H2')):
        for j,(kind,key,xlabel) in enumerate((('composition','phi','Relación de equivalencia'),('pressure','pressure_atm','Presión [atm]'),('temperature','temperature','Temperatura [K]'))):
            # Representative paths keep this overview legible; all transport times remain in CSV.
            ss=sorted([s for s in summary if s['fuel']==fuel and in_sweep(s,kind) and representative(s)],key=lambda s:s[key])
            ax=axs[i,j]; ax.set_title(f'{fuel}'); ax.set_xlabel(xlabel); ax.set_ylabel('Tiempo [s]')
            for backend in ('native','cantera'):
                vv=[s for s in ss if s['solvers'][backend]['median_s'] is not None]
                if not vv:continue
                x=[s[key] for s in vv]; y=[s['solvers'][backend]['median_s'] for s in vv]
                q=np.array([s['solvers'][backend]['quartiles_s'] for s in vv])
                ax.errorbar(x,y,yerr=np.array([np.array(y)-q[:,0],q[:,1]-np.array(y)]),fmt='-o',capsize=3,label=NAMES[backend],color=COLORS[backend])
                for xx,s in zip(x,vv):ax.scatter([xx]*len(s['solvers'][backend]['times']),s['solvers'][backend]['times'],s=12,alpha=.5,color=COLORS[backend])
    finish(fig,7,'Tiempos de las rutas representativas: transporte promediado en metano y multicomponente con Soret en hidrógeno. Puntos: ejecuciones; línea: mediana; barras: cuartiles, que describen dispersión temporal. Los demás transportes se recogen en las tablas.')

    fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for i,(fuel,variant) in enumerate((('CH4','blocks-off'),('H2','fits-off'))):
        rr=[r for r in good if r['phase']=='ablation' and r['condition']['fuel']==fuel]
        ax=axs[i]; ax.set_title(fuel)
        short_label='Sin sustitución\ncompilada' if variant=='blocks-off' else 'Sin reutilización\nmolecular'
        ax.set_xticks([0,1],[short_label,'KFLAME']); ax.set_xlim(-.4,1.4); ax.set_ylabel('Tiempo [s]')
        for rep in sorted(set(r['repetition'] for r in rr)):
            pair={r['variant']:r for r in rr if r['repetition']==rep}
            if variant in pair and 'native' in pair:ax.plot([0,1],[pair[variant]['time_s'],pair['native']['time_s']],'-o',alpha=.6,label=f'Par {rep+1}')
    finish(fig,8,'Ablaciones independientes: sustitución compilada en metano y reutilización molecular en hidrógeno. Cada segmento une dos ejecuciones pareadas; los errores de perfil se registran en el resumen de ablaciones.')
    fig,axs=plt.subplots(4,2,figsize=(8,10),layout='constrained')
    for j,mode in enumerate(modes):
        for i,fuel in enumerate(('CH4','H2')):
            ax=axs[j,i];ax.set_title(f'{fuel}: {mode_names[j]}');ax.set_xlabel('Relación de equivalencia');ax.set_ylabel('Tiempo [s]')
            ax.set_yscale('log')
            ss=sorted([s for s in summary if s['fuel']==fuel and in_sweep(s,'composition') and
                       (s['transport'],s['soret'])==mode],key=lambda s:s['phi'])
            for backend in ('native','cantera'):
                vv=[s for s in ss if s['solvers'][backend]['median_s'] is not None]
                if not vv:continue
                xx=np.array([s['phi'] for s in vv]); yy=np.array([s['solvers'][backend]['median_s'] for s in vv])
                q=np.array([s['solvers'][backend]['quartiles_s'] for s in vv])
                ax.errorbar(xx,yy,yerr=[yy-q[:,0],q[:,1]-yy],fmt='-o',capsize=3,label=NAMES[backend],color=COLORS[backend])
                for x,s in zip(xx,vv):
                    times=s['solvers'][backend]['times']
                    ax.scatter(x+np.linspace(-.012,.012,len(times)),times,s=10,alpha=.6,color=COLORS[backend])
    finish(fig,9,'Tiempos frente a composición a 300 K y 1 atm para ambos mecanismos y los cuatro transportes, con eje temporal logarítmico para comparar razones de coste. Se muestran las cinco observaciones, desplazadas ligeramente en horizontal para distinguirlas, la mediana y los cuartiles. Las repeticiones instrumentadas se excluyen de estas estadísticas.')
    fig,axs=plt.subplots(3,2,figsize=(8,8.5),layout='constrained')
    mesh_rows=[]
    for col,fuel in enumerate(('CH4','H2')):
        refs=[]
        for backend in ('native','cantera'):
            r=reference_record(good,fuel,backend,smoke)
            if r is None:continue
            f=profile(root,r);z=aligned(f)*1000
            progress=(f['T']-f['T'][0])/(f['T'][-1]-f['T'][0]);active=(progress>=.05)&(progress<=.95)
            refs.append((r,f,z,active))
        if not refs:continue
        lo=min(z[active].min() for r,f,z,active in refs);hi=max(z[active].max() for r,f,z,active in refs)
        pad=.3*(hi-lo);lo-=pad;hi+=pad
        for r,f,z,active in refs:
            backend=r['backend'];color=COLORS[backend];style='-' if backend=='native' else '--'
            name=NAMES[backend];dz=np.diff(f['z'])*1e6;mid=.5*(z[:-1]+z[1:])
            axs[0,col].plot(z,f['T'],style,color=color,label=f'{name}: N={len(z)}, L={r["width"]*1000:g} mm')
            axs[0,col].plot(z,np.full_like(z,.05 if backend=='native' else .13),'|',color=color,
                transform=axs[0,col].get_xaxis_transform(),markersize=5,alpha=.6)
            mask=(z>=lo)&(z<=hi)
            axs[1,col].plot(z[mask],f['T'][mask],style,marker='o' if backend=='native' else 'x',
                markersize=2.5,markerfacecolor='none',linewidth=.9,color=color,label=name)
            axs[2,col].semilogy(mid,dz,style,color=color,label=name)
            mesh_rows.append(dict(case_id=r['case_id'],backend=backend,repetition=r['repetition'],
                folder=r['folder'],profile_sha256=r['profile_sha256'],nodes=len(z),width_mm=r['width']*1000,
                dz_min_um=float(dz.min()),dz_max_um=float(dz.max()),nodes_5_95=int(active.sum()),
                fraction_nodes_5_95=float(active.mean()),thermal_width_5_95_mm=float(z[active].max()-z[active].min())))
        title='Caso 1: CH4, promediado sin Soret' if fuel=='CH4' else 'Caso 9: H2, multicomponente + Soret'
        axs[0,col].set_title(title,fontsize=9);axs[0,col].set_ylabel('Temperatura [K]')
        axs[1,col].set_title('Frente ampliado: nodos originales');axs[1,col].set_ylabel('Temperatura [K]')
        axs[2,col].set_title('Espaciado local de la malla');axs[2,col].set_ylabel(r'$\Delta z$ [µm]')
        for row in (1,2):axs[row,col].set_xlim(lo,hi)
        spacings=[]
        for r,f,z,active in refs:
            mid=.5*(z[:-1]+z[1:]);spacings.extend((np.diff(f['z'])*1e6)[(mid>=lo)&(mid<=hi)])
        if spacings:axs[2,col].set_ylim(min(spacings)/1.4,max(spacings)*1.4)
        for row in range(3):axs[row,col].set_xlabel('Posición respecto al punto térmico medio [mm]')
    finish(fig,10,'Mallas de los mismos perfiles de la figura de referencia: primera repetición de CH4/GRI30 promediado sin Soret (caso 1) e H2/h2o2 multicomponente con Soret (caso 9), a equivalencia uno, 300 K y 1 atm. Arriba se muestran el dominio completo y los nodos como marcas en dos bandas; en el centro, el frente ampliado con todos sus nodos originales; abajo, la distancia entre nodos consecutivos, en escala logarítmica. El desplazamiento del origen al punto medio térmico conserva el espaciado de cada malla. N es el número de nodos y L la longitud del dominio.')
    write_csv(out/'mallas_referencia.csv',mesh_rows)
    return outputs


def latex_tables(records,summary,manifest=None,split=False):
    """Compact manuscript tables; per-case detail remains in CSV/JSON."""
    def fmt(value):
        return '--' if value is None else f'{value:.3g}'
    def table(caption,head,rows):
        if not rows:
            return [r'\paragraph{'+caption+r'.} Pendiente de datos.']
        n=len(head)
        return [r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+'}',
                r'\begin{tabular}{'+'l'+'r'*(n-1)+'}',r'\hline',
                ' & '.join(head)+r'\\\hline',*[' & '.join(map(str,r))+r'\\' for r in rows],
                r'\hline\end{tabular}\end{table}']
    spatial=[]
    for fuel in ('CH4','H2'):
        for p,t in ((1,300),(10,300),(1,500)):
            for backend in ('native','cantera'):
                rr=sorted([r for r in records if r.get('usable') and r['phase'].startswith('verify-L') and
                           r['condition']['fuel']==fuel and r['condition']['pressure_atm']==p and
                           r['condition']['temperature']==t and r['backend']==backend],key=lambda r:r['settings']['level'])
                if not rr:continue
                last=rr[-1]; delta=100*abs(last['Su']-rr[-2]['Su'])/abs(rr[-2]['Su']) if len(rr)>1 else None
                if manifest and manifest.get('schema',1)>=2:
                    base=next((r for r in rr if r['settings']['level']==4),None)
                    ref=next((r for r in rr if r['settings']['level']==5),None)
                    if base is None:continue
                    last=base
                    delta=100*abs(ref['Su']-base['Su'])/abs(base['Su']) if ref else None
                domains=[r for r in records if r.get('usable') and r['phase']==f'domain-L{last["settings"]["level"]}' and r['case_id']==last['case_id'] and r['backend']==backend]
                domain_delta=100*abs(domains[-1]['Su']-last['Su'])/abs(last['Su']) if domains else None
                spatial.append([f'{fuel}, {p} atm, {t} K / {NAMES[backend]}',last['nodes'],fmt(delta),fmt(domain_delta)])
    a=table('Sensibilidad de L4 frente a L5 y al dominio ampliado, normalizada por la velocidad de L4; objetivo inferior al 0.5 por ciento.' if manifest and manifest.get('schema',1)>=2 else 'Sensibilidad del último nivel disponible; los cambios de velocidad se expresan en porcentaje.',
            ['Estado / solver','Nodos',r'Malla [\%]',r'Dominio [\%]'],spatial)
    if manifest and manifest.get('spatial_protocol',{}).get('verification_required') is False:
        a=[r'\paragraph{Malla de producción.} Se utiliza L3 por elección del usuario; esta campaña no incluye una verificación espacial previa. Los nodos y dominios efectivos se conservan por ejecución.']
    errors=[]
    for fuel in ('CH4','H2'):
        for mode,soret in (('mixture-averaged',False),('mixture-averaged',True),('multicomponent',False),('multicomponent',True)):
            ss=[s for s in summary if s['fuel']==fuel and s['transport']==mode and s['soret']==soret]
            ee=[e for s in ss for e in s['profile_errors']]
            if not ee:continue
            label=f'{fuel} / '+('Prom.' if mode=='mixture-averaged' else 'Multi.')+(' + Soret' if soret else '')
            errors.append([label,len(ee),fmt(max(e['Su_percent'] for e in ee)),fmt(max(e['T_L2_percent'] for e in ee)),fmt(max(e['max_species_L2_percent'] for e in ee))])
    b=table('Máximos de discrepancia entre solvers sobre los pares disponibles; especies activas con pico de fracción másica de al menos $10^{-5}$.',
            ['Sistema / transporte','Pares',r'$S_u$ [\%]',r'$E_2(T)$ [\%]',r'$E_2(Y)$ [\%]'],errors)
    timing=[]
    for s in summary:
        if not representative(s) or s['phi']!=1 or (s['pressure_atm'],s['temperature']) not in ((1,300),(10,300),(1,500)):
            continue
        if not s['statistics']['n_pairs']:continue
        st=s['statistics']; ci=st['ci95']
        timing.append([f'{s["fuel"]}, {s["pressure_atm"]:g} atm, {s["temperature"]:g} K',
                       st['n_pairs'],fmt(st['paired_medians']['native']),fmt(st['paired_medians']['cantera']),fmt(st['ratio']),
                       '['+fmt(ci[0])+', '+fmt(ci[1])+']' if ci else '--'])
    c=table(r'Rendimiento en los estados de control. Tiempos medianos de pares completos; C/K es la razón Cantera/KFLAME y el intervalo es bootstrap pareado al 95\%.',
            ['Estado','Pares','K [s]','C [s]','C/K',r'IC 95\%'],timing)
    conservation=[]
    for fuel in ('CH4','H2'):
        for backend in ('native','cantera'):
            rr=[r for r in records if r.get('usable') and r['phase']=='main' and r['condition']['fuel']==fuel and r['backend']==backend]
            if rr:
                conservation.append([fuel+' / '+NAMES[backend],*[fmt(max(r[k] for r in rr)) for k in
                    ('mass_error','species_sum_error','elemental_error_mass_scaled','energy_error_sensible_scaled')]])
    d=table(r'Máximos de los diagnósticos adimensionales sobre todas las ejecuciones principales. Los balances elemental y energético se reconstruyen sobre la malla guardada.',
        ['Sistema / solver',r'$e_m$',r'$e_Y$',r'$e_Z$',r'$e_H$'],conservation)
    d.append(r'Aquí $e_m=\max|\dot m-\overline{\dot m}|/|\overline{\dot m}|$, $e_Y=\max|\sum_kY_k-1|$, '
             r'$e_Z=\max_{e,j}|F_{e,j}-F_{e,\mathrm u}|/|\dot m_{\mathrm u}|$ y '
             r'$e_H=(\max F_H-\min F_H)/(|\dot m_{\mathrm u}|\,\max c_p\,|T_{\mathrm b}-T_{\mathrm u}|)$. '
             r'$\dot m=\rho u$ es el flujo másico, la barra indica su media espacial, $F_e$ es el flujo del elemento $e$ y $F_H$ el flujo total de entalpía, incluida la conducción; el subíndice $j$ identifica una cara de la malla. Los flujos se reconstruyen con las propiedades de cada solver y conservan sensibilidad a la malla.')
    parts=dict(spatial=a,errors=b,performance=c,conservation=d)
    return parts if split else a+b+c+d


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root=args.input.resolve(); out=args.output.resolve()
    if out==root or root in out.parents and out.name in ('inputs','main','verify','ablation','smoke'):
        parser.error('Use a separate report directory')
    manifest=load_manifest(root); records=load_records(root)
    out.mkdir(parents=True,exist_ok=True)
    summary=summarize(root,records,manifest)
    main_rows=[r for r in records if r['phase']=='main']
    pending=[dict(case_id=s['id'],missing_pairs=manifest['pairs']-s['statistics']['n_pairs']) for s in summary if s['statistics']['n_pairs']<manifest['pairs']]
    ablations=[]
    for fuel,variant in (('CH4','blocks-off'),('H2','fits-off')):
        rr=[r for r in records if r['phase']=='ablation' and r['condition']['fuel']==fuel]
        diffs=[]
        for rep in range(manifest['pairs']):
            pair={r['variant']:r for r in rr if r['repetition']==rep and r.get('usable')}
            if 'native' in pair and variant in pair:diffs.append(dict(repetition=rep,**comparison(root,pair['native'],pair[variant])))
        ablations.append(dict(fuel=fuel,statistics=paired_statistics(rr,right=variant),profile_errors=diffs))
    report=dict(smoke=manifest['smoke'],expected_main=len(manifest['cases'])*manifest['pairs']*2,
                native_strategies=manifest.get('native_strategies',{}),
                spatial_protocol=manifest.get('spatial_protocol',{}),
                verification=read(root/'verification.json') if (root/'verification.json').exists() else {},
                recorded_main=len(main_rows),accepted_main=sum(bool(r.get('accepted')) for r in main_rows),
                usable_main=sum(bool(r.get('usable')) for r in main_rows),pending=pending,
                conditions=summary,ablations=ablations,
                interpretation='Temporal intervals describe repeated measurements, not discretization or chemical uncertainty.')
    json_write(out/'summary.json',report)
    write_csv(out/'configuracion.csv',manifest['cases'])
    write_csv(out/'ejecuciones.csv',[{k:r.get(k) for k in ('phase','case_id','repetition','variant','nonlinear_strategy','status','accepted','usable','time_s','process_s','Su','Tb','nodes','width','mass_error','species_sum_error','elemental_error_mass_scaled','energy_error_sensible_scaled','folder')} for r in records])
    write_csv(out/'rendimiento.csv',[dict(case_id=s['id'],**s['statistics'],solvers=s['solvers']) for s in summary])
    write_csv(out/'errores.csv',[dict(case_id=s['id'],**e) for s in summary for e in s['profile_errors']])
    spatial=[r for r in records if r['phase'].startswith(('verify-','domain-'))]
    write_csv(out/'sensibilidad.csv',[dict(case_id=r['case_id'],phase=r['phase'],solver=r['backend'],strategy=r.get('nonlinear_strategy'),level=r.get('settings',{}).get('level'),Su=r.get('Su'),nodes=r.get('nodes'),width=r.get('width'),accepted=r.get('accepted')) for r in spatial])
    write_csv(out/'verificacion.csv',[dict(fuel=fuel,**check) for fuel,v in report['verification'].items() for check in v.get('checks',[])])
    figures=plots(root,out,records,manifest,summary)
    # Independent fragments let the manuscript place each result next to its interpretation.
    json_write(out/'figures.json',figures)
    for name,part in latex_tables(records,summary,manifest,split=True).items():
        fragment='\n'.join(part)
        fragment=fragment.replace(r'\begin{tabular}',r'\label{tab:sweep-'+name+'}\n'+r'\begin{tabular}',1)
        (out/f'table_{name}.tex').write_text(fragment+'\n',encoding='utf-8')
    ablation_rows=[]
    for item in ablations:
        stats=item['statistics']; variant=stats.get('numerator')
        native=stats.get('paired_medians',{}).get('native')
        reference=stats.get('paired_medians',{}).get(variant)
        ci=stats.get('ci95')
        errors=item['profile_errors']
        max_su=max((entry['Su_percent'] for entry in errors),default=None)
        ablation_rows.append([
            'Sustitución compilada, CH$_4$' if item['fuel']=='CH4' else 'Reutilización molecular, H$_2$',
            stats.get('n_pairs',0),
            '--' if reference is None else f'{reference:.3f}',
            '--' if native is None else f'{native:.3f}',
            '--' if reference is None or native is None else f'{100*(1-native/reference):.1f}',
            '--' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]',
            '--' if max_su is None else f'{max_su:.1e}',
        ])
    ablation_table=[r'\begin{table}[htbp]\centering\small',
        r'\caption{Ablaciones pareadas con malla L3. La variante de referencia desactiva solamente el componente indicado; la columna optimizada es KFLAME. Los tiempos son medianas de cinco ejecuciones por variante; la reducción es $100(1-t_{\mathrm{KFLAME}}/t_{\mathrm{ref}})$, usando esas medianas. El intervalo bootstrap pareado del 95\% corresponde a su razón referencia/KFLAME, estimada con cinco pares. El último campo recoge la máxima discrepancia de velocidad entre las variantes sobre los cinco pares.}',
        r'\label{tab:sweep-ablations}\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lrrrrrr}\hline',
        r'Ensayo & Pares & Ref. [s] & KFLAME [s] & Red. [\%] & IC 95\% & $\max\Delta S_u$ [\%]\\\hline',
        *[' & '.join(map(str,row))+r'\\' for row in ablation_rows],
        r'\hline\end{tabular}}\end{table}']
    (out/'table_ablations.tex').write_text('\n'.join(ablation_table)+'\n',encoding='utf-8')
    for i,item in enumerate(figures,1):
        fragment='\n'.join([r'\begin{figure}[htbp]\centering',
            r'\includegraphics[width=\textwidth,height=.74\textheight,keepaspectratio]{\SweepReportRoot/'+item['file']+'}',
            r'\caption{'+item['caption']+'}',r'\label{fig:sweep-auto-'+str(i)+'}',r'\end{figure}']) if item['available'] else '% No data available.\n'
        (out/f'figure_{i:02d}.tex').write_text(fragment+'\n',encoding='utf-8')
    lines=[r'% Generated offline; paths are relative to \SweepReportRoot.',
           r'\paragraph{Estado de la campaña.}',
           f'Se han registrado {len(main_rows)} de {report["expected_main"]} resoluciones principales; '
           f'{report["usable_main"]} disponen de solución aceptada y diagnósticos completos. '
           f'Quedan {len(pending)} condiciones con pares pendientes o incompletos.']
    if manifest['smoke']:
        lines.append(r'\textbf{Este informe corresponde a una prueba técnica y no constituye evidencia de la campaña científica.}')
    lines.extend([r'\begin{center}\begin{tabular}{lrr}\hline',r'Sistema & Condiciones & Resoluciones previstas\\\hline'])
    for fuel, label in [('CH4', 'CH$_4$'), ('H2', 'H$_2$')]:
        count = sum(c['fuel'] == fuel for c in manifest['cases'])
        lines.append(f'{label}--aire & {count} & {count*manifest["pairs"]*2}'+r'\\')
    lines.append(r'\hline\end{tabular}\end{center}')
    if manifest.get('spatial_protocol',{}).get('verification_required') is False:
        lines.append(r'\paragraph{Configuración numérica.} GRI30 emplea Newton con PTC--SER y rescate Euler; h2o2 emplea Newton con Euler implícito directo. Se utiliza L3, con \texttt{slope}=0.01, \texttt{curve}=0.02 y \texttt{ratio}=2.5, elegido sin verificación espacial previa para esta campaña. Se mantienen los controles internos de aceptación y los diagnósticos por ejecución.')
    elif manifest.get('schema',1)>=2:
        lines.append(r'\paragraph{Configuración numérica.} GRI30 emplea Newton con PTC--SER y rescate Euler; h2o2 emplea Newton con Euler implícito directo. Se selecciona L4, con \texttt{slope}=0.005 y \texttt{curve}=0.01. La sensibilidad se comprueba frente a L5 y al ampliar el dominio, con un objetivo inferior al 0.5\% referido a L4.')
        passed=all(report['verification'].get(f,{}).get('passed') for f in ('CH4','H2'))
        lines.append('Verificación espacial: '+('superada en ambos sistemas.' if passed else 'pendiente o no superada; consultar los controles individuales.'))
    lines.extend(latex_tables(records,summary,manifest))
    for i,item in enumerate(figures,1):
        if not item['available']:
            continue
        lines.extend([r'\begin{figure}[htbp]\centering',
                      r'\includegraphics[width=\textwidth,height=.76\textheight,keepaspectratio]{\SweepReportRoot/'+item['file']+'}',
                      r'\caption{'+item['caption']+'}',r'\label{fig:sweep-auto-'+str(i)+'}',r'\end{figure}'])
    if not any(r['phase']=='ablation' for r in records):
        lines.append(r'\paragraph{Ablaciones.} La campaña de ablaciones está pendiente de ejecución; su contribución causal al rendimiento se evaluará con los pares previstos.')
    (out/'report.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'README.md').write_text(f'# Informe de barridos\n\n{len(main_rows)}/{report["expected_main"]} ejecuciones principales registradas. '
        f'{len(pending)} condiciones incompletas.\n\nLas figuras con datos ausentes lo indican expresamente. '
        'CSV: configuración, sensibilidad, errores, tiempos y conservación. JSON: estadísticas completas y ablaciones. '
        'Las bandas y cuartiles temporales no representan incertidumbre física.\n',encoding='utf-8')
    print(f'Informe offline: {out}; {len(main_rows)}/{report["expected_main"]} registros principales; {len(pending)} condiciones pendientes.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
