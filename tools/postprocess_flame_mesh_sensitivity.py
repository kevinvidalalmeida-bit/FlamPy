"""Offline spatial-study figure and table. Never imports or invokes flame solvers."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import postprocess_flame_sweeps as pp


def spatial_profile_metrics(parent_root, study_root, base, row):
    """Compare an actual refined profile with its archived L3 reference."""
    if not row.get('usable'):
        return dict(T_L2_percent=None, qdot_L2_percent=None)
    try:
        reference = pp.profile(parent_root, base)
        candidate = pp.profile(study_root, row)
        zr, zc = pp.aligned(reference), pp.aligned(candidate)
        left, right = max(zr.min(), zc.min()), min(zr.max(), zc.max())
        if right <= left:
            raise ValueError('No common aligned profile interval')
        grid = np.linspace(left, right, 2001)
        result = {}
        for field in ('T', 'qdot'):
            a = np.interp(grid, zc, candidate[field])
            b = np.interp(grid, zr, reference[field])
            result[field + '_L2_percent'] = 100 * float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1.e-30)
        return result
    except (FileNotFoundError, KeyError, ValueError):
        # A partial study remains reportable without manufacturing profile errors.
        return dict(T_L2_percent=None, qdot_L2_percent=None)


def analyze(manifest,records,parent_root=None,study_root=None):
    rows=[]
    for c in manifest['cases']:
        for backend in ('native','cantera'):
            base=next(r for r in manifest['baseline'] if r['case_id']==c['id'] and r['backend']==backend)
            levels={3:base}
            for r in records:
                if r['case_id']==c['id'] and r['backend']==backend and r['phase'].startswith('verify-L'):
                    levels[r['settings']['level']]=r
            for level in manifest['levels']:
                r=levels.get(level,{})
                prev=levels.get(level-1,{})
                usable=bool(r.get('usable'))
                change=100*(r['Su']-base['Su'])/abs(base['Su']) if usable else None
                successive=100*abs(r['Su']-prev['Su'])/abs(prev['Su']) if usable and prev.get('usable') else None
                metrics=(spatial_profile_metrics(parent_root,study_root,base,r)
                    if parent_root is not None and study_root is not None and usable else {})
                rows.append(dict(case_id=c['id'],fuel=c['fuel'],pressure_atm=c['pressure_atm'],
                    temperature=c['temperature'],backend=backend,test=f'L{level}',level=level,
                    usable=usable,status=r.get('status','pending'),nodes=r.get('nodes'),Su=r.get('Su'),Tb=r.get('Tb'),
                    qdot_peak=r.get('qdot_peak'),
                    width=r.get('width'),signed_change_L3_percent=change,successive_change_percent=successive,
                    reused=level==3,**metrics))
            dom=next((r for r in records if r['case_id']==c['id'] and r['backend']==backend and r['phase']=='domain-L3'),{})
            metrics=(spatial_profile_metrics(parent_root,study_root,base,dom)
                if parent_root is not None and study_root is not None and dom.get('usable') else {})
            rows.append(dict(case_id=c['id'],fuel=c['fuel'],pressure_atm=c['pressure_atm'],
                temperature=c['temperature'],backend=backend,test='domain-L3',level=3,
                usable=bool(dom.get('usable')),status=dom.get('status','pending'),nodes=dom.get('nodes'),
                Su=dom.get('Su'),Tb=dom.get('Tb'),qdot_peak=dom.get('qdot_peak'),width=dom.get('width'),signed_change_L3_percent=
                100*(dom['Su']-base['Su'])/abs(base['Su']) if dom.get('usable') else None,
                successive_change_percent=None,reused=False,**metrics))
    return rows


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('runs/thesis_flames_L3_sensitivity'))
    ap.add_argument('--output',type=Path,default=Path('runs/thesis_flames_L3/report'))
    args=ap.parse_args(argv);out=args.output;out.mkdir(parents=True,exist_ok=True)
    m=pp.read(args.input/'manifest.json')
    if m.get('kind')!='L3_spatial_sensitivity':raise ValueError('Expected independent L3 spatial study')
    rows=analyze(m,pp.load_records(args.input),Path(m.get('parent',args.input)),args.input)
    usable=sum(r['usable'] for r in rows);new_usable=sum(r['usable'] and not r['reused'] for r in rows)
    pp.write_csv(out/'sensibilidad_L3.csv',rows)
    pp.json_write(out/'sensibilidad_L3.json',dict(rows=rows,usable=usable,expected=60,new_usable=new_usable,
        target_percent=.5,scope='Changes relative to original L3, not exact discretization error. Signed values retain nonmonotonicity.'))
    if not new_usable:
        (out/'sensitivity_report.tex').write_text(r'\paragraph{Estado del estudio espacial.} Pendiente de nuevas soluciones utilizables; se conservan los doce perfiles L3 de referencia.'+'\n',encoding='utf-8')
        print('No new usable spatial profiles yet; no synthetic plot generated.');return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(8,6.5),layout='constrained')
    for i,fuel in enumerate(('CH4','H2')):
        for j,(p,t) in enumerate(((1,300),(10,300),(1,500))):
            ax=axes[i,j];ax.set_title(f'{fuel}: {p} atm, {t} K')
            for backend in ('native','cantera'):
                rr=[r for r in rows if r['fuel']==fuel and r['backend']==backend and r['pressure_atm']==p and r['temperature']==t and r['usable']]
                mesh=sorted((r for r in rr if r['test']!='domain-L3'),key=lambda r:r['level'])
                ax.plot([r['level'] for r in mesh],[r['signed_change_L3_percent'] for r in mesh],
                    '-o' if backend=='native' else '--s',label=pp.NAMES[backend],color=pp.COLORS[backend],markersize=4)
                dom=[r for r in rr if r['test']=='domain-L3']
                if dom:ax.scatter([6],[dom[0]['signed_change_L3_percent']],marker='x',color=pp.COLORS[backend],s=45)
            ax.axhline(0,color='.55',lw=.6)
            for value in (-.5,.5):ax.axhline(value,color='#48754a',linestyle=':',lw=.8)
            ax.set_xticks([2,3,4,5,6],['L2','L3','L4','L5','L3\n2L'])
            ax.set_ylabel(r'$100(S_u-S_{u,L3})/|S_{u,L3}|$ [%]')
            ax.grid(alpha=.15);ax.legend(fontsize=7)
    fig.suptitle(f'Sensibilidad espacial respecto a la producción L3\n{usable}/60 soluciones utilizables; líneas punteadas: ±0,5 %',fontsize=11)
    fig.savefig(out/'11_sensibilidad_L3.pdf',bbox_inches='tight')
    fig.savefig(out/'11_sensibilidad_L3.png',dpi=180,bbox_inches='tight');plt.close(fig)
    lines=[r'\begin{figure}[htbp]\centering',
        r'\includegraphics[width=\textwidth,height=.7\textheight,keepaspectratio]{\SweepReportRoot/11_sensibilidad_L3.pdf}',
        r'\caption{Sensibilidad de la velocidad respecto a L3 en seis estados estequiométricos: CH4/GRI30 promediado sin Soret e H2/h2o2 multicomponente con Soret. Cada solver conserva su propia referencia L3. L2, L3, L4 y L5 emplean slope=0.02, 0.01, 0.005 y 0.0025, con curve igual al doble. El punto 2L es una resolución L3 cuyo dominio inicial duplica el mayor dominio final del par original. Las variaciones se muestran con signo y escala lineal para conservar posibles cambios de tendencia.}',
        r'\label{fig:L3-sensitivity}\end{figure}',
        r'\begin{table}[htbp]\centering\footnotesize',
        r'\caption{Sensibilidad referida a L3. Las cuatro primeras columnas son cambios absolutos de velocidad en porcentaje; $\Delta T_b$ y $E_2(T)$ comparan L5 con el perfil L3 alineado. Dominio corresponde a L3 con dominio inicial duplicado. Un guion indica un dato pendiente o no utilizable. El umbral descriptivo de velocidad es 0.5\%.}',
        r'\label{tab:L3-sensitivity}\begin{tabular}{lrrrrrr}\hline',
        r'Estado / solver & L4--L3 & L5--L3 & L5--L4 & Dominio & $\Delta T_b$ [K] & $E_2(T)$ [\%]\\\hline']
    for c in m['cases']:
        for backend in ('native','cantera'):
            lookup={r['test']:r for r in rows if r['case_id']==c['id'] and r['backend']==backend}
            l5=lookup['L5'];base=lookup['L3']
            values=[lookup['L4']['signed_change_L3_percent'],l5['signed_change_L3_percent'],
                    l5['successive_change_percent'],lookup['domain-L3']['signed_change_L3_percent'],
                    None if l5.get('Tb') is None or base.get('Tb') is None else abs(l5['Tb']-base['Tb']),
                    l5.get('T_L2_percent')]
            label=f"{c['fuel']}, {c['pressure_atm']:g} atm, {c['temperature']:g} K / "+('K' if backend=='native' else 'C')
            lines.append(' & '.join([label,*['--' if v is None else f'{abs(v):.3f}' for v in values]])+r'\\')
    lines.append(r'\hline\end{tabular}\end{table}')
    (out/'sensitivity_report.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Spatial figure and table: {usable}/60 usable profiles; {out}')


if __name__=='__main__':main()
