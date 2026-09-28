"""Offline mesh-selection evidence; never changes acceptance or runs a solver."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from postprocess_flame_sweeps import load_records

COLORS = {'native': '#1766a1', 'cantera': '#c05032'}
NAMES = {'native': 'KFLAME', 'cantera': 'Cantera'}
STATES = [(1, 300), (10, 300), (1, 500)]


def analyze(records):
    mesh = [r for r in records if r['phase'].startswith('verify-L')]
    groups = {}
    for r in mesh:
        key = (r['case_id'], r['backend'])
        groups.setdefault(key, {})[r['settings']['level']] = r
    rows = []
    for (case_id, backend), levels in sorted(groups.items()):
        for level, r in sorted(levels.items()):
            prev = levels.get(level-1, {})
            change = (100*abs(r['Su']-prev['Su'])/abs(prev['Su'])
                      if r.get('usable') and prev.get('usable') else None)
            rows.append(dict(case_id=case_id, fuel=r['condition']['fuel'],
                pressure_atm=r['condition']['pressure_atm'], temperature_K=r['condition']['temperature'],
                backend=backend, level=level, slope=r['settings']['slope'],
                usable=bool(r.get('usable')), Su_m_s=r.get('Su'), nodes=r.get('nodes'),
                width_m=r.get('width'), time_s=r.get('time_s'), change_percent=change))
    assessment = []
    for fuel in ('CH4', 'H2'):
        for level in sorted({r['level'] for r in rows if r['fuel']==fuel}):
            rr = [r for r in rows if r['fuel']==fuel and r['level']==level]
            previous = [r for r in rows if r['fuel']==fuel and r['level']==level-1]
            expected = {(p,t,b) for p,t in STATES for b in NAMES}
            def complete(items):
                return ({(r['pressure_atm'],r['temperature_K'],r['backend']) for r in items}==expected
                        and all(r['usable'] and r['change_percent'] is not None for r in items))
            current = max(r['change_percent'] for r in rr) if complete(rr) else None
            both = max(current, max(r['change_percent'] for r in previous)) if current is not None and complete(previous) else None
            assessment.append(dict(fuel=fuel, level=level, max_last_change_percent=current,
                max_two_changes_percent=both,
                two_changes_below_0p2=level>=3 and both is not None and both<.2,
                two_changes_below_0p5=level>=3 and both is not None and both<.5))
    return rows, assessment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('runs/thesis_flames'))
    parser.add_argument('--output', type=Path, default=Path('runs/thesis_flames/mesh_review'))
    args = parser.parse_args(argv)
    root, out = args.input.resolve(), args.output.resolve()
    if out==root:
        parser.error('Use a separate output directory')
    records = load_records(root)
    rows, assessment = analyze(records)
    if not rows:
        parser.error('No saved mesh verification records')
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [('niveles',rows), ('criterios',assessment)]:
        with (out/f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    report = dict(source=str(root), recorded_mesh_solves=len(rows),
        usable_mesh_solves=sum(r['usable'] for r in rows), criteria=assessment,
        domain_solves=sum(r['phase'].startswith('domain-L') for r in records),
        definition='100 * abs(Su(L)-Su(L-1)) / abs(Su(L-1)); strict threshold; two consecutive changes; at least four levels',
        status='Analysis only: no acceptance criteria or verification settings changed')
    (out/'analisis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':180})
    levels = sorted({r['level'] for r in rows})
    for metric, title, ylabel, filename in [
        ('change_percent','Sensibilidad de la velocidad al refinamiento','Cambio respecto al nivel anterior [%]','sensibilidad_malla'),
        ('Su_m_s','Velocidad de llama en las mallas calculadas',r'Velocidad $S_u$ [m/s]','velocidad_malla'),
        ('time_s','Coste de las resoluciones de verificación','Tiempo medido [s]','coste_malla')]:
        fig, axes = plt.subplots(2,3,figsize=(14,8))
        for i,fuel in enumerate(('CH4','H2')):
            for j,(p,t) in enumerate(STATES):
                ax=axes[i,j]
                for backend in NAMES:
                    rr=sorted([r for r in rows if r['fuel']==fuel and r['pressure_atm']==p
                        and r['temperature_K']==t and r['backend']==backend and r['usable']
                        and r[metric] is not None],key=lambda r:r['level'])
                    ax.plot([r['level'] for r in rr],[r[metric] for r in rr],
                            '-o' if backend=='native' else '--s',color=COLORS[backend],
                            markersize=4,label=NAMES[backend],linewidth=1.5)
                if metric=='change_percent':
                    ax.axhline(.5,color='#327547',linestyle='--',linewidth=1.2,label='Umbral 0,5 %')
                    ax.axhline(.2,color='#717171',linestyle=':',linewidth=1.2,label='Umbral 0,2 %')
                    ax.set_yscale('log')
                    ax.set_ylim(.025, max(6.,ax.get_ylim()[1]))
                    ax.set_yticks([.05,.1,.2,.5,1,2,5],labels=['0,05','0,1','0,2','0,5','1','2','5'])
                elif metric=='time_s':
                    ax.set_yscale('log')
                ax.set_title(f'{fuel} · {p} atm · {t} K')
                ax.set_xticks(levels,labels=[f'L{l}' for l in levels])
                ax.set_xlabel('Nivel de refinamiento')
                if j==0: ax.set_ylabel(ylabel)
                ax.grid(alpha=.22)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.935),ncol=len(labels),frameon=False)
        fig.suptitle(title,fontsize=17,y=.98)
        caption=('CH₄: transporte promediado. H₂: multicomponente + Soret. Todos los estados: φ = 1.\n'
                 'L0 → L5: slope = 0,08; 0,04; 0,02; 0,01; 0,005; 0,0025. curve = 2·slope.\n')
        if metric=='change_percent':
            caption+='Cada punto compara dos mallas sucesivas; el criterio exige dos cambios consecutivos bajo el umbral en ambos solvers. Dominio pendiente.'
        elif metric=='time_s':
            caption+='Una observación por solver, estado y nivel; sin calentamiento ni exportación. Las cinco repeticiones pertenecen a la campaña principal.'
        else:
            caption+='Resultados aceptados de la verificación. Cada solver adapta su malla y puede ampliar el dominio inicial de 0,03 m.'
        fig.text(.5,.025,caption,ha='center',va='bottom',fontsize=9,linespacing=1.5)
        fig.subplots_adjust(top=.85,bottom=.18,hspace=.38,wspace=.25)
        fig.savefig(out/f'{filename}.png')
        fig.savefig(out/f'{filename}.pdf')
        plt.close(fig)
    lines=['# Selección de malla: evidencia guardada', '',
           'Análisis sin nuevas simulaciones. No modifica el manifiesto ni la aceptación de la campaña.', '',
           'Cambio = 100 |Su(L) − Su(L−1)| / |Su(L−1)|. Se evalúan dos cambios consecutivos, '
           'ambos solvers y los tres estados de cada combustible. Los umbrales expresan sensibilidad '
           'entre mallas, no una cota del error respecto a la solución continua.', '',
           '| Sistema | Nivel | Máximo último cambio (%) | Máximo de los dos cambios (%) | Cumple 0,5 % en malla |',
           '|---|---:|---:|---:|---|']
    for a in assessment:
        if a['level']<3: continue
        def fmt(v):return 'pendiente' if v is None else f'{v:.5f}'
        lines.append(f'| {a["fuel"]} | L{a["level"]} | {fmt(a["max_last_change_percent"])} | '
                     f'{fmt(a["max_two_changes_percent"])} | {"Sí" if a["two_changes_below_0p5"] else "No"} |')
    lines += ['',f'Comprobaciones de dominio guardadas: {report["domain_solves"]}.', '',
              'Las figuras de tiempo tienen una observación por punto, sin intervalos de incertidumbre. '
              'Los CSV conservan nodos, dominios, tiempos y velocidades por ejecución.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(rows)} mesh records analyzed offline. Figures and tables: {out}')
    for a in assessment:
        if a['level']>=3: print(a)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
