"""Regenerate the isolated H2 strategy comparison without solving flames."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('runs/h2_euler_probe_L4_p10'))
    root=parser.parse_args().input
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    rows=summary['results']
    times=[r['time_s'] for r in rows]
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    bars=ax.bar(['KFLAME\nNewton + PTC–SER / BE','KFLAME\nNewton + Euler','Cantera'],
                times,color=['#1766a1','#2a8a67','#c05032'],width=.55)
    ax.bar_label(bars,labels=[f'{t:.3f} s' for t in times],padding=5)
    ax.set_ylim(0,max(times)*1.2)
    ax.set_ylabel('Tiempo medido [s]')
    ax.set_title('H₂/h2o2 · 10 atm · 300 K · φ = 1\nMulticomponente + Soret · L4')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.025,
        'Una ejecución medida por opción; calentamiento excluido y perfiles iniciales independientes.\n'
        'Todas aceptadas. KFLAME: 1391 nodos en ambas variantes; Cantera: 1396 nodos.\n'
        'Diagnóstico aislado: la reducción observada requiere más casos para generalizarse.',
        ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.25,top=.85)
    fig.savefig(root/'comparacion_tiempos.png',dpi=180)
    fig.savefig(root/'comparacion_tiempos.pdf')
    plt.close(fig)
    lines=['# Prueba breve Newton + Euler para H2/h2o2','',
        'Condición: phi=1, 300 K, 10 atm, multicomponente/Soret, L4. '
        'Una resolución medida por estrategia y un calentamiento separado por proceso.','',
        '| Estrategia | Tiempo [s] | Velocidad [m/s] | Nodos | Aceptada |',
        '|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {r["folder"]} | {r["time_s"]:.6f} | {r["Su"]:.12f} | {r["nodes"]} | {r["usable"]} |')
    lines.extend(['',f'Reducción observada Euler/PTC: {100*(1-times[1]/times[0]):.2f} %. '
        f'Razón Cantera/Euler: {times[2]/times[1]:.3f}.',
        'Sin intervalos estadísticos: una observación por estrategia, orden fijo.', '',
        'Llamadas a Newton por etapa (incluyen intentos fallidos; no son iteraciones individuales):'])
    for r in rows:
        if r.get('probe_newton_calls'):lines.append(f'- {r["folder"]}: {r["probe_newton_calls"]}')
    lines.extend(['',
        'Se aplica el mismo contador ligero a ambas variantes nativas. Los archivos '
        'hybrid_original.py y hybrid_euler.py documentan el cambio: selección directa '
        'de BE, manteniendo convergencia, adaptación del paso y retorno a Newton '
        'estacionario. Se conserva la inicialización con transporte promediado. '
        'La variante solo se instala en el proceso diagnóstico; producción y campaña '
        'original permanecen intactas.', '',
        'summary.json guarda comparaciones de perfiles. Cada carpeta conserva perfil NPZ, '
        'diagnósticos, logs y calentamiento; inputs contiene mecanismos y copias de código. '
        'Esta prueba no valida toda la matriz física ni la sensibilidad al dominio.'])
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
