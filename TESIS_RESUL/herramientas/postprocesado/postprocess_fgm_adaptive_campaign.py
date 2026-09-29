"""Offline tables and figures for adaptive FGM sweeps; never solves a flame."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'TESIS_RESUL/herramientas/campanas'))
from benchmark_fgm_campaign import atomic, completed, latest, verify_artifacts

LABELS = dict(P='Promediado', PS='Promediado + Soret', M='Multicomponente', MS='Multicomponente + Soret')
COLORS = dict(P='#246091', PS='#bc6732', M='#388065', MS='#86549b')
MARKERS = dict(P='o', PS='s', M='^', MS='D')
TIME_KEYS = ['setup_s', 'family_s', 'coordinate_checks_s', 'tabulation_s', 'selection_s', 'compute_s', 'process_wall_s']


def collect(source, expected_kind='adaptive-fgm-sweeps'):
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != expected_kind: raise ValueError('Unexpected FGM campaign kind.')
    rows, attempts = [], []
    for c in manifest['conditions']:
        job = source / 'cases' / c['id']
        result = completed(job)
        row = dict(c, status='pending', n_flames=None, n_inserted=None, rounds=None,
                   max_defect=None, **{k: None for k in TIME_KEYS}, reason='', attempt='')
        if result:
            attempt = latest(job)
            verify_artifacts(attempt, result)
            row.update({k: result[k] for k in row if k in result})
            row['attempt'] = attempt.relative_to(source).as_posix()
        elif latest(job): row['status'] = 'incomplete'
        rows.append(row)
        for attempt in sorted(job.glob('attempt-*')):
            record = dict(condition=c['id'], attempt=attempt.relative_to(source).as_posix())
            for name in ('result', 'state'):
                path = attempt / f'{name}.json'
                if path.exists(): record[name] = json.loads(path.read_text(encoding='utf-8'))
            attempts.append(record)
    return manifest, rows, attempts


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def figures(out, manifest, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), layout='constrained', sharex='col')
    for j, (sweep, coordinate, xlabel) in enumerate([
            ('pressure', 'p_atm', 'Presión [atm], a 300 K'),
            ('temperature', 'T_in_K', 'Temperatura de entrada [K], a 1 atm')]):
        for tag in LABELS:
            group = sorted([r for r in rows if r['transport'] == tag and r['sweep'] in ('base', sweep)],
                           key=lambda r: r[coordinate])
            x = [r[coordinate] for r in group]
            for i, key in enumerate(('compute_s', 'n_flames')):
                y = [r[key] if r['status'] == 'accepted' else np.nan for r in group]
                axes[i, j].plot(x, y, linestyle='-', marker=MARKERS[tag], color=COLORS[tag],
                                label=LABELS[tag], markersize=5, markerfacecolor='none', linewidth=1.3)
        axes[1, j].set_xlabel(xlabel)
        axes[0, j].set_title(('Barrido de presión', 'Barrido de temperatura')[j])
        vals = [r['compute_s'] for r in rows if r['status'] == 'accepted' and r['sweep'] in ('base', sweep)]
        if vals and min(vals) > 0 and max(vals)/min(vals) > 10:
            axes[0, j].set_yscale('log')
        axes[0, j].set_ylabel('Tiempo individual [s]')
        axes[1, j].set_ylabel('Llamas finales, $N_f$')
        axes[1, j].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, min_n_ticks=1))
        nodes = [r['n_flames'] for r in rows if r['status'] == 'accepted' and r['sweep'] in ('base', sweep)]
        if nodes and min(nodes) == max(nodes): axes[1, j].set_ylim(nodes[0]-1, nodes[0]+1)
        if x and min(x) == max(x): axes[1, j].set_xticks([x[0]])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    if manifest['technical_check']: fig.suptitle('Prueba técnica: fuera de la campaña científica')
    for ext in ('png', 'pdf'): fig.savefig(out / f'01_coste_adaptativo.{ext}', dpi=180)
    plt.close(fig)
    caption = (r'Construcción adaptativa de FGM de CH$_4$--aire, GRI-Mech~3.0. '
               r'Cada punto representa una ejecución individual, con calentamiento previo y escritura '
               r'excluidos del tiempo. Se incluyen preparación, resolución y propiedades, comprobación '
               r'del progreso, tabulación y selección de filas. Las cinco composiciones iniciales, '
               r'$N_c=241$ y el objetivo interno del 1\% son comunes; $N_f$ es el tamaño final adaptativo. '
               r'La base de 300 K y 1 atm es compartida. Las líneas unen puntos contiguos completados; '
               r'los pendientes o fallidos quedan sin punto. Hay una observación por condición, '
               r'sin intervalos estadísticos.')
    if manifest['technical_check']:
        caption = r'Prueba técnica, sin valor de resultado científico: tres composiciones iniciales, $N_c=61$ y objetivo del 10\%.'
    (out / '01_coste_adaptativo.tex').write_text(
        '\\begin{figure}[!htbp]\n\\centering\n'
        '\\includegraphics[width=\\textwidth,height=.58\\textheight,keepaspectratio]{\\FGMSweepReportRoot/01_coste_adaptativo.pdf}\n'
        f'\\caption{{{caption}}}\n\\label{{fig:fgm-adaptive-sweeps}}\n\\end{{figure}}\n', encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path,
                   default=Path('TESIS_RESUL/corridas/FGM/thesis_fgm_comparison'))
    p.add_argument('--output', type=Path,
                   default=Path('TESIS_RESUL/corridas/reproduccion/thesis_fgm_adaptive_report'))
    args = p.parse_args(argv)
    source = args.input.resolve()
    if not (source / 'manifest.json').exists():
        print('Campaign not created yet; no files written.'); return 0
    manifest, rows, attempts = collect(source)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'conditions.csv', rows)
    atomic(out / 'conditions.json', rows)
    atomic(out / 'attempts.json', attempts)
    counts = {s: sum(r['status'] == s for r in rows) for s in ('accepted', 'failed', 'incomplete', 'pending')}
    atomic(out / 'status.json', dict(counts, expected=len(rows), repetitions=1,
                                    technical_check=manifest['technical_check']))
    lines = [r'\begin{table}[!htbp]\centering\small',
             r'\caption{Cobertura disponible de la campaña adaptativa. P: promediado; M: multicomponente; S: Soret. Cada condición tiene una ejecución. Los intervalos indican mínimos y máximos entre estados físicos, no dispersión entre repeticiones.}',
             r'\label{tab:fgm-adaptive-coverage}',
             r'\begin{tabular}{lrrrrr}\toprule',
             r'Transporte & Aceptados & Fallidos & Pendientes$^*$ & $N_f$ & Tiempo [s]\\\midrule']
    for tag in LABELS:
        group = [r for r in rows if r['transport'] == tag]
        accepted = [r for r in group if r['status'] == 'accepted']
        def span(key):
            values = [r[key] for r in accepted]
            if not values: return '--'
            fmt = (lambda v: f'{v:.2f}') if key == 'compute_s' else (lambda v: str(v))
            return fmt(min(values)) if min(values) == max(values) else fmt(min(values))+'--'+fmt(max(values))
        lines.append(f'{dict(P="P",PS="P+S",M="M",MS="M+S")[tag]} & {len(accepted)} & '
                     f'{sum(r["status"]=="failed" for r in group)} & '
                     f'{sum(r["status"] in ("pending","incomplete") for r in group)} & '
                     f'{span("n_flames")} & {span("compute_s")} '+r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\par\smallskip\footnotesize $^*$Incluye construcciones interrumpidas.', r'\end{table}']
    (out / 'table_coverage.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if counts['accepted']: figures(out, manifest, rows)
    else: (out / '01_coste_adaptativo.tex').unlink(missing_ok=True)
    (out / 'README.md').write_text(
        '# Campaña FGM adaptativa\n\n'
        f'{counts["accepted"]}/{len(rows)} aceptadas; {counts["failed"]} fallidas; '
        f'{counts["pending"]+counts["incomplete"]} pendientes o interrumpidas.\n\n'
        'Una observación por condición, sin medianas ni intervalos de confianza. '
        'conditions.csv contiene todos los estados y tiempos por etapa; attempts.json conserva el historial. '
        'El indicador interno de selección de filas no sustituye la validación independiente. '
        'Las figuras se regeneran únicamente desde archivos, sin resolver llamas.\n', encoding='utf-8')
    print(json.dumps(counts)); print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
