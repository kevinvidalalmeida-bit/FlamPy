"""Offline operation counts for the eight central diagnostic flames of Chapter 4."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import numpy as np
import postprocess_flame_sweeps as pp
from postprocess_flame_diagnostics import profile_totals, metrics
from postprocess_flame_recovery import classify, same_solution, MODES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('runs/thesis_flames_L3'))
    parser.add_argument('--diagnostics', type=Path, default=Path('runs/thesis_flames_L3_diagnostics'))
    parser.add_argument('--output', type=Path, default=Path('runs/thesis_flames_L3/report_expanded/revision_figures'))
    args = parser.parse_args(argv)
    root, diag, out = args.input.resolve(), args.diagnostics.resolve(), args.output.resolve()
    records = [r for r in pp.load_records(root) if r['phase'] == 'main' and r['variant'] in ('native', 'cantera') and r.get('usable')]
    rows, provenance = [], []
    for fuel in ('CH4', 'H2'):
        for transport, soret in MODES:
            cid = f'{fuel}_phi1_p1_T300_{transport}_S{int(soret)}'
            folder = diag / cid / 'native'
            path = folder / 'result.json'
            result = pp.read(path)
            if not result.get('usable') or not result.get('diagnostic_committed'):
                raise ValueError(f'Incomplete diagnostic: {cid}')
            ref = min((r for r in records if r['case_id'] == cid and r['variant'] == 'native'), key=lambda r: r['repetition'])
            if not same_solution(root / ref['folder'] / 'profile.npz', folder / 'profile.npz'):
                raise ValueError(f'Diagnostic differs from production: {cid}')
            prof = profile_totals(result['report'])
            calls = result['instrumentation']['function_calls']
            _, recovery = classify(result)
            measured = metrics(result)
            row = dict(case_id=cid, backend='native', fuel=fuel, transport=transport, soret=soret,
                       thermochemistry=prof['residual_nodal_thermochemistry']['count'],
                       face_transport=prof['residual_face_transport_and_flux']['count'],
                       chemical_derivatives=prof['jacobian_analytic_thermochemistry']['count'],
                       jacobian=calls['jacobian'], residual=calls['residual'],
                       factorization=calls['factorize'], linear_solve=calls['linear_solve'],
                       euler_accepted=measured['euler_accepted'], jacobian_s=measured['jacobian_s'],
                       jacobian_ms=1000*measured['jacobian_s']/calls['jacobian'],
                       **recovery)
            # These equalities permit one clearly named panel per pair of nested blocks.
            if not row['thermochemistry'] == row['face_transport'] == row['residual']:
                raise ValueError('Split residual/chemistry/transport counts before plotting this dataset.')
            if row['chemical_derivatives'] != row['jacobian']:
                raise ValueError('Split chemistry derivative and Jacobian counters before plotting.')
            rows.append(row)
            provenance.append(dict(case_id=cid, backend='native', source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_profile=str(root / ref['folder'] / 'profile.npz'),
                exact_profile_match=True, time_source='same instrumented execution as figure 16'))
            folder = diag / cid / 'cantera'
            path = folder / 'result.json'
            result = pp.read(path)
            if not result.get('usable') or not result.get('diagnostic_committed'):
                raise ValueError(f'Incomplete Cantera diagnostic: {cid}')
            ref = min((r for r in records if r['case_id'] == cid and r['variant'] == 'cantera'), key=lambda r: r['repetition'])
            if not same_solution(root / ref['folder'] / 'profile.npz', folder / 'profile.npz'):
                raise ValueError(f'Cantera diagnostic differs from production: {cid}')
            measured = metrics(result)
            rows.append(dict(case_id=cid, backend='cantera', fuel=fuel, transport=transport, soret=soret,
                             thermochemistry=None, face_transport=None, chemical_derivatives=None,
                             jacobian=measured['jacobian_calls'], residual=measured['residual_calls'],
                             factorization=None, linear_solve=None,
                             euler_accepted=measured['euler_accepted'], jacobian_s=measured['jacobian_s'],
                             jacobian_ms=1000*measured['jacobian_s']/measured['jacobian_calls']))
            provenance.append(dict(case_id=cid, backend='cantera', source=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_profile=str(root / ref['folder'] / 'profile.npz'), exact_profile_match=True,
                time_source='same instrumented execution as figure 16'))
    out.mkdir(parents=True, exist_ok=True)
    pp.write_csv(out / '17_trabajo_por_llama.csv', rows)
    pp.json_write(out / '17_trabajo_por_llama_provenance.json', provenance)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.grid': False,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    from matplotlib.patches import Patch
    fields = [('residual', 'Evaluaciones del residual registradas', 'Número de evaluaciones'),
              ('jacobian', 'Construcciones del Jacobiano registradas', 'Número de construcciones'),
              ('jacobian_ms', 'Tiempo medio por Jacobiano', 'Tiempo registrado / construcciones [ms]'),
              ('euler_accepted', 'Pasos Euler aceptados', 'Número de pasos aceptados')]
    labels = ['Promediado', 'Promediado + Soret', 'Multicomponente', 'Multi. + Soret']
    colors = {'native': '#1769aa', 'cantera': '#b84b32'}
    fig, axes = plt.subplots(4, 2, figsize=(9.1, 10.3), layout='constrained')
    for i, (field, title, xlabel) in enumerate(fields):
        largest = max(r[field] for r in rows)
        for j, fuel in enumerate(('CH4', 'H2')):
            ax = axes[i, j]
            for backend, offset in [('native', -.17), ('cantera', .17)]:
                values = [r[field] for r in rows if r['fuel'] == fuel and r['backend'] == backend]
                bars = ax.barh(np.arange(4)+offset, values, color=colors[backend], height=.30)
                texts = [(f'{v:.2f}' if v < 10 else f'{v:.1f}') if field == 'jacobian_ms' else str(v) for v in values]
                ax.bar_label(bars, labels=texts, padding=3, fontsize=9)
            ax.set_yticks(range(4), labels if j == 0 else [''] * 4)
            ax.set_ylim(3.6, -.6)
            limit = max(r[field] for r in rows if r['fuel'] == fuel) if field == 'jacobian_ms' else largest
            ax.set_xlim(0, limit * 1.19)
            ax.set_title(title, fontsize=9.5, pad=8)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', labelsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
            if i == 0:
                ax.text(.5, 1.27, r'CH$_4$ / GRI-Mech 3.0' if j == 0 else r'H$_2$ / h2o2',
                        transform=ax.transAxes, ha='center', fontsize=12, weight='bold')
    fig.legend(handles=[Patch(facecolor=colors[b], label=n) for b, n in
                        [('native', 'KFLAME'), ('cantera', 'Cantera')]],
               loc='outside lower center', ncol=2, frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'17_trabajo_por_llama.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    caption = (r'Trabajo registrado de KFLAME y Cantera en los ocho casos centrales: '
        r'$\phi=1$, 300 K y 1 atm. Una ejecución instrumentada por solver y caso, la misma '
        r'del desglose temporal. Los recuentos comparten escala entre mecanismos; el tiempo '
        r'por Jacobiano tiene escala propia por mecanismo, común a ambos solvers. El tiempo medio por '
        r'Jacobiano es $t_J/N_J$ de esa ejecución; los demás paneles muestran recuentos. '
        r'Cantera excluye del residual las evaluaciones internas de diferencias finitas '
        r'y sus estadísticas finales pueden omitir etapas interrumpidas al ampliar el dominio. '
        r'KFLAME incluye las etapas preliminares. Los pasos Euler aceptados de Cantera '
        r'proceden de callbacks; KFLAME combina Euler con PTC en metano. El desglose químico '
        r'y lineal específico se conserva en los cuadros, con su disponibilidad por solver.')
    (out / '17_trabajo_por_llama.tex').write_text(
        '\\clearpage\n\\begin{figure}[p]\\centering\n'
        '\\includegraphics[width=\\textwidth,height=.76\\textheight,keepaspectratio]'
        '{\\SweepReportRoot/revision_figures/17_trabajo_por_llama.pdf}\n'
        '\\caption[Contadores de trabajo de los ocho casos centrales]{' + caption + '}\n'
        '\\label{fig:rev-17-trabajo-por-llama}\n\\end{figure}\n\\clearpage\n', encoding='utf-8')
    print(f'16 diagnostic profiles verified across eight central cases. Figure: {out / "17_trabajo_por_llama.pdf"}')


if __name__ == '__main__':
    main()
