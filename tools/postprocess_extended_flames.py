"""Offline combined report for the base L3 campaign and its transport extension."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import postprocess_flame_sweeps as pp


def write_coverage_table(out, summary, records):
    lines = [r'\begin{table}[htbp]\centering\small',
        r'\caption{Cobertura y rendimiento por transporte. P: promediado; M: multicomponente; '
        r'+S: Soret. Se indican condiciones, resoluciones utilizables/previstas y rango de '
        r'$S_t$ entre condiciones, donde $S_t$ es la mediana del tiempo de Cantera dividida '
        r'por la de KFlame, con cinco ejecuciones por solver y condición. '
        r'Las tres últimas columnas cuentan intervalos bootstrap '
        r'del 95\,\% completamente por encima de uno (K), por debajo (C) o que incluyen uno (=). '
        r'Cada intervalo procede de cinco pares; las condiciones tienen el mismo peso.}',
        r'\label{tab:expanded-coverage}',
        r'\begin{tabular}{llrrrrrr}\toprule',
        r'Sistema & Transp. & Estados & Útiles/total & Rango $S_t$ & K & C & =\\\midrule']
    for fuel in ('CH4', 'H2'):
        for tr, so, label in [('mixture-averaged',False,'P'),('mixture-averaged',True,'P+S'),
                              ('multicomponent',False,'M'),('multicomponent',True,'M+S')]:
            group = [s for s in summary if s['fuel']==fuel and s['transport']==tr and s['soret']==so]
            ids = {s['id'] for s in group}
            usable = sum(bool(r.get('usable')) for r in records if r['phase']=='main' and r['case_id'] in ids)
            ratios = [s['statistics']['ratio'] for s in group if s['statistics']['ratio'] is not None]
            intervals = [s['statistics']['ci95'] for s in group if s['statistics']['ci95'] is not None]
            bounds = f'{min(ratios):.2f}--{max(ratios):.2f}' if ratios else '--'
            counts = [sum(ci[0]>1 for ci in intervals), sum(ci[1]<1 for ci in intervals),
                      sum(ci[0]<=1<=ci[1] for ci in intervals)]
            cells = [r'CH$_4$' if fuel=='CH4' else r'H$_2$', label, str(len(group)),
                     f'{usable}/{10*len(group)}', bounds, *map(str,counts)]
            lines.append(' & '.join(cells)+r'\\')
        if fuel=='CH4': lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (out/'table_coverage.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def write_sensitivity_table(source, out):
    rows = pp.read(source)['rows']
    lines = [r'\begin{table}[!htbp]\centering\footnotesize',
        r'\caption{Control espacial de los seis estados de referencia. K: KFlame; C: Cantera. '
        r'$|\Delta S_u|$ es el cambio porcentual respecto a L3; $N_3/N_5$ presenta los nodos '
        r'de ambos niveles. 2L identifica el ensayo de dominio ampliado. '
        r'$\Delta T_b^{L3}=100(T_b^{L5}-T_b^{L3})/T_b^{L3}$ conserva el signo y $E_2(T)$ compara el perfil L5 '
        r'alineado con L3. CH$_4$: promediado sin Soret; H$_2$: multicomponente con Soret; '
        r'$\phi=1$ en todos los estados. Se utiliza una resolución individual por estado, '
        r'nivel o dominio y solver.}',
        r'\label{tab:L3-sensitivity}\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{lrrrrr}\toprule',
        r'Estado / solver & $|\Delta S_u|_{L5}$ [\%] & $N_3/N_5$ & $|\Delta S_u|_{2L}$ [\%] & $\Delta T_b$ [\%] & $E_2(T)$ [\%]\\\midrule']
    for fuel in ('CH4','H2'):
        for p,t in ((1,300),(10,300),(1,500)):
            for backend, label in (('native','K'),('cantera','C')):
                group = {r['test']:r for r in rows if r['fuel']==fuel and r['pressure_atm']==p
                         and r['temperature']==t and r['backend']==backend and r['usable']}
                if not all(k in group for k in ('L3','L5','domain-L3')):
                    raise ValueError(f'Missing spatial control: {fuel}/{p}/{t}/{backend}')
                base, fine, domain = (group[k] for k in ('L3','L5','domain-L3'))
                cells = [f'{fuel}, {p} atm, {t} K / {label}',
                         f"{abs(fine['signed_change_L3_percent']):.3f}",
                         f"{base['nodes']}/{fine['nodes']}",
                         r'\num{'+f"{abs(domain['signed_change_L3_percent']):.2e}"+'}',
                         f"{100*(fine['Tb']-base['Tb'])/base['Tb']:+.4f}",f"{fine['T_L2_percent']:.3f}"]
                lines.append(' & '.join(cells)+r'\\')
        if fuel=='CH4': lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (out/'table_sensitivity.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def write_sensitivity_table_compact(source, out):
    """Write the paired KFLAME/Cantera sensitivity table used in the thesis."""
    rows = pp.read(source)['rows']
    lines = [
        r'\begin{table}[!htbp]\centering\scriptsize',
        r'\caption{Sensibilidad espacial respecto a L3. Las columnas K/C corresponden a '
        r'KFlame/Cantera. $N_{L3}$ es el numero de nodos de la malla de campana; '
        r'$|\Delta S_u^{L5-L3}|$ y $|\Delta S_u^{2L-L3}|$ son cambios relativos de velocidad. '
        r'$\Delta T_b^{L5-L3}=100(T_b^{L5}-T_b^{L3})/T_b^{L3}$ y $E_2(T)$ se expresan en porcentaje. '
        r'CH$_4$: promediado sin Soret; H$_2$: multicomponente con Soret; $\phi=1$. '
        r'Cada valor procede de una resolucion individual por solver.}',
        r'\label{tab:L3-sensitivity}\setlength{\tabcolsep}{3pt}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lrrrrr}\toprule',
        r'Estado & $N_{L3}$ K/C & $|\Delta S_u^{L5-L3}|$ [\%] K/C & '
        r'$\Delta T_b^{L5-L3}$ [\%] K/C & $E_2(T)$ [\%] K/C & '
        r'max $|\Delta S_u^{2L-L3}|$ [\%] K/C\\\midrule',
    ]
    for fuel in ('CH4', 'H2'):
        for pressure, temperature in ((1, 300), (10, 300), (1, 500)):
            data = {}
            for backend, label in (('native', 'K'), ('cantera', 'C')):
                group = {r['test']: r for r in rows if r['fuel'] == fuel
                         and r['pressure_atm'] == pressure and r['temperature'] == temperature
                         and r['backend'] == backend and r['usable']}
                if not all(test in group for test in ('L3', 'L5', 'domain-L3')):
                    raise ValueError(f'Missing spatial control: {fuel}/{pressure}/{temperature}/{backend}')
                data[label] = tuple(group[test] for test in ('L3', 'L5', 'domain-L3'))
            k_base, k_fine, k_domain = data['K']
            c_base, c_fine, c_domain = data['C']
            temp_k = 100 * (k_fine['Tb'] - k_base['Tb']) / k_base['Tb']
            temp_c = 100 * (c_fine['Tb'] - c_base['Tb']) / c_base['Tb']
            cells = [
                f'{fuel}, {pressure} atm, {temperature} K',
                f"{k_base['nodes']} / {c_base['nodes']}",
                f"{abs(k_fine['signed_change_L3_percent']):.3f} / {abs(c_fine['signed_change_L3_percent']):.3f}",
                f'{temp_k:+.4f} / {temp_c:+.4f}',
                f"{k_fine['T_L2_percent']:.3f} / {c_fine['T_L2_percent']:.3f}",
                f"{abs(k_domain['signed_change_L3_percent']):.1e} / {abs(c_domain['signed_change_L3_percent']):.1e}",
            ]
            lines.append(' & '.join(cells) + r'\\')
        if fuel == 'CH4':
            lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}}\end{table}']
    (out / 'table_sensitivity.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('runs/thesis_flames_L3'))
    parser.add_argument('--output', type=Path, default=Path('runs/thesis_flames_L3/report_expanded'))
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args(argv)
    root, out = args.input.resolve(), args.output.resolve()
    if out == root or out == root/'report' or out in (root/'main', root/'inputs', root/'code'):
        parser.error('Use a separate report_expanded directory; preserve scientific data and the original report.')
    manifest = pp.load_manifest(root)
    if 'transport_extension' not in manifest:
        parser.error('Start the extension first; transport_extension.json is not present.')
    records = pp.load_records(root)
    summary = pp.summarize(root, records, manifest)
    out.mkdir(parents=True, exist_ok=True)
    pending = [dict(case_id=s['id'], complete_pairs=s['statistics']['n_pairs'],
                    missing_pairs=manifest['pairs']-s['statistics']['n_pairs'])
               for s in summary if s['statistics']['n_pairs'] < manifest['pairs']]
    main_rows = [r for r in records if r['phase'] == 'main']
    base_ids = {c['id'] for c in pp.read(root/'manifest.json')['cases']}
    pp.json_write(out/'summary.json', dict(
        expected_conditions=len(manifest['cases']), expected_main=len(manifest['cases'])*manifest['pairs']*2,
        recorded_main=len(main_rows), usable_main=sum(bool(r.get('usable')) for r in main_rows),
        pending=pending, conditions=summary,
        provenance={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                    for name in ('manifest.json', 'transport_extension.json')},
        interpretation='Paired timing comparisons within each condition. Base and extension ran in separate sessions; spatial sensitivity retains the original six anchors.'))
    pp.write_csv(out/'configuracion.csv', manifest['cases'])
    pp.write_csv(out/'pendientes.csv', pending)
    pp.write_csv(out/'rendimiento.csv', [dict(case_id=s['id'], **s['statistics'], solvers=s['solvers']) for s in summary])
    pp.write_csv(out/'errores.csv', [dict(case_id=s['id'], **e) for s in summary for e in s['profile_errors']])
    pp.write_csv(out/'ejecuciones.csv', [dict(
        {k:r.get(k) for k in ('phase','case_id','repetition','variant','nonlinear_strategy','status','accepted',
                             'usable','time_s','process_s','Su','Tb','nodes','width','mass_error','species_sum_error',
                             'elemental_error_mass_scaled','energy_error_sensible_scaled','folder')},
        campaign_batch='base' if r['case_id'] in base_ids else 'transport_extension') for r in records])
    for name, lines in pp.latex_tables(records, summary, manifest, split=True).items():
        fragment = '\n'.join(lines).replace(r'\begin{tabular}',
            r'\label{tab:sweep-'+name+'}\n'+r'\begin{tabular}', 1)
        (out/f'table_{name}.tex').write_text(fragment+'\n', encoding='utf-8')
    write_coverage_table(out, summary, records)
    # These analyses concern the unchanged central cases and retain their provenance.
    for source in (root/'report').glob('table_diagnostic_*.tex'):
        shutil.copy2(source, out/source.name)
    for name in ('table_ablations.tex',):
        source = root/'report'/name
        if source.exists(): shutil.copy2(source, out/name)
    sensitivity = root/'report/sensibilidad_L3.json'
    if sensitivity.exists():
        write_sensitivity_table_compact(sensitivity, out)
    print(f'{len(main_rows)}/{len(manifest["cases"])*manifest["pairs"]*2} resoluciones registradas; '
          f'{len(pending)} condiciones con pares pendientes o fallidos.')
    (out/'README.md').write_text(
        '# Campaña ampliada L3\n\n'
        f'{len(main_rows)} resoluciones registradas; {len(pending)} condiciones incompletas.\n\n'
        'summary.json y pendientes.csv conservan los pares ausentes o fallidos. Las estadísticas '
        'usan pares utilizables. Los gráficos finales se generan cuando todas las condiciones '
        'tienen sus cinco pares. La columna campaign_batch distingue la sesión original y la ampliación. '
        'Los perfiles, mallas, coste y ablaciones centrales se reutilizan; la sensibilidad espacial '
        'se refiere a los seis estados originales, sin extrapolar su alcance a todos los transportes.\n', encoding='utf-8')
    if pending or args.tables_only:
        print('Tablas y estadísticas guardadas; figuras finales pendientes de completar los cinco pares por condición.')
        return 0
    import redesign_thesis_result_figures as figures
    figures.main(['--input', str(root), '--report', str(out), '--output', str(out/'revision_figures'),
                  '--sensitivity', str(root/'report/sensibilidad_L3.json')])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
