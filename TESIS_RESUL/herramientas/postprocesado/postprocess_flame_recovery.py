"""Offline recovery-frequency figure for the 104 L3 conditions; no solver imports."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import postprocess_flame_sweeps as pp

EVENTS = ('newton_only', 'ptc', 'be', 'thermal', 'domain', 'mesh')
EVENT_LABELS = ('Solo Newton', 'PTC–SER', 'Euler implícito / BE', 'Rescate térmico',
                'Ampliación de dominio', 'Adaptación de malla')
MODES = [('mixture-averaged', False), ('mixture-averaged', True),
         ('multicomponent', False), ('multicomponent', True)]


def report_tree(report):
    yield report
    if report.get('transport_bootstrap'):
        yield from report_tree(report['transport_bootstrap'])


def classify(result):
    """Count activation (accepted OR rejected attempts), never configured options.

    Use instrumentation.phases once: solver_trace and report.passes can duplicate
    those histories. Interrupted domain passes are retained by the instrumenter.
    """
    phases = result.get('instrumentation', {}).get('phases', [])
    if not phases or any(not p.get('history') for p in phases):
        raise ValueError('Incomplete phase histories; absence cannot be interpreted as zero.')
    histories = [h for phase in phases for h in phase['history']]
    counts = Counter()
    for h in histories:
        if h.get('phase') == 'steady':
            counts['steady_attempts'] += 1
        elif h.get('phase') == 'transient':
            scheme = h.get('scheme')
            if scheme not in ('PTC-SER', 'BE-direct', 'BE-fallback'):
                raise ValueError(f'Unknown transient scheme: {scheme}')
            key = 'ptc' if scheme == 'PTC-SER' else 'be'
            counts[key + '_attempts'] += 1
            counts[key + ('_accepted' if h.get('ok') else '_rejected')] += 1
        else:
            raise ValueError(f'Unknown phase: {h.get("phase")}')
    thermal = any(p.get('energy') is False for p in phases)
    # Profiles compare actual final domain; nested reports cover bootstrap adaptation.
    reports = list(report_tree(result['report']))
    mesh = any(isinstance(step, dict) and step.get('changed', False)
               for rep in reports for name in ('refine', 'post_expand_refine')
               for step in (rep.get(name, []) if isinstance(rep.get(name, []), list) else []))
    domain = result['width'] > result['settings']['width'] * (1 + 1e-10)
    ptc, be = counts['ptc_attempts'] > 0, counts['be_attempts'] > 0
    flags = dict(newton_only=not (ptc or be or thermal), ptc=ptc, be=be,
                 thermal=thermal, domain=bool(domain), mesh=bool(mesh))
    return flags, dict(counts, thermal_phases=sum(p.get('energy') is False for p in phases),
                       interrupted_domain_phases=sum(p.get('interrupted') == 'DomainTooNarrowError' for p in phases))


def same_solution(reference, diagnostic):
    with np.load(reference, allow_pickle=False) as a, np.load(diagnostic, allow_pickle=False) as b:
        return all(k in a and k in b and np.array_equal(a[k], b[k]) for k in ('z', 'T', 'u', 'Y'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3'))
    parser.add_argument('--diagnostics', type=Path,
                        default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3_recovery'))
    parser.add_argument('--output', type=Path,
                        default=Path('TESIS_RESUL/corridas/reproduccion/figuras_llamas'))
    args = parser.parse_args(argv)
    root, diag, out = args.input.resolve(), args.diagnostics.resolve(), args.output.resolve()
    manifest = pp.load_manifest(root)
    source = pp.read(diag / 'manifest.json')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if source['kind'] != 'flame-recovery-104' or source['parent_manifest_sha256'] != digest(root / 'manifest.json') or source['extension_sha256'] != digest(root / 'transport_extension.json'):
        raise ValueError('Diagnostic campaign does not match the production campaign.')
    if {c['id'] for c in source['cases']} != {c['id'] for c in manifest['cases']} or len(source['cases']) != 104:
        raise ValueError('Expected identical sets of 104 unique conditions.')
    rows, evidence = [], []
    for c in source['cases']:
        folder = diag / c['id'] / 'native'
        path = folder / 'result.json'
        row = dict(case_id=c['id'], fuel=c['fuel'], transport=c['transport'], soret=c['soret'],
                   phi=c['phi'], pressure_atm=c['pressure_atm'], temperature=c['temperature'],
                   available=False, reason='pending', **{k: None for k in EVENTS})
        if path.exists():
            result = pp.read(path)
            if not result.get('diagnostic_committed'):
                row['reason'] = 'interrupted'
            elif not result.get('usable'):
                row['reason'] = result.get('status', 'failed')
            else:
                ref = source['references'][c['id']]
                original = root / ref['folder'] / 'profile.npz'
                profile = folder / 'profile.npz'
                if digest(original) != ref['profile_sha256'] or digest(profile) != result['profile_sha256']:
                    raise ValueError(f'Profile checksum changed: {c["id"]}')
                equal = same_solution(original, profile)
                row['profile_identical'] = equal
                if not equal:
                    row['reason'] = 'diagnostic_profile_differs'
                else:
                    try:
                        flags, counts = classify(result)
                        row.update(flags, available=True, reason='verified', **counts)
                    except ValueError as exc:
                        row['reason'] = str(exc)
                evidence.append(dict(case_id=c['id'], result=str(path), result_sha256=digest(path),
                                     profile_sha256=digest(profile), reference=str(original),
                                     reference_sha256=digest(original)))
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    pp.write_csv(out / '17_recuperacion_condiciones.csv', rows)
    pp.json_write(out / '17_recuperacion_evidencia.json', evidence)
    groups = []
    for fuel in ('CH4', 'H2'):
        for transport, soret in MODES:
            group = [r for r in rows if (r['fuel'], r['transport'], r['soret']) == (fuel, transport, soret)]
            known = [r for r in group if r['available']]
            if len(group) != 13:
                raise ValueError('Every mechanism/transport must contain 13 conditions.')
            groups.append(dict(fuel=fuel, transport=transport, soret=soret, planned=len(group),
                               available=len(known), **{k: sum(r[k] for r in known) for k in EVENTS}))
    complete = all(r['available'] for r in rows)
    pp.json_write(out / '17_recuperacion_resumen.json', dict(complete=complete,
        available=sum(r['available'] for r in rows), expected=104, groups=groups,
        scope='One verified instrumented KFLAME resolution per condition, including bootstrap and interrupted domain passes. Activation counts include rejected attempts. Events overlap; configured methods are not counted as activations.',
        pending=[dict(case_id=r['case_id'], reason=r['reason']) for r in rows if not r['available']]))
    if not complete:
        # Never publish an incomplete frequency as the scientific 104-condition result.
        print(f'Recovery audit: {sum(r["available"] for r in rows)}/104 verified; '
              'CSV/JSON saved. Chapter figure will be published when complete.')
        return 2

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.grid': False, 'pdf.fonttype': 42})
    from matplotlib.patches import Patch
    colors = ['#1769aa', '#d07b28', '#2e8b70', '#8b5897']
    mode_labels = ['Promediado', 'Promediado + Soret', 'Multicomponente', 'Multicomponente + Soret']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True, layout='constrained')
    y = np.array([0., 1., 2., 3., 4.5, 5.5])
    for ax, fuel, title in zip(axes, ('CH4', 'H2'), (r'CH$_4$ / GRI-Mech 3.0', r'H$_2$ / h2o2')):
        data = np.array([[g[k] for g in groups if g['fuel'] == fuel] for k in EVENTS])
        left = np.zeros(6)
        for j, color in enumerate(colors):
            ax.barh(y, data[:, j], left=left, height=.55, color=color,
                    edgecolor='white', linewidth=.6)
            left += data[:, j]
        for yi, total in zip(y, left.astype(int)):
            ax.text(total + .8, yi, f'{total} llamas' if total else '0 · ninguna',
                    va='center', fontsize=10, color='#243443')
        ax.set_title(title + '\n52 condiciones', pad=13)
        ax.set_yticks(y, ['Solo Newton\nsin recuperación', 'PTC–SER', 'Euler implícito / BE',
                         'Rescate térmico', 'Ampliación de dominio', 'Adaptación de malla'])
        ax.set_xlim(0, 68)
        ax.set_xticks([0, 13, 26, 39, 52])
        ax.set_xlabel('Número de llamas que utilizaron la etapa', labelpad=9)
        ax.set_ylim(6.05, -.6)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_bounds(0, 52)
    fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(colors, mode_labels)],
               loc='outside lower center', ncol=2, frameon=False, fontsize=9)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'17_recuperacion_global.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    caption = (r'Etapas utilizadas por KFLAME en las 104 condiciones L3: 52 por mecanismo. '
        r'La longitud y la etiqueta de cada barra indican el número de llamas que utilizaron esa etapa. '
        r'Los segmentos distinguen los cuatro transportes, con 13 condiciones por transporte. '
        r'Las barras son independientes: una llama puede utilizar varias etapas. Se utiliza una ejecución '
        r'instrumentada por condición, cuyo perfil reproduce exactamente la referencia de producción. '
        r'Solo Newton identifica una trayectoria sin PTC, Euler implícito (BE) ni rescate térmico. '
        r'PTC y BE cuentan cualquier intento, aceptado o rechazado; el rescate térmico identifica '
        r'la etapa con temperatura fijada. Dominio y malla describen controles espaciales. '
        r'Las categorías pueden coexistir y abarcan también la inicialización multicomponente '
        r'y los intentos interrumpidos para ampliar el dominio.')
    (out / '17_recuperacion_global.tex').write_text(
        '\\begin{figure}[!htbp]\n\\centering\n'
        '\\includegraphics[width=\\textwidth,height=.58\\textheight,keepaspectratio]'
        '{\\SweepReportRoot/revision_figures/17_recuperacion_global.pdf}\n'
        '\\caption[Rutas de resolución en las 104 condiciones L3]{' + caption + '}\n'
        '\\label{fig:rev-17-recuperacion-global}\n\\end{figure}\n', encoding='utf-8')
    direct = sum(r['newton_only'] for r in rows)
    text = (r'La \cref{fig:rev-17-recuperacion-global} reúne las rutas observadas en '
            r'una resolución instrumentada por condición, con los mismos perfiles finales '
            r'de producción. ')
    if direct == 0:
        text += (r'En todas las condiciones se activa al menos un mecanismo de recuperación '
                 r'no lineal, lo que sitúa su papel en la obtención de la solución desde la '
                 r'estimación física inicial. ')
    else:
        text += (f'{direct} de las 104 condiciones convergen mediante Newton sin activar '
                 r'PTC, Euler implícito ni rescate térmico; las restantes combinan Newton con '
                 r'alguna de esas rutas. ')
    if all(r['be'] for r in rows) and all(r['ptc'] == (r['fuel'] == 'CH4') for r in rows):
        text += (r'Las llamas de metano utilizan PTC y Euler durante su trayectoria, mientras '
                 r'las de hidrógeno emplean Euler, de acuerdo con la estrategia configurada por '
                 r'mecanismo. ')
    if not any(r['thermal'] for r in rows):
        text += (r'El rescate con temperatura fijada permanece disponible, pero no se activa '
                 r'en estas ejecuciones. ')
    text += (r'La ampliación del dominio y la adaptación de malla atienden la resolución '
             r'espacial, por lo que se presentan como controles complementarios. La frecuencia '
             r'indica utilización; el coste de cada intervención se examina mediante los '
             r'contadores de los casos representativos.' + '\n\n'
             r'\input{\SweepReportRoot/revision_figures/17_recuperacion_global.tex}' + '\n'
             r'\FloatBarrier' + '\n')
    (out / '17_recuperacion_analisis.tex').write_text(text, encoding='utf-8')
    print(f'104/104 verified. Figure and chapter fragment saved: {out / "17_recuperacion_global.pdf"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
