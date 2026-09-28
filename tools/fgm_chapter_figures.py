"""Figures from the eight completed adaptive families; no flame solves."""
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter, MaxNLocator


def page_wrapper(out, name, caption, label):
    (out / f'{name}.tex').write_text(
        '\\clearpage\n\\begin{figure}[p]\\centering\n'
        f'\\includegraphics[width=\\textwidth,height=.77\\textheight,keepaspectratio]'
        f'{{\\FGMChapterRoot/{name}.pdf}}\n'
        f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
        '\\end{figure}\n\\clearpage\n', encoding='utf-8')


def current_figures(comparison, rows, out, labels):
    provenance = []
    # Chronological index is essential: adaptive insertion revisits composition intervals.
    fig, axes = plt.subplots(4, 2, figsize=(8.7, 10.0), sharex=True,
                             sharey=True, layout='constrained')
    exported, summaries = [], []
    markers = {'cold': 'D', 'copy': 's', 'secant': 'o'}
    for i, (tag, title) in enumerate(labels.items()):
        for j, solver in enumerate(('kflame', 'cantera')):
            row = next(r for r in rows if r['transport'] == tag and r['solver'] == solver)
            source = comparison / row['attempt'] / 'trace.json'
            trace = json.loads(source.read_text(encoding='utf-8'))
            provenance.append(source)
            ax = axes[i, j]
            color = '#246091' if j == 0 else '#bd682e'
            times = np.array([t['measured_s'] for t in trace])
            indices = np.arange(1, len(trace) + 1)
            ax.plot(indices, times, color=color, lw=1.05)
            kinds = [('cold' if t['predictor_kind'] == 'cold' else
                      'secant' if 'secant' in t['predictor_kind'] else 'copy') for t in trace]
            for kind, marker in markers.items():
                mask = np.array([k == kind for k in kinds])
                ax.scatter(indices[mask], times[mask], marker=marker, s=23,
                           facecolors=color if kind == 'cold' else 'white',
                           edgecolors=color, linewidths=.9, zorder=3)
            ax.set_yscale('log')
            ax.set_ylim(.35, 45)
            ax.set_yticks([.5, 1, 3, 10, 30], ['0.5', '1', '3', '10', '30'])
            ax.minorticks_off()
            ax.set_xlim(0, 42)
            ax.set_xticks([1, 10, 20, 30, 40])
            ax.set_title(f'{title} · {len(trace)} llamas', fontsize=10)
            if j == 0:
                ax.set_ylabel('Tiempo por llama [s]')
            if i == 3:
                ax.set_xlabel('Orden de resolución')
            if i == 0:
                ax.text(.5, 1.24, 'KFLAME' if j == 0 else 'Cantera',
                        transform=ax.transAxes, ha='center', fontsize=12, weight='bold')
            summaries.append(dict(solver=solver, transport=tag, first_s=float(times[0]),
                                  later_median_s=float(np.median(times[1:])),
                                  first_to_later_median=float(times[0]/np.median(times[1:]))))
            for index, t in enumerate(trace, 1):
                exported.append(dict(solver=solver, transport=tag, order=index,
                                     phi=t['phi'], round=t['round'], time_s=t['measured_s'],
                                     predictor=t['predictor_kind'], previous_phi=t['previous_phi']))
    handles = [Line2D([], [], color='.3', ls='', marker=markers[k],
                      markerfacecolor='.3' if k == 'cold' else 'white', label=label)
               for k, label in [('cold', 'Arranque físico'), ('copy', 'Perfil previo'),
                                ('secant', 'Predictor secante')]]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'01_familia.{ext}', dpi=190)
    plt.close(fig)
    with (out / '01_familia_datos.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader(); writer.writerows(exported)
    (out / '01_familia_resumen.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    page_wrapper(out, '01_familia',
        r'Tiempos individuales de las 318 llamas de los ocho FGM: CH$_4$--aire, 300 K y 1 atm. '
        r'Cada punto incluye resolución y propiedades; las líneas siguen el orden real de ejecución. '
        r'Los ejes temporales comparten escala logarítmica. El símbolo identifica la inicialización '
        r'efectivamente utilizada. Las primeras cinco llamas forman la familia inicial; las siguientes '
        r'son inserciones adaptativas, por lo que el orden no equivale a una composición creciente. '
        r'Cada panel corresponde a una construcción, sin promedios ni repeticiones.',
        'fig:fgm-01_familia')

    row = next(r for r in rows if r['transport'] == 'P' and r['solver'] == 'kflame')
    source = comparison / row['attempt'] / 'fgm_table.npz'
    provenance.append(source)
    with np.load(source, allow_pickle=False) as d:
        table = dict(d)
    z = table['Z_grid']; c = table['c_grid']
    zstar = (z - z.min()) / (z.max() - z.min())
    assert np.all(np.diff(zstar) > 0) and np.all(np.diff(c) > 0)
    ns = table['species_names'].tolist()
    # Same family and coordinates in every panel; preserve the approved profile/map composition.
    profiles = [('Temperatura', table['T'], r'$T$ [K]'),
                ('Velocidad local', table['u'], r'$u$ [m s$^{-1}$]'),
                ('Dióxido de carbono', table['Y'][:, ns.index('CO2'), :], r'$Y_{\mathrm{CO_2}}$ [kg kg$^{-1}$]'),
                ('Monóxido de carbono', table['Y'][:, ns.index('CO'), :], r'$Y_{\mathrm{CO}}$ [kg kg$^{-1}$]')]
    maps = [('Densidad', table['rho'], r'$\rho$ [kg m$^{-3}$]', 'viridis'),
            ('Conductividad térmica', table['conductivity'], r'$\lambda$ [W m$^{-1}$ K$^{-1}$]', 'viridis'),
            ('Liberación de calor', table['qdot'] / 1e9, r'$\dot q$ [GW m$^{-3}$]', 'magma'),
            ('Fuente química de progreso', table['omega_c'], r'$\dot\omega_c$ [kg m$^{-3}$ s$^{-1}$]', 'magma')]
    fig, axes = plt.subplots(4, 2, figsize=(8.7, 10.6), layout='constrained')
    cmap = plt.get_cmap('turbo')
    for i, ((title, values, ylabel), (mtitle, field, unit, palette)) in enumerate(zip(profiles, maps)):
        ax, mx = axes[i]
        for k, value in enumerate(values):
            ax.plot(c, value, color=cmap(zstar[k]), lw=.85)
        ax.set(xlim=(0, 1), xlabel='Progreso, $c$', ylabel=ylabel)
        ax.set_title(title, fontsize=10)
        # Filled isolines on the saved nonuniform grid avoid cell-edge rendering seams.
        mesh = mx.contourf(zstar, c, field.T,
                           levels=np.linspace(float(field.min()), float(field.max()), 65),
                           cmap=palette, antialiased=False)
        mesh.set_edgecolor('face')
        mesh.set_linewidth(.25)
        mx.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'Composición, $Z^\star$', ylabel='Progreso, $c$')
        mx.set_title(mtitle, fontsize=10)
        mx.plot(zstar, np.ones_like(zstar), '|', ms=4, color='black', clip_on=False)
        cb = fig.colorbar(mesh, ax=mx, fraction=.046, pad=.03)
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        for a in (ax, mx):
            a.tick_params(labelsize=9)
            a.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                      ax=list(axes[:, 0]), location='bottom', fraction=.035, pad=.02, aspect=35)
    cb.set_label(r'Color de los perfiles: composición $Z^\star$', fontsize=9)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'02_mapa.{ext}', dpi=190)
    plt.close(fig)
    page_wrapper(out, '02_mapa',
        r'Manifold actual de KFLAME, transporte promediado sin Soret: '
        f'{len(z)} llamas adaptativas y {len(c)} nodos de progreso, '
        r'CH$_4$--aire, 300 K y 1 atm. Los ocho campos proceden de la misma tabla. '
        r'Izquierda: perfiles coloreados por $Z^\star$, con escala común inferior. '
        r'Derecha: contornos de los campos tabulados con escala propia y unidades; las marcas superiores indican '
        r'las composiciones resueltas. $Z^\star$ normaliza la composición de entrada entre los extremos '
        r'de la familia; $c$ es el progreso normalizado. La velocidad $u$ es local dentro de la llama.',
        'fig:fgm-02-mapa')
    return provenance
