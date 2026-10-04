"""Reproduce four scientific figures from computed nonadiabatic data.

python examples/plot_nonadiabatic_3d.py runs/nonadiabatic --validation runs/validation3d \
    --output output/figures/nonadiabatic3d --pdf output/pdf/FGM_no_adiabatico_validacion.pdf
Published bundle: omit --validation when using docs/assets/nonadiabatic3d.
--export-bundle copies portable plot/lookup data; it never reruns a simulation.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM, OutsideManifoldError


COLORS = ['#0077BB', '#CC3311', '#009988', '#AA3377', '#EE7733', '#33BBEE', '#666666', '#331144', '#88AA00']


def training_summary(folder, family):
    cached = folder / 'training_summary.npz'
    if cached.exists():
        with np.load(cached, allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}
    shape = (len(family['phis']), len(family['mass_flux_fractions']) + 1)
    result = {k: np.full(shape, np.nan) for k in ('T_burned', 'deficit', 'q_burner', 'closure', 'mass_flux')}
    for row in family['rows']:
        i, j = row['composition_index'], row['loss_index']
        with np.load(folder / row['output'] / 'flame.npz', allow_pickle=False) as f:
            result['T_burned'][i, j] = f['T'][-1]
        result['mass_flux'][i, j] = row.get('mass_flux', row['adiabatic_mass_flux'])
        if j:
            loss = row['heat_loss']
            result['deficit'][i, j] = loss['burned_enthalpy_deficit_J_kg']
            result['q_burner'][i, j] = loss['burner_heat_loss_W_m2']
            result['closure'][i, j] = loss['relative_energy_closure_error']
    return result


def export_bundle(folder, validation, output, family, summary):
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(folder / 'nonadiabatic_fgm.npz', output / 'nonadiabatic_fgm.npz')
    # Plot/lookup metadata have no local continuation paths or private source files.
    public = {k: v for k, v in family.items() if k not in ('flame_settings', 'failures')}
    (output / 'metadata.json').write_text(json.dumps(public, indent=2) + '\n', encoding='utf-8', newline='\n')
    np.savez_compressed(output / 'training_summary.npz', **summary)
    shutil.copyfile(validation / 'validation.json', output / 'validation.json')
    for path in validation.glob('case_*_comparison.*'):
        shutil.copyfile(path, output / path.name)


def figures(folder, validation, output, pdf_path=None, export=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize

    folder, validation, output = Path(folder), Path(validation), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    family = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    report = json.loads((validation / 'validation.json').read_text(encoding='utf-8'))
    summary = training_summary(folder, family)
    model = NonAdiabaticFGM(folder)
    profiles = []
    for i in range(len(report['cases'])):
        with np.load(validation / f'case_{i:02d}_comparison.npz', allow_pickle=False) as saved:
            profiles.append({k: saved[k] for k in saved.files})
    if export is not None:
        export_bundle(folder, validation, Path(export), family, summary)
    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None

    def save(fig, name, number, note):
        fig.subplots_adjust(left=.10, right=.95, bottom=.14, top=.87, wspace=.34, hspace=.47)
        fig.text(.10, .045, note, fontsize=9, va='bottom')
        fig.text(.95, .025, str(number), ha='right', fontsize=9, color='.4')
        fig.savefig(output / f'{name}.png', dpi=200)
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'mathtext.fontset': 'dejavusans',
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlesize': 11}):
        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        fig.suptitle('CH₄-aire: pérdidas de calor hacia un quemador isotérmico', fontsize=16, y=.97)
        fig.text(.5, .915, 'Entrada y superficie: 300 K | 101 325 Pa | GRI-Mech 3.0 | Transporte promediado por mezcla',
                 ha='center', fontsize=10)
        ratio = np.array([1., *family['mass_flux_fractions']])
        for i, phi in enumerate(family['phis']):
            color = COLORS[i % len(COLORS)]
            axes[0, 0].plot(ratio, summary['T_burned'][i], 'o-', color=color, label=rf'$\phi={phi:g}$', ms=4)
            axes[0, 1].plot(ratio[1:], summary['deficit'][i, 1:] / 1000., 'o-', color=color, ms=4)
            axes[1, 0].plot(ratio[1:], summary['q_burner'][i, 1:] / 1000., 'o-', color=color, ms=4)
            axes[1, 1].plot(ratio[1:], 100. * summary['closure'][i, 1:], 'o-', color=color, ms=4)
        titles = ['(a) Temperatura de salida', '(b) Pérdida de entalpía por unidad de masa',
                  '(c) Flujo de calor hacia el quemador', '(d) Cierre del balance energético']
        ylabels = [r'$T_b$ [K]', r'$h_{entrada}-h_b$ [kJ/kg]', r'$q_{quemador}$ [kW/m²]', 'Error relativo [%]']
        for ax, title, ylabel in zip(axes.flat, titles, ylabels):
            ax.set(title=title, ylabel=ylabel, xlabel=r'Fracción de caudal $r=\dot{m}/(\rho_u S_u)$', xlim=(0., 1.02))
            ax.grid(alpha=.18)
        axes[0, 0].legend(fontsize=8, ncol=3)
        save(fig, '01_familia_perdidas', 1,
             f'{len(family["rows"])} llamas nativas aceptadas: {len(family["phis"])} adiabáticas y '
             f'{len(family["rows"])-len(family["phis"])} en quemador. El punto r = 1 es la referencia adiabática.\n'
             r'Error de cierre = $|q_{quemador}-\dot m(h_{entrada}-h_b)|/\max(|q_{quemador}|,|\dot m(h_{entrada}-h_b)|)$.')

        # Z is horizontal, C vertical. A blank cell is unresolved, never a zero.
        zmin, zmax = np.min(model.table['Z']), np.max(model.table['Z'])
        z_grid = np.linspace(zmin, zmax, 61)
        c_grid = np.linspace(0., np.max(model.table['C']), 91)
        Zg, Cg = np.meshgrid(z_grid, c_grid)
        deficits = [150000., 600000.]
        values = []
        for deficit in deficits:
            fields = {k: np.full(Zg.shape, np.nan) for k in ('T', 'omega_C')}
            for index in np.ndindex(Zg.shape):
                Z, C = Zg[index], Cg[index]
                try:
                    h = model.reference_enthalpy(Z=Z, C=C) - deficit
                    result = model.lookup(Z=Z, C=C, h=h)
                except OutsideManifoldError:
                    continue
                for k in fields:
                    fields[k][index] = result[k]
            values.append(fields)
        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        fig.suptitle('FGM no adiabático: cortes a pérdida de entalpía constante', fontsize=16, y=.97)
        fig.text(.5, .915, r'$\Delta h=h_{ad}(Z,C)-h$ | $C=Y_{CO_2}+Y_{H_2O}+Y_{CO}+0.5Y_{H_2}$ | $Z$ de Bilger local',
                 ha='center', fontsize=11)
        np.savez_compressed(output / 'fgm_cuts.npz', Z=Zg, C=Cg, delta_h=np.array(deficits),
                            T=np.array([v['T'] for v in values]), omega_C=np.array([v['omega_C'] for v in values]))
        norms = dict(T=Normalize(300., 100.*np.ceil(max(np.nanmax(v['T']) for v in values)/100.)),
                     omega_C=Normalize(min(0., min(np.nanmin(v['omega_C']) for v in values)),
                                       100.*np.ceil(max(np.nanmax(v['omega_C']) for v in values)/100.)))
        for i, (deficit, fields) in enumerate(zip(deficits, values)):
            for j, key in enumerate(('T', 'omega_C')):
                ax = axes[i, j]
                ax.set_facecolor('#e6e6e6')
                levels = np.linspace(norms[key].vmin, norms[key].vmax, 65)
                mesh = ax.contourf(Zg, Cg, np.ma.masked_invalid(fields[key]), levels=levels,
                                   cmap='cividis', norm=norms[key])
                ax.set_rasterization_zorder(0.)
                mesh.set_zorder(-1.)
                isolines = [800., 1200., 1600., 2000.] if key == 'T' else [10., 25., 100., 200.]
                contours = ax.contour(Zg, Cg, fields[key], levels=isolines, colors='.15', linewidths=.45, alpha=.55)
                ax.clabel(contours, fontsize=7, inline=True, fmt='%g')
                label = 'Temperatura [K]' if key == 'T' else r'Fuente $\Omega_C$ [kg/(m³ s)]'
                ticks = np.arange(300., norms[key].vmax+1., 300.) if key == 'T' else np.arange(0., norms[key].vmax+1., 100.)
                fig.colorbar(mesh, ax=ax, label=label, pad=.025, fraction=.055, ticks=ticks)
                ax.set(title=rf'$\Delta h={deficit/1000.:g}$ kJ/kg', xlabel=r'Fracción de mezcla local $Z$ [-]',
                       ylabel=r'Progreso común $C$ [-]')
        save(fig, '02_fgm_enthalpia', 2,
             'Cada columna comparte escala entre ambos cortes. Gris: estados fuera de las celdas calculadas o sin referencia adiabática.\n'
             'C conserva una definición común; no se normaliza por llama. La temperatura se recupera de la entalpía total y las especies.')

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        fig.suptitle('Validación: composiciones y caudales ausentes de la tabla', fontsize=16, y=.97)
        fig.text(.5, .915, 'Llama nativa detallada y referencia Cantera independientes; FGM consultado con sus controles locales',
                 ha='center', fontsize=10)
        for i, (ax, case, profile) in enumerate(zip(axes.flat, report['cases'], profiles)):
            ax.plot(1000. * profile['z'], profile['T'], color=COLORS[0], label='Nativa detallada', lw=2.2)
            ax.plot(1000. * profile['reference_z'], profile['reference_T'], color='.2', ls=':', label='Cantera', lw=1.8)
            ax.plot(1000. * profile['z'], profile['fgm_T'], color=COLORS[1], ls='--', label='FGM (interpolación)', lw=1.4)
            ax.set(title=rf'$\phi={case["phi"]:.3f}$; $r={case["fraction"]:g}$; $\dot{{m}}={case["mass_flux_kg_m2_s"]:.4f}$ kg/(m² s)',
                   xlabel='Distancia desde el quemador [mm]', ylabel='Temperatura [K]', xlim=(0., 3.))
            ax.grid(alpha=.18)
            ax.text(.97, .07, f'Cobertura: {100*case["interpolation"]["coverage_fraction"]:.1f} %',
                    transform=ax.transAxes, ha='right', fontsize=9)
            if i == 0:
                ax.legend(fontsize=9)
        save(fig, '03_validacion_perfiles', 3,
             'La ventana muestra los primeros 3 mm; los errores y la cobertura se calculan en todo el dominio (30 mm).\n'
             'Los huecos del FGM se conservan. Esta prueba mide tabulación y consistencia numérica; no resuelve transporte reducido.')

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        fig.suptitle('Errores de tabulación y verificación de las fuentes', fontsize=16, y=.97)
        fig.text(.5, .915, 'Máximos sobre estados cubiertos | Fuentes volumétricas | Colores: cuatro casos retenidos', ha='center', fontsize=10)
        for i, (case, profile) in enumerate(zip(report['cases'], profiles)):
            mask = profile['covered']
            color = COLORS[i]
            axes[0, 0].scatter(profile['T'][mask], profile['fgm_T'][mask], s=9, color=color, alpha=.65,
                               label=rf'$\phi={case["phi"]:.3f}, r={case["fraction"]:g}$')
            axes[0, 1].plot(profile['C'], profile['omega_C'], color=color, lw=1.4)
            axes[0, 1].plot(profile['C'], profile['fgm_omega_C'], color=color, ls='--', lw=1.4)
        axes[0, 0].plot([300., 2400.], [300., 2400.], 'k:', lw=1.)
        axes[0, 0].set(title='(a) Temperatura: FGM frente a llama detallada', xlabel='Temperatura nativa [K]', ylabel='Temperatura FGM [K]')
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].set(title='(b) Fuente de progreso', xlabel=r'Progreso común $C$ [-]', ylabel=r'$\Omega_C$ [kg/(m³ s)]')
        axes[0, 1].text(.03, .95, 'Continua: nativa\nDiscontinua: FGM', transform=axes[0, 1].transAxes, va='top', fontsize=9)
        x = np.arange(len(report['cases']))
        table_error = [c['interpolation']['temperature_Linf_K'] for c in report['cases']]
        ref_error = [c['reference']['temperature_Linf_K'] for c in report['cases']]
        axes[1, 0].bar(x-.18, table_error, .36, color=COLORS[1], label='FGM / nativa')
        axes[1, 0].bar(x+.18, ref_error, .36, color=COLORS[0], label='Nativa / Cantera')
        axes[1, 0].set(title='(c) Máximo error de temperatura', ylabel='Error absoluto [K]', xlabel='Caso retenido')
        axes[1, 0].legend(fontsize=9)
        omega_error = [100*c['interpolation']['omega_C_Linf_over_native_peak'] for c in report['cases']]
        q_error = [100*c['interpolation']['qdot_Linf_over_native_peak'] for c in report['cases']]
        axes[1, 1].bar(x-.18, omega_error, .36, color=COLORS[2], label=r'$\Omega_C$')
        axes[1, 1].bar(x+.18, q_error, .36, color=COLORS[3], label=r'$\dot{q}$')
        axes[1, 1].axhline(15., color='.3', ls=':', label='Umbral fijado: 15 %')
        axes[1, 1].set(title='(d) Error de fuentes / pico nativo', ylabel='Error relativo al pico [%]', xlabel='Caso retenido')
        axes[1, 1].legend(fontsize=9, ncol=2)
        for ax in axes[1]:
            ax.set_xticks(x, [str(i+1) for i in x])
        for ax in axes.flat:
            ax.grid(alpha=.15)
        save(fig, '04_validacion_errores', 4,
             'Error de fuentes = max|FGM - nativa| / max|nativa|; no es un error relativo punto a punto.\n'
             f'Los cuatro casos cambian composición y caudal simultáneamente; no forman parte de las {len(family["rows"])} llamas de entrenamiento.')
    if pdf is not None:
        pdf.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/figures/nonadiabatic3d'))
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--export-bundle', type=Path)
    args = parser.parse_args()
    print(figures(args.folder, args.validation or args.folder, args.output, args.pdf, args.export_bundle))


if __name__ == '__main__':
    main()
