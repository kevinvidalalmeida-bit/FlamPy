import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

import postprocess_flame_sweeps as pp


# =============================================================================
# CONFIGURACIÓN GRÁFICA
# =============================================================================

# Colores utilizados en el TFM
AZUL_TFM = '#155C8A'
NARANJA_TFM = '#D97706'

# Ancho útil aproximado de una página A4 con márgenes de 2.5 cm.
# La figura queda preparada para insertarse con width=\textwidth.
FIG_WIDTH = 6.30
FIG_HEIGHT = 4.55

# Posiciones horizontales.
# Se deja un espacio adicional antes del caso L3 con dominio 2L.
X_LEVELS = [0.0, 1.0, 2.0, 3.0]
X_DOMAIN = 4.65

# Correspondencia entre condiciones y columnas
CONDITIONS = {
    (1, 300): 0,
    (10, 300): 1,
    (1, 500): 2,
}

# Estilo de cada solver
BACKEND_STYLE = {
    'native': {
        'label': 'KFLAME',
        'color': AZUL_TFM,
        'linestyle': '-',
        'marker': 'o',
    },
    'cantera': {
        'label': 'Cantera',
        'color': NARANJA_TFM,
        'linestyle': '--',
        'marker': 's',
    },
}


def configure_matplotlib():
    """Configura Matplotlib con un estilo compatible con la tesis."""

    plt.rcParams.update({

        # Tipografía
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'font.size': 8.2,

        'axes.titlesize': 8.5,
        'axes.labelsize': 8.5,

        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,

        'legend.fontsize': 7.8,

        # Ejes
        'axes.linewidth': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,

        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,

        # PDF vectorial
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        # Matemática
        'mathtext.fontset': 'stix',
    })


def collect_levels(manifest, records, case_id, backend):
    """
    Reúne los resultados disponibles para un caso y solver.

    Claves:
        2 -> L2
        3 -> L3
        4 -> L4
        5 -> L5
        6 -> L3 con longitud inicial del dominio 2L
    """

    levels = {}

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
    # Se utiliza como L3 si existe.
    # Una eventual ejecución verify-L3 tendrá prioridad posteriormente.
    # -------------------------------------------------------------------------

    for rec in manifest.get('baseline', []):

        if (
            rec.get('case_id') == case_id
            and rec.get('backend') == backend
        ):

            if rec.get('time_s') is not None:
                levels[3] = rec

            break

    # -------------------------------------------------------------------------
    # Registros de sensibilidad
    # -------------------------------------------------------------------------

    for rec in records:

        if rec.get('case_id') != case_id:
            continue

        if rec.get('backend') != backend:
            continue

        accepted = rec.get(
            'accepted',
            rec.get('status') == 'accepted'
        )

        if not accepted:
            continue

        if rec.get('time_s') is None:
            continue

        phase = rec.get('phase', '')

        # Refinamiento de malla
        if phase.startswith('verify-L'):

            level = rec.get(
                'settings', {}
            ).get('level')

            if level in (2, 3, 4, 5):
                levels[int(level)] = rec

        # Sensibilidad al dominio
        elif phase == 'domain-L3':

            levels[6] = rec

    return levels


def main():

    # =========================================================================
    # ARGUMENTOS
    # =========================================================================

    parser = argparse.ArgumentParser(
        description=(
            'Genera la figura de tiempo de resolución frente al '
            'nivel de refinamiento y longitud inicial del dominio.'
        )
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path(
            'runs/thesis_flames_L3_sensitivity'
        ),
        help='Directorio de entrada de la campaña de sensibilidad.',
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path(
            'runs/thesis_flames_L3/report'
        ),
        help='Directorio de salida.',
    )

    args = parser.parse_args()

    out = args.output
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================================
    # CARGA DE DATOS
    # =========================================================================

    manifest = pp.read(
        args.input / 'manifest.json'
    )

    records = pp.load_records(
        args.input
    )

    csv_rows = []

    configure_matplotlib()

    # =========================================================================
    # FIGURA
    # =========================================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        ),
        sharex=True,
    )

    # Todos los paneles se ocultan inicialmente.
    # Solo se activan cuando existe la condición correspondiente.
    for ax in axes.flat:
        ax.set_visible(False)

    # =========================================================================
    # CASOS
    # =========================================================================

    for case in manifest['cases']:

        fuel = case['fuel']
        pressure = case['pressure_atm']
        temperature = case['temperature']

        # Solo se representan los dos mecanismos del estudio
        if fuel not in ('CH4', 'H2'):
            continue

        col = CONDITIONS.get(
            (pressure, temperature)
        )

        if col is None:
            continue

        row = 0 if fuel == 'CH4' else 1

        ax = axes[row, col]
        ax.set_visible(True)

        # ---------------------------------------------------------------------
        # Título de las columnas
        # ---------------------------------------------------------------------
        # La condición termodinámica se escribe una sola vez, en la fila
        # superior. El combustible/mecanismo se identifica lateralmente.
        # ---------------------------------------------------------------------

        if row == 0:

            ax.set_title(
                f'{pressure:g} atm, {temperature:g} K',
                pad=6.0,
            )

        panel_times = []

        # ---------------------------------------------------------------------
        # Tiempos necesarios para calcular el speed-up
        # ---------------------------------------------------------------------
        speedup_times = {}

        # =====================================================================
        # SOLVERS
        # =====================================================================

        for backend in ('native', 'cantera'):

            style = BACKEND_STYLE[backend]

            levels = collect_levels(
                manifest=manifest,
                records=records,
                case_id=case['id'],
                backend=backend,
            )

            # -----------------------------------------------------------------
            # Refinamiento L2--L5
            # -----------------------------------------------------------------

            xs = []
            ys = []

            for level, xpos in zip(
                (2, 3, 4, 5),
                X_LEVELS
            ):

                if level not in levels:
                    continue

                value = float(
                    levels[level]['time_s']
                )

                if value <= 0:
                    continue

                xs.append(xpos)
                ys.append(value)

                panel_times.append(value)

                # -------------------------------------------------------------
                # Guardar los tiempos de cada nivel para calcular el speed-up
                # -------------------------------------------------------------

                if level not in speedup_times:
                    speedup_times[level] = {}

                speedup_times[level][backend] = value

                csv_rows.append({
                    'combustible': fuel,
                    'presion': pressure,
                    'temperatura': temperature,
                    'backend': backend,
                    'malla': f'L{level}',
                    'tiempo_s': value,
                })

            if xs:

                ax.plot(
                    xs,
                    ys,

                    color=style['color'],
                    linestyle=style['linestyle'],

                    linewidth=1.10,

                    marker=style['marker'],
                    markersize=3.5,
                    markeredgewidth=0.65,

                    zorder=3,
                )

            # -----------------------------------------------------------------
            # L3 con dominio inicial 2L
            # -----------------------------------------------------------------

            if 6 in levels:

                value = float(
                    levels[6]['time_s']
                )

                if value > 0:

                    panel_times.append(value)

                    ax.plot(
                        [X_DOMAIN],
                        [value],

                        linestyle='none',

                        marker='x',
                        markersize=5.0,
                        markeredgewidth=1.0,

                        color=style['color'],

                        zorder=4,
                    )

                    csv_rows.append({
                        'combustible': fuel,
                        'presion': pressure,
                        'temperatura': temperature,
                        'backend': backend,
                        'malla': 'L3_2L',
                        'tiempo_s': value,
                    })

        # =====================================================================
        # CONFIGURACIÓN DEL PANEL
        # =====================================================================

        ax.set_yscale('log')

        # Límites horizontales
        ax.set_xlim(
            -0.35,
            5.05
        )

        # Etiquetas del eje x
        ax.set_xticks(
            [
                *X_LEVELS,
                X_DOMAIN,
            ],
            [
                'L2',
                'L3',
                'L4',
                'L5',
                'L3\n$2L$',
            ]
        )

        # ---------------------------------------------------------------------
        # Límites verticales
        # ---------------------------------------------------------------------
        # Se añade un margen multiplicativo porque el eje es logarítmico.
        # ---------------------------------------------------------------------

        if panel_times:

            ymin = min(panel_times)
            ymax = max(panel_times)

            if ymin > 0:

                ax.set_ylim(
                    ymin / 1.35,
                    ymax * 1.35,
                )

        # =====================================================================
        # SPEED-UP
        # =====================================================================
        #
        # S_t = t_Cantera / t_KFLAME
        #
        # Se muestran únicamente L2 y L5 para visualizar cómo evoluciona
        # la separación temporal al aumentar el refinamiento.
        #
        # La barra vertical une los tiempos de ambos solvers y el valor
        # del speed-up se coloca en el centro logarítmico de la barra.
        # =====================================================================

        for level in (2, 5):

            if (
                level not in speedup_times
                or 'native' not in speedup_times[level]
                or 'cantera' not in speedup_times[level]
            ):
                continue

            t_native = speedup_times[level]['native']
            t_cantera = speedup_times[level]['cantera']

            speedup = t_cantera / t_native

            xpos = X_LEVELS[level - 2]

            y_low = min(
                t_native,
                t_cantera
            )

            y_high = max(
                t_native,
                t_cantera
            )

            # Centro correcto para un eje logarítmico
            y_mid = (
                y_low * y_high
            ) ** 0.5

            # Barra vertical entre ambos puntos
            ax.annotate(
                '',
                xy=(
                    xpos,
                    y_high
                ),
                xytext=(
                    xpos,
                    y_low
                ),
                arrowprops=dict(
                    arrowstyle='<->',
                    color='0.42',
                    linewidth=0.70,
                    shrinkA=2.5,
                    shrinkB=2.5,
                ),
                zorder=2,
            )

            # Número del speed-up junto a la barra y centrado verticalmente
            ax.text(
                xpos + 0.11,
                y_mid,

                rf'${speedup:.1f}\times$',

                rotation=90,

                ha='left',
                va='center',

                fontsize=6.6,
                color='0.30',

                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.90,
                    pad=0.10,
                ),

                clip_on=False,
                zorder=6,
            )

        # ---------------------------------------------------------------------
        # Ticks logarítmicos
        # ---------------------------------------------------------------------

        ax.yaxis.set_major_locator(
            LogLocator(
                base=10,
                numticks=5,
            )
        )

        ax.yaxis.set_minor_locator(
            LogLocator(
                base=10,
                subs=(
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ),
                numticks=12,
            )
        )

        ax.yaxis.set_minor_formatter(
            NullFormatter()
        )

        ax.tick_params(
            axis='both',
            which='both',
            direction='out',
        )

        # ---------------------------------------------------------------------
        # Sin grid
        # ---------------------------------------------------------------------

        ax.grid(False)


    # =========================================================================
    # IDENTIFICACIÓN DE LAS FILAS
    # =========================================================================
    #
    # Se utiliza la misma lógica visual de la figura de sensibilidad espacial:
    #
    #   CH4 / GRI-Mech 3.0
    #   H2  / h2o2.yaml
    #
    # Las etiquetas se colocan independientemente del ylabel general para
    # evitar superposiciones.
    # =========================================================================

    fig.text(
        0.022,
        0.625,
        r'CH$_4$ / GRI-Mech 3.0',

        rotation=90,

        va='center',
        ha='center',

        fontsize=8.5,
    )

    fig.text(
        0.022,
        0.315,
        r'H$_2$ / h2o2.yaml',

        rotation=90,

        va='center',
        ha='center',

        fontsize=8.5,
    )


    # =========================================================================
    # ETIQUETAS GENERALES
    # =========================================================================

    fig.supylabel(
        'Tiempo de resolución [s]',
        x=0.070,
        fontsize=8.5,
    )

    fig.supxlabel(
        'Nivel de refinamiento / dominio',
        x=0.565,
        y=0.025,
        fontsize=8.5,
    )


    # =========================================================================
    # LEYENDA GLOBAL
    # =========================================================================

    legend_handles = [

        Line2D(
            [0],
            [0],

            color=AZUL_TFM,

            linewidth=1.10,

            linestyle='-',

            marker='o',
            markersize=3.5,

            label='KFLAME',
        ),

        Line2D(
            [0],
            [0],

            color=NARANJA_TFM,

            linewidth=1.10,

            linestyle='--',

            marker='s',
            markersize=3.5,

            label='Cantera',
        ),

        Line2D(
            [0],
            [0],

            color='0.25',

            linestyle='none',

            marker='x',
            markersize=5.0,
            markeredgewidth=1.0,

            label=r'L3 con $2L$',
        ),
    ]

    fig.legend(
        handles=legend_handles,

        loc='upper center',

        bbox_to_anchor=(
            0.565,
            0.985,
        ),

        ncol=3,

        frameon=False,

        handlelength=2.3,
        handletextpad=0.55,
        columnspacing=1.45,

        borderaxespad=0.0,
    )


    # =========================================================================
    # ESPACIADO FINAL
    # =========================================================================
    #
    # Se utilizan márgenes manuales para controlar exactamente la composición.
    #
    # No utilizar:
    #
    #     tight_layout()
    #     constrained_layout=True
    #     bbox_inches='tight'
    #
    # porque pueden desplazar las etiquetas externas y modificar el tamaño
    # efectivo del PDF.
    # =========================================================================

    fig.subplots_adjust(
        left=0.165,
        right=0.990,

        bottom=0.135,
        top=0.830,

        wspace=0.34,
        hspace=0.38,
    )


    # =========================================================================
    # ARCHIVOS DE SALIDA
    # =========================================================================

    pdf_path = (
        out / 'tiempo_vs_malla.pdf'
    )

    png_path = (
        out / 'tiempo_vs_malla.png'
    )

    csv_path = (
        out / 'tiempo_vs_malla.csv'
    )


    # PDF vectorial para la tesis
    fig.savefig(
        pdf_path
    )

    # PNG únicamente para inspección rápida
    fig.savefig(
        png_path,
        dpi=240
    )

    plt.close(fig)


    # =========================================================================
    # EXPORTACIÓN DE DATOS
    # =========================================================================

    with csv_path.open(
        'w',
        newline='',
        encoding='utf-8',
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                'combustible',
                'presion',
                'temperatura',
                'backend',
                'malla',
                'tiempo_s',
            ],
        )

        writer.writeheader()
        writer.writerows(csv_rows)


    # =========================================================================
    # RESUMEN
    # =========================================================================

    print(
        f'Gráfico PDF guardado en: {pdf_path}'
    )

    print(
        f'Gráfico PNG guardado en: {png_path}'
    )

    print(
        f'Datos exportados a: {csv_path}'
    )


if __name__ == '__main__':
    main()