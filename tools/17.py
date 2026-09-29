#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figura de trabajo computacional por llama.

Genera una figura 4x2 con:
  1) Evaluaciones del residual
  2) Construcciones del Jacobiano
  3) Tiempo medio por Jacobiano
  4) Pasos Euler aceptados

Las columnas corresponden a CH4 / GRI-Mech 3.0 y H2 / h2o2.
La leyenda KFLAME / Cantera aparece una sola vez en la parte superior.
Los títulos de cada fila se colocan verticalmente y se centran usando
la posición real de los ejes, evitando desfases visuales.

Uso:
    python plot_trabajo_por_llama_limpio.py \
        --input 17_trabajo_por_llama.csv \
        --output-dir revision_figures

Salidas:
    revision_figures/17_trabajo_por_llama.pdf
    revision_figures/17_trabajo_por_llama.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = [
    ("mixture-averaged", "False", "Promediado"),
    ("mixture-averaged", "True",  "Promediado + Soret"),
    ("multicomponent",   "False", "Multicomponente"),
    ("multicomponent",   "True",  "Multi. + Soret"),
]

FIELDS = [
    ("residual",       "Evaluaciones\ndel residual",      "Número de evaluaciones"),
    ("jacobian",       "Construcciones\ndel Jacobiano",   "Número de construcciones"),
    ("jacobian_ms",    "Tiempo medio\npor Jacobiano",     "Tiempo por construcción [ms]"),
    ("euler_accepted", "Pasos Euler\naceptados",          "Número de pasos aceptados"),
]


def _as_float(value: str | None) -> float:
    if value in (None, ""):
        return np.nan
    return float(value)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _values(
    rows: list[dict[str, str]],
    fuel: str,
    backend: str,
    field: str,
) -> list[float]:
    values: list[float] = []

    for transport, soret, _ in MODES:
        row = next(
            r for r in rows
            if r["fuel"] == fuel
            and r["backend"] == backend
            and r["transport"] == transport
            and r["soret"] == soret
        )
        values.append(_as_float(row[field]))

    return values


def make_figure(rows: list[dict[str, str]], out_pdf: Path, out_png: Path) -> None:
    # Tipografía y exportación vectorial compatibles con el documento LaTeX.
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.grid": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(4, 2, figsize=(10.6, 10.0))

    # Márgenes fijos: permiten ubicar títulos laterales con precisión.
    plt.subplots_adjust(
        left=0.255,
        right=0.97,
        bottom=0.075,
        top=0.865,
        hspace=0.92,
        wspace=0.20,
    )

    legend_handles = None

    # Escalas comunes para magnitudes de conteo.
    shared_limits: dict[str, float] = {}
    for field in ("residual", "jacobian", "euler_accepted"):
        shared_limits[field] = (
            max(
                _as_float(r[field])
                for r in rows
                if r.get(field) not in ("", None)
            )
            * 1.18
        )

    for i, (field, _, xlabel) in enumerate(FIELDS):
        for j, fuel in enumerate(("CH4", "H2")):
            ax = axes[i, j]
            y = np.arange(4)

            kflame = _values(rows, fuel, "native", field)
            cantera = _values(rows, fuel, "cantera", field)

            bars_k = ax.barh(
                y - 0.17,
                kflame,
                height=0.30,
                label="KFLAME",
            )
            bars_c = ax.barh(
                y + 0.17,
                cantera,
                height=0.30,
                label="Cantera",
            )

            if legend_handles is None:
                legend_handles = [bars_k[0], bars_c[0]]

            ax.set_yticks(y)
            ax.set_yticklabels(
                [m[2] for m in MODES] if j == 0 else [""] * 4
            )
            ax.invert_yaxis()

            ax.set_xlabel(xlabel, fontsize=9)
            ax.tick_params(axis="x", labelsize=8.5)
            ax.tick_params(axis="y", length=0, labelsize=9, pad=3)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            if field in shared_limits:
                ax.set_xlim(0, shared_limits[field])
            else:
                ax.set_xlim(0, max(kflame + cantera) * 1.18)

            if field == "jacobian_ms":
                labels_k = [
                    f"{v:.2f}" if v < 10 else f"{v:.1f}"
                    for v in kflame
                ]
                labels_c = [
                    f"{v:.2f}" if v < 10 else f"{v:.1f}"
                    for v in cantera
                ]
            else:
                labels_k = [f"{int(round(v))}" for v in kflame]
                labels_c = [f"{int(round(v))}" for v in cantera]

            ax.bar_label(
                bars_k,
                labels=labels_k,
                padding=3,
                fontsize=8.5,
            )
            ax.bar_label(
                bars_c,
                labels=labels_c,
                padding=3,
                fontsize=8.5,
            )

    # Obliga a Matplotlib a calcular la geometría definitiva de los ejes.
    fig.canvas.draw()

    # Centros horizontales exactos de las dos columnas.
    left_box = axes[0, 0].get_position()
    right_box = axes[0, 1].get_position()

    x_left = (left_box.x0 + left_box.x1) / 2
    x_right = (right_box.x0 + right_box.x1) / 2
    x_center = (x_left + x_right) / 2

    # Leyenda única.
    fig.legend(
        legend_handles,
        ["KFLAME", "Cantera"],
        loc="upper center",
        bbox_to_anchor=(x_center, 0.988),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    # Encabezados de mecanismo una sola vez.
    fig.text(
        x_left,
        0.912,
        r"CH$_4$ / GRI-Mech 3.0",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        x_right,
        0.912,
        r"H$_2$ / h2o2",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    # Títulos laterales centrados exactamente en cada fila.
    x_row_title = 0.095

    for i, (_, row_title, _) in enumerate(FIELDS):
        box = axes[i, 0].get_position()
        y_center = (box.y0 + box.y1) / 2

        fig.text(
            x_row_title,
            y_center,
            row_title,
            ha="center",
            va="center",
            rotation=90,
            rotation_mode="anchor",
            multialignment="center",
            fontsize=10.5,
            linespacing=1.05,
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        out_png,
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/thesis_flames_L3/report_expanded/revision_figures/17_trabajo_por_llama.csv"),
        help="CSV generado por el diagnóstico.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/thesis_flames_L3/report_expanded/revision_figures"),
        help="Directorio de salida.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    rows = _load_rows(input_path)

    out_pdf = output_dir / "17_trabajo_por_llama.pdf"
    out_png = output_dir / "17_trabajo_por_llama.png"

    make_figure(rows, out_pdf, out_png)

    print(f"PDF: {out_pdf}")
    print(f"PNG: {out_png}")


if __name__ == "__main__":
    main()
