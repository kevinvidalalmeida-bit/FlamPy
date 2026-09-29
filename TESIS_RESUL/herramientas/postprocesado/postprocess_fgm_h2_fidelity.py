"""Create the independent H2 FGM fidelity figure from saved profiles only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "TESIS_RESUL/herramientas/campanas"))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from benchmark_fgm_campaign import digest
from kflame.fgm.common import monotonicize_on_c

FIELDS = ("T", "H2O", "OH", "omega_c")
LABELS = (r"$T$", r"$Y_{H_2O}$", r"$Y_{OH}$", r"$\dot\omega_c$")
UNITS = (r"$T$ [K]", r"$Y_{H_2O}$ [kg/kg]", r"$Y_{OH}$ [kg/kg]", r"$\dot\omega_c$ [kg m$^{-3}$ s$^{-1}$]")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def scalar(value):
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--construction", type=Path,
                   default=Path("TESIS_RESUL/corridas/FGM/thesis_fgm_h2_comparison/cases/H2_T300_p1_P_kflame/attempt-001"))
    p.add_argument("--holdouts", type=Path,
                   default=Path("TESIS_RESUL/corridas/FGM/thesis_fgm_h2_holdouts"))
    p.add_argument("--output", type=Path,
                   default=Path("TESIS_RESUL/corridas/reproduccion/figuras_FGM_H2"))
    args = p.parse_args(argv)
    construction, holdouts, out = args.construction.resolve(), args.holdouts.resolve(), args.output.resolve()
    result = json.loads((holdouts / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "accepted":
        raise ValueError("The H2 holdout campaign is incomplete.")
    table = load_npz(construction / "fgm_table.npz")
    names = [str(x) for x in table["species_names"]]
    values = np.stack([table["T"], table["Y"][:, names.index("H2O")],
                       table["Y"][:, names.index("OH")], table["omega_c"]], axis=-1)
    interp = RegularGridInterpolator((table["Z_grid"], table["c_grid"]), values, bounds_error=True)
    c = np.linspace(0, 1, 1001)
    truth, prediction, phi, z = [], [], [], []
    for path in sorted((holdouts / "profiles").glob("*.npz")):
        record = {key: scalar(value) for key, value in load_npz(path).items()}
        if not (record["solve_ok"] and record["final_accepted"]):
            raise ValueError(f"Unaccepted holdout: {path}")
        names_r = [str(x) for x in record["species_names"]]
        if names_r != names:
            raise ValueError("Species list differs between construction and retained flame.")
        raw_c = (record["beta"] - record["beta"][0]) / (record["beta"][-1] - record["beta"][0])
        if np.min(np.diff(raw_c)) < -1e-7:
            raise ValueError(f"Nonmonotone progress in {path}")
        cu, fields = monotonicize_on_c(record["c"], dict(T=record["T"], H2O=record["Y"][names.index("H2O")],
                                                           OH=record["Y"][names.index("OH")], omega_c=record["omega_c"]))
        truth.append(np.column_stack([np.interp(c, cu, fields[name]) for name in FIELDS]))
        prediction.append(interp(np.column_stack([np.full(c.size, record["Z"]), c])))
        phi.append(float(record["phi"])); z.append(float(record["Z"]))
    truth, prediction, phi, z = np.asarray(truth), np.asarray(prediction), np.asarray(phi), np.asarray(z)
    if np.isclose(phi[:, None], table["phi_grid"][None, :], rtol=0, atol=1e-12).any():
        raise ValueError("Holdout leakage: retained and construction compositions overlap.")
    scale = np.max(np.abs(truth), axis=(0, 1)); scale[0] = np.max(np.ptp(truth[:, :, 0], axis=1))
    error = 100 * np.abs(prediction - truth) / scale
    metrics = [dict(field=name, scale=float(scale[i]), mean_percent=float(error[:, :, i].mean()),
                    p95_percent=float(np.quantile(error[:, :, i], .95)), max_percent=float(error[:, :, i].max()))
               for i, name in enumerate(FIELDS)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "H2_table_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    (out / "H2_fidelity_summary.json").write_text(json.dumps(dict(source_construction=str(construction),
        source_table_sha256=digest(construction / "fgm_table.npz"), holdout_manifest_sha256=digest(holdouts / "manifest.json"),
        n_holdouts=len(phi), phi=phi.tolist(), scales=scale.tolist(), metrics=metrics), indent=2), encoding="utf-8")
    lines = [r"\begin{table}[H]\centering\small", r"\caption{Error de interpolacion en doce llamas de retencion independientes de H$_2$--aire. Cada fila usa 1001 posiciones uniformes de $c$ por llama, con igual peso por llama. Los valores se expresan como porcentaje de la escala de referencia de cada campo.}", r"\label{tab:fgm-h2-errors}", r"\begin{tabular}{lrrr}\toprule Magnitud & Media [\%] & P95 [\%] & Maximo [\%]\\\midrule"]
    pretty = {"T": "$T$", "H2O": "$Y_{H_2O}$", "OH": "$Y_{OH}$", "omega_c": "$\\dot\\omega_c$"}
    lines += [f"{pretty[row['field']]} & {row['mean_percent']:.3g} & {row['p95_percent']:.3g} & {row['max_percent']:.3g}\\\\" for row in metrics]
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (out / "H2_table_errors.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.grid": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    chosen = int(np.argmin(np.abs(np.log(phi))))
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 7.8), layout="constrained")
    for i, ax in enumerate(axes[:2].flat):
        ax.plot(c, truth[chosen, :, i], color="#27758b", lw=1.8, label="Llama retenida")
        ax.plot(c, prediction[chosen, :, i], color="#bd7940", ls="--", lw=1.5, label="FGM")
        ax.set(xlabel=r"Progreso $c$", ylabel=UNITS[i], xlim=(0, 1))
        ax.set_title(f"({chr(97 + i)})", loc="left", fontsize=10); ax.tick_params(top=False, right=False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 1].text(.03, .92, rf"$\phi={phi[chosen]:.4f}$", transform=axes[0, 1].transAxes, fontsize=10)
    axes[2, 0].boxplot([error[:, :, i].ravel() for i in range(4)], tick_labels=LABELS, showfliers=False, whis=(5, 95))
    p95 = float(np.quantile(error, .95))
    axes[2, 0].set(ylabel="Error normalizado [%]", title="(e) Todas las llamas retenidas", ylim=(0, max(1e-8, 1.15 * p95)))
    zstar = (z - table["Z_grid"][0]) / np.ptp(table["Z_grid"])
    xx, yy = np.meshgrid(zstar, c[::10], indexing="ij")
    cloud = axes[2, 1].scatter(xx.ravel(), yy.ravel(), c=error[:, ::10, :].max(axis=2).ravel(), s=5, cmap="magma")
    fig.colorbar(cloud, ax=axes[2, 1], label="Mayor error [%]", fraction=.06, pad=.03)
    axes[2, 1].set(xlabel=r"$Z^\star$", ylabel=r"$c$", xlim=(0, 1), ylim=(0, 1), title="(f) Localizacion del error")
    for ax in axes[2]: ax.tick_params(top=False, right=False)
    fig.savefig(out / "H2_03_fidelidad.pdf", bbox_inches="tight")
    fig.savefig(out / "H2_03_fidelidad.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'H2_03_fidelidad.pdf'} from {len(phi)} independent H2 retained flames.")


if __name__ == "__main__":
    main()
