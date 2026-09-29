"""Render the H2/h2o2 eight-FGM figures from saved campaign data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from postprocess_fgm_adaptive_campaign import collect, LABELS


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/thesis_fgm_h2_comparison"))
    parser.add_argument("--output", type=Path, default=Path(".local/research/thesis/figuras/FGM_H2"))
    args = parser.parse_args(argv)
    campaign, out = args.input.resolve(), args.output.resolve()
    manifest, rows, _ = collect(campaign, expected_kind="fgm-h2-solver-comparison")
    if len(rows) != 8 or any(row["status"] != "accepted" for row in rows):
        raise ValueError("The H2 figure package requires eight accepted FGM.")
    report = campaign / "report"
    if not (report / "table_concordance.json").exists():
        raise ValueError("Run postprocess_fgm_solver_comparison.py --h2 first.")
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "axes.grid": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})

    # 01: actual solve chronology, one panel per transport and solver.
    fig, axes = plt.subplots(4, 2, figsize=(8.7, 10), sharex=True, layout="constrained")
    traces, all_times = {}, []
    for row in rows:
        trace = json.loads((campaign / row["attempt"] / "trace.json").read_text(encoding="utf-8"))
        traces[(row["transport"], row["solver"])] = trace
        all_times.extend(t["measured_s"] for t in trace)
    lo, hi = min(all_times), max(all_times)
    ymin, ymax = 10 ** np.floor(np.log10(lo) - .2), 10 ** np.ceil(np.log10(hi) + .2)
    marks = {"cold": "D", "copy": "s", "secant": "o"}
    for i, (tag, title) in enumerate(LABELS.items()):
        for j, solver in enumerate(("kflame", "cantera")):
            trace = traces[(tag, solver)]; ax = axes[i, j]
            times = np.array([t["measured_s"] for t in trace]); x = np.arange(1, len(trace)+1)
            color = "#246091" if solver == "kflame" else "#bd682e"
            ax.plot(x, times, color=color, lw=1.05)
            kinds = ["cold" if t["predictor_kind"] == "cold" else ("secant" if "secant" in t["predictor_kind"] else "copy") for t in trace]
            for kind, marker in marks.items():
                mask = np.array([v == kind for v in kinds])
                ax.scatter(x[mask], times[mask], marker=marker, s=23, facecolors=color if kind == "cold" else "white", edgecolors=color)
            ax.set(yscale="log", ylim=(ymin, ymax), xlim=(0, max(len(t) for t in traces.values())+1), title=f"{title} · {len(trace)} llamas")
            if j == 0: ax.set_ylabel("Tiempo por llama [s]")
            if i == 3: ax.set_xlabel("Orden de resolucion")
            if i == 0: ax.text(.5, 1.22, "KFLAME" if solver == "kflame" else "Cantera", transform=ax.transAxes, ha="center", weight="bold", fontsize=12)
    fig.legend(handles=[Line2D([], [], marker=m, ls="", color=".3", label=l) for m, l in (("D", "Arranque fÃ­sico"), ("s", "Perfil previo"), ("o", "Predictor secante"))], loc="outside lower center", ncol=3, frameon=False)
    save(fig, out, "01_familia"); plt.close(fig)

    # 02: one saved KFLAME/P table, with H2-specific species panels.
    row = next(r for r in rows if r["transport"] == "P" and r["solver"] == "kflame")
    with np.load(campaign / row["attempt"] / "fgm_table.npz", allow_pickle=False) as data: table = dict(data)
    z, c, names = table["Z_grid"], table["c_grid"], table["species_names"].tolist()
    zstar = (z-z.min())/(z.max()-z.min())
    profiles = [("Temperatura", table["T"], "$T$ [K]"), ("Velocidad local", table["u"], "$u$ [m s$^{-1}$]"), ("Hidrogeno", table["Y"][:, names.index("H2"), :], "$Y_{H_2}$"), ("Oxigeno", table["Y"][:, names.index("O2"), :], "$Y_{O_2}$")]
    maps = [("Densidad", table["rho"], "$\\rho$ [kg m$^{-3}$]", "viridis"), ("Conductividad termica", table["conductivity"], "$\\lambda$ [W m$^{-1}$ K$^{-1}$]", "viridis"), ("Liberacion de calor", table["qdot"]/1e9, "$\\dot q$ [GW m$^{-3}$]", "magma"), ("Fuente de progreso", table["omega_c"], "$\\dot\\omega_c$ [kg m$^{-3}$ s$^{-1}$]", "magma")]
    fig, axes = plt.subplots(4, 2, figsize=(8.7, 10.6), layout="constrained"); cmap = plt.get_cmap("turbo")
    for i, ((title, values, unit), (mtitle, field, munit, palette)) in enumerate(zip(profiles, maps)):
        ax, mx = axes[i]
        for k, values_k in enumerate(values): ax.plot(c, values_k, color=cmap(zstar[k]), lw=.85)
        ax.set(xlim=(0,1), xlabel="Progreso, $c$", ylabel=unit, title=title)
        mesh = mx.contourf(zstar, c, field.T, levels=64, cmap=palette)
        mx.set(xlim=(0,1), ylim=(0,1), xlabel="Composicion, $Z^\\star$", ylabel="Progreso, $c$", title=mtitle)
        cb = fig.colorbar(mesh, ax=mx, fraction=.046, pad=.03); cb.set_label(munit, fontsize=8); cb.locator=MaxNLocator(5); cb.update_ticks()
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1), cmap=cmap), ax=list(axes[:,0]), location="bottom", fraction=.035, pad=.02, aspect=35).set_label("Color de perfiles: $Z^\\star$")
    save(fig, out, "02_mapa"); plt.close(fig)

    # 04: adaptive indicator histories.
    fig, axes = plt.subplots(2, 2, figsize=(9.3,6), sharex=True, sharey=True, layout="constrained")
    for ax, (tag, title) in zip(axes.flat, LABELS.items()):
        for solver, label, color, marker in (("kflame", "KFLAME", "#246091", "o"), ("cantera", "Cantera", "#bd682e", "s")):
            row = next(r for r in rows if r["transport"] == tag and r["solver"] == solver)
            rounds = json.loads((campaign / row["attempt"] / "rounds.json").read_text())
            ax.semilogy([r["n_flames"] for r in rounds], [100*r["max_defect"] for r in rounds], marker=marker, color=color, label=label, markerfacecolor="none")
        ax.axhline(1, color=".4", ls=":"); ax.set_title(title)
    for ax in axes[1]: ax.set_xlabel("Llamas resueltas, $N_f$")
    for ax in axes[:,0]: ax.set_ylabel("Indicador mÃ¡ximo [%]")
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    save(fig, out, "04_refinamiento"); plt.close(fig)

    shutil.copy2(report / "01_coste_comparado.pdf", out / "05_coste_comparado.pdf")
    shutil.copy2(report / "01_coste_comparado.png", out / "05_coste_comparado.png")
    shutil.copy2(report / "table_cost.tex", out / "table_cost.tex")
    shutil.copy2(report / "table_concordance.csv", out / "table_concordance.csv")
    (out / "README.md").write_text("# Figuras H2/h2o2\n\n01: familias; 02: manifold; 04: refinamiento; 05: coste.\nFig. 03 requiere validaciÃ³n independiente con llamas H2 retenidas y no se genera aquÃ­.\n", encoding="utf-8")
    print(f"H2 figure package written to {out}")


if __name__ == "__main__":
    main()
