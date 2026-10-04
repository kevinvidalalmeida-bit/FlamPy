"""Regenerate the six FGM figures from verified, saved numerical snapshots.

Run from any directory: python path/to/docs/assets/generate_fgm_figures.py
Only NumPy and Matplotlib are required; no solver runs or private paths.
"""
from pathlib import Path
import argparse
import hashlib
import json
import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator
import numpy as np

ASSETS = Path(__file__).resolve().parent
DATA = ASSETS / "data"
STYLE = {
    "font.family": "DejaVu Serif", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.15, "lines.markersize": 3,
    "pdf.fonttype": 42, "mathtext.fontset": "stix",
    "savefig.facecolor": "white",
}
SPECIES_TEX = {"CO2": r"\mathrm{CO_2}", "CO": r"\mathrm{CO}",
               "H2O": r"\mathrm{H_2O}", "OH": r"\mathrm{OH}"}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def save(fig, folder, name, pad=.035):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Fixed PDF dates permit byte-for-byte checks within one software version.
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight", pad_inches=pad,
                dpi=220, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(folder / (name + ".png"), bbox_inches="tight", pad_inches=pad, dpi=220)
    plt.close(fig)

def manifold(fuel, table, folder=ASSETS, name=None):
    """Plot the unscaled inlet Bilger coordinate horizontally and progress vertically."""
    z, c = table["Z_grid"], table["c_grid"]
    species = ("CO2", "CO") if fuel == "CH4" else ("H2O", "OH")
    names = table["species_names"].tolist()
    fields = [
        (table["T"], r"$T$ [K]", [600, 1200, 1800]),
        (table["Y"][:, names.index(species[0]), :],
         rf"$Y_{{{SPECIES_TEX[species[0]]}}}$ [kg kg$^{{-1}}$]", None),
        (table["Y"][:, names.index(species[1]), :] * (1e3 if fuel == "H2" else 1),
         rf"$Y_{{{SPECIES_TEX[species[1]]}}}$ [" +
         (r"$10^{-3}$ kg kg$^{-1}$]" if fuel == "H2" else r"kg kg$^{-1}$]"), None),
        (table["omega_c"] / 1e3, r"$\dot\omega_c$ [$10^3$ kg m$^{-3}$ s$^{-1}$]", None),
    ]
    assert np.all(np.diff(z) > 0) and np.all(np.diff(c) > 0)
    assert np.all(table["final_accepted"])
    with plt.rc_context(dict(STYLE, **{"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9})):
        fig, axes = plt.subplots(2, 2, figsize=(6.05, 3.3), layout="constrained")
        zst = float(np.interp(1., table["phi_grid"], z))
        for i, (ax, (field, label, contours)) in enumerate(zip(axes.flat, fields, strict=True)):
            assert np.isfinite(field).all()
            limits = (300., 2500.) if i == 0 else (float(field.min()), float(field.max()))
            fill = ax.contourf(z, c, field.T, levels=np.linspace(*limits, 49),
                               cmap="viridis", antialiased=False, extend="neither")
            fill.set_rasterized(True)  # Text and isolines remain vector objects.
            if contours is None:
                contours = np.linspace(max(0., limits[0]), limits[1], 5)[1:-1]
            contours = [v for v in contours if field.min() < v < field.max()]
            lines = ax.contour(z, c, field.T, levels=contours, colors="white", linewidths=.45, alpha=.9)
            positions = []
            for segments in lines.allsegs:
                segment = max(segments, key=len)
                px, py = (segment[:, 0]-z[0])/(z[-1]-z[0]), segment[:, 1]
                candidates = np.flatnonzero((px > .10) & (px < .90) & (py > .10) & (py < .92))
                if not len(candidates): candidates = np.arange(len(segment))
                chosen = candidates[np.argmin((px[candidates]-.60)**2 + (py[candidates]-.65)**2)]
                positions.append(tuple(segment[chosen]))
            labels = ax.clabel(lines, inline=True, fontsize=7.2,
                fmt=(lambda v: f"{v:.0f}") if i == 0 else (lambda v: f"{v:.2g}"),
                inline_spacing=2, manual=positions)
            for label_artist in labels:
                label_artist.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground=".25")])
            ax.axvline(zst, color="white", ls="--", lw=.65)
            ax.plot(z, np.ones_like(z), "|", markersize=2.5, mew=.45, color=".25", clip_on=False)
            ax.set(xlim=(float(z[0]), float(z[-1])), ylim=(0, 1), xlabel=r"$Z_{\mathrm{in}}$",
                   ylabel=r"Progreso, $c$", title=f"({chr(97+i)}) " + label, yticks=[0, .5, 1])
            ticks = ([float(z[0]), *[v for v in (.05, .09) if z[0] < v < z[-1]], float(z[-1])]
                     if fuel == "H2" else [float(z[0]), zst, float(z[-1])])
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f" if fuel == "H2" else "%.4f"))
            cb = fig.colorbar(fill, ax=ax, pad=.025, fraction=.05)
            cb.ax.tick_params(labelsize=8); cb.locator = MaxNLocator(4); cb.update_ticks()
            if i == 0: cb.set_ticks([300, 1000, 1800, 2500])
            if i == 3:
                minimum = float(field.min())
                cb.set_ticks([0. if abs(minimum) < 1e-10 else minimum,
                              float(field.max()/2), float(field.max())])
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        save(fig, folder, name or f"fgm-{fuel.lower()}-map", pad=.04)

def fidelity(fuel, data, folder=ASSETS, name=None):
    """Compare interpolated fields with 12 independent held-out flames."""
    c, phi, z, error = [data[k] for k in ("c", "phi", "Z", "error_percent")]
    species = ("CO2", "CO") if fuel == "CH4" else ("H2O", "OH")
    labels = [r"$T$ [K]", *[rf"$Y_{{{SPECIES_TEX[s]}}}$ [kg kg$^{{-1}}$]" for s in species],
              r"$\dot\omega_c$ [kg m$^{-3}$ s$^{-1}$]"]
    boxlabels = [r"$T$", *[rf"$Y_{{{SPECIES_TEX[s]}}}$" for s in species], r"$\dot\omega_c$"]
    chosen = int(np.argmin(abs(np.log(phi))))
    assert error.shape == (12, 1001, 4)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(6.05, 4.8), layout="constrained")
        for i, ax in enumerate(axes[:2].flat):
            ax.plot(c, data["truth"][chosen, :, i], color="#155C8A", label="Llama retenida")
            ax.plot(c, data["prediction"][chosen, :, i], color="#D97706", ls="--", label="FGM")
            ax.set(xlabel=r"Progreso, $c$", ylabel=labels[i], xlim=(0, 1), xticks=[0, .5, 1])
            ax.set_title(f"({chr(97+i)})"); ax.yaxis.set_major_locator(MaxNLocator(3))
        axes[0, 0].legend(frameon=False)
        axes[0, 1].text(.05, .90, rf"$\phi={phi[chosen]:.4f}$", transform=axes[0, 1].transAxes)
        tick_option = "tick_labels" if "tick_labels" in inspect.signature(axes[2, 0].boxplot).parameters else "labels"
        axes[2, 0].boxplot([error[:, :, i].ravel() for i in range(4)],
                          **{tick_option: boxlabels}, showfliers=False, whis=(5, 95))
        axes[2, 0].set(ylabel="Error normalizado [%]", title="(e) Doce llamas retenidas",
                      ylim=(0, 1.15*np.quantile(error, .95, axis=(0, 1)).max()))
        axes[2, 0].yaxis.set_major_locator(MaxNLocator(3))
        cc, zz = np.meshgrid(c, z)
        cloud = axes[2, 1].scatter(zz.ravel(), cc.ravel(), c=error.max(axis=2).ravel(),
            s=2, cmap="viridis", vmin=0, vmax=float(error.max()), rasterized=True)
        bounds = data["Z_bounds"]
        axes[2, 1].set(xlim=bounds, ylim=(0, 1), xlabel=r"$Z_{\mathrm{in}}$",
            ylabel=r"Progreso, $c$", title="(f) Localización del error", yticks=[0, .5, 1],
            xticks=[.015, .07, .128] if fuel == "H2" else np.linspace(*bounds, 3))
        if fuel == "CH4": axes[2, 1].xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        cb = fig.colorbar(cloud, ax=axes[2, 1], pad=.025, fraction=.055)
        cb.set_label("Mayor error [%]"); cb.locator=MaxNLocator(3); cb.update_ticks()
        save(fig, folder, name or f"fgm-{fuel.lower()}-fidelity")

def families(fuel, rows, folder=ASSETS, name=None):
    """Plot recorded time including properties, retaining the actual solve order."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(6.05, 3.3), sharex=True, sharey=True)
        fig.subplots_adjust(left=.11, right=.98, top=.86, bottom=.20, hspace=.60, wspace=.35)
        values=[]
        for i, tag in enumerate(("P", "PS", "M", "MS")):
            for j, solver in enumerate(("kflame", "cantera")):
                row = next(r for r in rows if r["transport"] == tag and r["solver"] == solver)
                trace = row["trace"]
                times = np.asarray([t["measured_s"] for t in trace]); values.extend(times)
                kinds = ["cold" if t["predictor_kind"] == "cold" else "secant"
                         if "secant" in t["predictor_kind"] else "copy" for t in trace]
                ax = axes[i, j]; x=np.arange(1, len(trace)+1); color=("#155C8A", "#D97706")[j]
                ax.plot(x, times, color=color, lw=.9)
                for kind, marker in (("cold", "D"), ("copy", "s"), ("secant", "o")):
                    mask = np.asarray([v == kind for v in kinds])
                    ax.scatter(x[mask], times[mask], marker=marker, s=12,
                        facecolors=color if kind == "cold" else "white", edgecolors=color, linewidths=.6)
                title = {"P": "P", "PS": "P+S", "M": "M", "MS": "M+S"}[tag]
                ax.set_yscale("log"); ax.set_title(f"{title} / {len(trace)} llamas", pad=2)
                ax.set_xticks([1, 35, 71] if fuel == "H2" else [1, 20, 40])
                ax.tick_params(labelleft=True)
        for ax in axes.flat: ax.set_ylim(min(values)/1.3, max(values)*1.3)
        for ax in axes[-1]: ax.set_xlabel("Orden de resolución")
        fig.text(.30, .975, "FlamPy", ha="center"); fig.text(.78, .975, "Cantera", ha="center")
        fig.text(.015, .53, "Tiempo [s]", rotation=90, va="center")
        handles = [Line2D([], [], color=".3", ls="", marker=m, mfc=".3" if k == "cold" else "white", label=l)
            for k, m, l in (("cold", "D", "Arranque físico"), ("copy", "s", "Perfil previo"),
                            ("secant", "o", "Predictor secante"))]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.015), ncol=3, frameon=False)
        for t in fig.findobj(Text): t.set_fontfamily("DejaVu Serif"); t.set_fontsize(8.5)
        for ax in fig.axes:
            ax.tick_params(axis="both", labelsize=8)
            ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        save(fig, folder, name or f"fgm-{fuel.lower()}-family")

def generate_all(output=ASSETS):
    manifest = json.loads((DATA/"provenance.json").read_text(encoding="utf8"))
    for snapshot in manifest["snapshots"]:
        path=DATA/snapshot["snapshot"]
        if digest(path) != snapshot["sha256"]: raise ValueError(f"Snapshot checksum mismatch: {path.name}")
    for fuel in ("CH4", "H2"):
        manifold(fuel, load(DATA/f"fgm-{fuel.lower()}.npz"), output)
        fidelity(fuel, load(DATA/f"fgm-{fuel.lower()}-fidelity.npz"), output)
        rows=json.loads((DATA/f"fgm-{fuel.lower()}-family.json").read_text(encoding="utf8"))
        families(fuel, rows, output)

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ASSETS, help="Destination for PNG/PDF figures")
    args=cli.parse_args()
    generate_all(args.output.resolve())
    print(f"Six FGM figures regenerated in {args.output.resolve()}")

if __name__ == "__main__": main()
