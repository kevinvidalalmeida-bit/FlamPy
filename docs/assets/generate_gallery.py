"""Reproduce the README figures from compact, saved numerical snapshots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np


ASSETS = Path(__file__).resolve().parent
DATA = ASSETS / "data"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.titlesize": 13,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8893a1",
    "axes.labelcolor": "#223347",
    "text.color": "#223347",
    "xtick.color": "#405269",
    "ytick.color": "#405269",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def tidy(axis):
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.grid(alpha=0.15)
    axis.set_axisbelow(True)


def flame_figure():
    with np.load(DATA / "flame-ch4.npz", allow_pickle=False) as data:
        z, temperature, velocity = data["z"], data["T"], data["u"]
        heat, names, mass = data["qdot"], data["species_names"], data["Y"]
    midpoint = np.interp(0.5 * (temperature[0] + temperature[-1]), temperature, z)
    x = 1e3 * (z - midpoint)
    # Show the resolved front. All curves use the original saved nodes.
    active = np.flatnonzero((temperature >= temperature[0] + 0.05 * np.ptp(temperature)) &
                            (temperature <= temperature[-1] - 0.05 * np.ptp(temperature)))
    span = max(0.5, x[active[-1]] - x[active[0]])
    limits = (x[active[0]] - 0.15 * span, x[active[-1]] + 0.15 * span)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), layout="constrained")
    fig.suptitle("FlamPy · llama de CH₄–aire\nφ = 1 · entrada a 300 K y 1 atm", fontsize=16)
    axes[0, 0].plot(x, temperature, color="#df6b28", linewidth=2)
    axes[0, 0].set(title="Temperatura", ylabel="T [K]")
    axes[0, 1].plot(x, velocity, color="#206ba4", linewidth=2)
    axes[0, 1].set(title="Velocidad local", ylabel="u [m/s]")
    axes[1, 0].plot(x, heat / 1e9, color="#bc4048", linewidth=2)
    axes[1, 0].set(title="Liberación de calor", ylabel="q̇ [GW/m³]")
    colors = ("#206ba4", "#328c89", "#df6b28", "#bc4048")
    for name, values, color in zip(names, mass, colors, strict=True):
        axes[1, 1].plot(x, values, label=str(name), color=color, linewidth=1.8)
    axes[1, 1].set(title="Composición", ylabel="Fracción másica")
    axes[1, 1].legend(ncol=2, frameon=False)
    for axis in axes.flat:
        axis.set_xlim(*limits)
        axis.set_xlabel("Distancia al centro térmico [mm]")
        tidy(axis)
    fig.savefig(ASSETS / "flame-ch4.png", dpi=180)
    plt.close(fig)


def fgm_figure():
    tables = []
    for fuel in ("ch4", "h2"):
        with np.load(DATA / f"fgm-{fuel}.npz", allow_pickle=False) as data:
            tables.append({k: data[k] for k in data.files})
    vmin = min(float(t["T"].min()) for t in tables)
    vmax = np.ceil(max(float(t["T"].max()) for t in tables) / 100) * 100
    norm = Normalize(vmin, vmax)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    fig.suptitle("FlamPy · temperatura en tablas FGM\nEntrada a 300 K y 1 atm · φ entre 0,7 y 1,4", fontsize=16)
    for axis, table, fuel in zip(axes, tables, ("CH₄", "H₂"), strict=True):
        z = table["Z_grid"]
        z_star = (z - z[0]) / np.ptp(z)
        image = axis.pcolormesh(z_star, table["c_grid"], table["T"].T,
                                shading="auto", cmap="magma", norm=norm, rasterized=True)
        axis.set(title=f"{fuel}–aire · {len(z)} flamelets", xlabel="Composición normalizada, Z⋆",
                 ylabel="Progreso, c", xlim=(0, 1), ylim=(0, 1))
        axis.set_xticks((0, 0.5, 1))
        axis.set_yticks((0, 0.5, 1))
    colorbar = fig.colorbar(image, ax=list(axes), fraction=0.035, pad=0.025)
    colorbar.set_label("Temperatura [K]")
    colorbar.locator = MaxNLocator(4)
    colorbar.update_ticks()
    fig.savefig(ASSETS / "fgm-temperature.png", dpi=180)
    plt.close(fig)


def main():
    with plt.rc_context(STYLE):
        flame_figure()
        fgm_figure()
    print(f"Gallery regenerated in {ASSETS}")


if __name__ == "__main__":
    main()
