"""Build, save, and plot a small methane/air FGM table.

Run from the repository root after installing the optional plotting extras:

    python -m pip install -e . matplotlib
    python examples/example_fgm.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

import kflame


# %% FGM CASE CONFIGURATION
# These five flamelets start the adaptive family.  The solver adds and resolves
# logarithmic bridge flamelets until the interpolation-defect target is met.
FGM_CASE = {
    "mechanism": "gri30.yaml",  # Or "h2o2.yaml" with fuel="H2".
    "fuel": "CH4",
    "oxidizer": "O2:1, N2:3.76",
    "phis": (0.7, 0.9, 1.0, 1.1, 1.4),
    "temperature": 300.0,  # Unburned-gas temperature [K].
    "pressure": 101325.0,  # Pressure [Pa].
    "width": 0.03,  # Initial domain width [m].
    "transport": "mixture-averaged",  # Or "multicomponent".
    "soret": False,  # Thermal diffusion; requires multicomponent transport.
    # C4: the validated CH4 progress variable used in the thesis workflow.
    "progress_species": "CO2:1.0,H2O:1.0,CO:1.0,H2:0.5",
    "progress_points": 241,  # Number of adaptive c points.
    "target_defect": 0.01,  # 1% leave-one-out interpolation defect.
    "max_bridges_per_round": 10,
    # The FGM generator always prints one summary per flamelet.  Keep the
    # nonlinear solver trace disabled so the console remains readable.
    "verbose": False,
    "output": None,  # Or Path("runs/my_fgm_table").
}

# The fields below reproduce the CH4/air manifold layout used in the thesis.
PLOT_SPECIES = ("CO2", "CO")


# %% BUILD, LOAD, AND SAVE NUMERICAL DATA
print("[1/3] Building the FGM table...", flush=True)
started = time.perf_counter()
output_dir = Path(kflame.generate_fgm(**FGM_CASE, export=False, plots=False))
print(f"[1/3] FGM table accepted in {time.perf_counter() - started:.1f} s.", flush=True)

print("[2/3] Loading arrays and writing the numerical matrix...", flush=True)
with np.load(output_dir / "fgm_table.npz", allow_pickle=False) as data:
    phi = data["phi_grid"]
    burning_velocity = data["Su"]
    mixture_fraction = data["Z_grid"]
    progress = data["c_grid"]
    temperature = data["T"]
    velocity = data["u"]
    density = data["rho"]
    conductivity = data["conductivity"]
    heat_release = data["qdot"]
    progress_source = data["omega_c"]
    species_names = data["species_names"].astype(str).tolist()
    species_indices = [species_names.index(name) for name in PLOT_SPECIES]
    mass_fractions = data["Y"][:, species_indices, :]

# Z_star labels the flamelet family independently of the chosen phi values.
z_span = mixture_fraction[-1] - mixture_fraction[0]
if z_span <= 0.0:
    raise RuntimeError("The FGM mixture-fraction coordinate must be increasing.")
mixture_fraction_star = (mixture_fraction - mixture_fraction[0]) / z_span

# One matrix row corresponds to one FGM state (Z, Z_star, c).
Z_matrix, c_matrix = np.meshgrid(mixture_fraction, progress, indexing="ij")
Z_star_matrix, _ = np.meshgrid(mixture_fraction_star, progress, indexing="ij")
fgm_matrix = np.column_stack(
    (
        Z_matrix.ravel(),
        Z_star_matrix.ravel(),
        c_matrix.ravel(),
        temperature.ravel(),
        density.ravel(),
        heat_release.ravel(),
        *(mass_fractions[:, index, :].ravel() for index in range(len(PLOT_SPECIES))),
    )
)
matrix_header = ",".join(
    ("Z", "Z_star", "c", "T_K", "rho_kg_m3", "qdot_W_m3", *PLOT_SPECIES)
)
np.savetxt(
    output_dir / "fgm_matrix.csv",
    fgm_matrix,
    delimiter=",",
    header=matrix_header,
    comments="",
)


# %% PLOTS
# Modify this section freely: it uses only the arrays loaded above.
print("[3/3] Creating plots...", flush=True)
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.65,
    }
):
    figure = plt.figure(figsize=(8.27, 11.69))
    grid = figure.add_gridspec(5, 2, width_ratios=(1.0, 1.12), hspace=0.62, wspace=0.58)
    profile_axes = tuple(figure.add_subplot(grid[row, 0]) for row in range(4))
    flamelet_axis = figure.add_subplot(grid[4, 0])
    map_axes = (
        figure.add_subplot(grid[:2, 1]),
        figure.add_subplot(grid[2:4, 1]),
    )
    profile_color_axis = figure.add_subplot(grid[4, 1])
    profile_colormap = plt.get_cmap("turbo")

    profile_fields = (
        ("Temperature", temperature, "T [K]"),
        ("Local velocity", velocity, "u [m s$^{-1}$]"),
        ("Carbon dioxide", mass_fractions[:, 0, :], r"$Y_{CO_2}$ [kg kg$^{-1}$]"),
        ("Carbon monoxide", mass_fractions[:, 1, :], r"$Y_{CO}$ [kg kg$^{-1}$]"),
    )
    map_fields = (
        ("Heat release", heat_release / 1.0e9, r"$\dot q$ [GW m$^{-3}$]", "magma"),
        ("Progress source", progress_source, r"$\dot\omega_c$ [kg m$^{-3}$ s$^{-1}$]", "magma"),
    )

    for profile_axis, (profile_title, profile_values, profile_label) in zip(
        profile_axes, profile_fields, strict=True
    ):
        for index, values in enumerate(profile_values):
            profile_axis.plot(progress, values, color=profile_colormap(mixture_fraction_star[index]), linewidth=0.75)
        profile_axis.set(xlim=(0.0, 1.0), xlabel="c", ylabel=profile_label, title=profile_title)

    flamelet_axis.plot(phi, burning_velocity, "o-", color="tab:blue", markersize=3, linewidth=0.9)
    flamelet_axis.set(
        xlabel="Equivalence ratio, phi",
        ylabel="Burning velocity [m s$^{-1}$]",
        title="Flamelets used to build the table",
    )

    profile_colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=profile_colormap),
        cax=profile_color_axis,
        orientation="horizontal",
    )
    profile_colorbar.set_label(r"Profile color: $Z^\star$", fontsize=8)
    profile_colorbar.ax.tick_params(labelsize=7)

    for map_axis, (map_title, map_values, map_label, map_colormap) in zip(
        map_axes, map_fields, strict=True
    ):
        contour = map_axis.contourf(
            mixture_fraction_star,
            progress,
            map_values.T,
            levels=np.linspace(float(map_values.min()), float(map_values.max()), 65),
            cmap=map_colormap,
            antialiased=False,
        )
        contour.set_edgecolor("face")
        map_axis.plot(mixture_fraction_star, np.ones_like(mixture_fraction_star), "|", markersize=3, color="black", clip_on=False)
        map_axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel=r"$Z^\star$", ylabel="c", title=map_title)
        colorbar = figure.colorbar(contour, ax=map_axis, fraction=0.046, pad=0.03)
        colorbar.locator = MaxNLocator(nbins=4)
        colorbar.update_ticks()
        colorbar.set_label(map_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    for axis in (*profile_axes, flamelet_axis, *map_axes):
        axis.tick_params(direction="in", top=True, right=True, length=3, width=0.65)
        axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    figure.subplots_adjust(left=0.12, right=0.93, bottom=0.07, top=0.97)
    figure.savefig(output_dir / "custom_fgm_plots.pdf")

plt.close(figure)

print(f"Numerical matrix: {output_dir / 'fgm_matrix.csv'}")
print(f"Custom plots: {output_dir / 'custom_fgm_plots.pdf'}")
