"""Build, save, and plot a small methane/air FGM table.

Run from the repository root after installing the optional plotting extras:

    python -m pip install -e . matplotlib
    python examples/example_fgm.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import kflame


# %% FGM CASE CONFIGURATION
# This intentionally small set of flamelets is suitable for an example. Add
# more phi values for a denser table in a production calculation.
FGM_CASE = {
    "mechanism": "gri30.yaml",  # Or "h2o2.yaml" with fuel="H2".
    "fuel": "CH4",
    "oxidizer": "O2:1, N2:3.76",
    "phis": (0.8, 1.0, 1.2),
    "temperature": 300.0,  # Unburned-gas temperature [K].
    "pressure": 101325.0,  # Pressure [Pa].
    "width": 0.03,  # Initial domain width [m].
    "transport": "mixture-averaged",  # Or "multicomponent".
    "soret": False,  # Thermal diffusion; requires multicomponent transport.
    "progress_variable": "CO2 + H2O",
    "progress_points": 241,
    "verbose": True,  # Show flamelet progress in the console.
    "output": None,  # Or Path("runs/my_fgm_table").
}

PLOT_SPECIES = ("CO2", "H2O")


# %% BUILD, LOAD, AND SAVE NUMERICAL DATA
print("[1/3] Building the FGM table...", flush=True)
started = time.perf_counter()
output_dir = Path(kflame.generate_fgm(**FGM_CASE, export=True, plots=False))
print(f"[1/3] FGM table accepted in {time.perf_counter() - started:.1f} s.", flush=True)

print("[2/3] Loading arrays and writing the numerical matrix...", flush=True)
with np.load(output_dir / "fgm_table.npz", allow_pickle=False) as data:
    phi = data["phi_grid"]
    burning_velocity = data["Su"]
    mixture_fraction = data["Z_grid"]
    progress = data["c_grid"]
    temperature = data["T"]
    density = data["rho"]
    heat_release = data["qdot"]
    species_names = data["species_names"].astype(str).tolist()
    species_indices = [species_names.index(name) for name in PLOT_SPECIES]
    mass_fractions = data["Y"][:, species_indices, :]

# One matrix row corresponds to one FGM state (Z, c).
Z_matrix, c_matrix = np.meshgrid(mixture_fraction, progress, indexing="ij")
fgm_matrix = np.column_stack(
    (
        Z_matrix.ravel(),
        c_matrix.ravel(),
        temperature.ravel(),
        density.ravel(),
        heat_release.ravel(),
        *(mass_fractions[:, index, :].ravel() for index in range(len(PLOT_SPECIES))),
    )
)
matrix_header = ",".join(("Z", "c", "T_K", "rho_kg_m3", "qdot_W_m3", *PLOT_SPECIES))
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
figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")

plot_fields = (
    (temperature, "Temperature [K]"),
    (heat_release, "Heat release [W m$^{-3}$]"),
    (mass_fractions[:, 0, :], f"{PLOT_SPECIES[0]} mass fraction"),
)
for axis, (field, title) in zip(axes.flat[:3], plot_fields, strict=True):
    mesh = axis.pcolormesh(progress, mixture_fraction, field, shading="auto")
    axis.set(xlabel="Progress variable, c", ylabel="Mixture fraction, Z", title=title)
    figure.colorbar(mesh, ax=axis)

axes[1, 1].plot(phi, burning_velocity, marker="o")
axes[1, 1].set(
    xlabel="Equivalence ratio, phi",
    ylabel="Burning velocity [m s$^{-1}$]",
    title="Flamelets used to build the table",
)

figure.savefig(output_dir / "custom_fgm_plots.pdf", bbox_inches="tight")
# Uncomment in an interactive session to display the figure immediately.
# plt.show()
plt.close(figure)

print(f"Numerical matrix: {output_dir / 'fgm_matrix.csv'}")
print(f"Custom plots: {output_dir / 'custom_fgm_plots.pdf'}")
