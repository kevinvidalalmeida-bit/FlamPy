"""Build, save, and plot a small methane/air FGM table.

Run from the repository root after installing the optional plotting extras:

    pip install ".[plot]"
    python examples/example_fgm.py

The example keeps the numerical table separate from its presentation: it first
creates an FGM case, then loads fgm_table.npz and makes explicit user-editable
plots from the saved arrays.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kflame


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
    "output": None,  # Or Path("runs/my_fgm_table").
}

PLOT_SPECIES = ("CO2", "H2O")


def load_table(output_dir: Path) -> dict[str, np.ndarray]:
    """Load the FGM arrays and build a row-wise matrix over (Z, c)."""
    with np.load(output_dir / "fgm_table.npz", allow_pickle=False) as data:
        species_names = data["species_names"].astype(str).tolist()
        species_indices = [species_names.index(name) for name in PLOT_SPECIES]
        mixture_fraction, progress = np.meshgrid(
            data["Z_grid"], data["c_grid"], indexing="ij"
        )
        mass_fractions = data["Y"][:, species_indices, :]
        table_matrix = np.column_stack(
            (
                mixture_fraction.ravel(),
                progress.ravel(),
                data["T"].ravel(),
                data["rho"].ravel(),
                data["qdot"].ravel(),
                *(mass_fractions[:, index, :].ravel() for index in range(len(PLOT_SPECIES))),
            )
        )

        return {
            "phi": data["phi_grid"],
            "burning_velocity": data["Su"],
            "Z": data["Z_grid"],
            "c": data["c_grid"],
            "temperature": data["T"],
            "heat_release": data["qdot"],
            "mass_fractions": mass_fractions,
            "matrix": table_matrix,
        }


def save_matrix(output_dir: Path, table: dict[str, np.ndarray]) -> None:
    """Write one row per FGM state, with coordinates and selected fields."""
    header = ",".join(("Z", "c", "T_K", "rho_kg_m3", "qdot_W_m3", *PLOT_SPECIES))
    np.savetxt(
        output_dir / "fgm_matrix.csv",
        table["matrix"],
        delimiter=",",
        header=header,
        comments="",
    )


def plot_table(output_dir: Path, table: dict[str, np.ndarray]) -> None:
    """Create custom figures directly from the saved FGM arrays."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")

    fields = (
        (table["temperature"], "Temperature [K]"),
        (table["heat_release"], "Heat release [W m$^{-3}$]"),
        (table["mass_fractions"][:, 0, :], f"{PLOT_SPECIES[0]} mass fraction"),
    )
    for axis, (field, title) in zip(axes.flat[:3], fields, strict=True):
        mesh = axis.pcolormesh(table["c"], table["Z"], field, shading="auto")
        axis.set(xlabel="Progress variable, c", ylabel="Mixture fraction, Z", title=title)
        figure.colorbar(mesh, ax=axis)

    axes[1, 1].plot(table["phi"], table["burning_velocity"], marker="o")
    axes[1, 1].set(
        xlabel="Equivalence ratio, phi",
        ylabel="Burning velocity [m s$^{-1}$]",
        title="Flamelets used to build the table",
    )

    figure.savefig(output_dir / "custom_fgm_plots.pdf", bbox_inches="tight")
    # Uncomment in an interactive session to display the figure immediately.
    # plt.show()
    plt.close(figure)


if __name__ == "__main__":
    output = kflame.generate_fgm(**FGM_CASE, export=True, plots=False)
    output_dir = Path(output)
    table = load_table(output_dir)
    save_matrix(output_dir, table)
    plot_table(output_dir, table)

    print(f"Numerical matrix: {output_dir / 'fgm_matrix.csv'}")
    print(f"Custom plots: {output_dir / 'custom_fgm_plots.pdf'}")
