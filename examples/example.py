"""Solve, save, and plot one neutral methane/air flame.

Run from the repository root after installing the optional plotting extras:

    pip install ".[plot]"
    python examples/example.py

The workflow is deliberately explicit: configure a case, solve it, load the
saved numerical arrays, and make the plots needed for the study.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kflame


# Change this dictionary to define another case. Keep fuel and mechanism
# compatible: for example, use fuel="H2" with mechanism="h2o2.yaml".
CASE = {
    "mechanism": "gri30.yaml",  # Or "h2o2.yaml".
    "fuel": "CH4",  # Or "H2" when using h2o2.yaml.
    "oxidizer": "O2:1, N2:3.76",
    "phi": 1.0,
    "temperature": 300.0,  # Unburned-gas temperature [K].
    "pressure": 101325.0,  # Pressure [Pa].
    "width": 0.03,  # Initial domain width [m].
    "initial_points": 8,
    "transport": "mixture-averaged",  # Or "multicomponent".
    "soret": False,  # Thermal diffusion; requires multicomponent transport.
    "ratio": 2.5,
    "slope": 0.04,
    "curve": 0.08,
    "prune": 0.01,
    "output": None,  # Or Path("runs/my_neutral_flame").
}

# These species are saved in profiles.csv and used in the custom plot below.
PLOT_SPECIES = ("CH4", "O2", "CO2", "H2O", "OH")


def load_solution(output_dir: Path) -> dict[str, np.ndarray | list[str]]:
    """Load the named arrays produced by KFLAME and create a row-wise matrix."""
    with np.load(output_dir / "flame.npz", allow_pickle=False) as data:
        species_names = data["species_names"].astype(str).tolist()
        species_indices = [species_names.index(name) for name in PLOT_SPECIES]
        mass_fractions = data["Y"][species_indices]
        solution_matrix = np.column_stack(
            (
                data["z"],
                data["T"],
                data["u"],
                data["qdot"],
                mass_fractions.T,
            )
        )

        return {
            "z": data["z"],
            "temperature": data["T"],
            "velocity": data["u"],
            "heat_release": data["qdot"],
            "mass_fractions": mass_fractions,
            "matrix": solution_matrix,
        }


def save_matrix(output_dir: Path, solution: dict[str, np.ndarray | list[str]]) -> None:
    """Write a conventional numerical matrix: one row for every grid point."""
    header = ",".join(("z_m", "T_K", "u_m_s", "qdot_W_m3", *PLOT_SPECIES))
    np.savetxt(
        output_dir / "solution_matrix.csv",
        solution["matrix"],
        delimiter=",",
        header=header,
        comments="",
    )


def plot_solution(output_dir: Path, solution: dict[str, np.ndarray | list[str]]) -> None:
    """Create an editable set of plots from the saved arrays."""
    z_mm = 1.0e3 * solution["z"]
    mass_fractions = solution["mass_fractions"]

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    axes[0, 0].plot(z_mm, solution["temperature"], color="tab:red")
    axes[0, 0].set(xlabel="Distance [mm]", ylabel="Temperature [K]")

    axes[0, 1].plot(z_mm, solution["velocity"], color="tab:blue")
    axes[0, 1].set(xlabel="Distance [mm]", ylabel="Velocity [m s$^{-1}$]")

    axes[1, 0].plot(z_mm, solution["heat_release"], color="tab:orange")
    axes[1, 0].set(xlabel="Distance [mm]", ylabel="Heat release [W m$^{-3}$]")

    for name, values in zip(PLOT_SPECIES, mass_fractions, strict=True):
        axes[1, 1].plot(z_mm, values, label=name)
    axes[1, 1].set(xlabel="Distance [mm]", ylabel="Mass fraction")
    axes[1, 1].legend(ncol=2, fontsize="small")

    figure.savefig(output_dir / "custom_flame_plots.pdf", bbox_inches="tight")
    # Uncomment in an interactive session to display the figure immediately.
    # plt.show()
    plt.close(figure)


if __name__ == "__main__":
    result = kflame.solve_flame(**CASE, species=PLOT_SPECIES, plots=False)
    output_dir = Path(result["output"])
    solution = load_solution(output_dir)
    save_matrix(output_dir, solution)
    plot_solution(output_dir, solution)

    print(f"Burning velocity: {result['Su']:.6g} m/s")
    print(f"Numerical matrix: {output_dir / 'solution_matrix.csv'}")
    print(f"Custom plots: {output_dir / 'custom_flame_plots.pdf'}")
