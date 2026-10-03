"""Build an FGM family or plot a saved table with the thesis contour layout.

    python examples/example_fgm.py
    python examples/example_fgm.py --table docs/assets/data/fgm-h2.npz --fuel H2

The second command reproduces the published H2 map without solving flames.
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np

import kflame
from kflame.chemistry.initialization import NativeMixture
from kflame.chemistry.mechanism import load_mechanism

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "docs/assets"))
from generate_fgm_figures import manifold


# These five flamelets start an adaptive family. Its final size depends on
# the interpolation defect. The published H2 family is a separate fixed grid.
FGM_CASE = {
    "mechanism": "gri30.yaml",
    "fuel": "CH4",
    "oxidizer": "O2:1, N2:3.76",
    "phis": (0.7, 0.9, 1.0, 1.1, 1.4),
    "temperature": 300.0,  # Unburned gas [K].
    "pressure": 101325.0,  # [Pa].
    "width": 0.03,  # Initial domain [m].
    "transport": "mixture-averaged",
    "soret": False,
    "progress_species": "CO2:1.0,H2O:1.0,CO:1.0,H2:0.5",
    "progress_points": 241,
    "target_defect": 0.01,  # 1% leave-one-out interpolation defect.
    "max_bridges_per_round": 10,
    "verbose": False,
    "output": None,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, help="Plot a saved NPZ; skip solving.")
    parser.add_argument("--fuel", choices=("CH4", "H2"), default=FGM_CASE["fuel"])
    parser.add_argument("--output", type=Path, help="New output directory.")
    args = parser.parse_args()
    case = dict(FGM_CASE, fuel=args.fuel)
    if args.fuel == "H2":
        case.update(mechanism="h2o2.yaml", progress_species="H2O:1,HO2:10,OH:-1")

    if args.table:
        table_path = args.table.resolve()
        output_dir = args.output or REPO_ROOT / "runs" / f"fgm_plot_{time.time_ns()}"
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if args.output:
            case["output"] = args.output
        print("Building the FGM table...", flush=True)
        started = time.perf_counter()
        output_dir = Path(kflame.generate_fgm(**case, export=False, plots=False))
        print(f"FGM family built in {time.perf_counter() - started:.1f} s.")
        table_path = output_dir / "fgm_table.npz"

    with np.load(table_path, allow_pickle=False) as saved:
        table = {key: saved[key] for key in saved.files}
    if not np.all(table["final_accepted"]):
        raise RuntimeError("The table contains flamelets that were not accepted.")

    # Use Bilger Z for the stated fuel/air mole streams. Older full tables used
    # mass-stream labels: change only the display coordinate, never saved data.
    mixture = NativeMixture(load_mechanism(case["mechanism"]))
    physical, historical = [], []
    for phi in table["phi_grid"]:
        mixture.set_equivalence_ratio(float(phi), args.fuel, case["oxidizer"])
        physical.append(mixture.mixture_fraction(args.fuel, case["oxidizer"], basis="mole"))
        historical.append(mixture.mixture_fraction(args.fuel, case["oxidizer"], basis="mass"))
    if np.allclose(table["Z_grid"], physical, rtol=0.0, atol=1e-13):
        z = table["Z_grid"]
    elif np.allclose(table["Z_grid"], historical, rtol=0.0, atol=1e-13):
        z = np.asarray(physical)
    else:
        raise ValueError("Z labels do not match the configured fuel and oxidizer.")
    displayed = dict(table, Z_grid=z)

    species = ("CO2", "CO") if args.fuel == "CH4" else ("H2O", "OH")
    names = table["species_names"].astype(str).tolist()
    zz, cc = np.meshgrid(z, table["c_grid"], indexing="ij")
    columns = {"Z_in": zz, "c": cc, "T_K": table["T"],
               "omega_c_kg_m3_s": table["omega_c"]}
    for name in species:
        columns[f"Y_{name}"] = table["Y"][:, names.index(name), :]
    for field, label in (("rho", "rho_kg_m3"), ("qdot", "qdot_W_m3")):
        if field in table:
            columns[label] = table[field]
    np.savetxt(output_dir / "fgm_matrix.csv",
               np.column_stack([array.ravel() for array in columns.values()]),
               delimiter=",", header=",".join(columns), comments="")

    manifold(args.fuel, displayed, output_dir, "custom_fgm_plots")
    print(f"Flamelets: {len(z)}; progress points: {len(table['c_grid'])}")
    print(f"Numerical matrix: {output_dir / 'fgm_matrix.csv'}")
    print(f"Plots: {output_dir / 'custom_fgm_plots.pdf'} and .png")


if __name__ == "__main__":
    main()
