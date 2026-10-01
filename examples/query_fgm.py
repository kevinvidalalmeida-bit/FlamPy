"""Query an existing FlamPy table without solving any new flames."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def query_table(
    path: Path,
    *,
    z: float | None = None,
    progress: tuple[float, ...] = (0.25, 0.5, 0.75),
    species: str = "H2O",
) -> dict[str, np.ndarray]:
    """Return temperature and one mass fraction at a fixed inlet Z and several c."""
    with np.load(path, allow_pickle=False) as table:
        z_grid = table["Z_grid"]
        c_grid = table["c_grid"]
        names = table["species_names"].astype(str).tolist()
        temperature = table["T"]
        mass_fractions = np.moveaxis(table["Y"], 1, -1)
    if species not in names:
        raise ValueError(f"Species {species!r} is not present in the table.")
    z = float(0.5 * (z_grid[0] + z_grid[-1]) if z is None else z)
    c = np.asarray(progress, dtype=float)
    if c.ndim != 1 or c.size == 0 or not np.isfinite(c).all() or not np.isfinite(z):
        raise ValueError("Z and c must be finite; provide at least one progress value.")
    points = np.column_stack((np.full(c.size, z), c))
    interpolate_temperature = RegularGridInterpolator(
        (z_grid, c_grid), temperature, method="linear", bounds_error=True
    )
    interpolate_composition = RegularGridInterpolator(
        (z_grid, c_grid), mass_fractions, method="linear", bounds_error=True
    )
    return {
        "Z": points[:, 0],
        "c": c,
        "T": interpolate_temperature(points),
        "Y": interpolate_composition(points)[:, names.index(species)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path, help="Path to fgm_table.npz")
    parser.add_argument("--Z", type=float, default=None, help="Inlet mixture fraction; default: interval midpoint")
    parser.add_argument("--c", nargs="+", type=float, default=[0.25, 0.5, 0.75], help="Normalised progress values")
    parser.add_argument("--species", default="H2O", help="Species mass fraction to report")
    args = parser.parse_args()
    result = query_table(args.table, z=args.Z, progress=tuple(args.c), species=args.species)
    print(f"Z,c,T_K,Y_{args.species}")
    for z, c, temperature, mass_fraction in zip(
        result["Z"], result["c"], result["T"], result["Y"], strict=True
    ):
        print(f"{z:.8g},{c:.8g},{temperature:.8g},{mass_fraction:.8g}")


if __name__ == "__main__":
    main()
