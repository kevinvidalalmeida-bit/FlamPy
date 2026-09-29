"""Generate one neutral CH4/air FGM table and save its data and plots.

Run from the repository root after ``python -m pip install ".[plot]"``:
    python examples/example_fgm.py
"""

from pathlib import Path

import kflame


if __name__ == "__main__":
    output = kflame.generate_fgm(
        mechanism="gri30.yaml",  # "gri30.yaml" or "h2o2.yaml"
        temperature=300.0,  # Fresh-mixture temperature [K]
        pressure=101325.0,  # Constant pressure [Pa]
        fuel="CH4",  # Fuel stream on a mole basis
        oxidizer="O2:1, N2:3.76",  # Oxidizer stream on a mole basis
        phis=(0.8, 1.0, 1.2),  # At least two positive, strictly increasing values
        width=0.03,  # Initial domain width [m]
        initial_points=8,  # Initial grid nodes for every flamelet
        transport="mixture-averaged",  # "mixture-averaged" or "multicomponent"
        soret=False,  # False or True with either transport model
        ratio=2.5,  # Mesh refinement ratio criterion
        slope=0.04,  # Mesh refinement slope criterion
        curve=0.08,  # Mesh refinement curvature criterion
        prune=0.003,  # Mesh coarsening criterion
        max_points=1600,  # Maximum adaptive grid nodes
        max_time=180.0,  # Per-flamelet time budget [s]
        species=("CO2", "H2O"),  # Exactly two valid species for the FGM plots
        plots=True,  # Write the standard FGM PDF and PNG figures
        export=True,  # Write NPZ, CSV, and FlameMaster-compatible tables
        output=None,  # Or Path("runs/my_fgm"); the directory must not exist
    )

    print(f"FGM table, flamelets, exports, and plots: {Path(output)}")
