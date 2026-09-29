"""Solve one neutral CH4/air flame and save its data and plots.

Run from the repository root after ``python -m pip install ".[plot]"``:
    python examples/example.py
"""

from pathlib import Path

import kflame


if __name__ == "__main__":
    result = kflame.solve_flame(
        mechanism="gri30.yaml",  # "gri30.yaml" or "h2o2.yaml"
        temperature=300.0,  # Fresh-mixture temperature [K]
        pressure=101325.0,  # Constant pressure [Pa]
        phi=1.0,  # Equivalence ratio
        fuel="CH4",  # Fuel stream on a mole basis
        oxidizer="O2:1, N2:3.76",  # Oxidizer stream on a mole basis
        width=0.03,  # Initial domain width [m]; the solver may expand it
        initial_points=8,  # Initial grid nodes; adaptive refinement follows
        transport="mixture-averaged",  # "mixture-averaged" or "multicomponent"
        soret=False,  # False or True with either transport model
        ratio=2.5,  # Mesh refinement ratio criterion
        slope=0.04,  # Mesh refinement slope criterion
        curve=0.08,  # Mesh refinement curvature criterion
        prune=0.003,  # Mesh coarsening criterion
        max_points=1600,  # Maximum adaptive grid nodes
        rtol=1e-4,  # Steady relative tolerance
        atol=1e-9,  # Steady absolute tolerance
        max_time=180.0,  # Per-flame time budget [s]
        species=("CH4", "O2", "CO2", "H2O", "OH"),  # Species written to profiles.csv
        plots=True,  # Write profiles.pdf and profiles.png
        output=None,  # Or Path("runs/my_flame"); the directory must not exist
    )

    output = Path(result["output"])
    print(f"Flame speed: {result['Su']:.8f} m/s")
    print(f"Adaptive nodes: {len(result['z'])}")
    print(f"Results and plots: {output}")
