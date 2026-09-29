# Examples

Install the package with plotting support from the repository root:

```powershell
python -m pip install -e ".[plot]"
```

The examples use the same transparent workflow as a typical Cantera script:

1. edit one case dictionary;
2. solve and save the numerical output;
3. reload named NumPy arrays from the saved `.npz` file;
4. assemble a conventional row-wise CSV matrix; and
5. create explicit Matplotlib figures that you can modify.

Nothing chooses figures implicitly. Edit the plotting functions to add fields,
change limits, or replace the supplied layout.

## One-dimensional flame

```powershell
python examples/example.py
```

`example.py` defines `CASE`, runs a neutral CH4/air flame, and writes these
additional files into the generated run directory:

- `solution_matrix.csv`: one row per grid point, with position, temperature,
  velocity, heat release, and the selected species mass fractions;
- `custom_flame_plots.pdf`: temperature, velocity, heat-release, and species
  plots made directly from `flame.npz`.

Use `mechanism="h2o2.yaml"` and `fuel="H2"` to adapt the same script to a
hydrogen flame. Valid transport choices are `"mixture-averaged"` and
`"multicomponent"`. Soret diffusion requires multicomponent transport.

## FGM table

```powershell
python examples/example_fgm.py
```

`example_fgm.py` defines `FGM_CASE`, constructs a deliberately small FGM table,
and writes:

- `fgm_matrix.csv`: one row for every `(Z, c)` table state, with selected
  thermo-chemical fields;
- `custom_fgm_plots.pdf`: maps of temperature, heat release, a selected species,
  and the burning velocity of the flamelets.

Increase `phis` in `FGM_CASE` before using the workflow for a production table.
