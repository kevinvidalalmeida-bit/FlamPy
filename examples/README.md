# User examples

Install KFLAME with plotting support from the repository root:

```sh
python -m pip install ".[plot]"
```

Run one premixed flame:

```sh
python examples/example.py
```

Run one small FGM construction:

```sh
python examples/example_fgm.py
```

Both examples use neutral CH4/air conditions at 300 K and 1 atm. Every keyword
is documented beside its value, including the accepted transport and Soret
options. The output directory contains the numerical data, metadata, selected
profiles, and PDF/PNG plots. Set `output=Path("runs/my_case")` only when that
directory does not already exist.

The thesis-specific adaptive-map workflow is intentionally kept outside these
examples in `TESIS_RESUL/herramientas/campanas/`.
