# Compatibility matrix

KFLAME supports Python 3.11 and newer. The project constrains its CPU
dependencies to versions that work together; it does not upgrade the user's
global Python installation. Use a virtual environment per matrix below.

| Component | Python 3.11 baseline | Python 3.13 current validated stack |
|---|---:|---:|
| Python | 3.11.0 | 3.13.x |
| NumPy | 2.4.6 | 2.4.6 |
| SciPy | 1.17.1 | 1.17.1 |
| Numba | 0.65.1 | 0.67.x |
| llvmlite | 0.47.0 | 0.49.x |
| PyYAML | 6.0.3 | 6.0.3 |
| Cantera (optional reference) | 3.2.0 | 3.2.0 |
| Matplotlib (optional plots) | 3.11.1 | 3.11.2 |

The NumPy 2.4 and SciPy 1.17 lines work with both supported Numba lines.
`pip install .` resolves the CPU stack; `pip install ".[reference]"` adds the
reference-comparison tools.

KFLAME is CPU/Numba by design. GPU and historical optimization experiments are
not part of the maintained numerical workflow.
