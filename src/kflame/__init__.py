"""FlamPy: native premixed flames and FGM tables. References load explicitly."""
import os as _os

# Configure before any scientific import; respect explicit user overrides.
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('NUMBA_NUM_THREADS', '4')
__version__ = '0.1.0'


def __getattr__(name):
    # Keep CLI/help imports light; scientific modules load on first API access.
    if name == 'generate_adaptive_nonadiabatic_fgm':
        from .fgm.adaptive_nonadiabatic import generate_adaptive_nonadiabatic_fgm
        return generate_adaptive_nonadiabatic_fgm
    if name == 'solve_reduced_burner_fgm':
        from .fgm.reduced_burner import solve_reduced_burner_fgm
        return solve_reduced_burner_fgm
    if name in ('solve_flame', 'solve_burner_flame', 'generate_fgm', 'generate_burner_fgm', 'generate_nonadiabatic_fgm'):
        from . import api
        return getattr(api, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['solve_flame', 'solve_burner_flame', 'generate_fgm', 'generate_burner_fgm', 'generate_nonadiabatic_fgm', 'generate_adaptive_nonadiabatic_fgm', 'solve_reduced_burner_fgm']
