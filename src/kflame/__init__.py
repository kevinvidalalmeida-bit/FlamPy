"""FlamPy: native premixed/counterflow flames and FGM tables."""
import os as _os
import sys as _sys

# Thread policy: one BLAS thread inside FlamPy's own threads (multithreaded
# LAPACK on its small blocks is slower and oversubscribes the cores) and four
# Numba threads. Explicit user settings of these variables always win.
_EXPLICIT = {name: name in _os.environ for name in ('OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')}
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('NUMBA_NUM_THREADS', '4')
__version__ = '0.1.0'


def _apply_thread_policy():
    """Apply the policy to libraries imported before FlamPy.

    Environment variables only act before NumPy/Numba initialize their
    pools; when they are already loaded, limit them at run time instead.
    """
    if not _EXPLICIT['OPENBLAS_NUM_THREADS'] and 'numpy' in _sys.modules:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(int(_os.environ['OPENBLAS_NUM_THREADS']), user_api='blas')
        except Exception:  # threadpoolctl missing or no BLAS pool exposed
            pass
    if not _EXPLICIT['NUMBA_NUM_THREADS'] and 'numba' in _sys.modules:
        try:
            import numba
            numba.set_num_threads(min(int(_os.environ['NUMBA_NUM_THREADS']),
                                      int(numba.config.NUMBA_NUM_THREADS)))
        except Exception:
            pass


def effective_threads():
    """Threads actually in use: Numba, BLAS pools and FlamPy's heavy kernels."""
    info = {}
    try:
        import numba
        info['numba'] = int(numba.get_num_threads())
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_info
        info['blas'] = {pool['internal_api']: int(pool['num_threads'])
                        for pool in threadpool_info() if pool['user_api'] == 'blas'}
    except Exception:
        pass
    try:
        from kflame.flame.diffusion import heavy_threads
        info['heavy_kernels'] = int(heavy_threads())
    except Exception:
        pass
    return info


_apply_thread_policy()


def __getattr__(name):
    # Keep CLI/help imports light; scientific modules load on first API access.
    if name in ('generate_diffusion_fgm','DiffusionFGM','validate_diffusion_fgm',
                'generate_diffusion_heat_loss_fgm','DiffusionHeatLossFGM','validate_diffusion_heat_loss_fgm'):
        from .fgm import diffusion
        return getattr(diffusion,name)
    if name == 'solve_diffusion_flame':
        from .diffusion import solve_diffusion_flame
        return solve_diffusion_flame
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


__all__ = ['solve_diffusion_flame', 'generate_diffusion_fgm', 'DiffusionFGM', 'validate_diffusion_fgm',
           'generate_diffusion_heat_loss_fgm', 'DiffusionHeatLossFGM', 'validate_diffusion_heat_loss_fgm', 'solve_flame', 'solve_burner_flame', 'generate_fgm', 'generate_burner_fgm', 'generate_nonadiabatic_fgm', 'generate_adaptive_nonadiabatic_fgm', 'solve_reduced_burner_fgm']
