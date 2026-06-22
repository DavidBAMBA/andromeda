"""
===============================================================================
Schwarzschild shadow renderer -- compatibility shim over _shadow_numba
===============================================================================
The integrator suite (DP45 / CK45 / RKF45 / BS / Verlet) now lives, once, in
scr/common/_shadow_numba.py and is shared by every spacetime.  This module keeps
the historical Schwarzschild entry points (``render_shadow_numba`` / ``warmup``)
so ex24-ex26 import unchanged; both delegate to ``_shadow_numba.render_shadow``
with ``spacetime="schwarzschild"``.
===============================================================================
"""

import math

from numba import config  # noqa: F401  (re-export: ex26 reads gn.config.NUMBA_NUM_THREADS)

from scr.common._shadow_numba import render_shadow, _BS_SEQ  # noqa: F401  (re-export)
from scr.common import _shadow_numba as _sn


def render_shadow_numba(alphas, betas, *, D, iota, method="DP45", nthreads=None,
                        lam_max=None, atol=1e-9, rtol=1e-9, EH=2.0, h0=0.1,
                        verlet_h=0.1, chunk=256, shuffle=True, sort="random",
                        timed=True):
    """Render the Schwarzschild shadow; return (flag, |H|, elapsed_s).

    Thin wrapper over :func:`scr.common._shadow_numba.render_shadow`.
    """
    return render_shadow(alphas, betas, spacetime="schwarzschild", method=method,
                         D=D, iota=iota, nthreads=nthreads, lam_max=lam_max,
                         atol=atol, rtol=rtol, EH=EH, h0=h0, verlet_h=verlet_h,
                         chunk=chunk, shuffle=shuffle, sort=sort, timed=timed)


def warmup(D=100.0, iota=math.pi / 2):
    """Compile every kernel and warm the parallel thread pool (Schwarzschild)."""
    _sn.warmup(spacetime="schwarzschild", D=D, iota=iota)
