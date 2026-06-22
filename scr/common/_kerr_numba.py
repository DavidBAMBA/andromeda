"""
===============================================================================
Kerr shadow renderer -- compatibility shim over _shadow_numba
===============================================================================
The integrator suite (DP45 / CK45 / RKF45 / BS / Verlet) now lives, once, in
scr/common/_shadow_numba.py and is shared by every spacetime.  Kerr therefore
gains every integrator for free (previously DP45-only).  ``render_kerr_shadow``
keeps its historical signature -- so ex27 imports unchanged -- and now accepts a
``method=`` argument (default "DP45" for back-compat); both it and ``warmup``
delegate to ``_shadow_numba.render_shadow`` with ``spacetime="kerr"``.
===============================================================================
"""

import math

from scr.common._shadow_numba import render_shadow  # noqa: F401  (re-export)
from scr.common import _shadow_numba as _sn


def render_kerr_shadow(alphas, betas, *, D, iota, a, method="DP45", nthreads=None,
                       lam_max=None, atol=1e-9, rtol=1e-9, h0=0.5,
                       chunk=256, shuffle=True):
    """Render the Kerr shadow; return (flag, |H|, elapsed_s).

    Thin wrapper over :func:`scr.common._shadow_numba.render_shadow`.  ``method``
    is any of DP45/CK45/RKF45/BS/Verlet.  ``sort=None`` is forwarded so the
    historical plain-``shuffle`` semantics are preserved.
    """
    return render_shadow(alphas, betas, spacetime="kerr", a=a, method=method,
                         D=D, iota=iota, nthreads=nthreads, lam_max=lam_max,
                         atol=atol, rtol=rtol, h0=h0, chunk=chunk,
                         shuffle=shuffle, sort=None)


def warmup(D=1000.0, iota=math.pi / 2, a=0.98,
           methods=("DP45", "CK45", "RKF45", "BS", "Verlet")):
    """Compile every kernel and warm the parallel thread pool (Kerr, spin a)."""
    _sn.warmup(spacetime="kerr", a=a, D=D, iota=iota, methods=methods)
