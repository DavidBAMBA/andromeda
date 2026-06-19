"""
===============================================================================
Single-photon integration dispatcher for the project's OWN integrators
===============================================================================
Mirrors the method set of the numba-refactor `integrate()` API and
`benchmark_integrators.py`:

    "LSODA"  : scipy.solve_ivp(method="LSODA")   -- legacy default (Adams/BDF)
    "DOP853" : scipy.solve_ivp(method="DOP853")  -- 8th-order DP, reference
    "RK45"   : in-house adaptive Dormand-Prince  (integrator.rk_adaptive DP45)
    "Verlet" : fixed-step symmetric midpoint      (integrator.verlet)

These are the integrators that actually exist in this codebase -- NOT the
paper's RKCK45 / RKF45 / Bulirsch-Stoer.
===============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp

from scr.common import integrator

# legend labels + a stable color per method
NICE = {
    "LSODA": "LSODA (scipy)",
    "DOP853": "DOP853 (scipy)",
    "RK45": "RK45 (Dormand-Prince, in-house)",
    "Verlet": "Verlet (symmetric midpoint)",
}
COLOR = {"LSODA": "tab:purple", "DOP853": "tab:orange",
         "RK45": "tab:blue", "Verlet": "tab:green"}
METHODS = ["LSODA", "DOP853", "RK45", "Verlet"]


def _stop_factory(r_stop, r_esc):
    if r_stop is None and r_esc is None:
        return None

    def stop(t, y):
        if r_stop is not None and y[1] <= r_stop:
            return True
        if r_esc is not None and y[1] >= r_esc:
            return True
        return False
    return stop


def integrate_photon(rhs, y0, lam_span, method, *, atol=1e-9, rtol=1e-9,
                     r_stop=None, r_esc=None, verlet_steps=10000):
    """Integrate one photon with ``method``; return (T, Y) numpy arrays.

    ``rhs(lam, y)`` is the geodesic RHS. ``r_stop`` / ``r_esc`` add terminal
    horizon / escape events (radius thresholds).
    """
    t0, t1 = float(lam_span[0]), float(lam_span[1])

    if method in ("LSODA", "DOP853"):
        events = []
        if r_stop is not None:
            ev = lambda t, y, rs=r_stop: y[1] - rs
            ev.terminal = True
            ev.direction = -1
            events.append(ev)
        if r_esc is not None:
            ev2 = lambda t, y, re=r_esc: y[1] - re
            ev2.terminal = True
            ev2.direction = 1
            events.append(ev2)
        sol = solve_ivp(rhs, (t0, t1), y0, method=method, atol=atol, rtol=rtol,
                        events=events or None)
        return sol.t, sol.y.T

    stop = _stop_factory(r_stop, r_esc)
    if method == "RK45":
        return integrator.rk_adaptive(rhs, t0, y0, t1, method="DP45",
                                      atol=atol, rtol=rtol, h_max=5.0, stop=stop)
    if method == "Verlet":
        return integrator.verlet(rhs, t0, y0, t1, n_steps=verlet_steps, stop=stop)
    raise ValueError(f"Unknown method: {method}")
