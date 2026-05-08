"""
===============================================================================
Unified geodesic integrator with event detection
===============================================================================
API:
    integrate(f, y0, lmbda_span, method="DOP853", events=None, ...)
        -> IntegrationResult

Backends:
    - "LSODA"  : scipy.integrate.solve_ivp with LSODA
    - "DOP853" : scipy.integrate.solve_ivp with DOP853
    - "RK45"   : in-house adaptive Dormand-Prince with Brent event refinement
    - "Verlet" : 2nd-order Strang splitting (experimental)

Events:
    Each event is a callable g(lmbda, y) -> float. Integration stops at the
    first zero crossing matching its direction attribute. Use make_events()
    to build the standard (horizon, disk, escape) set.
===============================================================================
"""
from numpy import cos

from scr.common._solvers import (IntegrationResult,
                                  _solve_scipy, _solve_rk45, _solve_verlet)
from scr.common._numba_kernels import (_null_omega_nb, _solve_photon_nb,
                                        _compute_pixel_nb, _render_image_nb)

# Re-export so parallel.py and other callers don't need to change imports.
__all__ = [
    "integrate", "make_events", "IntegrationResult",
    "_null_omega_nb", "_solve_photon_nb", "_compute_pixel_nb", "_render_image_nb",
]


def integrate(f, y0, lmbda_span, *, method="DOP853", events=None,
              rtol=1e-9, atol=1e-11, max_step=None,
              first_step=None, max_steps=1_000_000):
    """
    Integrate y'(lmbda) = f(lmbda, y) over lmbda_span (backward allowed).

    method  : "LSODA" | "DOP853" | "RK45" | "Verlet"
    events  : callables g(lmbda, y) -> float with .terminal and .direction attrs.
    rtol/atol ignored by Verlet (fixed step).

    Returns IntegrationResult(t, y, t_events, y_events, status, nfev, wall_time).
    """
    events = list(events) if events else []
    t0, t1 = float(lmbda_span[0]), float(lmbda_span[1])

    # "auto"/"RK45_numba" are served by _solve_photon_nb directly in
    # parallel.py; here they degrade to DOP853 for the scipy path.
    if method in ("auto", "RK45_numba"):
        method = "DOP853"

    if method in ("DOP853", "RK45_scipy", "LSODA"):
        scipy_method = "LSODA" if method == "LSODA" else (
            "RK45" if method == "RK45_scipy" else "DOP853")
        result = _solve_scipy(f, y0, (t0, t1), scipy_method, events,
                              rtol, atol, max_step, first_step)
    elif method == "RK45":
        result = _solve_rk45(f, y0, (t0, t1), events, rtol, atol,
                             max_step, first_step, max_steps)
    elif method == "Verlet":
        result = _solve_verlet(f, y0, (t0, t1), events, max_step,
                               first_step, max_steps)
    else:
        raise ValueError(f"Unknown method: {method}")

    result.method = method
    return result


def make_events(blackhole, acc_structure=None, r_escape=None, eps_horizon=1e-3):
    """
    Build the standard set of events for photon geodesics.

    Returns
    -------
    events : list of callables with .name, .direction, .terminal attributes
        [horizon, disk, escape] — disk and escape are omitted when the
        corresponding argument is None.
    """
    r_plus = float(blackhole.EH)

    def horizon(lmbda, y):
        return y[1] - (r_plus + eps_horizon)
    horizon.terminal = True
    horizon.direction = -1
    horizon.name = "horizon"

    events = [horizon]

    if acc_structure is not None:
        # NON-TERMINAL: record every cos(theta) sign change so the caller can
        # pick the first crossing inside the disk annulus. Terminal here
        # would kill photon-ring trajectories that cross the equator at
        # r ~ 3 (photon sphere) before reaching the disk.
        def disk(lmbda, y):
            return cos(y[2])
        disk.terminal = False
        disk.direction = 0
        disk.name = "disk"
        events.append(disk)

    if r_escape is not None:
        r_esc = float(r_escape)

        def escape(lmbda, y):
            return y[1] - r_esc
        escape.terminal = True
        escape.direction = 1
        escape.name = "escape"
        events.append(escape)

    return events


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    def f(t, y):
        return [-y[0]]

    print("Backend  |    y(5)       |  abs err   | nfev | wall (ms)")
    print("-" * 60)
    for m in ("LSODA", "DOP853", "RK45", "Verlet"):
        kwargs = dict(rtol=1e-10, atol=1e-12)
        if m == "Verlet":
            kwargs = dict(first_step=1e-3)
        res = integrate(f, [1.0], (0.0, 5.0), method=m, **kwargs)
        err = abs(res.y[-1, 0] - math.exp(-5.0))
        print(f"{m:8s} | {res.y[-1,0]:.12f} | {err:8.2e} | "
              f"{res.nfev:4d} | {1000*res.wall_time:7.2f}")

    def g(t, y):
        return [y[1], -9.8]
    def hit_ground(t, y):
        return y[0]
    hit_ground.terminal = True
    hit_ground.direction = -1
    hit_ground.name = "ground"

    print("\nEvent test (freefall):")
    for m in ("DOP853", "RK45"):
        res = integrate(g, [10.0, 0.0], (0.0, 10.0), method=m,
                        events=[hit_ground], rtol=1e-9, atol=1e-11)
        t_hit = res.t[-1]
        exact = math.sqrt(2 * 10.0 / 9.8)
        print(f"  {m}: status={res.status}, t_hit={t_hit:.6f}, "
              f"exact={exact:.6f}, err={abs(t_hit-exact):.2e}")
