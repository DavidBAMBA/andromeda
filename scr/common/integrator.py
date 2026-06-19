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


# ===========================================================================
# Embedded Runge-Kutta family (DP45 / Cash-Karp / Fehlberg) + Bulirsch-Stoer
# ---------------------------------------------------------------------------
# Added to reproduce the integrator comparison of Fig. 4 in the OSIRIS paper
# (Hamiltonian-constraint preservation for null geodesics). The three adaptive
# RK pairs below share a single driver and differ only in their Butcher tableau.
# The 5th-order weights are used to propagate the solution (local extrapolation)
# and the embedded 4th-order weights provide the error estimate for step control.
# All drivers return numpy arrays (T, Y) of the accepted steps, with the SAME
# state layout odeint produces -- so common.Hamiltonian(Y, bh) works directly.
# ===========================================================================
import numpy as np

# Each tableau: c, A (lower-triangular), b5 (propagated 5th order),
#               b4 (embedded 4th order).
_RK_TABLEAUX = {
    "DP45": {  # Dormand-Prince (same core method as SciPy's RK45)
        "c": [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0],
        "A": [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
        ],
        "b5": [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0],
        "b4": [5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40],
    },
    "CK45": {  # Cash-Karp
        "c": [0.0, 1/5, 3/10, 3/5, 1.0, 7/8],
        "A": [
            [],
            [1/5],
            [3/40, 9/40],
            [3/10, -9/10, 6/5],
            [-11/54, 5/2, -70/27, 35/27],
            [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
        ],
        "b5": [37/378, 0.0, 250/621, 125/594, 0.0, 512/1771],
        "b4": [2825/27648, 0.0, 18575/48384, 13525/55296, 277/14336, 1/4],
    },
    "RKF45": {  # Runge-Kutta-Fehlberg
        "c": [0.0, 1/4, 3/8, 12/13, 1.0, 1/2],
        "A": [
            [],
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8.0, 3680/513, -845/4104],
            [-8/27, 2.0, -3544/2565, 1859/4104, -11/40],
        ],
        "b5": [16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55],
        "b4": [25/216, 0.0, 1408/2565, 2197/4104, -1/5, 0.0],
    },
}


def rk_adaptive(f, t0, y0, t1, method="DP45", *, atol=1e-10, rtol=1e-10,
                h0=1e-2, h_min=1e-12, h_max=10.0, max_steps=2_000_000,
                stop=None):
    """Adaptive embedded Runge-Kutta integrator (numpy backend).

    Integrates y'(t) = f(t, y) from t0 to t1 using one of the embedded 4(5)
    pairs in ``_RK_TABLEAUX``.

    Parameters
    ----------
    method : {"DP45", "CK45", "RKF45"}
        Dormand-Prince, Cash-Karp or Fehlberg tableau.
    stop : callable(t, y) -> bool, optional
        Event callback. After each accepted step it is queried; when it
        returns True the integration halts (used to stop a plunging photon
        before the horizon coordinate singularity).

    Returns
    -------
    T, Y : ndarray, ndarray
        Accepted-step times (M,) and states (M, n), including t0.
    """
    tab = _RK_TABLEAUX[method]
    c, A = tab["c"], tab["A"]
    b5 = np.asarray(tab["b5"], float)
    b4 = np.asarray(tab["b4"], float)
    s = len(c)

    y = np.asarray(y0, float).copy()
    n = y.size
    t = float(t0)
    t_end = float(t1)
    forward = t_end >= t
    sgn = 1.0 if forward else -1.0

    safety, min_scale, max_scale = 0.9, 0.2, 5.0
    h = sgn * min(max(h0, h_min), h_max)

    T = [t]
    Y = [y.copy()]
    nstep = 0
    while (t < t_end if forward else t > t_end) and nstep < max_steps:
        h = sgn * min(max(abs(h), h_min), h_max)
        if forward and t + h > t_end:
            h = t_end - t
        elif (not forward) and t + h < t_end:
            h = t_end - t

        k = np.empty((s, n))
        k[0] = np.asarray(f(t, y), float)
        for i in range(1, s):
            yi = y.copy()
            Ai = A[i]
            for j in range(i):
                if Ai[j] != 0.0:
                    yi = yi + (h * Ai[j]) * k[j]
            k[i] = np.asarray(f(t + c[i] * h, yi), float)

        y5 = y + h * (b5 @ k)
        y4 = y + h * (b4 @ k)
        err = y5 - y4
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        en = np.sqrt(np.mean((err / sc) ** 2))

        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            t += h
            y = y5
            T.append(t)
            Y.append(y.copy())
            nstep += 1
            if stop is not None and stop(t, y):
                break
            if en == 0.0:
                fac = max_scale
            else:
                fac = min(max(safety * en ** -0.2, min_scale), max_scale)
            h = sgn * min(max(abs(h) * fac, h_min), h_max)
        else:
            fac = min(max(safety * en ** -0.25, min_scale), 1.0)
            h = sgn * min(max(abs(h) * fac, h_min), h_max)

    return np.asarray(T), np.asarray(Y)


def bulirsch_stoer(f, t0, y0, t1, *, atol=1e-10, rtol=1e-10, h0=1e-1,
                   h_min=1e-10, h_max=10.0, max_steps=500_000, stop=None):
    """Adaptive Gragg-Bulirsch-Stoer integrator.

    Modified-midpoint substepping followed by polynomial (Richardson)
    extrapolation in h^2, with error-controlled macro steps. Returns
    (T, Y) over the accepted macro-steps, same layout as ``rk_adaptive``.

    ``stop`` is an optional callback(t, y) -> bool that halts the run after
    an accepted macro-step (e.g. to stop a plunging photon at the horizon).
    """
    n_seq = [2, 4, 6, 8, 10, 12, 14, 16]
    KMAX = len(n_seq)
    safety = 0.9

    y = np.asarray(y0, float).copy()
    n = y.size
    t = float(t0)
    t_end = float(t1)
    forward = t_end >= t
    sgn = 1.0 if forward else -1.0
    H = sgn * min(max(h0, h_min), h_max)

    def mmid(t, y, Htot, nsub):
        """Modified midpoint rule: nsub substeps across Htot."""
        h = Htot / nsub
        ym = y.copy()
        ym1 = y + h * np.asarray(f(t, y), float)
        for m in range(1, nsub):
            ym, ym1 = ym1, ym + 2.0 * h * np.asarray(f(t + m * h, ym1), float)
        return 0.5 * (ym1 + ym + h * np.asarray(f(t + Htot, ym1), float))

    T = [t]
    Y = [y.copy()]
    nstep = 0
    while (t < t_end if forward else t > t_end) and nstep < max_steps:
        if forward and t + H > t_end:
            H = t_end - t
        elif (not forward) and t + H < t_end:
            H = t_end - t

        tbl = [[None] * KMAX for _ in range(KMAX)]
        en = float("inf")
        k_conv = None
        for kk in range(KMAX):
            tbl[kk][0] = mmid(t, y, H, n_seq[kk])
            for m in range(1, kk + 1):
                ratio = (n_seq[kk] / n_seq[kk - m]) ** 2
                tbl[kk][m] = tbl[kk][m - 1] + (tbl[kk][m - 1] - tbl[kk - 1][m - 1]) / (ratio - 1.0)
            if kk >= 1:
                diff = tbl[kk][kk] - tbl[kk][kk - 1]
                sc = atol + rtol * np.maximum(np.abs(y), np.abs(tbl[kk][kk]))
                en = np.sqrt(np.mean((diff / sc) ** 2))
                if en <= 1.0:
                    k_conv = kk
                    break

        if k_conv is not None or abs(H) <= 1.0001 * h_min:
            kc = k_conv if k_conv is not None else KMAX - 1
            t += H
            y = tbl[kc][kc]
            T.append(t)
            Y.append(y.copy())
            nstep += 1
            if stop is not None and stop(t, y):
                break
            if en == 0.0 or en != en:  # zero error or NaN guard
                fac = 4.0
            else:
                fac = safety * en ** (-1.0 / (2 * kc + 1))
            fac = min(max(fac, 0.2), 4.0)
            H = sgn * min(max(abs(H) * fac, h_min), h_max)
        else:
            H = sgn * max(abs(H) * 0.25, h_min)

    return np.asarray(T), np.asarray(Y)


def verlet(f, t0, y0, t1, *, n_steps=10000, h0=None, max_steps=2_000_000,
           stop=None):
    """Fixed-step 2nd-order symmetric midpoint splitting.

    Faithful port of the "Verlet" method in the numba-refactor branch
    (_solvers._solve_verlet):

        y_mid = y + (h/2) f(t,        y)
        y_new = y + h     f(t + h/2,  y_mid)

    Not strictly symplectic for the (non-separable) geodesic Hamiltonian, but
    symmetric / time-reversible, so it tends to *bound* the Hamiltonian drift
    rather than let it grow secularly. Fixed step h = (t1-t0)/n_steps unless
    ``h0`` is given. Returns (T, Y) numpy arrays of the steps.
    """
    y = np.asarray(y0, float).copy()
    t = float(t0)
    t_end = float(t1)
    forward = t_end >= t
    sgn = 1.0 if forward else -1.0
    h = abs(h0) * sgn if h0 is not None else (t_end - t) / n_steps

    T = [t]
    Y = [y.copy()]
    steps = 0
    while (t < t_end if forward else t > t_end) and steps < max_steps:
        if forward and t + h > t_end:
            h = t_end - t
        elif (not forward) and t + h < t_end:
            h = t_end - t
        k1 = np.asarray(f(t, y), float)
        y_mid = y + 0.5 * h * k1
        k2 = np.asarray(f(t + 0.5 * h, y_mid), float)
        y = y + h * k2
        t += h
        T.append(t)
        Y.append(y.copy())
        steps += 1
        if stop is not None and stop(t, y):
            break
    return np.asarray(T), np.asarray(Y)


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
