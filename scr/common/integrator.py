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
    - "Verlet" : 2nd-order Yoshida-style splitting (experimental for non-
                 separable Hamiltonians such as Kerr)

Events:
    Each event is a callable g(lmbda, y) -> float. Integration stops at the
    first zero crossing matching its direction attribute. Use make_events()
    to build the standard (horizon, disk, escape) set.
===============================================================================
"""
from math import sqrt, isfinite, cos as mcos, sin as msin

import numpy as np
from numpy import asarray, array, isfinite as np_isfinite, cos, sin
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import numba
from numba import njit, prange


MAX_DISK_HITS_NB = 32


@njit(cache=True)
def _null_omega_nb(r):
    '''Stub for BHs without a Doppler-compatible Omega(r).

    Must never be invoked in practice — _compute_pixel_nb only calls omega
    when mode_code == 1 (doppler). Exists to satisfy numba's first-class
    function typing when passing a non-None omega hook is required.
    '''
    return 0.0


@njit(cache=True, fastmath=False)
def _solve_photon_nb(rhs, y0, lmbda_end, r_hor, r_esc, rtol, atol):
    '''Numba-compiled RK45 (Dormand-Prince) with inline event detection.

    Events: horizon (terminal, direction=-1), escape (terminal, direction=+1),
    disk (non-terminal cos(theta) sign changes, up to MAX_DISK_HITS_NB).

    Returns
    -------
    status   : int8  (0=max_lambda, 1=horizon, 2=escape, -1=nonfinite)
    y_final  : float64[8]
    n_hits   : int32
    y_hits   : float64[MAX_DISK_HITS_NB, 8]
    '''
    n = 8
    y = y0.copy()
    t0 = 0.0
    t1 = lmbda_end
    sgn = 1.0 if t1 >= t0 else -1.0
    forward = sgn > 0.0

    h_max = abs(t1 - t0)
    h_min = 1e-14
    h = max(h_min, min(1e-2, h_max)) * sgn

    safety = 0.9
    min_scale = 0.2
    max_scale = 5.0
    max_steps = 1_000_000
    eps_horizon = 1e-3

    # Dormand-Prince coefficients (Butcher tableau).
    A21 = 1.0/5.0
    A31 = 3.0/40.0;   A32 = 9.0/40.0
    A41 = 44.0/45.0;  A42 = -56.0/15.0;  A43 = 32.0/9.0
    A51 = 19372.0/6561.0;   A52 = -25360.0/2187.0
    A53 = 64448.0/6561.0;   A54 = -212.0/729.0
    A61 = 9017.0/3168.0;    A62 = -355.0/33.0;   A63 = 46732.0/5247.0
    A64 = 49.0/176.0;       A65 = -5103.0/18656.0
    A71 = 35.0/384.0;       A73 = 500.0/1113.0;  A74 = 125.0/192.0
    A75 = -2187.0/6784.0;   A76 = 11.0/84.0
    B41 = 5179.0/57600.0;   B43 = 7571.0/16695.0
    B44 = 393.0/640.0;      B45 = -92097.0/339200.0
    B46 = 187.0/2100.0;     B47 = 1.0/40.0

    status = 0
    n_hits = 0
    y_hits = np.zeros((MAX_DISK_HITS_NB, n))

    # Pre-allocated buffers.
    y_tmp = np.empty(n)
    y_new = np.empty(n)
    y_event = np.empty(n)

    # Event value at start.
    horizon_prev = y[1] - (r_hor + eps_horizon)
    escape_prev  = y[1] - r_esc
    disk_prev    = mcos(y[2])

    # First RHS evaluation (k1). FSAL reuses this across accepted steps.
    k1 = rhs(y)
    for m in range(n):
        if not np.isfinite(k1[m]):
            return -1, y, 0, y_hits

    t = t0
    accepted = 0
    while accepted < max_steps:
        # Clamp step and check if we would overshoot t1.
        if abs(h) < h_min:
            h = h_min * sgn
        elif abs(h) > h_max:
            h = h_max * sgn
        if forward and t + h > t1:
            h = t1 - t
        if (not forward) and t + h < t1:
            h = t1 - t
        # Are we done?
        if (forward and t >= t1) or ((not forward) and t <= t1):
            status = 0
            break

        # Stage 2 .. 6
        for m in range(n):
            y_tmp[m] = y[m] + h * A21 * k1[m]
        k2 = rhs(y_tmp)
        for m in range(n):
            y_tmp[m] = y[m] + h * (A31*k1[m] + A32*k2[m])
        k3 = rhs(y_tmp)
        for m in range(n):
            y_tmp[m] = y[m] + h * (A41*k1[m] + A42*k2[m] + A43*k3[m])
        k4 = rhs(y_tmp)
        for m in range(n):
            y_tmp[m] = y[m] + h * (A51*k1[m] + A52*k2[m] + A53*k3[m] + A54*k4[m])
        k5 = rhs(y_tmp)
        for m in range(n):
            y_tmp[m] = y[m] + h * (A61*k1[m] + A62*k2[m] + A63*k3[m] + A64*k4[m] + A65*k5[m])
        k6 = rhs(y_tmp)

        # 5th-order y_new (=B5 row = A7 row in DP)
        for m in range(n):
            y_new[m] = y[m] + h * (A71*k1[m] + A73*k3[m] + A74*k4[m] + A75*k5[m] + A76*k6[m])
        # Stage 7 for error estimate (FSAL → next step's k1)
        k7 = rhs(y_new)
        # 4th-order estimate for error
        err_acc = 0.0
        ok_finite = True
        for m in range(n):
            y4 = y[m] + h * (B41*k1[m] + B43*k3[m] + B44*k4[m] + B45*k5[m] + B46*k6[m] + B47*k7[m])
            if not np.isfinite(y_new[m]):
                ok_finite = False
                break
            sc = atol + rtol * max(abs(y[m]), abs(y_new[m]))
            err_acc += ((y_new[m] - y4) / sc) ** 2
        if not ok_finite:
            return -1, y, n_hits, y_hits
        en = (err_acc / n) ** 0.5

        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            # Accept. Check events before committing.
            horizon_new = y_new[1] - (r_hor + eps_horizon)
            escape_new  = y_new[1] - r_esc
            disk_new    = mcos(y_new[2])

            event_kind = 0  # 0=none, 1=horizon, 2=escape

            # Horizon (direction -1: prev > 0, new <= 0)
            if horizon_prev > 0.0 and horizon_new <= 0.0:
                s_lo = 0.0; s_hi = 1.0
                g_lo = horizon_prev
                for _ in range(30):
                    s_mid = 0.5 * (s_lo + s_hi)
                    h00 = 2.0*s_mid*s_mid*s_mid - 3.0*s_mid*s_mid + 1.0
                    h10 = s_mid*s_mid*s_mid - 2.0*s_mid*s_mid + s_mid
                    h01 = -2.0*s_mid*s_mid*s_mid + 3.0*s_mid*s_mid
                    h11 = s_mid*s_mid*s_mid - s_mid*s_mid
                    r_mid = h00*y[1] + h10*h*k1[1] + h01*y_new[1] + h11*h*k7[1]
                    g_mid = r_mid - (r_hor + eps_horizon)
                    if g_mid * g_lo < 0.0:
                        s_hi = s_mid
                    else:
                        s_lo = s_mid
                        g_lo = g_mid
                s_star = 0.5 * (s_lo + s_hi)
                h00 = 2.0*s_star*s_star*s_star - 3.0*s_star*s_star + 1.0
                h10 = s_star*s_star*s_star - 2.0*s_star*s_star + s_star
                h01 = -2.0*s_star*s_star*s_star + 3.0*s_star*s_star
                h11 = s_star*s_star*s_star - s_star*s_star
                for m in range(n):
                    y_event[m] = h00*y[m] + h10*h*k1[m] + h01*y_new[m] + h11*h*k7[m]
                event_kind = 1
            elif escape_prev < 0.0 and escape_new >= 0.0:
                s_lo = 0.0; s_hi = 1.0
                g_lo = escape_prev
                for _ in range(30):
                    s_mid = 0.5 * (s_lo + s_hi)
                    h00 = 2.0*s_mid*s_mid*s_mid - 3.0*s_mid*s_mid + 1.0
                    h10 = s_mid*s_mid*s_mid - 2.0*s_mid*s_mid + s_mid
                    h01 = -2.0*s_mid*s_mid*s_mid + 3.0*s_mid*s_mid
                    h11 = s_mid*s_mid*s_mid - s_mid*s_mid
                    r_mid = h00*y[1] + h10*h*k1[1] + h01*y_new[1] + h11*h*k7[1]
                    g_mid = r_mid - r_esc
                    if g_mid * g_lo < 0.0:
                        s_hi = s_mid
                    else:
                        s_lo = s_mid
                        g_lo = g_mid
                s_star = 0.5 * (s_lo + s_hi)
                h00 = 2.0*s_star*s_star*s_star - 3.0*s_star*s_star + 1.0
                h10 = s_star*s_star*s_star - 2.0*s_star*s_star + s_star
                h01 = -2.0*s_star*s_star*s_star + 3.0*s_star*s_star
                h11 = s_star*s_star*s_star - s_star*s_star
                for m in range(n):
                    y_event[m] = h00*y[m] + h10*h*k1[m] + h01*y_new[m] + h11*h*k7[m]
                event_kind = 2

            # Disk (non-terminal): record every equatorial crossing this step.
            if event_kind == 0 and disk_prev * disk_new < 0.0 and n_hits < MAX_DISK_HITS_NB:
                s_lo = 0.0; s_hi = 1.0
                g_lo = disk_prev
                for _ in range(30):
                    s_mid = 0.5 * (s_lo + s_hi)
                    h00 = 2.0*s_mid*s_mid*s_mid - 3.0*s_mid*s_mid + 1.0
                    h10 = s_mid*s_mid*s_mid - 2.0*s_mid*s_mid + s_mid
                    h01 = -2.0*s_mid*s_mid*s_mid + 3.0*s_mid*s_mid
                    h11 = s_mid*s_mid*s_mid - s_mid*s_mid
                    theta_mid = h00*y[2] + h10*h*k1[2] + h01*y_new[2] + h11*h*k7[2]
                    g_mid = mcos(theta_mid)
                    if g_mid * g_lo < 0.0:
                        s_hi = s_mid
                    else:
                        s_lo = s_mid
                        g_lo = g_mid
                s_star = 0.5 * (s_lo + s_hi)
                h00 = 2.0*s_star*s_star*s_star - 3.0*s_star*s_star + 1.0
                h10 = s_star*s_star*s_star - 2.0*s_star*s_star + s_star
                h01 = -2.0*s_star*s_star*s_star + 3.0*s_star*s_star
                h11 = s_star*s_star*s_star - s_star*s_star
                for m in range(n):
                    y_hits[n_hits, m] = h00*y[m] + h10*h*k1[m] + h01*y_new[m] + h11*h*k7[m]
                n_hits += 1

            if event_kind == 1:
                status = 1
                for m in range(n):
                    y[m] = y_event[m]
                break
            if event_kind == 2:
                status = 2
                for m in range(n):
                    y[m] = y_event[m]
                break

            # Commit step; reuse k7 as next k1 (FSAL).
            t = t + h
            for m in range(n):
                y[m] = y_new[m]
                k1[m] = k7[m]
            horizon_prev = horizon_new
            escape_prev  = escape_new
            disk_prev    = disk_new
            accepted += 1

            # Step-size update.
            if en == 0.0:
                sc_h = max_scale
            else:
                sc_h = safety * (1.0 / en) ** 0.2
                if sc_h < min_scale:
                    sc_h = min_scale
                if sc_h > max_scale:
                    sc_h = max_scale
            h = h * sc_h
            if abs(h) < h_min:
                h = h_min * sgn
            if abs(h) > h_max:
                h = h_max * sgn
        else:
            # Reject step; shrink h.
            sc_h = safety * (1.0 / en) ** 0.25
            if sc_h < min_scale:
                sc_h = min_scale
            if sc_h > 1.0:
                sc_h = 1.0
            h = h * sc_h
            if abs(h) < h_min:
                h = h_min * sgn
    return status, y, n_hits, y_hits


@njit(cache=False, fastmath=False)
def _compute_pixel_nb(rhs, metric, omega,
                     y0, lmbda_end, r_hor, r_esc,
                     r_tbl, I_tbl, in_edge, out_edge,
                     rtol, atol, mode_code):
    '''Full per-pixel numba pipeline: integrate, pick first in-annulus hit,
    return scalar intensity (with Doppler if mode_code==1).

    mode_code: 0=no_doppler, 1=doppler, 2=shadow.
    '''
    status, y_final, n_hits, y_hits = _solve_photon_nb(
        rhs, y0, lmbda_end, r_hor, r_esc, rtol, atol)

    if mode_code == 2:
        return 0.0 if status == 1 else 100.0

    for k in range(n_hits):
        r_hit = y_hits[k, 1]
        if in_edge <= r_hit <= out_edge:
            I0 = np.interp(r_hit, r_tbl, I_tbl)
            if mode_code == 1:
                g_tt, g_rr, g_thth, g_phph, g_tph = metric(y_hits[k, :4])
                Om = omega(r_hit)
                gshift = ((-g_tt - 2.0*g_tph*Om - g_phph*Om*Om) ** 0.5) \
                         / (1.0 + y_hits[k, 7] * Om / y_hits[k, 4])
                return I0 * gshift * gshift * gshift
            return I0
    return 0.0


@njit(parallel=True, nogil=True, cache=False)
def _render_image_nb(rhs, metric, omega,
                     y0_batch, lmbda_end, r_hor, r_esc,
                     r_tbl, I_tbl, in_edge, out_edge,
                     rtol, atol, mode_code):
    '''Parallel-over-photons render kernel. Each iteration is independent.'''
    n = y0_batch.shape[0]
    values = np.zeros(n)
    for idx in prange(n):
        values[idx] = _compute_pixel_nb(
            rhs, metric, omega,
            y0_batch[idx], lmbda_end, r_hor, r_esc,
            r_tbl, I_tbl, in_edge, out_edge,
            rtol, atol, mode_code)
    return values


class IntegrationResult:
    """Lightweight result container mimicking a subset of scipy's Bunch."""
    __slots__ = ("t", "y", "t_events", "y_events",
                 "status", "nfev", "wall_time", "method", "success")

    def __init__(self, t, y, t_events, y_events, status, nfev, wall_time,
                 method, success=True):
        self.t = t
        self.y = y
        self.t_events = t_events
        self.y_events = y_events
        self.status = status
        self.nfev = nfev
        self.wall_time = wall_time
        self.method = method
        self.success = success


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def integrate(f, y0, lmbda_span, *, method="DOP853", events=None,
              rtol=1e-9, atol=1e-11, max_step=None, dense=False,
              first_step=None, max_steps=1_000_000):
    """
    Integrate y'(lmbda) = f(lmbda, y) from lmbda_span[0] to lmbda_span[1].

    Parameters
    ----------
    f : callable
        f(lmbda, y) -> list/array of derivatives.
    y0 : sequence
        Initial state.
    lmbda_span : (lmbda0, lmbda1)
        Integration interval. lmbda1 < lmbda0 is allowed (backward).
    method : {"LSODA", "DOP853", "RK45", "Verlet"}
    events : list of callables, optional
        Each must have attributes .terminal (bool, default True) and
        .direction (int, default 0). Returns scalar g; a sign change
        between steps triggers the event.
    rtol, atol : float
        Tolerances. For Verlet these are ignored (fixed step).
    max_step : float, optional
        Upper bound on |h|.
    dense : bool
        If True, request all accepted steps (for Hamiltonian verification).
    first_step : float, optional
        Initial step size guess.
    max_steps : int
        Safety cap.

    Returns
    -------
    IntegrationResult with:
        t         : 1D array of output lambdas
        y         : 2D array shape (n_points, n_state)
        t_events  : list of 1D arrays, one per event
        y_events  : list of 2D arrays, one per event
        status    : "horizon"|"disk"|"escape"|"nonfinite"|"max_lambda"|"error"
                    The string is chosen from the triggered event's .name if
                    available, otherwise from the fallback.
        nfev      : number of RHS evaluations
        wall_time : seconds
    """
    events = list(events) if events else []
    t0, t1 = float(lmbda_span[0]), float(lmbda_span[1])

    # "auto"/"RK45_numba" are served by _solve_photon_nb directly (see
    # parallel.py); here they degrade to DOP853 for the scipy path.
    if method in ("auto", "RK45_numba"):
        method = "DOP853"
    if method in ("DOP853", "RK45_scipy", "LSODA"):
        scipy_method = "LSODA" if method == "LSODA" else (
            "RK45" if method == "RK45_scipy" else "DOP853")
        result = _solve_scipy(f, y0, (t0, t1), scipy_method, events,
                              rtol, atol, max_step, dense, first_step)
    elif method == "RK45":
        result = _solve_rk45(f, y0, (t0, t1), events, rtol, atol,
                             max_step, dense, first_step, max_steps)
    elif method == "Verlet":
        result = _solve_verlet(f, y0, (t0, t1), events, max_step, dense,
                               first_step, max_steps)
    else:
        raise ValueError(f"Unknown method: {method}")
    result.method = method
    return result


# ----------------------------------------------------------------------------
# SciPy backend (LSODA, DOP853, RK45)
# ----------------------------------------------------------------------------
def _solve_scipy(f, y0, span, method, events, rtol, atol, max_step,
                 dense, first_step):
    kwargs = dict(method=method, rtol=rtol, atol=atol, dense_output=False)
    if max_step is not None:
        kwargs["max_step"] = max_step
    if first_step is not None:
        kwargs["first_step"] = first_step
    if events:
        kwargs["events"] = events

    sol = solve_ivp(f, span, asarray(y0, dtype=float), **kwargs)

    # Determine which event (if any) terminated the integration.
    status = "max_lambda"
    if sol.status == 1:  # a termination event was hit
        for i, te in enumerate(sol.t_events):
            if te.size > 0:
                status = getattr(events[i], "name", f"event_{i}")
                break
    elif sol.status < 0:
        status = "error"

    # Check for non-finite state
    if not np_isfinite(sol.y).all():
        status = "nonfinite"

    return IntegrationResult(
        t=sol.t,
        y=sol.y.T,
        t_events=list(sol.t_events) if sol.t_events is not None else [],
        y_events=[ye for ye in sol.y_events] if sol.y_events is not None else [],
        status=status,
        nfev=int(sol.nfev),
        wall_time=0.0,
        method=method,
        success=(sol.status >= 0),
    )


# ----------------------------------------------------------------------------
# In-house adaptive Dormand-Prince RK45 with Brent event refinement
# ----------------------------------------------------------------------------
_DP_C = (0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0)
_DP_A = (
    (),
    (1/5,),
    (3/40, 9/40),
    (44/45, -56/15, 32/9),
    (19372/6561, -25360/2187, 64448/6561, -212/729),
    (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
    (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84),
)
_DP_B5 = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)
_DP_B4 = (5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40)


def _solve_rk45(f, y0, span, events, rtol, atol, max_step, dense,
                first_step, max_steps):
    t0, t1 = span
    forward = t1 >= t0
    sgn = 1.0 if forward else -1.0

    y = [float(v) for v in y0]
    n = len(y)
    t = float(t0)

    h_max = max_step if max_step is not None else abs(t1 - t0)
    h_min = 1e-14
    h0 = first_step if first_step is not None else max(h_min,
                                                       min(1e-2, h_max))
    h = h0 * sgn

    # Count nfev
    counter = [0]
    def ff(tt, yy):
        counter[0] += 1
        return list(f(tt, yy))

    # Evaluate events once at t0
    prev_g = [float(ev(t, y)) for ev in events] if events else []

    T_out = [t]
    Y_out = [list(y)]
    t_events = [[] for _ in events]
    y_events = [[] for _ in events]

    status = "max_lambda"
    accepted = 0
    safety, min_scale, max_scale = 0.9, 0.2, 5.0

    def done(tt):
        return tt >= t1 if forward else tt <= t1

    while not done(t) and accepted < max_steps:
        # Clamp step so we don't overshoot t1
        h = max(h_min, min(abs(h), h_max)) * sgn
        if forward and t + h > t1:
            h = t1 - t
        if (not forward) and t + h < t1:
            h = t1 - t

        # Stages
        k = [None] * 7
        k[0] = ff(t, y)
        if not all(isfinite(v) for v in k[0]):
            status = "nonfinite"
            break
        for i in range(1, 7):
            yi = list(y)
            for j, aij in enumerate(_DP_A[i], start=1):
                for m in range(n):
                    yi[m] += h * aij * k[j-1][m]
            k[i] = ff(t + _DP_C[i] * h, yi)

        y5 = list(y)
        y4 = list(y)
        for i in range(7):
            for m in range(n):
                y5[m] += h * _DP_B5[i] * k[i][m]
                y4[m] += h * _DP_B4[i] * k[i][m]

        # Error norm
        acc = 0.0
        for m in range(n):
            sc = atol + rtol * max(abs(y[m]), abs(y5[m]))
            acc += ((y5[m] - y4[m]) / sc) ** 2
        en = sqrt(acc / n)

        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            # Accept
            t_new = t + h
            y_new = y5

            if not all(isfinite(v) for v in y_new):
                status = "nonfinite"
                break

            # Event detection on the accepted interval [t, t_new]
            event_hit = None
            if events:
                # Cubic Hermite interpolation using k[0] (dy at t) and
                # k[6] (dy at t_new, thanks to FSAL of Dormand-Prince).
                k0 = k[0]
                k1 = k[6]
                def interp(s, y=y, y_new=y_new, k0=k0, k1=k1, h=h, n=n):
                    h00 = 2*s**3 - 3*s**2 + 1
                    h10 = s**3 - 2*s**2 + s
                    h01 = -2*s**3 + 3*s**2
                    h11 = s**3 - s**2
                    return [h00*y[m] + h10*h*k0[m] + h01*y_new[m] + h11*h*k1[m]
                            for m in range(n)]

                new_g = [float(ev(t_new, y_new)) for ev in events]
                for idx, (g_old, g_new, ev) in enumerate(zip(prev_g, new_g, events)):
                    direction = getattr(ev, "direction", 0)
                    # sign change?
                    if g_old * g_new < 0.0:
                        if direction == 1 and g_old >= 0.0:
                            continue
                        if direction == -1 and g_old <= 0.0:
                            continue
                        # Refine with brentq on the interpolant
                        def root_fn(s, ev=ev):
                            ys = interp(s)
                            return float(ev(t + s * (t_new - t), ys))
                        try:
                            s_star = brentq(root_fn, 0.0, 1.0,
                                            xtol=1e-12, rtol=1e-12)
                        except ValueError:
                            s_star = 1.0
                        t_star = t + s_star * (t_new - t)
                        y_star = interp(s_star)
                        t_events[idx].append(t_star)
                        y_events[idx].append(y_star)
                        if getattr(ev, "terminal", True):
                            event_hit = (idx, t_star, y_star)
                            break

            if event_hit is not None:
                idx, t_star, y_star = event_hit
                T_out.append(t_star)
                Y_out.append(y_star)
                ev = events[idx]
                status = getattr(ev, "name", f"event_{idx}")
                t = t_star
                y = y_star
                break

            T_out.append(t_new)
            Y_out.append(y_new)
            accepted += 1
            t = t_new
            y = y_new
            if events:
                prev_g = new_g

            # Step size update
            if en == 0.0:
                sc = max_scale
            else:
                sc = safety * (1.0 / en) ** 0.2
                sc = min(max(sc, min_scale), max_scale)
            h = max(h_min, min(abs(h) * sc, h_max)) * sgn
        else:
            sc = safety * (1.0 / en) ** 0.25
            sc = min(max(sc, min_scale), 1.0)
            h = max(h_min, min(abs(h) * sc, h_max)) * sgn

    return IntegrationResult(
        t=array(T_out),
        y=array(Y_out),
        t_events=[array(te) for te in t_events],
        y_events=[array(ye) if ye else array([]).reshape(0, n)
                  for ye in y_events],
        status=status,
        nfev=counter[0],
        wall_time=0.0,
        method="RK45",
        success=True,
    )


# ----------------------------------------------------------------------------
# Verlet / Yoshida-style splitting (experimental, fixed step)
# ----------------------------------------------------------------------------
def _solve_verlet(f, y0, span, events, max_step, dense, first_step,
                  max_steps):
    """
    Second-order splitting that treats the full RHS symmetrically:
        y_{n+1/2} = y_n + (h/2) * f(t_n,       y_n)
        y_{n+1}   = y_{n+1/2} + (h/2) * f(t_{n+1}, y_{n+1/2})  (approx)

    For non-separable Hamiltonians this is not strictly symplectic, but
    preserves H better than RK45 at equal step count in many cases.
    """
    t0, t1 = span
    forward = t1 >= t0
    sgn = 1.0 if forward else -1.0

    y = list(asarray(y0, dtype=float))
    n = len(y)
    t = float(t0)

    if first_step is not None:
        h = abs(first_step) * sgn
    elif max_step is not None:
        h = abs(max_step) * sgn
    else:
        h = (t1 - t0) / 10_000

    counter = [0]
    def ff(tt, yy):
        counter[0] += 1
        return list(f(tt, yy))

    prev_g = [float(ev(t, y)) for ev in events] if events else []
    T_out = [t]
    Y_out = [list(y)]
    t_events = [[] for _ in events]
    y_events = [[] for _ in events]
    status = "max_lambda"
    accepted = 0

    def done(tt):
        return tt >= t1 if forward else tt <= t1

    while not done(t) and accepted < max_steps:
        if forward and t + h > t1:
            h = t1 - t
        if (not forward) and t + h < t1:
            h = t1 - t

        # Half-step with RHS at (t, y)
        k1 = ff(t, y)
        if not all(isfinite(v) for v in k1):
            status = "nonfinite"
            break
        y_mid = [y[m] + 0.5 * h * k1[m] for m in range(n)]

        # Second half-step using mid-point RHS (Strang-like)
        k2 = ff(t + 0.5 * h, y_mid)
        if not all(isfinite(v) for v in k2):
            status = "nonfinite"
            break
        y_new = [y[m] + h * k2[m] for m in range(n)]
        t_new = t + h

        # Event detection
        event_hit = None
        if events:
            def interp(s):
                return [y[m] + s * (y_new[m] - y[m]) for m in range(n)]

            new_g = [float(ev(t_new, y_new)) for ev in events]
            for idx, (g_old, g_new, ev) in enumerate(zip(prev_g, new_g, events)):
                direction = getattr(ev, "direction", 0)
                if g_old * g_new < 0.0:
                    if direction == 1 and g_old >= 0.0:
                        continue
                    if direction == -1 and g_old <= 0.0:
                        continue
                    def root_fn(s, ev=ev):
                        ys = interp(s)
                        return float(ev(t + s * h, ys))
                    try:
                        s_star = brentq(root_fn, 0.0, 1.0,
                                        xtol=1e-12, rtol=1e-12)
                    except ValueError:
                        s_star = 1.0
                    t_star = t + s_star * h
                    y_star = interp(s_star)
                    t_events[idx].append(t_star)
                    y_events[idx].append(y_star)
                    if getattr(ev, "terminal", True):
                        event_hit = (idx, t_star, y_star)
                        break

        if event_hit is not None:
            idx, t_star, y_star = event_hit
            T_out.append(t_star)
            Y_out.append(y_star)
            status = getattr(events[idx], "name", f"event_{idx}")
            break

        T_out.append(t_new)
        Y_out.append(y_new)
        t, y = t_new, y_new
        if events:
            prev_g = new_g
        accepted += 1

    return IntegrationResult(
        t=array(T_out),
        y=array(Y_out),
        t_events=[array(te) for te in t_events],
        y_events=[array(ye) if ye else array([]).reshape(0, n)
                  for ye in y_events],
        status=status,
        nfev=counter[0],
        wall_time=0.0,
        method="Verlet",
        success=True,
    )


# ----------------------------------------------------------------------------
# Standard event factory for BH geodesics
# ----------------------------------------------------------------------------
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
        # Pure equatorial crossing, NON-TERMINAL: record every sign change
        # of cos(θ) so the caller can pick the first crossing whose radius
        # falls inside the disk annulus. Terminal + annulus-check kills the
        # photon ring because ring photons spiral through the photon sphere
        # (r ≈ 3) and cross the equator there first — outside the annulus.
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

    # 1. Simple ODE y' = -y against exp(-t)
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

    # 2. Event: simple falling object hitting ground
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
