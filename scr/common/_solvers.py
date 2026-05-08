"""
Python-based ODE backends for the geodesic integrator.

Three solvers:
    _solve_scipy  — wraps scipy.integrate.solve_ivp (LSODA, DOP853, RK45)
    _solve_rk45   — in-house adaptive Dormand-Prince with Brent event refinement
    _solve_verlet — 2nd-order Strang splitting (experimental, fixed step)

All return an IntegrationResult.
"""
from math import sqrt, isfinite

import numpy as np
from numpy import asarray, array, isfinite as np_isfinite, cos
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


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
# SciPy backend (LSODA, DOP853, RK45)
# ----------------------------------------------------------------------------

def _solve_scipy(f, y0, span, method, events, rtol, atol, max_step, first_step):
    kwargs = dict(method=method, rtol=rtol, atol=atol, dense_output=False)
    if max_step is not None:
        kwargs["max_step"] = max_step
    if first_step is not None:
        kwargs["first_step"] = first_step
    if events:
        kwargs["events"] = events

    sol = solve_ivp(f, span, asarray(y0, dtype=float), **kwargs)

    status = "max_lambda"
    if sol.status == 1:
        for i, te in enumerate(sol.t_events):
            if te.size > 0:
                status = getattr(events[i], "name", f"event_{i}")
                break
    elif sol.status < 0:
        status = "error"

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


def _solve_rk45(f, y0, span, events, rtol, atol, max_step, first_step, max_steps):
    t0, t1 = span
    forward = t1 >= t0
    sgn = 1.0 if forward else -1.0

    y = [float(v) for v in y0]
    n = len(y)
    t = float(t0)

    h_max = max_step if max_step is not None else abs(t1 - t0)
    h_min = 1e-14
    h0 = first_step if first_step is not None else max(h_min, min(1e-2, h_max))
    h = h0 * sgn

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
    safety, min_scale, max_scale = 0.9, 0.2, 5.0

    def done(tt):
        return tt >= t1 if forward else tt <= t1

    while not done(t) and accepted < max_steps:
        h = max(h_min, min(abs(h), h_max)) * sgn
        if forward and t + h > t1:
            h = t1 - t
        if (not forward) and t + h < t1:
            h = t1 - t

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

        acc = 0.0
        for m in range(n):
            sc = atol + rtol * max(abs(y[m]), abs(y5[m]))
            acc += ((y5[m] - y4[m]) / sc) ** 2
        en = sqrt(acc / n)

        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            t_new = t + h
            y_new = y5

            if not all(isfinite(v) for v in y_new):
                status = "nonfinite"
                break

            event_hit = None
            if events:
                k0, k1 = k[0], k[6]
                interp = lambda s: [
                    (2*s**3 - 3*s**2 + 1)*y[m] + (s**3 - 2*s**2 + s)*h*k0[m]
                    + (-2*s**3 + 3*s**2)*y_new[m] + (s**3 - s**2)*h*k1[m]
                    for m in range(n)
                ]

                new_g = [float(ev(t_new, y_new)) for ev in events]
                for idx, (g_old, g_new, ev) in enumerate(zip(prev_g, new_g, events)):
                    direction = getattr(ev, "direction", 0)
                    if g_old * g_new >= 0.0:
                        continue
                    if direction == 1 and g_old >= 0.0:
                        continue
                    if direction == -1 and g_old <= 0.0:
                        continue
                    try:
                        s_star = brentq(
                            lambda s, ev=ev: float(ev(t + s*(t_new - t), interp(s))),
                            0.0, 1.0, xtol=1e-12, rtol=1e-12
                        )
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
                status = getattr(events[idx], "name", f"event_{idx}")
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
        y_events=[array(ye) if ye else array([]).reshape(0, n) for ye in y_events],
        status=status,
        nfev=counter[0],
        wall_time=0.0,
        method="RK45",
        success=True,
    )


# ----------------------------------------------------------------------------
# Verlet / Strang splitting (experimental, fixed step)
# ----------------------------------------------------------------------------

def _solve_verlet(f, y0, span, events, max_step, first_step, max_steps):
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

        k1 = ff(t, y)
        if not all(isfinite(v) for v in k1):
            status = "nonfinite"
            break
        y_mid = [y[m] + 0.5 * h * k1[m] for m in range(n)]

        k2 = ff(t + 0.5 * h, y_mid)
        if not all(isfinite(v) for v in k2):
            status = "nonfinite"
            break
        y_new = [y[m] + h * k2[m] for m in range(n)]
        t_new = t + h

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
                    try:
                        s_star = brentq(
                            lambda s, ev=ev: float(ev(t + s * h, interp(s))),
                            0.0, 1.0, xtol=1e-12, rtol=1e-12
                        )
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
        y_events=[array(ye) if ye else array([]).reshape(0, n) for ye in y_events],
        status=status,
        nfev=counter[0],
        wall_time=0.0,
        method="Verlet",
        success=True,
    )
