from math import sqrt, isfinite

def rk45(f, t0, y0, t1, *, atol=1e-9, rtol=1e-9, h0=1e-2, h_min=1e-12, h_max=1.0,
         max_steps=1_000_000, t_eval=None):
    """
    Integrate y'(t) = f(t, y) from t0 to t1 using adaptive Dormand–Prince RK45.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) -> dy with the same shape as y.
    t0 : float
        Initial time.
    y0 : sequence of floats
        Initial state vector.
    t1 : float
        Final time (can be less than t0 for backward integration).
    atol, rtol : float
        Absolute and relative tolerances for adaptive control.
    h0 : float
        Initial step size guess (positive magnitude; direction inferred).
    h_min, h_max : float
        Minimum and maximum allowed step magnitudes.
    max_steps : int
        Safety cap on the number of *accepted* steps.
    t_eval : None or sequence of floats
        If provided, returns solution interpolated at these times (must be
        monotonic and within [min(t0, t1), max(t0, t1)]).

    Returns
    -------
    T, Y : list[float], list[list[float]]
        If t_eval is None:
            T contains the accepted step times (including t0 and t1),
            Y contains the corresponding states.
        If t_eval is provided:
            T == list(t_eval), Y contains states at those times.

    Notes
    -----
    - Uses Dormand–Prince tableau (same core method as SciPy's RK45).
    - Simple step-by-step integration; when t_eval is provided, the solver
      advances internally and performs *piecewise*-constant interpolation to
      the last accepted state before/at each query (good enough for many uses).
      For high-accuracy dense output, a Hermite or DP dense interpolant can be added.
    """
    # --- helpers for vector operations without requiring NumPy ---
    def to_list(y):
        try:
            # Supports lists, tuples, numpy arrays
            return [float(yi) for yi in y]
        except TypeError:
            return [float(y)]
    def add(a, b):  # a+b
        return [ai + bi for ai, bi in zip(a, b)]
    def axpy(a, x, y):  # a*x + y
        return [a*xi + yi for xi, yi in zip(x, y)]
    def scale(s, x):
        return [s*xi for xi in x]
    def err_norm(y_old, y_new, err):
        acc = 0.0
        for i in range(n):
            sc = atol + rtol * max(abs(y_old[i]), abs(y_new[i]))
            acc += (err[i] / sc) ** 2
        return sqrt(acc / n)

    # Direction & initializations
    y = to_list(y0)
    n = len(y)
    t = float(t0)
    t_end = float(t1)
    forward = (t_end >= t)
    sgn = 1.0 if forward else -1.0

    # Validate t_eval
    using_eval = t_eval is not None
    if using_eval:
        T_eval = list(t_eval)
        if (forward and any(T_eval[i] > T_eval[i+1] for i in range(len(T_eval)-1))) or \
           ((not forward) and any(T_eval[i] < T_eval[i+1] for i in range(len(T_eval)-1))):
            raise ValueError("t_eval must be monotonic in the integration direction.")
        lo, hi = (t, t_end) if forward else (t_end, t)
        if any((ti < lo) or (ti > hi) for ti in T_eval):
            raise ValueError("All t_eval points must lie within [min(t0,t1), max(t0,t1)].")
        Ti, Yi = [], []

    # Dormand–Prince coefficients
    C = (0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0)
    A = (
        (),
        (1/5,),
        (3/40, 9/40),
        (44/45, -56/15, 32/9),
        (19372/6561, -25360/2187, 64448/6561, -212/729),
        (9017/3168,  -355/33,     46732/5247,  49/176, -5103/18656),
        (35/384,     0.0,         500/1113,    125/192, -2187/6784, 11/84),
    )
    B5 = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)  # 5th order
    B4 = (5179/57600, 0.0, 7571/16695, 393/640,
          -92097/339200, 187/2100, 1/40)  # embedded 4th

    # Step-size control params
    safety = 0.9
    min_scale, max_scale = 0.2, 5.0
    h = max(h_min, min(h0, h_max)) * sgn

    # Storage for raw solution if no t_eval
    T = [t]
    Y = [y[:]]

    accepted_steps = 0

    # Function to decide if we've reached/passed t_end
    def done(t, h):
        return (t >= t_end) if forward else (t <= t_end)

    # Main loop
    while not done(t, h) and accepted_steps < max_steps:
        # Clamp final step to land exactly on t_end
        h = max(h_min, min(abs(h), h_max)) * sgn
        if forward:
            if t + h > t_end:
                h = t_end - t
        else:
            if t + h < t_end:
                h = t_end - t

        # Stage derivatives k_i
        k = [None] * 7
        k[0] = to_list(f(t, y))
        if not all(isfinite(ki) for ki in k[0]):
            raise FloatingPointError("Non-finite derivative encountered at initial stage.")

        def stage(i):
            yi = y[:]
            for j, aij in enumerate(A[i], start=1):
                yi = axpy(h * aij, k[j-1], yi)
            return t + C[i] * h, yi

        for i in range(1, 7):
            ti, yi = stage(i)
            k[i] = to_list(f(ti, yi))

        # Candidate solutions (5th and 4th order)
        y5 = y[:]
        y4 = y[:]
        for i in range(7):
            y5 = axpy(h * B5[i], k[i], y5)
            y4 = axpy(h * B4[i], k[i], y4)

        # Error estimate
        err = [y5[i] - y4[i] for i in range(n)]
        en = err_norm(y, y5, err)

        # Accept/reject step
        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            # Accept
            t_new = t + h
            y_new = y5
            accepted_steps += 1

            # Record output
            if using_eval:
                # Push any t_eval points passed by this accepted step,
                # using piecewise-constant y_new (simple but stable).
                while T_eval and ((forward and T_eval[0] <= t_new) or ((not forward) and T_eval[0] >= t_new)):
                    Ti.append(T_eval.pop(0))
                    Yi.append(y_new[:])
            else:
                T.append(t_new)
                Y.append(y_new[:])

            # Prepare next step
            # Scale factor ~ (1/en)^(1/5)
            if en == 0.0:
                scale = max_scale
            else:
                scale = safety * (1.0 / en) ** 0.2
                scale = min(max(scale, min_scale), max_scale)
            h = max(h_min, min(abs(h) * scale, h_max)) * sgn

            t, y = t_new, y_new
        else:
            # Reject and shrink step
            scale = safety * (1.0 / en) ** 0.25
            scale = min(max(scale, min_scale), 1.0)
            h = max(h_min, min(abs(h) * scale, h_max)) * sgn

    # Finalization / output packaging
    if using_eval:
        # Ensure we output exactly the requested evaluation times.
        # If some points lie after the final accepted step due to tight tolerances,
        # fill them with the last known state y (at t_end).
        while T_eval:
            Ti.append(T_eval.pop(0))
            Yi.append(y[:])
        return Ti, Yi
    else:
        # Ensure last point is exactly t1
        if T[-1] != t_end:
            T.append(t_end)
            Y.append(y[:])
        return T, Y


# --------------------------- Small usage example ---------------------------
if __name__ == "__main__":
    # Solve y' = -y, y(0)=1 on [0, 5]; exact y=exp(-t)
    import math

    def f(t, y):
        return [-y[0]]

    T, Y = rk45(f, 0.0, [1.0], 5.0, atol=1e-10, rtol=1e-10, h0=0.1)
    err = abs(Y[-1][0] - math.exp(-T[-1]))
    print(f"y(5) ≈ {Y[-1][0]:.12f}, exact={math.exp(-5):.12f}, abs err={err:.2e}, steps={len(T)-1}")