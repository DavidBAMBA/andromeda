"""
Numba-compiled kernels for geodesic integration and image rendering.

All functions are @njit — no Python overhead at call time.
Imported by integrator.py and (indirectly) parallel.py.
"""
from math import cos as mcos, sin as msin, exp as mexp, sqrt as msqrt

import numpy as np
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


# ============================================================================
# Hermite interpolation helpers
# ============================================================================

@njit(cache=True, fastmath=False)
def _hermite_coef(s):
    '''Cubic Hermite basis functions at parameter s ∈ [0,1].

    Returns (h00, h10, h01, h11) for interpolation:
        y(s) = h00*y0 + h10*h*dy0 + h01*y1 + h11*h*dy1
    '''
    s2 = s * s
    s3 = s2 * s
    return (
        2.0*s3 - 3.0*s2 + 1.0,
        s3 - 2.0*s2 + s,
        -2.0*s3 + 3.0*s2,
        s3 - s2
    )


@njit(cache=True, fastmath=False)
def _hermite_interp(y0, dy0, y1, dy1, h, s, n):
    '''Cubic Hermite interpolation at fractional step s ∈ [0,1].'''
    h00, h10, h01, h11 = _hermite_coef(s)
    y = np.empty(n)
    for m in range(n):
        y[m] = h00*y0[m] + h10*h*dy0[m] + h01*y1[m] + h11*h*dy1[m]
    return y


# ============================================================================
# Event bisection
# ============================================================================

@njit(cache=True, fastmath=False)
def _bisect_r_threshold(y0, dy0, y1, dy1, h, g_lo, threshold, n):
    '''Bisect where y[1] - threshold crosses zero (horizon or escape).'''
    s_lo, s_hi = 0.0, 1.0
    for _ in range(30):
        s_mid = 0.5 * (s_lo + s_hi)
        y_mid = _hermite_interp(y0, dy0, y1, dy1, h, s_mid, n)
        g_mid = y_mid[1] - threshold
        if g_mid * g_lo < 0.0:
            s_hi = s_mid
        else:
            s_lo = s_mid
            g_lo = g_mid
    return _hermite_interp(y0, dy0, y1, dy1, h, 0.5*(s_lo + s_hi), n)


@njit(cache=True, fastmath=False)
def _bisect_disk(y0, dy0, y1, dy1, h, g_lo, n):
    '''Bisect where cos(theta) crosses zero (equatorial plane).'''
    s_lo, s_hi = 0.0, 1.0
    for _ in range(30):
        s_mid = 0.5 * (s_lo + s_hi)
        y_mid = _hermite_interp(y0, dy0, y1, dy1, h, s_mid, n)
        g_mid = mcos(y_mid[2])
        if g_mid * g_lo < 0.0:
            s_hi = s_mid
        else:
            s_lo = s_mid
            g_lo = g_mid
    return _hermite_interp(y0, dy0, y1, dy1, h, 0.5*(s_lo + s_hi), n)


@njit(cache=True, fastmath=False)
def _check_horizon_event(y0, dy0, y1, dy1, h, horizon_prev, horizon_new,
                         r_hor, eps_horizon, n):
    '''Check and refine horizon crossing (terminal, inward direction).'''
    if horizon_prev > 0.0 and horizon_new <= 0.0:
        y_event = _bisect_r_threshold(y0, dy0, y1, dy1, h, horizon_prev,
                                      r_hor + eps_horizon, n)
        return True, y_event
    return False, np.empty(n)


@njit(cache=True, fastmath=False)
def _check_escape_event(y0, dy0, y1, dy1, h, escape_prev, escape_new,
                        r_esc, n):
    '''Check and refine escape crossing (terminal, outward direction).'''
    if escape_prev < 0.0 and escape_new >= 0.0:
        y_event = _bisect_r_threshold(y0, dy0, y1, dy1, h, escape_prev,
                                      r_esc, n)
        return True, y_event
    return False, np.empty(n)


@njit(cache=True, fastmath=False)
def _check_disk_event(y0, dy0, y1, dy1, h, disk_prev, disk_new, n):
    '''Check and refine equatorial crossing (non-terminal).'''
    if disk_prev * disk_new < 0.0:
        y_event = _bisect_disk(y0, dy0, y1, dy1, h, disk_prev, n)
        return True, y_event
    return False, np.empty(n)


# ============================================================================
# RK45 photon integrator
# ============================================================================

@njit(cache=True, fastmath=False)
def _solve_photon_nb(rhs, y0, lmbda_end, r_hor, r_esc, rtol, atol):
    '''Numba-compiled RK45 (Dormand-Prince) with inline event detection.

    Events: horizon (terminal, direction=-1), escape (terminal, direction=+1),
    disk (non-terminal cos(theta) sign changes, up to MAX_DISK_HITS_NB).

    Returns
    -------
    status   : int  (0=max_lambda, 1=horizon, 2=escape, -1=nonfinite)
    y_final  : float64[8]
    n_hits   : int
    y_hits   : float64[MAX_DISK_HITS_NB, 8]
    '''
    n = 8
    y = y0.copy()
    t0, t1 = 0.0, float(lmbda_end)
    sgn = 1.0 if t1 >= t0 else -1.0
    forward = sgn > 0.0

    h_max = abs(t1 - t0)
    h_min = 1e-14
    h = max(h_min, min(1e-2, h_max)) * sgn

    safety, min_scale, max_scale = 0.9, 0.2, 5.0
    max_steps = 1_000_000
    eps_horizon = 1e-3

    # Dormand-Prince RK45 Butcher tableau
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
    y_tmp = np.empty(n)
    y_new = np.empty(n)

    # Event values at start
    horizon_prev = y[1] - (r_hor + eps_horizon)
    escape_prev = y[1] - r_esc
    disk_prev = mcos(y[2])

    # First RHS evaluation (FSAL reuse)
    k1 = rhs(y)
    for m in range(n):
        if not np.isfinite(k1[m]):
            return -1, y, 0, y_hits

    t, accepted = t0, 0
    while accepted < max_steps:
        # Clamp step, don't overshoot t1
        if abs(h) < h_min:
            h = h_min * sgn
        elif abs(h) > h_max:
            h = h_max * sgn
        if forward and t + h > t1:
            h = t1 - t
        if (not forward) and t + h < t1:
            h = t1 - t

        if (forward and t >= t1) or ((not forward) and t <= t1):
            status = 0
            break

        # RK45 stages (k1 from FSAL)
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

        # 5th-order solution
        for m in range(n):
            y_new[m] = y[m] + h * (A71*k1[m] + A73*k3[m] + A74*k4[m] + A75*k5[m] + A76*k6[m])

        # 7th stage for FSAL and error estimate
        k7 = rhs(y_new)

        # Error norm
        err_acc = 0.0
        ok_finite = True
        for m in range(n):
            if not np.isfinite(y_new[m]):
                ok_finite = False
                break
            y4 = y[m] + h * (B41*k1[m] + B43*k3[m] + B44*k4[m] + B45*k5[m] + B46*k6[m] + B47*k7[m])
            sc = atol + rtol * max(abs(y[m]), abs(y_new[m]))
            err_acc += ((y_new[m] - y4) / sc) ** 2

        if not ok_finite:
            return -1, y, n_hits, y_hits

        en = (err_acc / n) ** 0.5

        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            horizon_new = y_new[1] - (r_hor + eps_horizon)
            escape_new = y_new[1] - r_esc
            disk_new = mcos(y_new[2])

            event_kind = 0

            hit_h, y_h = _check_horizon_event(y, k1, y_new, k7, h,
                                              horizon_prev, horizon_new,
                                              r_hor, eps_horizon, n)
            if hit_h:
                event_kind = 1
                y[:] = y_h

            if event_kind == 0:
                hit_e, y_e = _check_escape_event(y, k1, y_new, k7, h,
                                                 escape_prev, escape_new,
                                                 r_esc, n)
                if hit_e:
                    event_kind = 2
                    y[:] = y_e

            if event_kind == 0 and disk_prev * disk_new < 0.0 and n_hits < MAX_DISK_HITS_NB:
                hit_d, y_d = _check_disk_event(y, k1, y_new, k7, h,
                                               disk_prev, disk_new, n)
                if hit_d:
                    y_hits[n_hits, :] = y_d
                    n_hits += 1

            if event_kind == 1:
                status = 1
                break
            if event_kind == 2:
                status = 2
                break

            # Commit accepted step (FSAL: k7 → next k1)
            t = t + h
            y[:] = y_new
            k1[:] = k7
            horizon_prev = horizon_new
            escape_prev = escape_new
            disk_prev = disk_new
            accepted += 1

            sc_h = max_scale if en == 0.0 else max(min_scale, min(safety * (1.0/en)**0.2, max_scale))
            h = np.sign(h) * max(h_min, min(abs(h) * sc_h, h_max))
        else:
            sc_h = max(min_scale, min(safety * (1.0/en)**0.25, 1.0))
            h = np.sign(h) * max(h_min, min(abs(h) * sc_h, h_max))

    return status, y, n_hits, y_hits


# ============================================================================
# Asymptotic extrapolation to a source plane (gravitational-lensing mode)
# ============================================================================

@njit(cache=True, fastmath=False)
def _eval_profile_nb_kernel(xs, ys, kind, params):
    '''In-kernel copy of sources.light_profiles._eval_profile_nb.

    Kept here so _compute_pixel_nb remains a single @njit callable without
    cross-module first-class function plumbing. Must stay in sync with
    scr/sources/light_profiles.py.
    '''
    if kind == 0:       # Gaussian
        x0 = params[0]; y0p = params[1]
        sigma = params[2]; I0 = params[3]
        dx = xs - x0; dy = ys - y0p
        return I0 * mexp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))
    elif kind == 1:     # Sersic
        x0 = params[0]; y0p = params[1]
        Re = params[2]; n = params[3]
        Ie = params[4]; bn = params[5]
        ell = params[6]; pa = params[7]
        c = mcos(pa); s = msin(pa)
        dx = xs - x0; dy = ys - y0p
        xr =  c*dx + s*dy
        yr = -s*dx + c*dy
        q = 1.0 - ell
        R = msqrt(xr*xr + (yr/q)*(yr/q))
        if R == 0.0:
            return Ie * mexp(bn)
        return Ie * mexp(-bn * ((R/Re) ** (1.0/n) - 1.0))
    return 0.0


@njit(cache=True, fastmath=False)
def _extrapolate_to_source_plane_nb(rhs, y_final, D_LS):
    '''Straight-line extrapolation from y_final to the source plane at
    x = -D_LS.

    Geometry (iota = pi/2): the detector is in parallel-projection mode; the
    observer is on the +x axis at distance D_L and pixels are arranged on
    a plane at constant x = D_L, so every photon exits the lens region with
    v_x < 0 (heading toward the source). The source plane is perpendicular
    to the observer-lens axis at x = -D_LS. Coordinates on the source plane
    are (y_src, z_src) returned here as (x_src, y_src) for symmetry with
    the detector's (alpha, beta) ordering.

    The spacetime is asymptotically flat in the escape region, so
    straight-line extrapolation in lens-frame Cartesian is accurate.

    The integrator traces backward (lambda decreasing), so rhs(y_final)
    points opposite to the physical trajectory. We therefore negate the
    Cartesian velocity to continue the trace onward to the source plane.

    Returns
    -------
    ok : bool
    x_src, y_src : float
        (y, z) Cartesian on the source plane when ok; (0, 0) otherwise.
    '''
    r = y_final[1]; th = y_final[2]; ph = y_final[3]
    sth = msin(th); cth = mcos(th)
    sph = msin(ph); cph = mcos(ph)

    xf = r * sth * cph
    yf = r * sth * sph
    zf = r * cth

    dq = rhs(y_final)
    dr_ = dq[1]; dth_ = dq[2]; dph_ = dq[3]

    # Cartesian velocity in the backward-lambda direction (trajectory onward
    # to the source plane).
    vx = -(dr_ * sth * cph + r * cth * cph * dth_ - r * sth * sph * dph_)
    vy = -(dr_ * sth * sph + r * cth * sph * dth_ + r * sth * cph * dph_)
    vz = -(dr_ * cth       - r * sth     * dth_)

    # Intersect with the source plane at x = -D_LS.
    if vx >= 0.0:
        return False, 0.0, 0.0
    s = (-D_LS - xf) / vx
    if s < 0.0:
        return False, 0.0, 0.0
    return True, yf + s * vy, zf + s * vz


# ============================================================================
# Per-pixel pipeline and parallel render kernel
# ============================================================================

@njit(cache=False, fastmath=False)
def _compute_pixel_nb(rhs, metric, omega,
                     y0, lmbda_end, r_hor, r_esc,
                     r_tbl, I_tbl, in_edge, out_edge,
                     rtol, atol, mode_code,
                     profile_kind, profile_params, D_LS):
    '''Full per-pixel pipeline: integrate, pick first in-annulus hit,
    return scalar intensity (with Doppler if mode_code==1).

    mode_code: 0=no_doppler, 1=doppler, 2=shadow, 3=lensing.

    For mode_code == 3, r_tbl/I_tbl/in_edge/out_edge are ignored and
    (profile_kind, profile_params, D_LS) determine the source-plane brightness.
    '''
    status, y_final, n_hits, y_hits = _solve_photon_nb(
        rhs, y0, lmbda_end, r_hor, r_esc, rtol, atol)

    if mode_code == 2:
        return 0.0 if status == 1 else 100.0

    if mode_code == 3:
        if status != 2:
            return 0.0
        ok, xs, ys = _extrapolate_to_source_plane_nb(rhs, y_final, D_LS)
        if not ok:
            return 0.0
        return _eval_profile_nb_kernel(xs, ys, profile_kind, profile_params)

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
                     rtol, atol, mode_code,
                     profile_kind, profile_params, D_LS):
    '''Parallel-over-photons render kernel. Each iteration is independent.'''
    n = y0_batch.shape[0]
    values = np.zeros(n)
    for idx in prange(n):
        values[idx] = _compute_pixel_nb(
            rhs, metric, omega,
            y0_batch[idx], lmbda_end, r_hor, r_esc,
            r_tbl, I_tbl, in_edge, out_edge,
            rtol, atol, mode_code,
            profile_kind, profile_params, D_LS)
    return values
