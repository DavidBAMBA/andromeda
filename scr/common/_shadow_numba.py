"""
===============================================================================
Generic Numba shadow renderer -- one integrator suite for every spacetime
===============================================================================
The integrator math (embedded Runge-Kutta, Bulirsch-Stoer, Verlet) is identical
for every metric; only the geodesic RHS, the Hamiltonian |H|, and the
image-plane initial conditions are spacetime-specific.

This module factors that out.  ``build_kernels(rhs, absH, ic)`` returns a set of
thread-parallel renderers (``rk`` / ``bs`` / ``verlet``) in which the three
spacetime functions are captured as closure freevars -- so Numba still inlines
them and the per-pixel throughput matches a hand-specialised kernel.

A spacetime "provider" supplies those three functions plus a small dict of
integration defaults:

    rhs(q, out)                         -- in-place geodesic RHS (no allocation)
    absH(q)                             -- |H| constraint residual
    ic(alpha, beta, D, sin_i, cos_i, q) -- image-plane -> state (exact null k_r)

Metric parameters (e.g. the Kerr spin ``a``) are captured by the closure, so the
``ic`` signature is uniform across spacetimes.  Built kernels are memoised per
(spacetime, params) -- a new spin recompiles once, then is cached.

The thin wrappers ``_geo_numba.render_shadow_numba`` and
``_kerr_numba.render_kerr_shadow`` delegate here; ex24-ex28 go through them.
===============================================================================
"""

import math
import time
import numpy as np
from numba import njit, prange, set_num_threads, config

from scr.common.integrator import _RK_TABLEAUX

NV = 8                      # state dimension [t, r, th, phi, kt, kr, kth, kphi]
_BS_SEQ = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=np.float64)

_METHOD_ALIASES = {"RKDP45": "DP45", "RKCK45": "CK45", "RKF45": "RKF45",
                   "DP45": "DP45", "CK45": "CK45", "BS": "BS",
                   "RK45": "DP45", "Verlet": "Verlet"}


def _tableau_arrays(method):
    tab = _RK_TABLEAUX[method]
    s = len(tab["c"])
    A = np.zeros((s, s))
    for i, row in enumerate(tab["A"]):
        for j, v in enumerate(row):
            A[i, j] = v
    c = np.asarray(tab["c"], float)
    b5 = np.asarray(tab["b5"], float)
    b4 = np.asarray(tab["b4"], float)
    return A, c, b5, b4, s


# ===========================================================================
# Kernel factory -- spacetime functions captured as closure freevars
# ===========================================================================
def build_kernels(rhs, absH, ic):
    """Build (rk / bs / verlet) thread-parallel renderers for one spacetime.

    ``rhs(q, out)``, ``absH(q)`` and ``ic(...)`` are njit functions; they are
    captured here as freevars so Numba inlines them into the pixel loops.
    Closures cannot use ``cache=True`` (numba limitation), so these compile once
    per process -- absorbed by ``warmup()``.
    """

    # -- Embedded RK pixel integrator (tableau passed as arrays) -------------
    @njit(fastmath=True)
    def _rk_pixel(q, A, c, b5, b4, s, lam_max, atol, rtol, EH_stop, R_esc, h0,
                  h_max, max_steps, k, ytmp, y5, y4, y):
        for d in range(NV):
            y[d] = q[d]
        sgn = -1.0                     # trace backward in the affine parameter
        h = sgn * h0
        lam = 0.0
        h_min = 1e-9
        safety = 0.9
        steps = 0
        while lam > -lam_max and steps < max_steps:
            if abs(h) < h_min:
                h = sgn * h_min
            elif abs(h) > h_max:
                h = sgn * h_max
            if lam + h < -lam_max:
                h = -lam_max - lam

            rhs(y, k[0])
            for ist in range(1, s):
                for d in range(NV):
                    ytmp[d] = y[d]
                for jj in range(ist):
                    aij = A[ist, jj]
                    if aij != 0.0:
                        ha = h * aij
                        for d in range(NV):
                            ytmp[d] += ha * k[jj, d]
                rhs(ytmp, k[ist])

            err = 0.0
            for d in range(NV):
                acc5 = 0.0
                acc4 = 0.0
                for ist in range(s):
                    acc5 += b5[ist] * k[ist, d]
                    acc4 += b4[ist] * k[ist, d]
                y5[d] = y[d] + h * acc5
                y4[d] = y[d] + h * acc4
                sc = atol + rtol * max(abs(y[d]), abs(y5[d]))
                e = (y5[d] - y4[d]) / sc
                err += e * e
            en = math.sqrt(err / NV)

            if not math.isfinite(en):
                return 0.0, absH(y)

            if en <= 1.0 or abs(h) <= 1.0001 * h_min:
                lam += h
                for d in range(NV):
                    y[d] = y5[d]
                steps += 1
                r = y[1]
                if r <= EH_stop:
                    return 1.0, absH(y)
                if r >= R_esc:
                    return 0.0, absH(y)
                if en == 0.0:
                    fac = 5.0
                else:
                    fac = safety * en ** -0.2
                    if fac < 0.2:
                        fac = 0.2
                    elif fac > 5.0:
                        fac = 5.0
                h = sgn * min(max(abs(h) * fac, h_min), h_max)
            else:
                fac = safety * en ** -0.25
                if fac < 0.2:
                    fac = 0.2
                elif fac > 1.0:
                    fac = 1.0
                h = sgn * min(max(abs(h) * fac, h_min), h_max)
        return 0.0, absH(y)

    # -- Bulirsch-Stoer pixel integrator ------------------------------------
    @njit(fastmath=True)
    def _mmid(yv, H, nsub, out, ym, ym1, tmp):
        h = H / nsub
        for d in range(NV):
            ym[d] = yv[d]
        rhs(yv, tmp)
        for d in range(NV):
            ym1[d] = yv[d] + h * tmp[d]
        for mm in range(1, nsub):
            rhs(ym1, tmp)
            for d in range(NV):
                new = ym[d] + 2.0 * h * tmp[d]
                ym[d] = ym1[d]
                ym1[d] = new
        rhs(ym1, tmp)
        for d in range(NV):
            out[d] = 0.5 * (ym1[d] + ym[d] + h * tmp[d])

    @njit(fastmath=True)
    def _bs_pixel(q, n_seq, lam_max, atol, rtol, EH_stop, R_esc, H0, h_max,
                  max_steps, prevrow, currow, ym, ym1, tmp, y):
        for d in range(NV):
            y[d] = q[d]
        KMAX = n_seq.shape[0]
        sgn = -1.0
        H = sgn * H0
        lam = 0.0
        h_min = 1e-7
        safety = 0.9
        steps = 0
        while lam > -lam_max and steps < max_steps:
            if abs(H) < h_min:
                H = sgn * h_min
            elif abs(H) > h_max:
                H = sgn * h_max
            if lam + H < -lam_max:
                H = -lam_max - lam

            kconv = -1
            en = 1e300
            for kk in range(KMAX):
                _mmid(y, H, int(n_seq[kk]), currow[0], ym, ym1, tmp)
                for m in range(1, kk + 1):
                    ratio = (n_seq[kk] / n_seq[kk - m]) ** 2
                    inv = 1.0 / (ratio - 1.0)
                    for d in range(NV):
                        currow[m, d] = currow[m - 1, d] + (currow[m - 1, d] - prevrow[m - 1, d]) * inv
                if kk >= 1:
                    err = 0.0
                    for d in range(NV):
                        sc = atol + rtol * max(abs(y[d]), abs(currow[kk, d]))
                        e = (currow[kk, d] - currow[kk - 1, d]) / sc
                        err += e * e
                    en = math.sqrt(err / NV)
                    if en <= 1.0:
                        kconv = kk
                        break
                for m in range(kk + 1):
                    for d in range(NV):
                        prevrow[m, d] = currow[m, d]

            if kconv >= 0 or abs(H) <= 1.0001 * h_min:
                kc = kconv if kconv >= 0 else KMAX - 1
                lam += H
                for d in range(NV):
                    y[d] = currow[kc, d]
                steps += 1
                r = y[1]
                if r <= EH_stop:
                    return 1.0, absH(y)
                if r >= R_esc:
                    return 0.0, absH(y)
                if en <= 0.0 or en != en:
                    fac = 4.0
                else:
                    fac = safety * en ** (-1.0 / (2 * kc + 1))
                    if fac < 0.2:
                        fac = 0.2
                    elif fac > 4.0:
                        fac = 4.0
                H = sgn * min(max(abs(H) * fac, h_min), h_max)
            else:
                H = sgn * max(abs(H) * 0.25, h_min)
        return 0.0, absH(y)

    # -- Verlet (fixed-step symmetric midpoint) pixel integrator ------------
    @njit(fastmath=True)
    def _verlet_pixel(q, lam_max, EH_stop, R_esc, h0, max_steps,
                      ymid, k1, k2, y):
        for d in range(NV):
            y[d] = q[d]
        sgn = -1.0
        h = sgn * h0
        lam = 0.0
        steps = 0
        while lam > -lam_max and steps < max_steps:
            if lam + h < -lam_max:
                h = -lam_max - lam
            rhs(y, k1)
            for d in range(NV):
                ymid[d] = y[d] + 0.5 * h * k1[d]
            rhs(ymid, k2)
            for d in range(NV):
                y[d] = y[d] + h * k2[d]
            lam += h
            steps += 1
            r = y[1]
            if r <= EH_stop:
                return 1.0, absH(y)
            if r >= R_esc:
                return 0.0, absH(y)
        return 0.0, absH(y)

    # -- Thread-parallel renderers over a flat pixel list -------------------
    @njit(parallel=True, fastmath=True)
    def _render_rk(alphas, betas, D, sin_i, cos_i, A, c, b5, b4, s,
                   lam_max, atol, rtol, EH_stop, R_esc, h0, h_max, max_steps,
                   chunk):
        # prange iterates over CHUNKS (scratch allocated once per chunk, not per
        # pixel -> avoids allocator contention).  Pixels are pre-shuffled so
        # every chunk is a uniform sample -> balanced load.
        N = alphas.size
        nchunks = (N + chunk - 1) // chunk
        flag = np.empty(N)
        Hf = np.empty(N)
        for ci in prange(nchunks):
            q = np.empty(NV)
            k = np.empty((s, NV))
            ytmp = np.empty(NV)
            y5 = np.empty(NV)
            y4 = np.empty(NV)
            y = np.empty(NV)
            start = ci * chunk
            end = min(start + chunk, N)
            for idx in range(start, end):
                ic(alphas[idx], betas[idx], D, sin_i, cos_i, q)
                fl, hh = _rk_pixel(q, A, c, b5, b4, s, lam_max, atol, rtol,
                                   EH_stop, R_esc, h0, h_max, max_steps,
                                   k, ytmp, y5, y4, y)
                flag[idx] = fl
                Hf[idx] = hh
        return flag, Hf

    @njit(parallel=True, fastmath=True)
    def _render_verlet(alphas, betas, D, sin_i, cos_i,
                       lam_max, EH_stop, R_esc, h0, max_steps, chunk):
        N = alphas.size
        nchunks = (N + chunk - 1) // chunk
        flag = np.empty(N)
        Hf = np.empty(N)
        for ci in prange(nchunks):
            q = np.empty(NV)
            ymid = np.empty(NV)
            k1 = np.empty(NV)
            k2 = np.empty(NV)
            y = np.empty(NV)
            start = ci * chunk
            end = min(start + chunk, N)
            for idx in range(start, end):
                ic(alphas[idx], betas[idx], D, sin_i, cos_i, q)
                fl, hh = _verlet_pixel(q, lam_max, EH_stop, R_esc, h0,
                                       max_steps, ymid, k1, k2, y)
                flag[idx] = fl
                Hf[idx] = hh
        return flag, Hf

    @njit(parallel=True, fastmath=True)
    def _render_bs(alphas, betas, D, sin_i, cos_i, n_seq,
                   lam_max, atol, rtol, EH_stop, R_esc, h0, h_max, max_steps,
                   chunk):
        N = alphas.size
        KMAX = n_seq.shape[0]
        nchunks = (N + chunk - 1) // chunk
        flag = np.empty(N)
        Hf = np.empty(N)
        for ci in prange(nchunks):
            q = np.empty(NV)
            prevrow = np.empty((KMAX, NV))
            currow = np.empty((KMAX, NV))
            ym = np.empty(NV)
            ym1 = np.empty(NV)
            tmp = np.empty(NV)
            y = np.empty(NV)
            start = ci * chunk
            end = min(start + chunk, N)
            for idx in range(start, end):
                ic(alphas[idx], betas[idx], D, sin_i, cos_i, q)
                fl, hh = _bs_pixel(q, n_seq, lam_max, atol, rtol, EH_stop,
                                   R_esc, h0, h_max, max_steps,
                                   prevrow, currow, ym, ym1, tmp, y)
                flag[idx] = fl
                Hf[idx] = hh
        return flag, Hf

    return {"rk": _render_rk, "bs": _render_bs, "verlet": _render_verlet}


# ===========================================================================
# Spacetime providers  (rhs, absH, ic, defaults)
# ===========================================================================
def schwarzschild_funcs():
    @njit(fastmath=True, inline="always")
    def rhs(q, out):
        r = q[1]
        sth = math.sin(q[2])
        cth = math.cos(q[2])
        f = 1.0 - 2.0 / r
        rm2 = r - 2.0
        out[0] = -q[4] / f
        out[1] = f * q[5]
        out[2] = q[6] / (r * r)
        out[3] = q[7] / ((r * sth) * (r * sth))
        out[4] = 0.0
        out[5] = (-(q[4] / rm2) * (q[4] / rm2) - (q[5] / r) * (q[5] / r)
                  + (q[6] * q[6]) / (r * r * r)
                  + (q[7] * q[7]) / (r * r * r * sth * sth))
        out[6] = (cth / (sth * sth * sth)) * (q[7] / r) * (q[7] / r)
        out[7] = 0.0

    @njit(fastmath=True, inline="always")
    def absH(q):
        r = q[1]
        sth = math.sin(q[2])
        f = 1.0 - 2.0 / r
        gtt = -1.0 / f
        grr = f
        gthth = 1.0 / (r * r)
        gphph = 1.0 / ((r * sth) * (r * sth))
        H = 0.5 * (gtt * q[4] * q[4] + grr * q[5] * q[5]
                   + gthth * q[6] * q[6] + gphph * q[7] * q[7])
        return abs(H)

    @njit(fastmath=True, inline="always")
    def ic(alpha, beta, D, sin_i, cos_i, q):
        """Initial conditions from the image plane (cf. image_plane.photon_coords).

        Direction (k_t, k_th, k_phi) is the screen mapping of photon_coords, but
        k_r is solved from the EXACT null condition H = 0 so |H(lambda)| measures
        integrator drift; the impact parameter b = L/E is unchanged.
        """
        r = math.sqrt(alpha * alpha + beta * beta + D * D)
        theta = math.acos((beta * sin_i + D * cos_i) / r)
        phi = math.atan(alpha / (D * sin_i - beta * cos_i))
        f = 1.0 - 2.0 / r
        sth = math.sin(theta)
        g_thth = r * r
        g_phph = (r * sth) * (r * sth)
        k_th = math.sqrt(g_thth) * beta / D
        k_ph = -math.sqrt(g_phph) * alpha / D
        k_t = -math.sqrt(1.0 / f)
        gtt_i = -1.0 / f
        grr_i = f
        gthth_i = 1.0 / g_thth
        gphph_i = 1.0 / g_phph
        k_r = math.sqrt(-(gtt_i * k_t * k_t + gthth_i * k_th * k_th
                          + gphph_i * k_ph * k_ph) / grr_i)
        q[0] = 0.0
        q[1] = r
        q[2] = theta
        q[3] = phi
        q[4] = k_t
        q[5] = k_r
        q[6] = k_th
        q[7] = k_ph

    params = {"h0": 0.1, "verlet_h": 0.1, "EH": 2.0, "EH_off": 0.01,
              "lam_factor": 2.0, "rk_hmax": 5.0, "rk_maxsteps": 200000,
              "bs_hmax": 10.0, "bs_maxsteps": 200000, "verlet_maxsteps": 2000000}
    return rhs, absH, ic, params


def kerr_funcs(a):
    a = float(a)

    @njit(fastmath=True, inline="always")
    def rhs(q, out):
        r = q[1]
        r2 = r * r
        a2 = a * a
        sin_th = math.sin(q[2])
        cos_th = math.cos(q[2])
        sin_th2 = sin_th * sin_th
        cos_th2 = cos_th * cos_th
        Sigma = r2 + a2 * cos_th2
        Sigma2 = Sigma * Sigma
        Delta = r2 - 2.0 * r + a2

        kt = q[4]
        kr = q[5]
        kth = q[6]
        kph = q[7]

        W = -kt * (r2 + a2) - a * kph
        partXi = (r2 + (kph + a * kt) ** 2 + a2 * (1.0 + kt * kt) * cos_th2
                  + kph * kph * cos_th2 / sin_th2)
        Xi = W * W - Delta * partXi

        dXidE = 2.0 * W * (r2 + a2) + 2.0 * a * Delta * (kph + a * kt * sin_th2)
        dXidL = -2.0 * a * W - 2.0 * a * kt * Delta - 2.0 * kph * Delta / sin_th2
        dXidr = -4.0 * r * kt * W - 2.0 * (r - 1.0) * partXi - 2.0 * r * Delta

        dAdr = (r - 1.0) / Sigma - (r * Delta) / Sigma2
        dBdr = -r / Sigma2
        dCdr = (dXidr / (2.0 * Delta * Sigma) - (Xi * (r - 1.0)) / (Sigma * Delta * Delta)
                - r * Xi / (Delta * Sigma2))

        auxth = a2 * cos_th * sin_th
        dAdth = Delta * auxth / Sigma2
        dBdth = auxth / Sigma2
        dCdth = (((1.0 + kt * kt) * auxth + kph * kph * cos_th / (sin_th2 * sin_th)) / Sigma
                 + (Xi / (Delta * Sigma2)) * auxth)

        out[0] = dXidE / (2.0 * Delta * Sigma)
        out[1] = (Delta / Sigma) * kr
        out[2] = kth / Sigma
        out[3] = -dXidL / (2.0 * Delta * Sigma)
        out[4] = 0.0
        out[5] = -dAdr * kr * kr - dBdr * kth * kth + dCdr
        out[6] = -dAdth * kr * kr - dBdth * kth * kth + dCdth
        out[7] = 0.0

    @njit(fastmath=True, inline="always")
    def absH(q):
        r = q[1]
        r2 = r * r
        a2 = a * a
        sin_th = math.sin(q[2])
        cos_th = math.cos(q[2])
        sin_th2 = sin_th * sin_th
        Delta = r2 - 2.0 * r + a2
        Sigma = r2 + a2 * cos_th * cos_th
        A = (r2 + a2) ** 2 - Delta * a2 * sin_th2
        gtt = -A / (Delta * Sigma)
        grr = Delta / Sigma
        gthth = 1.0 / Sigma
        gphph = (Delta - a2 * sin_th2) / (Delta * Sigma * sin_th2)
        gtph = -2.0 * a * r / (Delta * Sigma)
        H = 0.5 * (gtt * q[4] * q[4] + grr * q[5] * q[5] + gthth * q[6] * q[6]
                   + gphph * q[7] * q[7] + 2.0 * gtph * q[4] * q[7])
        return abs(H)

    @njit(fastmath=True, inline="always")
    def ic(alpha, beta, D, sin_i, cos_i, q):
        """Kerr initial conditions, identical to image_plane.photon_coords."""
        r = math.sqrt(alpha * alpha + beta * beta + D * D)
        theta = math.acos((beta * sin_i + D * cos_i) / r)
        phi = math.atan(alpha / (D * sin_i - beta * cos_i))
        r2 = r * r
        a2 = a * a
        sin_th = math.sin(theta)
        sin_th2 = sin_th * sin_th
        Sigma = r2 + a2 * math.cos(theta) ** 2
        Delta = r2 - 2.0 * r + a2
        g_tt = -(1.0 - 2.0 * r / Sigma)
        g_rr = Sigma / Delta
        g_thth = Sigma
        g_phph = (r2 + a2 + 2.0 * a2 * r * sin_th2 / Sigma) * sin_th2
        g_tph = -2.0 * a * r * sin_th2 / Sigma

        k_th = math.sqrt(g_thth) * beta / D
        k_ph = -math.sqrt(g_phph) * alpha / D
        k_t = (-math.sqrt(g_phph / (g_tph * g_tph - g_tt * g_phph))
               + alpha * g_tph / (D * math.sqrt(g_phph)))
        k_r = math.sqrt(g_rr * (1.0 - (k_th * k_th) / g_thth - (k_ph * k_ph) / g_phph))
        q[0] = 0.0
        q[1] = r
        q[2] = theta
        q[3] = phi
        q[4] = k_t
        q[5] = k_r
        q[6] = k_th
        q[7] = k_ph

    EH = 1.0 + math.sqrt(1.0 - a * a)
    params = {"h0": 0.5, "verlet_h": 0.1, "EH": EH, "EH_off": 0.1,
              "lam_factor": 2.2, "rk_hmax": 20.0, "rk_maxsteps": 500000,
              "bs_hmax": 20.0, "bs_maxsteps": 500000, "verlet_maxsteps": 2000000}
    return rhs, absH, ic, params


_PROVIDERS = {"schwarzschild": schwarzschild_funcs, "kerr": kerr_funcs}


# ===========================================================================
# Memoised kernel cache  (build once per spacetime + params)
# ===========================================================================
_KERNEL_CACHE = {}


def _get_kernels(spacetime, a):
    """Return (kernels_dict, params) for the requested spacetime, building and
    caching them on first use.  Keyed by (spacetime, a) because the spacetime
    functions capture the metric parameters by closure value."""
    key = (spacetime, float(a))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    provider = _PROVIDERS.get(spacetime)
    if provider is None:
        raise ValueError(f"Unknown spacetime: {spacetime!r} "
                         f"(available: {sorted(_PROVIDERS)})")
    if spacetime == "schwarzschild":
        rhs, absH, ic, params = provider()
    else:
        rhs, absH, ic, params = provider(a)
    kernels = build_kernels(rhs, absH, ic)
    _KERNEL_CACHE[key] = (kernels, params)
    return kernels, params


# ===========================================================================
# Public dispatcher
# ===========================================================================
def render_shadow(alphas, betas, *, spacetime="schwarzschild", method="DP45",
                  a=0.0, D, iota, nthreads=None, lam_max=None, atol=1e-9,
                  rtol=1e-9, EH=None, h0=None, verlet_h=None, chunk=256,
                  shuffle=True, sort="random", timed=True):
    """Render a shadow for a flat pixel list; return (flag, |H|, elapsed_s).

    ``spacetime`` selects the metric provider ("schwarzschild" | "kerr"); ``a``
    is the spin (ignored for Schwarzschild).  ``method`` is any of
    DP45/CK45/RKF45/BS/Verlet (aliases RKDP45/RKCK45/RK45 accepted).

    Integration defaults (lam_max, h0, EH, step caps) come from the spacetime's
    params dict unless overridden.  ``sort='random'`` shuffles pixels (fixed
    seed) so shadow-edge photons spread evenly across threads.
    """
    if method not in _METHOD_ALIASES:
        raise ValueError(
            f"Unknown Numba shadow method {method!r}; supported: "
            f"{sorted(set(_METHOD_ALIASES))}.")
    method = _METHOD_ALIASES[method]
    kernels, params = _get_kernels(spacetime, a)

    if nthreads is not None:
        set_num_threads(int(nthreads))
    if lam_max is None:
        lam_max = params["lam_factor"] * D
    if h0 is None:
        h0 = params["h0"]
    if verlet_h is None:
        verlet_h = params["verlet_h"]
    if EH is None:
        EH = params["EH"]

    sin_i = math.sin(iota)
    cos_i = math.cos(iota)
    EH_stop = EH + params["EH_off"]
    R_esc = 1.1 * D
    alphas = np.ascontiguousarray(alphas, float)
    betas = np.ascontiguousarray(betas, float)
    N = alphas.size

    if sort == "random" or (sort is None and shuffle):
        perm = np.random.RandomState(12345).permutation(N)
        a_in = np.ascontiguousarray(alphas[perm])
        b_in = np.ascontiguousarray(betas[perm])
    else:
        perm = None
        a_in, b_in = alphas, betas

    t0 = time.perf_counter()
    if method == "BS":
        flag_s, H_s = kernels["bs"](a_in, b_in, D, sin_i, cos_i, _BS_SEQ,
                                    lam_max, atol, rtol, EH_stop, R_esc, h0,
                                    params["bs_hmax"], params["bs_maxsteps"],
                                    chunk)
    elif method == "Verlet":
        flag_s, H_s = kernels["verlet"](a_in, b_in, D, sin_i, cos_i,
                                        lam_max, EH_stop, R_esc, verlet_h,
                                        params["verlet_maxsteps"], chunk)
    else:
        A, c, b5, b4, s = _tableau_arrays(method)
        flag_s, H_s = kernels["rk"](a_in, b_in, D, sin_i, cos_i, A, c, b5, b4, s,
                                    lam_max, atol, rtol, EH_stop, R_esc, h0,
                                    params["rk_hmax"], params["rk_maxsteps"],
                                    chunk)
    elapsed = time.perf_counter() - t0 if timed else None

    if perm is not None:
        flag = np.empty(N)
        Hf = np.empty(N)
        flag[perm] = flag_s
        Hf[perm] = H_s
    else:
        flag, Hf = flag_s, H_s
    return flag, Hf, elapsed


def warmup(spacetime="schwarzschild", a=0.0, D=100.0, iota=math.pi / 2,
           methods=("DP45", "CK45", "RKF45", "BS", "Verlet")):
    """Trigger JIT compilation and warm up the parallel thread pool.

    Runs every method at nthreads=1 (compile), then a subset at max threads
    (warm the pool) so the first timed call pays no startup cost.
    """
    al = np.array([0.5, 4.0])
    be = np.array([0.5, -3.0])
    for m in methods:
        render_shadow(al, be, spacetime=spacetime, a=a, D=D, iota=iota,
                      method=m, nthreads=1)
    max_p = config.NUMBA_NUM_THREADS
    for m in ("DP45", "BS", "Verlet"):
        if m in methods:
            render_shadow(al, be, spacetime=spacetime, a=a, D=D, iota=iota,
                          method=m, nthreads=max_p)
