"""
===============================================================================
Numba JIT kernels for Schwarzschild shadow ray tracing (performance benchmark)
===============================================================================
Thread-parallel (prange) renderer of the Schwarzschild black-hole shadow used
by ex26 to benchmark the four integrators (DP45 / CK45 / RKF45 / BS) and to
measure thread speedup / scaling.

Each photon is launched from the image plane (same construction as
detectors/image_plane.photon_coords), traced backward in the affine parameter,
and classified:
    * flag = 1.0  -> captured (crossed the horizon)  => shadow pixel
    * flag = 0.0  -> escaped  (reached R_esc) or hit the affine-parameter cap

The integrators reproduce the same methods as scr/common/integrator.py, but as
nopython kernels so the pixel loop releases the GIL and scales across threads.
===============================================================================
"""

import math
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads, config

from scr.common.integrator import _RK_TABLEAUX

NV = 8                      # state dimension [t, r, th, phi, kt, kr, kth, kphi]
_BS_SEQ = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=np.float64)


# ---------------------------------------------------------------------------
# Schwarzschild geodesic RHS, |H|, and initial conditions  (all nopython)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline="always")
def _rhs(q, out):
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


@njit(cache=True, fastmath=True, inline="always")
def _absH(q):
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


@njit(cache=True, fastmath=True, inline="always")
def _ic(alpha, beta, D, sin_i, cos_i, q):
    """Initial conditions from the image plane (cf. image_plane.photon_coords).

    Direction (k_t, k_th, k_phi) is the screen mapping of photon_coords, but
    k_r is solved from the EXACT null condition H = 0 (photon_coords' own k_r
    leaves a residual H = 1/2(1 - 1/f^2) ~ 0.02 in Schwarzschild). This makes
    the geodesics null to machine precision so |H(lambda)| measures integrator
    drift; the impact parameter b = L/E (hence the shadow) is unchanged.
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
    # null condition: g^tt k_t^2 + g^rr k_r^2 + g^th k_th^2 + g^ph k_ph^2 = 0
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


# ---------------------------------------------------------------------------
# Embedded RK pixel integrator (tableau passed as arrays)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _rk_pixel(q, A, c, b5, b4, s, lam_max, atol, rtol, EH_stop, R_esc, h0,
              k, ytmp, y5, y4, y):
    for d in range(NV):
        y[d] = q[d]
    sgn = -1.0                     # trace backward in the affine parameter
    h = sgn * h0
    lam = 0.0
    h_min = 1e-9
    h_max = 5.0
    safety = 0.9
    steps = 0
    max_steps = 200000
    while lam > -lam_max and steps < max_steps:
        if abs(h) < h_min:
            h = sgn * h_min
        elif abs(h) > h_max:
            h = sgn * h_max
        if lam + h < -lam_max:
            h = -lam_max - lam

        _rhs(y, k[0])
        for ist in range(1, s):
            for d in range(NV):
                ytmp[d] = y[d]
            for jj in range(ist):
                a = A[ist, jj]
                if a != 0.0:
                    ha = h * a
                    for d in range(NV):
                        ytmp[d] += ha * k[jj, d]
            _rhs(ytmp, k[ist])

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

        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            lam += h
            for d in range(NV):
                y[d] = y5[d]
            steps += 1
            r = y[1]
            if r <= EH_stop:
                return 1.0, _absH(y)
            if r >= R_esc:
                return 0.0, _absH(y)
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
    return 0.0, _absH(y)


# ---------------------------------------------------------------------------
# Bulirsch-Stoer pixel integrator
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline="always")
def _mmid(y, H, nsub, out, ym, ym1, tmp):
    h = H / nsub
    for d in range(NV):
        ym[d] = y[d]
    _rhs(y, tmp)
    for d in range(NV):
        ym1[d] = y[d] + h * tmp[d]
    for mm in range(1, nsub):
        _rhs(ym1, tmp)
        for d in range(NV):
            new = ym[d] + 2.0 * h * tmp[d]
            ym[d] = ym1[d]
            ym1[d] = new
    _rhs(ym1, tmp)
    for d in range(NV):
        out[d] = 0.5 * (ym1[d] + ym[d] + h * tmp[d])


@njit(cache=True, fastmath=True)
def _bs_pixel(q, n_seq, lam_max, atol, rtol, EH_stop, R_esc, H0,
              prevrow, currow, ym, ym1, tmp, y):
    for d in range(NV):
        y[d] = q[d]
    KMAX = n_seq.shape[0]
    sgn = -1.0
    H = sgn * H0
    lam = 0.0
    h_min = 1e-7
    h_max = 10.0
    safety = 0.9
    steps = 0
    max_steps = 200000
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
                return 1.0, _absH(y)
            if r >= R_esc:
                return 0.0, _absH(y)
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
    return 0.0, _absH(y)


# ---------------------------------------------------------------------------
# Verlet (fixed-step symmetric midpoint) pixel integrator
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _verlet_pixel(q, lam_max, EH_stop, R_esc, h0, ymid, k1, k2, y):
    for d in range(NV):
        y[d] = q[d]
    sgn = -1.0
    h = sgn * h0
    lam = 0.0
    steps = 0
    max_steps = 2000000
    while lam > -lam_max and steps < max_steps:
        if lam + h < -lam_max:
            h = -lam_max - lam
        _rhs(y, k1)
        for d in range(NV):
            ymid[d] = y[d] + 0.5 * h * k1[d]
        _rhs(ymid, k2)
        for d in range(NV):
            y[d] = y[d] + h * k2[d]
        lam += h
        steps += 1
        r = y[1]
        if r <= EH_stop:
            return 1.0, _absH(y)
        if r >= R_esc:
            return 0.0, _absH(y)
    return 0.0, _absH(y)


# ---------------------------------------------------------------------------
# Thread-parallel renderers over a flat pixel list
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def _render_rk(alphas, betas, D, sin_i, cos_i, A, c, b5, b4, s,
               lam_max, atol, rtol, EH_stop, R_esc, h0, chunk):
    # prange iterates over CHUNKS (scratch allocated once per chunk, not per
    # pixel -> avoids allocator contention across threads). Pixels are assumed
    # pre-shuffled so every chunk is a uniform sample -> balanced load.
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
            _ic(alphas[idx], betas[idx], D, sin_i, cos_i, q)
            fl, hh = _rk_pixel(q, A, c, b5, b4, s, lam_max, atol, rtol,
                               EH_stop, R_esc, h0, k, ytmp, y5, y4, y)
            flag[idx] = fl
            Hf[idx] = hh
    return flag, Hf


@njit(parallel=True, cache=True, fastmath=True)
def _render_verlet(alphas, betas, D, sin_i, cos_i,
                   lam_max, EH_stop, R_esc, h0, chunk):
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
            _ic(alphas[idx], betas[idx], D, sin_i, cos_i, q)
            fl, hh = _verlet_pixel(q, lam_max, EH_stop, R_esc, h0,
                                   ymid, k1, k2, y)
            flag[idx] = fl
            Hf[idx] = hh
    return flag, Hf


@njit(parallel=True, cache=True, fastmath=True)
def _render_bs(alphas, betas, D, sin_i, cos_i, n_seq,
               lam_max, atol, rtol, EH_stop, R_esc, h0, chunk):
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
            _ic(alphas[idx], betas[idx], D, sin_i, cos_i, q)
            fl, hh = _bs_pixel(q, n_seq, lam_max, atol, rtol, EH_stop, R_esc, h0,
                               prevrow, currow, ym, ym1, tmp, y)
            flag[idx] = fl
            Hf[idx] = hh
    return flag, Hf


# ---------------------------------------------------------------------------
# Python-facing wrapper
# ---------------------------------------------------------------------------
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


def _cyclic_perm(alphas, betas, chunk):
    """Return a pixel permutation that balances load across prange chunks.

    Pixels are sorted ascending by |b - b_crit| (smallest distance to the
    photon sphere = most integration steps) and then distributed cyclically:
    chunk j receives pixels at sorted ranks j, j+nchunks, j+2*nchunks, ...
    so every chunk spans the full cost range and no chunk is dominated by
    expensive shadow-edge photons.
    """
    b_crit = 3.0 * math.sqrt(3.0)          # Schwarzschild photon-sphere b
    cost = np.abs(np.sqrt(alphas**2 + betas**2) - b_crit)
    sorted_ranks = np.argsort(cost)         # rank 0 = most expensive pixel
    N = alphas.size
    nchunks = (N + chunk - 1) // chunk
    perm = np.empty(N, dtype=np.intp)
    ptr = 0
    for j in range(nchunks):               # chunk j gets ranks j, j+nchunks, ...
        px = sorted_ranks[j::nchunks]
        perm[ptr:ptr + len(px)] = px
        ptr += len(px)
    return perm


def render_shadow_numba(alphas, betas, *, D, iota, method="DP45", nthreads=None,
                        lam_max=None, atol=1e-9, rtol=1e-9, EH=2.0, h0=0.1,
                        verlet_h=0.1, chunk=32, shuffle=True, sort="cyclic",
                        timed=True):
    """Render the shadow for a flat pixel list; return (flag, |H|, elapsed_s).

    ``alphas``/``betas`` are paired flat arrays of image-plane coordinates.
    ``nthreads`` sets numba's thread count for this call.

    ``sort`` controls pixel ordering before the parallel kernel:
      * ``'cyclic'`` (default) — sorts by cost proxy (|b - b_crit|) and
        distributes cyclically so every chunk spans the full cost range.
        Minimises load imbalance; the main driver of sub-linear scaling.
      * ``'random'`` — legacy random shuffle (fixed seed).
      * ``None`` — no reordering (caller is responsible for balanced input).

    ``chunk`` is the number of pixels each prange task processes serially
    (scratch arrays are allocated once per chunk).  Smaller values give finer
    load-balancing granularity; 32 is a good default for most GPUs/CPUs.
    """
    method = _METHOD_ALIASES[method]
    if nthreads is not None:
        set_num_threads(int(nthreads))
    if lam_max is None:
        lam_max = 2.0 * D
    sin_i = math.sin(iota)
    cos_i = math.cos(iota)
    EH_stop = EH + 0.01
    R_esc = 1.1 * D
    alphas = np.ascontiguousarray(alphas, float)
    betas  = np.ascontiguousarray(betas,  float)
    N = alphas.size

    if sort == "cyclic" and N > 1:
        perm = _cyclic_perm(alphas, betas, chunk)
        a_in = np.ascontiguousarray(alphas[perm])
        b_in = np.ascontiguousarray(betas[perm])
    elif sort == "random" or (sort is None and shuffle):
        perm = np.random.RandomState(12345).permutation(N)
        a_in = np.ascontiguousarray(alphas[perm])
        b_in = np.ascontiguousarray(betas[perm])
    else:
        perm = None
        a_in, b_in = alphas, betas

    t0 = time.perf_counter()
    if method == "BS":
        flag_s, H_s = _render_bs(a_in, b_in, D, sin_i, cos_i, _BS_SEQ,
                                 lam_max, atol, rtol, EH_stop, R_esc, h0, chunk)
    elif method == "Verlet":
        flag_s, H_s = _render_verlet(a_in, b_in, D, sin_i, cos_i,
                                     lam_max, EH_stop, R_esc, verlet_h, chunk)
    else:
        A, c, b5, b4, s = _tableau_arrays(method)
        flag_s, H_s = _render_rk(a_in, b_in, D, sin_i, cos_i, A, c, b5, b4, s,
                                 lam_max, atol, rtol, EH_stop, R_esc, h0, chunk)
    elapsed = time.perf_counter() - t0 if timed else None

    if perm is not None:
        flag = np.empty(N)
        Hf   = np.empty(N)
        flag[perm] = flag_s
        Hf[perm]   = H_s
    else:
        flag, Hf = flag_s, H_s
    return flag, Hf, elapsed


def warmup(D=100.0, iota=math.pi / 2):
    """Trigger JIT compilation of every kernel (excluded from timings)."""
    a = np.array([0.5, 4.0])
    b = np.array([0.5, -3.0])
    for m in ("DP45", "CK45", "RKF45", "BS", "Verlet"):
        render_shadow_numba(a, b, D=D, iota=iota, method=m, nthreads=1)
