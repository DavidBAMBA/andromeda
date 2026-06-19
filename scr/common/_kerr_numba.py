"""
===============================================================================
Numba JIT Kerr shadow renderer -- apples-to-apples timing vs OSIRIS (Fig. 8)
===============================================================================
Ports scr/black_holes/kerr.py (geodesics in Hamiltonian form) and
detectors/image_plane.photon_coords to nopython kernels, and renders the Kerr
shadow with the production RK45 (Dormand-Prince) integrator in a thread-parallel
prange loop. Used by ex27 to reproduce the OSIRIS Fig. 8 setup
(a = 0.98, observer at r0 = 1000, image plane [-8, 8]) and compare wall time.
===============================================================================
"""

import math
import time
import numpy as np
from numba import njit, prange, set_num_threads, config

from scr.common.integrator import _RK_TABLEAUX

NV = 8


@njit(cache=True, fastmath=True, inline="always")
def _rhs_kerr(q, out, a):
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


@njit(cache=True, fastmath=True, inline="always")
def _absH_kerr(q, a):
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


@njit(cache=True, fastmath=True, inline="always")
def _ic_kerr(alpha, beta, D, sin_i, cos_i, a, q):
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


@njit(cache=True, fastmath=True)
def _rk_pixel_kerr(q, A, c, b5, b4, s, a, lam_max, atol, rtol, EH_stop, R_esc,
                   h0, k, ytmp, y5, y4, y):
    for d in range(NV):
        y[d] = q[d]
    sgn = -1.0
    h = sgn * h0
    lam = 0.0
    h_min = 1e-9
    h_max = 20.0
    safety = 0.9
    steps = 0
    max_steps = 500000
    while lam > -lam_max and steps < max_steps:
        if abs(h) < h_min:
            h = sgn * h_min
        elif abs(h) > h_max:
            h = sgn * h_max
        if lam + h < -lam_max:
            h = -lam_max - lam

        _rhs_kerr(y, k[0], a)
        for ist in range(1, s):
            for d in range(NV):
                ytmp[d] = y[d]
            for jj in range(ist):
                aij = A[ist, jj]
                if aij != 0.0:
                    ha = h * aij
                    for d in range(NV):
                        ytmp[d] += ha * k[jj, d]
            _rhs_kerr(ytmp, k[ist], a)

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
            return 0.0, _absH_kerr(y, a)

        if en <= 1.0 or abs(h) <= 1.0001 * h_min:
            lam += h
            for d in range(NV):
                y[d] = y5[d]
            steps += 1
            r = y[1]
            if r <= EH_stop:
                return 1.0, _absH_kerr(y, a)
            if r >= R_esc:
                return 0.0, _absH_kerr(y, a)
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
    return 0.0, _absH_kerr(y, a)


@njit(parallel=True, cache=True, fastmath=True)
def _render_kerr(alphas, betas, D, sin_i, cos_i, a, A, c, b5, b4, s,
                 lam_max, atol, rtol, EH_stop, R_esc, h0, chunk):
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
            _ic_kerr(alphas[idx], betas[idx], D, sin_i, cos_i, a, q)
            fl, hh = _rk_pixel_kerr(q, A, c, b5, b4, s, a, lam_max, atol, rtol,
                                    EH_stop, R_esc, h0, k, ytmp, y5, y4, y)
            flag[idx] = fl
            Hf[idx] = hh
    return flag, Hf


def _tableau_arrays(method="DP45"):
    tab = _RK_TABLEAUX[method]
    s = len(tab["c"])
    A = np.zeros((s, s))
    for i, row in enumerate(tab["A"]):
        for j, v in enumerate(row):
            A[i, j] = v
    return (A, np.asarray(tab["c"], float), np.asarray(tab["b5"], float),
            np.asarray(tab["b4"], float), s)


def render_kerr_shadow(alphas, betas, *, D, iota, a, nthreads=None,
                       lam_max=None, atol=1e-9, rtol=1e-9, h0=0.5,
                       chunk=256, shuffle=True):
    """Render the Kerr shadow; return (flag, |H|, elapsed_s)."""
    if nthreads is not None:
        set_num_threads(int(nthreads))
    if lam_max is None:
        lam_max = 2.2 * D
    sin_i, cos_i = math.sin(iota), math.cos(iota)
    EH = 1.0 + math.sqrt(1.0 - a * a)
    EH_stop = EH + 0.1
    R_esc = 1.1 * D
    A, c, b5, b4, s = _tableau_arrays("DP45")
    alphas = np.ascontiguousarray(alphas, float)
    betas = np.ascontiguousarray(betas, float)
    N = alphas.size

    if shuffle:
        perm = np.random.RandomState(12345).permutation(N)
        a_in = np.ascontiguousarray(alphas[perm])
        b_in = np.ascontiguousarray(betas[perm])
    else:
        a_in, b_in = alphas, betas

    t0 = time.perf_counter()
    flag_s, H_s = _render_kerr(a_in, b_in, D, sin_i, cos_i, a, A, c, b5, b4, s,
                               lam_max, atol, rtol, EH_stop, R_esc, h0, chunk)
    elapsed = time.perf_counter() - t0

    if shuffle:
        flag = np.empty(N)
        Hf = np.empty(N)
        flag[perm] = flag_s
        Hf[perm] = H_s
    else:
        flag, Hf = flag_s, H_s
    return flag, Hf, elapsed


def warmup(D=1000.0, iota=math.pi / 2, a=0.98):
    al = np.array([0.5, 4.0])
    be = np.array([0.5, -3.0])
    render_kerr_shadow(al, be, D=D, iota=iota, a=a, nthreads=1)
