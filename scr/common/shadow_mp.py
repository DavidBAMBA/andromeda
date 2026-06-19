"""
===============================================================================
Multiprocessing shadow renderer -- cross-check / process-based scaling
===============================================================================
Renders the Schwarzschild shadow by distributing pixels across PROCESSES, each
running the pure-Python integrators from scr/common/integrator.py. This is the
reference path: same methods, same adaptive control as the numba kernels, but
without the JIT -- used to (a) cross-validate the numba shadow and (b) show
process-based scaling (true parallelism despite the GIL, at the cost of
process startup / IPC overhead and ~100x slower per-photon work).
===============================================================================
"""

import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from scr.common import integrator
from scr.black_holes import schwarzschild

_METHOD_MAP = {"RKDP45": "DP45", "RKCK45": "CK45", "RKF45": "RKF45",
               "DP45": "DP45", "CK45": "CK45", "BS": "BS",
               "RK45": "DP45", "Verlet": "Verlet"}


def ic_py(alpha, beta, D, sin_i, cos_i):
    """Image-plane initial conditions with k_r solved from the null condition
    H = 0 (see scr/common/_geo_numba._ic). Identical construction to the numba
    path, so the two backends are bit-for-bit comparable."""
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
    gtt_i, grr_i = -1.0 / f, f
    gthth_i, gphph_i = 1.0 / g_thth, 1.0 / g_phph
    k_r = math.sqrt(-(gtt_i * k_t**2 + gthth_i * k_th**2
                      + gphph_i * k_ph**2) / grr_i)
    return [0.0, r, theta, phi, k_t, k_r, k_th, k_ph]


def shadow_chunk(args):
    """Worker: classify a chunk of pixels as shadow (1.0) / escape (0.0)."""
    alphas, betas, D, iota, method, lam_max, atol, rtol, EH = args
    sin_i, cos_i = math.sin(iota), math.cos(iota)
    EH_stop, R_esc = EH + 0.01, 1.1 * D
    m = _METHOD_MAP[method]
    bh = schwarzschild.BlackHole()
    rhs = lambda lam, q: bh.geodesics(q, lam)
    stop = lambda lam, q: (q[1] <= EH_stop) or (q[1] >= R_esc)

    out = np.empty(len(alphas))
    for i in range(len(alphas)):
        q0 = ic_py(alphas[i], betas[i], D, sin_i, cos_i)
        if m == "BS":
            _, Y = integrator.bulirsch_stoer(rhs, 0.0, q0, -lam_max,
                                             atol=atol, rtol=rtol, stop=stop)
        elif m == "Verlet":
            _, Y = integrator.verlet(rhs, 0.0, q0, -lam_max, h0=0.1, stop=stop)
        else:
            _, Y = integrator.rk_adaptive(rhs, 0.0, q0, -lam_max, method=m,
                                          atol=atol, rtol=rtol, h_max=5.0, stop=stop)
        out[i] = 1.0 if Y[-1][1] <= EH_stop else 0.0
    return out


def render_shadow_mp(alphas, betas, *, D, iota, method="DP45", nworkers=1,
                     lam_max=None, atol=1e-9, rtol=1e-9, EH=2.0,
                     chunks_per_worker=4):
    """Render via a process pool; return (flag, elapsed_s).

    Pixels are shuffled before splitting so each chunk carries a balanced load.
    """
    if lam_max is None:
        lam_max = 2.0 * D
    alphas = np.ascontiguousarray(alphas, float)
    betas = np.ascontiguousarray(betas, float)
    N = alphas.size

    perm = np.random.RandomState(12345).permutation(N)
    nchunks = max(1, nworkers * chunks_per_worker)
    splits = np.array_split(perm, nchunks)
    args = [(alphas[s], betas[s], D, iota, method, lam_max, atol, rtol, EH)
            for s in splits]

    t0 = time.perf_counter()
    if nworkers == 1:
        results = [shadow_chunk(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            results = list(ex.map(shadow_chunk, args))
    elapsed = time.perf_counter() - t0

    flag = np.empty(N)
    for s, res in zip(splits, results):
        flag[s] = res
    return flag, elapsed
