"""
===============================================================================
Parallel ray tracing using multiprocessing with dynamic load balancing
===============================================================================
Each photon is an independent task. Since central pixels (which fall into the
horizon) can cost 10x more than background pixels, we use Pool.imap_unordered
with a small chunksize so any idle worker immediately picks up the next chunk.

Public API:
    trace_parallel(tasks, blackhole, acc_structure, detector, ...) -> (img, stats)

Called from common.Image.create_image(). Can also be used standalone.
===============================================================================
"""
import random
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from numpy import sqrt, zeros
import numba

from scr.common.integrator import (integrate, make_events, _solve_photon_nb,
                                     _compute_pixel_nb, _render_image_nb,
                                     _null_omega_nb)
from scr.accretion_structures.thin_disk import _intensity_nb


# Module-level worker state — populated once per worker via Pool initializer.
_W = {}

_MODE_CODES = {"no_doppler": 0, "doppler": 1, "shadow": 2}


def _has_numba_hooks(blackhole):
    return getattr(blackhole, "_rhs_nb", None) is not None


def _resolve_geometry(detector, r_escape, final_lmbda):
    """Fill in default geometry bounds from detector if not specified."""
    if r_escape is None:
        r_escape = 1.1 * detector.D
    if final_lmbda is None:
        final_lmbda = 1.5 * detector.D
    return r_escape, final_lmbda


def _init_worker(blackhole, acc_structure, detector, method, rtol, atol, mode,
                 r_escape, final_lmbda):
    """Runs once per worker process. Builds events locally to avoid pickling closures."""
    _W["blackhole"] = blackhole
    _W["acc_structure"] = acc_structure
    _W["detector"] = detector
    _W["method"] = method
    _W["rtol"] = rtol
    _W["atol"] = atol
    _W["mode"] = mode
    _W["mode_code"] = _MODE_CODES.get(mode, 0)
    _W["r_escape"] = r_escape
    _W["final_lmbda"] = final_lmbda
    # Numba hot path: capture hooks once; events are implicit (fixed set).
    _W["use_numba"] = _has_numba_hooks(blackhole) and method in ("RK45_numba", "auto")
    if not _W["use_numba"]:
        acc = acc_structure if mode != "shadow" else None
        _W["events"] = make_events(blackhole, acc_structure=acc,
                                   r_escape=r_escape)


def _doppler_inline(fP, I0, blackhole):
    """Doppler shift without requiring the Photon class (for worker speed)."""
    g_tt, _, _, g_phph, g_tph = blackhole.metric(fP[:4])
    Omega = blackhole.Omega(fP[1])
    g = sqrt(-g_tt - 2*g_tph*Omega - g_phph*Omega**2) / (1 + fP[7]*Omega/fP[4])
    return I0 * g**3


def _doppler_inline_nb(fP, I0, blackhole):
    """Doppler shift using the BH's numba metric/omega hooks."""
    g_tt, g_rr, g_thth, g_phph, g_tph = blackhole._metric_nb(fP[:4])
    Omega = blackhole._omega_nb(fP[1])
    g = sqrt(-g_tt - 2*g_tph*Omega - g_phph*Omega**2) / (1 + fP[7]*Omega/fP[4])
    return I0 * g**3


def trace_threads_nb(tasks, blackhole, acc_structure, detector,
                     *, n_workers=None, method="auto", rtol=1e-9, atol=1e-11,
                     mode="doppler", progress=True,
                     r_escape=None, final_lmbda=None):
    """Parallel render using numba.prange threads (no multiprocessing).

    Shares memory, no pickle, no fork. Requires the BH to expose
    `_rhs_nb` and `_metric_nb` (and `_omega_nb` for doppler mode).
    """
    if n_workers is None:
        n_workers = cpu_count()
    n_workers = max(1, n_workers)
    r_escape, final_lmbda = _resolve_geometry(detector, r_escape, final_lmbda)

    tasks = list(tasks)
    n_tasks = len(tasks)

    # Pack (i, j, iC) lists into arrays.
    idx_i = np.empty(n_tasks, dtype=np.int32)
    idx_j = np.empty(n_tasks, dtype=np.int32)
    y0_batch = np.empty((n_tasks, 8), dtype=np.float64)
    for k, (i, j, iC) in enumerate(tasks):
        idx_i[k] = i
        idx_j[k] = j
        y0_batch[k, :] = iC

    mode_code = _MODE_CODES.get(mode, 0)
    omega_nb = blackhole._omega_nb if blackhole._omega_nb is not None \
                                   else _null_omega_nb

    numba.set_num_threads(n_workers)
    image = zeros([detector.x_pixels, detector.y_pixels])
    stats = {}
    t0 = time.perf_counter()
    if progress:
        sys.stdout.write(f"  prange: {n_tasks} photons / {n_workers} threads ...\n")
        sys.stdout.flush()

    values = _render_image_nb(
        blackhole._rhs_nb, blackhole._metric_nb, omega_nb,
        y0_batch, -float(final_lmbda),
        float(blackhole.EH), float(r_escape),
        acc_structure._r_tbl, acc_structure._I_tbl,
        acc_structure.in_edge, acc_structure.out_edge,
        rtol, atol, mode_code)

    # Scatter results back to image grid.
    for k in range(n_tasks):
        image[idx_i[k], idx_j[k]] = values[k]

    stats["wall_total"] = time.perf_counter() - t0
    stats["n_workers"] = n_workers
    stats["chunksize"] = 0  # not applicable
    stats["backend"] = "numba-threads"
    return image, stats


def _integrate_one_nb(task):
    """Numba hot path: single photon end-to-end in _compute_pixel_nb."""
    i, j, iC = task
    bh = _W["blackhole"]
    acc = _W["acc_structure"]
    y0 = np.asarray(iC, dtype=np.float64)

    omega_nb = bh._omega_nb if bh._omega_nb is not None else _null_omega_nb
    value = _compute_pixel_nb(
        bh._rhs_nb, bh._metric_nb, omega_nb,
        y0, -_W["final_lmbda"],
        float(bh.EH), float(_W["r_escape"]),
        acc._r_tbl, acc._I_tbl, acc.in_edge, acc.out_edge,
        _W["rtol"], _W["atol"], _W["mode_code"])
    return (i, j, value)


def _integrate_one(task):
    """Integrate a single photon in the current worker."""
    if _W.get("use_numba", False):
        return _integrate_one_nb(task)

    i, j, iC = task
    bh = _W["blackhole"]
    acc = _W["acc_structure"]
    mode = _W["mode"]

    def rhs(lmbda, q):
        return bh.geodesics(q, lmbda)

    res = integrate(rhs, iC, (0.0, -_W["final_lmbda"]),
                    method=_W["method"], events=_W["events"],
                    rtol=_W["rtol"], atol=_W["atol"])

    if mode == "shadow":
        value = 0.0 if res.status == "horizon" else 100.0
    else:
        # Disk event is non-terminal — scan recorded equator crossings for
        # the first one inside the annulus (real disk emission point).
        value = 0.0
        y_ev = res.y_events[1] if len(res.y_events) > 1 else None
        if y_ev is not None and len(y_ev) > 0:
            in_edge = acc.in_edge
            out_edge = acc.out_edge
            for yev in y_ev:
                r_hit = float(yev[1])
                if in_edge <= r_hit <= out_edge:
                    I0 = acc.intensity(r_hit)
                    value = _doppler_inline(yev, I0, bh) if mode == "doppler" else float(I0)
                    break

    return (i, j, value)


def _auto_chunksize(n_tasks, n_workers):
    """Pick chunksize that balances IPC overhead and load imbalance."""
    return max(1, n_tasks // (n_workers * 4))


def trace_parallel(tasks, blackhole, acc_structure, detector,
                   *, n_workers=None, chunksize=None,
                   method="auto", rtol=1e-9, atol=1e-11,
                   mode="doppler", progress=True,
                   r_escape=None, final_lmbda=None):
    """
    Distribute photon tasks across a process pool with dynamic load balancing.

    Parameters
    ----------
    tasks : list of (i, j, iC)
        Pixel indices and initial conditions for each photon.
    blackhole, acc_structure, detector : objects
        Must be picklable. Passed once to each worker at init.
    n_workers : int, optional
        Defaults to os.cpu_count(). Use 1 to run serial (still via Pool for
        uniform code path; set serial=True in caller for true serial loop).
    chunksize : int, optional
        Tasks dispatched together. Smaller = better balance, more IPC overhead.
        Default: len(tasks) // (n_workers * 4).
    method, rtol, atol : integrator config
    mode : {"doppler", "no_doppler", "shadow"}
    progress : bool
        Print progress every ~1% of tasks to stdout.

    Returns
    -------
    image_data : ndarray shape (detector.x_pixels, detector.y_pixels)
    stats : dict
        {"wall_total": float, "n_workers": int, "chunksize": int}
    """
    n_tasks = len(tasks)
    if n_workers is None:
        n_workers = cpu_count()
    n_workers = max(1, n_workers)
    if chunksize is None:
        chunksize = _auto_chunksize(n_tasks, n_workers)
    r_escape, final_lmbda = _resolve_geometry(detector, r_escape, final_lmbda)

    # Fast path: numba threads (no multiprocessing). Works when the BH
    # exposes numba hooks and thin_disk has the numba arrays.
    use_threads = (_has_numba_hooks(blackhole)
                   and method in ("auto", "RK45_numba")
                   and hasattr(acc_structure, "_r_tbl"))
    if use_threads:
        return trace_threads_nb(
            tasks, blackhole, acc_structure, detector,
            n_workers=n_workers, method=method, rtol=rtol, atol=atol,
            mode=mode, progress=progress,
            r_escape=r_escape, final_lmbda=final_lmbda)

    tasks = list(tasks)
    random.shuffle(tasks)

    image = zeros([detector.x_pixels, detector.y_pixels])
    stats = {}

    # Warm up numba kernel in the parent before forking workers, so the
    # compiled code is inherited (saves N_workers × compile_time).
    use_numba = _has_numba_hooks(blackhole) and method in ("auto", "RK45_numba")
    if use_numba:
        y0 = np.asarray(tasks[0][2], dtype=np.float64)
        _solve_photon_nb(blackhole._rhs_nb, y0, -float(final_lmbda),
                         float(blackhole.EH), float(r_escape), rtol, atol)
        if mode == "doppler" and blackhole._omega_nb is not None:
            _ = blackhole._metric_nb(y0[:4])
            _ = blackhole._omega_nb(float(y0[1]))

    init_args = (blackhole, acc_structure, detector,
                 method, rtol, atol, mode, r_escape, final_lmbda)

    t_global_start = time.perf_counter()
    progress_every = max(1, n_tasks // 100)
    done = 0

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=init_args) as pool:
        for i, j, value in pool.imap_unordered(_integrate_one, tasks,
                                               chunksize=chunksize):
            image[i, j] = value
            done += 1
            if progress and (done % progress_every == 0 or done == n_tasks):
                pct = 100.0 * done / n_tasks
                sys.stdout.write(f"\r  {done:6d}/{n_tasks} ({pct:5.1f}%) "
                                 f"| workers={n_workers} | "
                                 f"chunksize={chunksize}")
                sys.stdout.flush()

    if progress:
        sys.stdout.write("\n")

    stats["t_end_global"] = time.perf_counter()
    stats["wall_total"] = stats["t_end_global"] - t_global_start
    stats["n_workers"] = n_workers
    stats["chunksize"] = chunksize
    return image, stats


def trace_serial(tasks, blackhole, acc_structure, detector,
                 *, method="auto", rtol=1e-9, atol=1e-11,
                 mode="doppler", progress=True,
                 r_escape=None, final_lmbda=None):
    """Serial fallback with the same return signature. Useful for debug."""
    # Prefer numba threads path with n_workers=1 if available.
    use_threads = (_has_numba_hooks(blackhole)
                   and method in ("auto", "RK45_numba")
                   and hasattr(acc_structure, "_r_tbl"))
    if use_threads:
        return trace_threads_nb(
            tasks, blackhole, acc_structure, detector,
            n_workers=1, method=method, rtol=rtol, atol=atol,
            mode=mode, progress=progress,
            r_escape=r_escape, final_lmbda=final_lmbda)

    r_escape, final_lmbda = _resolve_geometry(detector, r_escape, final_lmbda)
    _init_worker(blackhole, acc_structure, detector,
                 method, rtol, atol, mode, r_escape, final_lmbda)
    image = zeros([detector.x_pixels, detector.y_pixels])
    stats = {}
    t_global_start = time.perf_counter()
    n_tasks = len(tasks)
    progress_every = max(1, n_tasks // 100)
    for k, task in enumerate(tasks, 1):
        i, j, value = _integrate_one(task)
        image[i, j] = value
        if progress and (k % progress_every == 0 or k == n_tasks):
            pct = 100.0 * k / n_tasks
            sys.stdout.write(f"\r  {k:6d}/{n_tasks} ({pct:5.1f}%) | serial")
            sys.stdout.flush()
    if progress:
        sys.stdout.write("\n")
    stats["t_end_global"] = time.perf_counter()
    stats["wall_total"] = stats["t_end_global"] - t_global_start
    stats["n_workers"] = 1
    stats["chunksize"] = 1
    return image, stats
