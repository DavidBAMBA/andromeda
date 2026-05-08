"""
===============================================================================
Benchmark de paralelización: scaling, chunksize sweep, balance de carga
===============================================================================
Run from repo root:
    conda run -n engrenage python -m scr.common.benchmark_parallel

Opciones:
    --scaling      solo el test de speedup 1->N cores
    --chunksize    solo el sweep de chunksize
    --balance      solo el histograma de carga
    (sin flags = los tres)
===============================================================================
"""
import argparse
import os
import time
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from scr.black_holes import kerr
from scr.accretion_structures import thin_disk
from scr.detectors import image_plane
from scr.common.common import Image
from scr.common.parallel import trace_parallel, trace_serial


# ─── Escena fija para benchmarks ─────────────────────────────────────────────
A_SPIN    = 0.9
X_PIXELS  = 64   # 64 x 36 = 2304 fotones
X_SIDE    = 25
D         = 100
IOTA      = pi/180 * 80

N_CPU     = os.cpu_count()


def build_scene():
    bh  = kerr.BlackHole(A_SPIN)
    det = image_plane.detector(D=D, iota=IOTA,
                               x_pixels=X_PIXELS, x_side=X_SIDE,
                               ratio="16:9")
    acc = thin_disk.structure(bh)
    img = Image(bh, acc, det)
    img.create_photons()
    return img


def photon_tasks(img):
    return [(p.i, p.j, p.iC) for p in img.photon_list]


# ─── A. Scaling 1 -> N ───────────────────────────────────────────────────────
def scaling_test():
    print(f"\n╔══ SCALING TEST ({X_PIXELS}x{int(X_PIXELS*9/16)} = "
          f"{X_PIXELS*int(X_PIXELS*9/16)} fotones, CPU={N_CPU}) ══╗")
    img = build_scene()
    tasks = photon_tasks(img)

    n_workers_list = sorted(set([1, 2, 4, 8, 16, N_CPU]))
    results = []
    t_ref = None
    for nw in n_workers_list:
        if nw > N_CPU:
            continue
        t0 = time.perf_counter()
        if nw == 1:
            _, stats = trace_serial(tasks, img.blackhole, img.acc_structure,
                                    img.detector, progress=False)
        else:
            _, stats = trace_parallel(tasks, img.blackhole, img.acc_structure,
                                      img.detector, n_workers=nw,
                                      progress=False)
        t = time.perf_counter() - t0
        if t_ref is None:
            t_ref = t
        speedup = t_ref / t
        eff = speedup / nw
        results.append((nw, t, speedup, eff))
        print(f"  n_workers={nw:3d}  wall={t:6.2f}s  "
              f"speedup={speedup:5.2f}x  eff={eff*100:5.1f}%")

    # Plot
    nw = np.array([r[0] for r in results])
    sp = np.array([r[2] for r in results])
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(nw, sp, "o-", label="medido", lw=2, ms=8)
    ax.plot(nw, nw, "k--", alpha=0.5, label="ideal (linear)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_workers"); ax.set_ylabel("speedup")
    ax.set_xticks(nw); ax.set_xticklabels(nw)
    ax.set_yticks(nw); ax.set_yticklabels(nw)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(f"Scaling, Kerr a={A_SPIN}, {len(tasks)} fotones")
    plt.tight_layout()
    plt.savefig("images/bench_scaling.png", dpi=120)
    print("  Plot → images/bench_scaling.png")
    return results


# ─── B. Chunksize sweep ──────────────────────────────────────────────────────
def chunksize_sweep():
    print(f"\n╔══ CHUNKSIZE SWEEP (n_workers = {N_CPU // 2}) ══╗")
    img = build_scene()
    tasks = photon_tasks(img)
    nw = max(2, N_CPU // 2)
    chunks = [1, 4, 16, 64, 256, 1024]
    chunks = [c for c in chunks if c <= len(tasks)]
    results = []
    for cs in chunks:
        t0 = time.perf_counter()
        _, _ = trace_parallel(tasks, img.blackhole, img.acc_structure,
                              img.detector, n_workers=nw, chunksize=cs,
                              progress=False)
        t = time.perf_counter() - t0
        results.append((cs, t))
        print(f"  chunksize={cs:5d}  wall={t:6.2f}s")

    cs_arr = np.array([r[0] for r in results])
    t_arr  = np.array([r[1] for r in results])
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(cs_arr, t_arr, "o-", lw=2, ms=8)
    ax.set_xscale("log")
    ax.set_xlabel("chunksize"); ax.set_ylabel("wall [s]")
    ax.set_title(f"Chunksize sweep, n_workers={nw}")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/bench_chunksize.png", dpi=120)
    print("  Plot → images/bench_chunksize.png")
    best_cs, best_t = min(results, key=lambda r: r[1])
    print(f"  Mejor: chunksize={best_cs} ({best_t:.2f}s)")
    return results


# ─── C. Histograma de carga ──────────────────────────────────────────────────
def balance_test(n_workers=None):
    if n_workers is None:
        n_workers = max(2, N_CPU // 2)
    print(f"\n╔══ BALANCE HISTOGRAM (n_workers={n_workers}) ══╗")
    img = build_scene()
    tasks = photon_tasks(img)
    _, stats = trace_parallel(tasks, img.blackhole, img.acc_structure,
                              img.detector, n_workers=n_workers,
                              progress=False)

    pids = np.array(stats["pid"])
    walls = np.array(stats["wall"])
    t_start = np.array(stats["t_start"])

    # Tiempo total por PID
    unique_pids = sorted(np.unique(pids))
    total_per_pid = {pid: walls[pids == pid].sum() for pid in unique_pids}
    n_tasks_per_pid = {pid: int(np.sum(pids == pid)) for pid in unique_pids}

    print("  PID           tareas   t_total[s]   mean[ms]")
    for pid in unique_pids:
        tp = total_per_pid[pid]
        n  = n_tasks_per_pid[pid]
        print(f"  {pid:8d}     {n:5d}   {tp:9.3f}   {1000*tp/n:7.2f}")

    totals = np.array(list(total_per_pid.values()))
    mean_t = float(totals.mean())
    std_t  = float(totals.std())
    print(f"\n  Balance: media={mean_t:.3f}s, std={std_t:.3f}s, "
          f"CV={100*std_t/mean_t:.1f}%  (ideal: CV → 0)")

    # Plot: (a) boxplot tiempos por worker, (b) timeline de ocupación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Distribución tiempos por task agrupados por PID
    data_by_pid = [walls[pids == pid] * 1000 for pid in unique_pids]
    ax1.boxplot(data_by_pid, labels=[f"W{i+1}" for i in range(len(unique_pids))],
                showfliers=True)
    ax1.set_ylabel("wall per task [ms]")
    ax1.set_title(f"Distribución de costo por worker (n={n_workers})")
    ax1.grid(alpha=0.3)

    # (b) Timeline: barras cuando cada worker está activo
    t0_global = float(t_start.min())
    for i, pid in enumerate(unique_pids):
        mask = pids == pid
        starts = t_start[mask] - t0_global
        durs   = walls[mask]
        ax2.barh([i]*int(mask.sum()), durs, left=starts,
                 height=0.8, alpha=0.7)
    ax2.set_yticks(range(len(unique_pids)))
    ax2.set_yticklabels([f"W{i+1}" for i in range(len(unique_pids))])
    ax2.set_xlabel("tiempo [s] desde inicio")
    ax2.set_title("Timeline de ocupación por worker")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/bench_balance.png", dpi=120)
    print("  Plot → images/bench_balance.png")
    return stats


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling",   action="store_true")
    parser.add_argument("--chunksize", action="store_true")
    parser.add_argument("--balance",   action="store_true")
    args = parser.parse_args()
    run_all = not (args.scaling or args.chunksize or args.balance)

    if run_all or args.scaling:   scaling_test()
    if run_all or args.chunksize: chunksize_sweep()
    if run_all or args.balance:   balance_test()


if __name__ == "__main__":
    main()
