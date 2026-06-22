"""
===============================================================================
ex26 - Strong-scaling benchmark (Schwarzschild shadow, RKDP45 Numba kernel)
===============================================================================
Thread strong-scaling of the production integrator (RKDP45, the Numba
Dormand-Prince kernel) on a fixed 1024^2 Schwarzschild shadow: speedup
S = T_1/T_p and efficiency E = S/p for p = 1..16 threads.

Parallel efficiency is essentially integrator-independent (the prange splits
pixels the same way for every method), so a single representative production
integrator is shown.

Run from the repository root:
    python ex26.performance_benchmark.py
===============================================================================
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

from scr.common import _geo_numba as gn

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
D = 100.0
IOTA = math.pi / 2
SIDE = 8.0
ATOL = RTOL = 1e-9
PROD = "DP45"               # production parallel integrator (numba Dormand-Prince)

S2_RES = 1024
S2_THREADS = [1, 2, 4, 8, 16]
S2_REPS = 3                 # best-of-N per measurement

NICE = {"DP45": "RKDP45"}
COLOR = {"DP45": "#5E96C8"}


def grid(n, side=SIDE):
    ax = np.linspace(-side, side, n)
    A, B = np.meshgrid(ax, ax, indexing="ij")
    return A.ravel(), B.ravel()


def _paper_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.65",
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def study2_strong():
    print("\n" + "=" * 70)
    print(f"STRONG SCALING - {PROD} (production), {S2_RES}^2 px, best-of-{S2_REPS}")
    print("=" * 70)
    al, be = grid(S2_RES)
    t1 = None
    sp, ef, tt = [], [], []
    for p in S2_THREADS:
        best = min(gn.render_shadow_numba(al, be, D=D, iota=IOTA, method=PROD,
                                          nthreads=p, atol=ATOL, rtol=RTOL)[2]
                   for _ in range(S2_REPS))
        if t1 is None:
            t1 = best
        tt.append(best)
        sp.append(t1 / best)
        ef.append(100 * t1 / best / p)
        print(f"  threads={p:2d}  time={best:7.3f}s  speedup={sp[-1]:5.2f}x  eff={ef[-1]:4.0f}%")
    return S2_THREADS, tt, sp, ef


def make_plot(s2):
    _paper_rc()
    th2, _, sp2, ef2 = s2

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    axa.plot(th2, sp2, "o-", color=COLOR[PROD], label=NICE[PROD])
    axa.plot(th2, th2, "k:", lw=1, label="ideal linear")
    axa.set_xlabel("workers")
    axa.set_ylabel(r"speedup  $S = T_1/T_p$")
    axa.set_xlim(1, 17)
    axa.legend(loc="lower right", frameon=True)

    axb.plot(th2, ef2, "o-", color=COLOR[PROD], label=NICE[PROD])
    axb.axhline(100, color="k", ls=":", lw=0.8)
    axb.set_xlabel("workers")
    axb.set_ylabel(r"efficiency  $E = S/p$  [%]")
    axb.set_xlim(1, 17)
    axb.legend(loc="lower right", frameon=True)

    fig.savefig("images/bench_strong_scaling.png", dpi=300, bbox_inches="tight")
    print("\nSaved: images/bench_strong_scaling.png")


def main():
    import numba
    print("=" * 70)
    print("SYSTEM DIAGNOSTICS")
    print("=" * 70)
    try:
        from numba import njit, prange
        @njit(parallel=True)
        def _probe(): return sum(1 for _ in prange(2))
        _probe()
        layer = numba.threading_layer()
    except Exception:
        layer = "unknown"
    print(f"  numba threading layer : {layer}")
    print(f"  numba threads         : {gn.config.NUMBA_NUM_THREADS}")
    print()
    print("Compiling numba kernels + warming up thread pool (excluded from timings)...")
    t = time.perf_counter()
    gn.warmup(D=D, iota=IOTA)
    print(f"  done in {time.perf_counter() - t:.1f}s")

    s2 = study2_strong()
    make_plot(s2)


if __name__ == "__main__":
    main()
