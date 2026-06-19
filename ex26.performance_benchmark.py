"""
===============================================================================
ex26 - Performance benchmark with THIS project's integrators (Schwarzschild)
===============================================================================
Uses the integrators that actually exist in this codebase -- LSODA, DOP853,
RK45 (in-house Dormand-Prince), Verlet -- NOT the paper's RKCK45/RKF45/BS.

  1) COST vs RESOLUTION   : numba production renderer, RK45 vs Verlet (the two
     methods that run inside the parallel @njit kernel), 64^2 -> 1024^2.

  2) STRONG SCALING       : RK45 (your production integrator), threads 1..32.
     Speedup S=T1/Tp, efficiency E=S/p. Compared against the multiprocessing
     (pure-Python) path. Expect saturation near the 16 physical cores.

  3) WEAK SCALING         : RK45, pixels-per-thread constant.

  4) INTEGRATOR PARETO    : all four methods (LSODA / DOP853 / RK45 / Verlet) on
     a small scene -- time per photon vs Hamiltonian-constraint accuracy. This
     mirrors your scr/common/tests/benchmark_integrators.py.

Run from the repository root:
    python ex26.performance_benchmark.py
===============================================================================
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

from scr.common import _geo_numba as gn
from scr.common import shadow_mp
from scr.common import photon_methods as pm
from scr.common.shadow_mp import ic_py
from scr.common.common import Hamiltonian
from scr.black_holes import schwarzschild

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
D = 100.0
IOTA = math.pi / 2
SIDE = 8.0
ATOL = RTOL = 1e-9
PROD = "RK45"            # production parallel integrator (= numba DP kernel)

S1_RES = [64, 128, 256, 512, 1024]
S1_METHODS = ["RK45", "Verlet"]      # parallel-capable methods
S1_THREADS = 16
S2_RES = 512
S2_THREADS = [1, 2, 4, 8, 16, 24, 32]
S3_PER_THREAD = 40000
S3_THREADS = [1, 2, 4, 8, 16, 32]
MP_RES = 48
MP_WORKERS = [1, 2, 4, 8, 16, 32]
PARETO_RES = 32         # small scene for the 4-method time-vs-precision study

NUMBA_NICE = {"RK45": "RK45 (Dormand-Prince)", "Verlet": "Verlet"}
NUMBA_COLOR = {"RK45": "tab:blue", "Verlet": "tab:green"}


def grid(n, side=SIDE):
    ax = np.linspace(-side, side, n)
    A, B = np.meshgrid(ax, ax, indexing="ij")
    return A.ravel(), B.ravel()


# ---------------------------------------------------------------------------
def study1_resolution():
    print("\n" + "=" * 70)
    print("STUDY 1 - cost vs resolution (numba), threads =", S1_THREADS)
    print("=" * 70)
    times = {m: [] for m in S1_METHODS}
    npx = [n * n for n in S1_RES]
    last_img = None
    for n in S1_RES:
        al, be = grid(n)
        row = f"  {n:>4}^2 = {n*n:>8} px:"
        for m in S1_METHODS:
            flag, H, el = gn.render_shadow_numba(al, be, D=D, iota=IOTA,
                                                 method=m, nthreads=S1_THREADS,
                                                 atol=ATOL, rtol=RTOL)
            times[m].append(el)
            row += f"  {NUMBA_NICE[m]}={el:7.3f}s"
            if m == PROD and n == S1_RES[-1]:
                last_img = flag.reshape(n, n)
        print(row)
    return npx, times, last_img


def study2_strong():
    print("\n" + "=" * 70)
    print(f"STUDY 2 - strong scaling, {PROD} (production), {S2_RES}^2 px")
    print("=" * 70)
    al, be = grid(S2_RES)
    t1 = None
    sp, ef, tt = [], [], []
    for p in S2_THREADS:
        best = min(gn.render_shadow_numba(al, be, D=D, iota=IOTA, method=PROD,
                                          nthreads=p, atol=ATOL, rtol=RTOL)[2]
                   for _ in range(3))
        if t1 is None:
            t1 = best
        tt.append(best)
        sp.append(t1 / best)
        ef.append(100 * t1 / best / p)
        print(f"  threads={p:2d}  time={best:7.3f}s  speedup={sp[-1]:5.2f}x  eff={ef[-1]:4.0f}%")
    return S2_THREADS, tt, sp, ef


def study3_weak():
    print("\n" + "=" * 70)
    print(f"STUDY 3 - weak scaling, {PROD}, {S3_PER_THREAD} px/thread")
    print("=" * 70)
    rng = np.random.RandomState(7)
    pmax = max(S3_THREADS)
    pool_a = rng.uniform(-SIDE, SIDE, S3_PER_THREAD * pmax)
    pool_b = rng.uniform(-SIDE, SIDE, S3_PER_THREAD * pmax)
    t1 = None
    tt, ef = [], []
    for p in S3_THREADS:
        N = S3_PER_THREAD * p
        al, be = pool_a[:N], pool_b[:N]
        best = min(gn.render_shadow_numba(al, be, D=D, iota=IOTA, method=PROD,
                                          nthreads=p, atol=ATOL, rtol=RTOL)[2]
                   for _ in range(2))
        if t1 is None:
            t1 = best
        tt.append(best)
        ef.append(100 * t1 / best)
        print(f"  threads={p:2d}  N={N:>9}  time={best:7.3f}s  weak-eff={ef[-1]:4.0f}%")
    return S3_THREADS, tt, ef


def study_mp():
    print("\n" + "=" * 70)
    print(f"MULTIPROCESSING - cross-check + process scaling, {MP_RES}^2 px")
    print("=" * 70)
    al, be = grid(MP_RES)
    print(f"  cross-check numba vs integrator.py ({PROD}):")
    f_nb, _, _ = gn.render_shadow_numba(al, be, D=D, iota=IOTA, method=PROD,
                                        nthreads=8, atol=ATOL, rtol=RTOL)
    f_mp, _ = shadow_mp.render_shadow_mp(al, be, D=D, iota=IOTA, method=PROD,
                                         nworkers=1, atol=ATOL, rtol=RTOL)
    print(f"    mismatched pixels: {int(np.sum(f_nb != f_mp))}/{al.size}")

    print(f"  process scaling ({PROD}, pure-Python integrators):")
    t1 = None
    sp, ef, tt = [], [], []
    for p in MP_WORKERS:
        _, el = shadow_mp.render_shadow_mp(al, be, D=D, iota=IOTA, method=PROD,
                                           nworkers=p, atol=ATOL, rtol=RTOL)
        if t1 is None:
            t1 = el
        tt.append(el)
        sp.append(t1 / el)
        ef.append(100 * t1 / el / p)
        print(f"    workers={p:2d}  time={el:7.3f}s  speedup={sp[-1]:5.2f}x  eff={ef[-1]:4.0f}%")
    return MP_WORKERS, tt, sp, ef


def study_pareto():
    print("\n" + "=" * 70)
    print(f"STUDY 4 - integrator Pareto (time/photon vs accuracy), {PARETO_RES}^2 px")
    print("=" * 70)
    bh = schwarzschild.BlackHole()
    rhs = lambda lam, q: bh.geodesics(q, lam)
    sin_i, cos_i = math.sin(IOTA), math.cos(IOTA)
    al, be = grid(PARETO_RES)
    lam_max = 2.0 * D
    EH_stop, R_esc = bh.EH + 0.01, 1.1 * D

    # Accuracy metric: MEDIAN |H| (robust). The mean is reported too, but for
    # fixed-step Verlet a few captured photons diverge near the horizon and
    # blow the mean up to ~1e23 -- the median reflects the typical photon.
    res = {}
    print(f"  {'method':<32}{'ms/photon':>12}{'median|H|':>12}{'mean|H|':>12}")
    for m in pm.METHODS:
        t0 = time.perf_counter()
        Hs = np.empty(al.size)
        for i in range(al.size):
            q0 = ic_py(al[i], be[i], D, sin_i, cos_i)
            _, Y = pm.integrate_photon(rhs, q0, (0.0, -lam_max), m,
                                       atol=ATOL, rtol=RTOL, r_stop=EH_stop,
                                       r_esc=R_esc, verlet_steps=int(lam_max / 0.1))
            h = abs(Hamiltonian(np.array([Y[-1]]), bh)[0])
            Hs[i] = h if np.isfinite(h) else np.inf
        dt = time.perf_counter() - t0
        med = float(np.median(Hs))
        mean = float(np.mean(Hs[np.isfinite(Hs)])) if np.any(np.isfinite(Hs)) else np.inf
        res[m] = (1000 * dt / al.size, med, mean)
        print(f"  {pm.NICE[m]:<32}{res[m][0]:>12.3f}{med:>12.2e}{mean:>12.2e}")
    return res


# ---------------------------------------------------------------------------
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


def make_plots(s1, s2, s3, mp, pareto):
    _paper_rc()
    npx, times, img = s1
    th2, _, sp2, ef2 = s2
    th3, tt3, ef3 = s3
    wk, _, spm, efm = mp

    # Fig 1: cost vs resolution
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for m in S1_METHODS:
        ax.loglog(npx, times[m], "o-", color=NUMBA_COLOR[m], label=NUMBA_NICE[m])
    ax.set_xlabel(r"$N_x \cdot N_y$ (pixels)")
    ax.set_ylabel("wall time [s]")
    ax.grid(visible=True, which="both")
    ax.legend(frameon=True)
    fig.savefig("images/bench_cost_vs_resolution.png", dpi=300, bbox_inches="tight")

    # Fig 2: strong scaling
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    axa.plot(th2, sp2, "o-", label=f"numba threads ({PROD})")
    axa.plot(wk, spm, "s--", label=f"multiprocessing ({PROD})")
    axa.plot(th2, th2, "k:", lw=1, label="ideal linear")
    axa.axvline(16, color="gray", ls=":", lw=0.8)
    axa.text(16.5, 1, "16 cores", color="gray", va="bottom", fontsize=9)
    axa.set_xlabel("workers (threads / processes)")
    axa.set_ylabel(r"speedup  $S = T_1/T_p$")
    axa.grid(visible=True, which="both")
    axa.legend(frameon=True)
    axb.plot(th2, ef2, "o-", label="numba threads")
    axb.plot(wk, efm, "s--", label="multiprocessing")
    axb.axhline(100, color="k", ls=":", lw=0.8)
    axb.axvline(16, color="gray", ls=":", lw=0.8)
    axb.set_xlabel("workers")
    axb.set_ylabel(r"efficiency  $E = S/p$  [%]")
    axb.grid(visible=True, which="both")
    axb.legend(frameon=True)
    fig.savefig("images/bench_strong_scaling.png", dpi=300, bbox_inches="tight")

    # Fig 3: weak scaling
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    axa.plot(th3, tt3, "o-", label="measured")
    axa.axhline(tt3[0], color="k", ls=":", lw=0.8, label="ideal (flat)")
    axa.set_ylim(0, max(tt3) * 1.3)
    axa.set_xlabel(r"threads (work $\propto$ threads)")
    axa.set_ylabel("wall time [s]")
    axa.grid(visible=True, which="both")
    axa.legend(frameon=True)
    axb.plot(th3, ef3, "o-")
    axb.axhline(100, color="k", ls=":", lw=0.8)
    axb.set_xlabel("threads")
    axb.set_ylabel(r"weak efficiency  $T_1/T_p$  [%]")
    axb.grid(visible=True, which="both")
    fig.savefig("images/bench_weak_scaling.png", dpi=300, bbox_inches="tight")

    # Fig 4: integrator Pareto — median |H| accuracy
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for m in pm.METHODS:
        tpp, med, mean = pareto[m]
        ax.scatter(tpp, med, s=80, color=pm.COLOR[m], zorder=3)
        note = pm.NICE[m].split(" (")[0]
        if not np.isfinite(mean) or mean > 1e3:
            note += "\n(diverges at horizon)"
        ax.annotate(note, (tpp, med), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time per photon [ms]")
    ax.set_ylabel(r"median $|H|$")
    ax.grid(visible=True, which="both")
    ax.text(0.5, -0.15,
            r"LSODA/DOP853: scipy (C); RK45/Verlet: pure Python. "
            r"In production, RK45 runs as the numba kernel ($\sim$1000$\times$ faster).",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.4")
    fig.savefig("images/bench_integrator_pareto.png", dpi=300, bbox_inches="tight")

    # Fig 5: shadow image
    if img is not None:
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.imshow(img.T, origin="lower", cmap="gray_r",
                  extent=[-SIDE, SIDE, -SIDE, SIDE])
        th = np.linspace(0, 2 * np.pi, 400)
        bc = 3 * np.sqrt(3)
        ax.plot(bc * np.cos(th), bc * np.sin(th), "r-", lw=1,
                label=r"$b_{\rm crit} = 3\sqrt{3}$")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")
        ax.legend(loc="upper right", frameon=True)
        fig.savefig("images/bench_shadow_image.png", dpi=300, bbox_inches="tight")

    print("\nSaved figures:")
    for f in ("bench_cost_vs_resolution", "bench_strong_scaling",
              "bench_weak_scaling", "bench_integrator_pareto",
              "bench_shadow_image"):
        print(f"  images/{f}.png")


def main():
    print("Compiling numba kernels (warmup, excluded from timings)...")
    t = time.perf_counter()
    gn.warmup(D=D, iota=IOTA)
    print(f"  done in {time.perf_counter() - t:.1f}s "
          f"(numba threads available: {gn.config.NUMBA_NUM_THREADS})")

    s1 = study1_resolution()
    s2 = study2_strong()
    s3 = study3_weak()
    mp = study_mp()
    pareto = study_pareto()
    make_plots(s1, s2, s3, mp, pareto)


if __name__ == "__main__":
    main()
