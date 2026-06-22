"""
===============================================================================
ex27 - Kerr shadow: all five integrators vs OSIRIS Fig. 8
===============================================================================
Replicates the OSIRIS calibration scene (Kerr a = 0.98, equatorial observer at
r0 = 1000, image plane [-8, 8]) and benchmarks the FIVE thread-parallel Numba
integrators now available on Kerr -- the direct analogue of OSIRIS Fig. 8:

    RKDP45 / RKCK45 / RKF45 / Bulirsch-Stoer / Verlet

against the OSIRIS serial reference.  OSIRIS is a serial FORTRAN code (no MPI,
no CUDA; see its abstract) that evolves 1024^2 ~ 1e6 geodesics in ~1e3 s.  Here
every TARTARUS integrator renders the same image in seconds, on commodity
hardware, with no GPU -- the wall times are plotted on the same axes.

(Single-core fairness: ex26's strong-scaling study reports the 1-thread time T1;
this figure shows the real wall time with all Numba threads, TARTARUS's actual
performance.)

Run from the repository root:
    python ex27.kerr_vs_osiris.py
===============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from scr.common import _kerr_numba as kn

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
A_SPIN = 0.98
D = 1000.0
IOTA = math.pi / 2
SIDE = 8.0
RES = [64, 128, 256, 512, 1024, 2048]    # image resolutions (OSIRIS Fig. 8 sweep)
THREADS = 1                     # None -> all numba threads (real wall time)
OSIRIS_1024 = 1.0e3                # ~1000 s, OSIRIS serial FORTRAN (paper)

# Shared 5-integrator naming + pastel palette (identical to ex24/ex25/ex26/ex28)
METHODS = ["DP45", "CK45", "RKF45", "BS", "Verlet"]
NICE = {"DP45": "RKDP45", "CK45": "RKCK45", "RKF45": "RKF45",
        "BS": "Bulirsch-Stoer", "Verlet": "Verlet"}
COLOR = {"DP45": "#5E96C8", "CK45": "#E0956B", "RKF45": "#A07BC8",
         "BS": "#C77B92", "Verlet": "#6CB48A"}
LS = {"DP45": "-", "CK45": "--", "RKF45": "-.", "BS": ":", "Verlet": (0, (5, 1))}

SAVENAME = "kerr_vs_osiris"


# ---------------------------------------------------------------------------
def grid(n):
    ax = np.linspace(-SIDE, SIDE, n)
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


def study():
    """Wall time for every integrator at increasing resolution (Kerr)."""
    print(f"\n{'res':>8}  {'N_pix':>10}" + "".join(f"{NICE[m]:>16}" for m in METHODS))
    print("-" * (20 + 16 * len(METHODS)))
    times = {m: [] for m in METHODS}
    for n in RES:
        al, be = grid(n)
        row = f"{n:>5}^2  {n*n:>10}"
        for m in METHODS:
            _, _, el = kn.render_kerr_shadow(al, be, D=D, iota=IOTA,
                                             a=A_SPIN, method=m,
                                             nthreads=THREADS)
            times[m].append(el)
            row += f"{el:>15.2f}s"
        print(row, flush=True)
    return times


# ---------------------------------------------------------------------------
def make_plot(times):
    _paper_rc()
    x = list(range(len(RES)))

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for m in METHODS:
        ax.semilogy(x, times[m], color=COLOR[m], marker="o",
                    linestyle=LS[m], label=NICE[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"${n}^2$" for n in RES])
    ax.set_xlabel(r"$N_x \times N_y$")
    ax.set_ylabel("wall time [s]")
    ax.set_xlim(-0.3, len(RES) - 0.7)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(f"images/{SAVENAME}_timing.png", dpi=300, bbox_inches="tight")

    print(f"\nSaved: images/{SAVENAME}_timing.png")


# ---------------------------------------------------------------------------
def main():
    print("Compiling Kerr kernels (warmup, all five integrators)...")
    kn.warmup(D=D, iota=IOTA, a=A_SPIN)
    EH = 1.0 + math.sqrt(1.0 - A_SPIN**2)
    print(f"Kerr a={A_SPIN}, r0={D:.0f}, r_EH={EH:.3f}")
    print(f"OSIRIS reference: ~{OSIRIS_1024:.0f} s at 1024^2 (serial FORTRAN)")

    times = study()

    # Speedup vs OSIRIS at 1024^2 (the resolution OSIRIS reports)
    idx = RES.index(1024) if 1024 in RES else len(RES) - 1
    print("\n" + "=" * 60)
    print(f"  {'integrator':<18}{f't@{RES[idx]}^2 [s]':>14}{'speedup vs OSIRIS':>20}")
    print("-" * 60)
    for m in METHODS:
        t = times[m][idx]
        print(f"  {NICE[m]:<18}{t:>14.2f}{OSIRIS_1024 / t:>18.0f}x")
    print(f"  {'OSIRIS (serial)':<18}{OSIRIS_1024:>14.0f}{1:>18}x")
    print("=" * 60)

    make_plot(times)
    plt.show()


if __name__ == "__main__":
    main()
