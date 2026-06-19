"""
===============================================================================
ex27 - Kerr shadow timing: apples-to-apples vs OSIRIS (Fig. 8 setup)
===============================================================================
Replicates the OSIRIS calibration scene -- Kerr a = 0.98, equatorial observer
at r0 = 1000, image plane [-8, 8] -- and times the production numba RK45
(Dormand-Prince) renderer at increasing resolution, BOTH single-thread (to
match serial OSIRIS) and at full thread count.

OSIRIS reference (paper text + Fig. 8): it is serial (no MPI/CUDA) and evolves
1024^2 ~ 1e6 geodesics in "the same order of time as GRay" ~ 1e3 s.

Run from the repository root:
    python ex27.kerr_vs_osiris.py
===============================================================================
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

from scr.common import _kerr_numba as kn

import warnings
warnings.filterwarnings("ignore")

A_SPIN = 0.98
D = 1000.0
IOTA = math.pi / 2
SIDE = 8.0
RES = [256, 512, 1024]
OSIRIS_1024 = 1.0e3          # seconds, order of magnitude from the paper


def grid(n):
    ax = np.linspace(-SIDE, SIDE, n)
    A, B = np.meshgrid(ax, ax, indexing="ij")
    return A.ravel(), B.ravel()


def main():
    print("Compiling Kerr kernel (warmup)...")
    t = time.perf_counter()
    kn.warmup(D=D, iota=IOTA, a=A_SPIN)
    print(f"  done in {time.perf_counter() - t:.1f}s")
    EH = 1.0 + math.sqrt(1.0 - A_SPIN**2)
    print(f"\nOSIRIS Fig. 8 setup: Kerr a={A_SPIN}, r0={D:.0f}, image [-{SIDE},{SIDE}]^2, "
          f"horizon r_EH={EH:.3f}\n")

    npx, t1, t32 = [], [], []
    img1024 = None
    print(f"{'resolution':>12}{'pixels':>12}{'t(1 thr)':>12}{'t(32 thr)':>12}"
          f"{'us/photon(1)':>14}")
    print("-" * 62)
    for n in RES:
        al, be = grid(n)
        # single thread (fair vs serial OSIRIS)
        flag, H, e1 = kn.render_kerr_shadow(al, be, D=D, iota=IOTA, a=A_SPIN,
                                            nthreads=1)
        # full machine
        e32 = min(kn.render_kerr_shadow(al, be, D=D, iota=IOTA, a=A_SPIN,
                                        nthreads=32)[2] for _ in range(2))
        npx.append(n * n)
        t1.append(e1)
        t32.append(e32)
        print(f"{n:>9}^2{n*n:>12}{e1:>11.2f}s{e32:>11.3f}s{1e6*e1/(n*n):>14.2f}")
        if n == 1024:
            img1024 = flag.reshape(n, n)

    # --- comparison vs OSIRIS at 1024^2 ---
    i1024 = RES.index(1024)
    print("\n" + "=" * 62)
    print("COMPARISON @ 1024^2 (~1e6 geodesics):")
    print(f"  OSIRIS (serial, Kerr a=0.98, r0=1000)   ~ {OSIRIS_1024:.0f} s   (paper)")
    print(f"  this code, 1 thread                       {t1[i1024]:.1f} s "
          f"  -> {OSIRIS_1024/t1[i1024]:.0f}x faster per core")
    print(f"  this code, 32 threads                     {t32[i1024]:.2f} s "
          f"  -> {OSIRIS_1024/t32[i1024]:.0f}x faster wall-clock")
    print("=" * 62)

    # --- plot: cost vs resolution with OSIRIS reference ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(npx, t1, "o-", label="this code (numba RK45, 1 thread)")
    ax.loglog(npx, t32, "s-", label="this code (numba RK45, 32 threads)")
    ax.scatter([1024**2], [OSIRIS_1024], marker="*", s=260, color="red",
               zorder=5, label="OSIRIS @1024$^2$ (serial, ~10$^3$ s)")
    ax.set_xlabel(r"$N_x \cdot N_y$ (pixels)")
    ax.set_ylabel("wall time [s]")
    ax.set_title(f"Kerr shadow cost vs resolution (a={A_SPIN}, r0={D:.0f}) vs OSIRIS")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/kerr_vs_osiris_timing.png", dpi=150)

    # --- shadow image ---
    if img1024 is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(img1024.T, origin="lower", cmap="gray_r",
                   extent=[-SIDE, SIDE, -SIDE, SIDE])
        ax2.set_title(f"Kerr shadow a={A_SPIN} (1024$^2$, r0={D:.0f}, equatorial)")
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(r"$\beta$")
        fig2.tight_layout()
        fig2.savefig("images/kerr_vs_osiris_shadow.png", dpi=150)

    print("\nSaved: images/kerr_vs_osiris_timing.png, images/kerr_vs_osiris_shadow.png")


if __name__ == "__main__":
    main()
