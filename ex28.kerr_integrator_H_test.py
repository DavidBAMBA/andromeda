"""
===============================================================================
ex28 - Hamiltonian-constraint drift on KERR for all in-house integrators
===============================================================================
Kerr (a = 0.98) analogue of the OSIRIS Fig. 4 test (a null geodesic that
ESCAPES, where H = 1/2 g^{mu nu} p_mu p_nu must stay 0), now that every
integrator runs on Kerr -- not just DP45.  Two complementary views:

  (1) |H(lambda)| along a single escaping equatorial geodesic, traced with the
      five in-house integrators that the Numba shadow kernels mirror
      (RKDP45 / RKCK45 / RKF45 / Bulirsch-Stoer / Verlet).  This is the direct
      Kerr counterpart of ex24 (Schwarzschild) and OSIRIS Fig. 4.

  (2) A wall-time table for the same five integrators rendering the full Kerr
      shadow with the thread-parallel Numba kernels (scr/common/_kerr_numba ->
      _shadow_numba), demonstrating that all of them now run on Kerr.

Run from the repository root:
    python "ex28.kerr_integrator_H_test.py"
===============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from scr.black_holes import kerr
from scr.common import integrator
from scr.common.common import Hamiltonian
from scr.common import _kerr_numba as kn

import warnings
warnings.filterwarnings("ignore")


# --- single-geodesic |H(lambda)| test --------------------------------------
A_SPIN = 0.98
R0 = 100.0
B = 6.0                 # impact parameter L/E (> b_crit -> escapes)
LAMBDA_MAX = 300.0
ATOL = RTOL = 1e-10
VERLET_STEPS = 30000

# --- Numba shadow timing test ----------------------------------------------
D_SHADOW = 1000.0
IOTA = math.pi / 2
SIDE = 8.0
TIMING_RES = [128, 256]

METHODS = ["DP45", "CK45", "RKF45", "BS", "Verlet"]
NICE = {"DP45": "RKDP45", "CK45": "RKCK45", "RKF45": "RKF45",
        "BS": "Bulirsch-Stoer", "Verlet": "Verlet"}
COLOR = {"DP45": "#5E96C8", "CK45": "#E0956B", "RKF45": "#A07BC8",
         "BS": "#C77B92", "Verlet": "#6CB48A"}

SAVENAME = "kerr_integrator_H_test"


def kerr_null_ic(blackhole, r0=R0, b=B):
    """Escaping equatorial Kerr null geodesic with H = 0 to machine precision.

    k_r is solved from the full Kerr null condition (incl. the g^{t phi} cross
    term); the photon dives in (k_r < 0) to periapsis and escapes (b > b_crit).
    """
    theta0, phi0 = np.pi / 2.0, 0.0
    x0 = [0.0, r0, theta0, phi0]
    k_t = -1.0                                  # E = 1
    k_th = 0.0                                  # equatorial stays equatorial
    k_ph = b                                    # L = b (prograde)
    gtt, grr, gthth, gphph, gtph = blackhole.inverse_metric(x0)
    k_r2 = -(gtt * k_t**2 + 2.0 * gtph * k_t * k_ph
             + gthth * k_th**2 + gphph * k_ph**2) / grr
    if k_r2 <= 0:
        raise ValueError("Non-physical k_r^2 < 0; adjust b / momenta.")
    return x0 + [k_t, -np.sqrt(k_r2), k_th, k_ph]


def _paper_rc():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.65", "lines.linewidth": 1.5,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.top": True, "ytick.right": True,
        "grid.alpha": 0.3, "grid.linestyle": ":", "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def trace_one(blackhole, y0, method):
    """Trace a single geodesic with ``method``; return (T, |H|, nsteps)."""
    rhs = lambda lam, q: blackhole.geodesics(q, lam)
    r_stop = blackhole.EH + 0.05
    stop = lambda t, y: y[1] <= r_stop
    if method in ("DP45", "CK45", "RKF45"):
        T, Y = integrator.rk_adaptive(rhs, 0.0, y0, LAMBDA_MAX, method=method,
                                      atol=ATOL, rtol=RTOL, stop=stop)
    elif method == "BS":
        T, Y = integrator.bulirsch_stoer(rhs, 0.0, y0, LAMBDA_MAX,
                                         atol=ATOL, rtol=RTOL, stop=stop)
    elif method == "Verlet":
        T, Y = integrator.verlet(rhs, 0.0, y0, LAMBDA_MAX,
                                 n_steps=VERLET_STEPS, stop=stop)
    else:
        raise ValueError(method)
    return T, np.abs(Hamiltonian(Y, blackhole)), len(T) - 1


def constraint_test(blackhole, y0):
    print("=" * 70)
    print(f"Kerr null geodesic - escaping photon (a = {A_SPIN})")
    print("=" * 70)
    H0 = Hamiltonian(np.array([y0]), blackhole)[0]
    print(f"  r0 = {y0[1]:.1f}, b = {B}, equatorial; r_EH = {blackhole.EH:.4f}")
    print(f"  H(lambda=0) = {H0:+.3e}  (machine zero)\n")

    results = {m: trace_one(blackhole, y0, m) for m in METHODS}

    print(f"{'method':<18}{'steps':>8}{'max|H|':>14}{'final|H|':>14}")
    print("-" * 54)
    for m in METHODS:
        lam, H, nsteps = results[m]
        print(f"{NICE[m]:<18}{nsteps:>8}{H.max():>14.3e}{H[-1]:>14.3e}")
    print()

    _paper_rc()
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for m in METHODS:
        lam, H, _ = results[m]
        ax.semilogy(lam, np.clip(H, 1e-18, None), color=COLOR[m], lw=1.5,
                    label=NICE[m])
    ax.set_xlim(0, max(results[m][0][-1] for m in METHODS))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$|H|$")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(f"images/{SAVENAME}_overlay.png", dpi=300, bbox_inches="tight")
    print(f"Saved: images/{SAVENAME}_overlay.png")


def grid(n, side=SIDE):
    ax = np.linspace(-side, side, n)
    A, B_ = np.meshgrid(ax, ax, indexing="ij")
    return A.ravel(), B_.ravel()


def timing_test():
    print("\n" + "=" * 70)
    print(f"Numba Kerr shadow wall time per integrator (a={A_SPIN}, r0={D_SHADOW:.0f})")
    print("=" * 70)
    print("Warming up all Kerr kernels (compile)...")
    kn.warmup(D=D_SHADOW, iota=IOTA, a=A_SPIN)

    header = f"{'res':>8}" + "".join(f"{NICE[m]:>16}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for n in TIMING_RES:
        al, be = grid(n)
        row = f"{n:>5}^2"
        for m in METHODS:
            _, _, el = kn.render_kerr_shadow(al, be, D=D_SHADOW, iota=IOTA,
                                             a=A_SPIN, method=m)
            row += f"{el:>14.2f}s"
        print(row)
    print("\n(all five integrators now render the Kerr shadow in Numba)")


def main():
    blackhole = kerr.BlackHole(A_SPIN)
    y0 = kerr_null_ic(blackhole)
    constraint_test(blackhole, y0)
    timing_test()
    plt.show()


if __name__ == "__main__":
    main()
