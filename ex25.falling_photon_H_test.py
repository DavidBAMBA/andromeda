"""
===============================================================================
ex25 - Hamiltonian-constraint drift for THIS project's integrators (falling)
===============================================================================
Schwarzschild analogue of the OSIRIS Fig. 5 test (a null geodesic that FALLS
into the horizon), comparing the integrators that ACTUALLY exist here:

    LSODA / DOP853 / RK45 (in-house DP) / Verlet

(not the paper's RKCK45 / RKF45 / Bulirsch-Stoer). Impact parameter is
sub-critical (b < b_crit = 3*sqrt(3)) so the photon is captured; integration
stops just outside the horizon (R_STOP) to avoid the coordinate singularity.

Run from the repository root:
    python "ex25.falling_photon_H_test.py"
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from scr.black_holes import schwarzschild
from scr.common import photon_methods as pm
from scr.common.common import Hamiltonian

import warnings
warnings.filterwarnings("ignore")


LAMBDA_MAX = 200.0
ATOL = RTOL = 1e-10
VERLET_STEPS = 10000
R0 = 100.0
B = 3.0
R_STOP = 2.05
SAVENAME = "falling_photon_H_test_schwarzschild"


def null_initial_conditions(blackhole, r0=R0, b=B):
    """Captured null geodesic with H = 0 to machine precision."""
    theta0 = phi0 = np.pi / 2.0
    x0 = [0.0, r0, theta0, phi0]
    k_t = -1.0
    k_th = 1.0
    k_ph = -np.sqrt(b**2 - k_th**2)
    gtt, grr, gthth, gphph, gtph = blackhole.inverse_metric(x0)
    k_r2 = -(gtt * k_t**2 + gthth * k_th**2 + gphph * k_ph**2) / grr
    if k_r2 <= 0:
        raise ValueError("Non-physical k_r^2 < 0; adjust b / momenta.")
    return x0 + [k_t, -np.sqrt(k_r2), k_th, k_ph]


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


def run_all(blackhole, y0):
    rhs = lambda lam, q: blackhole.geodesics(q, lam)
    out = {}
    for m in pm.METHODS:
        T, Y = pm.integrate_photon(rhs, y0, (0.0, LAMBDA_MAX), m,
                                   atol=ATOL, rtol=RTOL, r_stop=R_STOP,
                                   verlet_steps=VERLET_STEPS)
        out[m] = (T, np.abs(Hamiltonian(Y, blackhole)), len(T) - 1)
    return out


def main():
    blackhole = schwarzschild.BlackHole()
    y0 = null_initial_conditions(blackhole)

    print("=" * 70)
    print("Schwarzschild null geodesic - falling photon (this codebase's methods)")
    print("=" * 70)
    print(f"  r0 = {y0[1]:.1f}, b = {B} (b_crit = {3*np.sqrt(3):.4f}) => captured")
    print(f"  integration stops at r_stop = {R_STOP} (horizon r_EH = {blackhole.EH})")
    H0 = Hamiltonian(np.array([y0]), blackhole)[0]
    print(f"  H(lambda=0) = {H0:+.3e}  (machine zero)\n")

    results = run_all(blackhole, y0)

    print(f"{'method':<32}{'steps':>8}{'max|H|':>14}{'final|H|':>14}")
    print("-" * 68)
    for m in pm.METHODS:
        lam, H, nsteps = results[m]
        print(f"{pm.NICE[m]:<32}{nsteps:>8}{H.max():>14.3e}{H[-1]:>14.3e}")
    print()

    _paper_rc()

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for m in pm.METHODS:
        lam, H, _ = results[m]
        ax.semilogy(lam, np.clip(H, 1e-18, None), color=pm.COLOR[m], lw=1.5,
                    label=pm.NICE[m])
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$|H|$")
    ax.grid(visible=True, which="both")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(f"images/{SAVENAME}_overlay.png", dpi=300, bbox_inches="tight")

    print(f"Saved: images/{SAVENAME}_overlay.png")
    plt.show()


if __name__ == "__main__":
    main()
