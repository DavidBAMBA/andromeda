"""
===============================================================================
ex25 - Hamiltonian-constraint drift for THIS project's integrators (falling)
===============================================================================
Schwarzschild analogue of the OSIRIS Fig. 5 test (a null geodesic that FALLS
into the horizon), comparing the FIVE integrators that TARTARUS implements as
Numba shadow kernels (scr/common/_shadow_numba) and that run, identically, on
the in-house numpy backend (scr/common/integrator):

    RKDP45 / RKCK45 / RKF45 / Bulirsch-Stoer / Verlet

This is the exact same set OSIRIS compares in its Fig. 5 (RKDP45/RKCK45/RKF45/BS)
plus Verlet.  Impact parameter is sub-critical (b < b_crit = 3*sqrt(3)) so the
photon is captured; integration stops just outside the horizon (R_STOP) to avoid
the coordinate singularity.

Run from the repository root:
    python "ex25.falling_photon_H_test.py"
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from scr.black_holes import schwarzschild
from scr.common import integrator
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

# The five in-house integrators (1:1 twins of the Numba shadow kernels).
METHODS = ["DP45", "CK45", "RKF45", "BS", "Verlet"]
NICE = {"DP45": "RKDP45", "CK45": "RKCK45", "RKF45": "RKF45",
        "BS": "Bulirsch-Stoer", "Verlet": "Verlet"}
COLOR = {"DP45": "#5E96C8", "CK45": "#E0956B", "RKF45": "#A07BC8",
         "BS": "#C77B92", "Verlet": "#6CB48A"}
LS = {"DP45": "-", "CK45": "--", "RKF45": "-.", "BS": ":",
      "Verlet": (0, (5, 1))}


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


def trace_one(blackhole, y0, method):
    """Trace a single captured geodesic with ``method``; return (T, Y).

    Integration stops just outside the horizon (R_STOP) to avoid the
    coordinate singularity.
    """
    rhs = lambda lam, q: blackhole.geodesics(q, lam)
    stop = lambda t, y: y[1] <= R_STOP
    if method in ("DP45", "CK45", "RKF45"):
        return integrator.rk_adaptive(rhs, 0.0, y0, LAMBDA_MAX, method=method,
                                      atol=ATOL, rtol=RTOL, stop=stop)
    if method == "BS":
        return integrator.bulirsch_stoer(rhs, 0.0, y0, LAMBDA_MAX,
                                         atol=ATOL, rtol=RTOL, stop=stop)
    if method == "Verlet":
        return integrator.verlet(rhs, 0.0, y0, LAMBDA_MAX,
                                 n_steps=VERLET_STEPS, stop=stop)
    raise ValueError(method)


def run_all(blackhole, y0):
    out = {}
    for m in METHODS:
        T, Y = trace_one(blackhole, y0, m)
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
                    linestyle=LS[m], label=NICE[m])
    ax.set_xlim(0, max(results[m][0][-1] for m in METHODS))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$|H|$")
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                          numticks=500))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(f"images/{SAVENAME}_overlay.png", dpi=300, bbox_inches="tight")

    print(f"Saved: images/{SAVENAME}_overlay.png")
    plt.show()


if __name__ == "__main__":
    main()
