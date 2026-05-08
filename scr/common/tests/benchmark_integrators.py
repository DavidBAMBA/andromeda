"""
===============================================================================
Benchmark: compare integrators on Kerr geodesics (time vs. precision)
===============================================================================
Methods compared:
    - LSODA  (via solve_ivp, with events)
    - DOP853 (reference at tight tolerances)
    - RK45   (in-house adaptive Dormand-Prince with Hermite event refinement)
    - Verlet (2nd-order splitting, experimental)

For each (a, method) combination we measure:
    - wall time (total and per photon)
    - nfev (RHS evaluations)
    - intensity RMSE vs. DOP853 reference at rtol=atol=1e-13
    - max |H - H_0| on a sample of photons

Run from the repo root:
    python -m scr.common.benchmark_integrators
===============================================================================
"""
import time
import warnings
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from scr.black_holes import kerr
from scr.accretion_structures import thin_disk
from scr.detectors import image_plane
from scr.common import common
from scr.common.common import Image, Photon, Hamiltonian
from scr.common.integrator import integrate, make_events

warnings.filterwarnings("ignore")


METHODS = ("LSODA", "DOP853", "RK45", "Verlet")
TOL = dict(rtol=1e-9, atol=1e-11)
REF_TOL = dict(rtol=1e-13, atol=1e-13)
SPINS = (0.0, 0.5, 0.9)


def _build_scene(a, x_pixels=32):
    blackhole = kerr.BlackHole(a)
    detector = image_plane.detector(D=100, iota=pi/180 * 80,
                                    x_pixels=x_pixels, x_side=25,
                                    ratio="16:9")
    acc = thin_disk.structure(blackhole)
    scene = Image(blackhole, acc, detector)
    scene.create_photons()
    return scene


def _run_image(scene, method, tol):
    """Integrate every photon in the scene with the given method/tol."""
    common.set_integrator(method=method, **tol)
    nfev_total = 0
    intensities = np.zeros(len(scene.photon_list))

    t0 = time.perf_counter()
    for idx, p in enumerate(scene.photon_list):
        final_lmbda = 1.5 * scene.detector.D
        r_escape = 1.1 * scene.detector.D
        events = make_events(scene.blackhole,
                             acc_structure=scene.acc_structure,
                             r_escape=r_escape)
        res = integrate(common._rhs(scene.blackhole), p.iC,
                        (0.0, -final_lmbda), method=method,
                        events=events, **tol)
        nfev_total += res.nfev
        if res.status == "disk":
            p.fP = list(res.y[-1])
            I_0 = scene.acc_structure.intensity(p.fP[1])
            intensities[idx] = common.doppler_shift(p, I_0, scene.blackhole)
    wall = time.perf_counter() - t0
    return intensities, nfev_total, wall


def _hamiltonian_drift(scene, method, tol, n_sample=10):
    common.set_integrator(method=method, **tol)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(scene.photon_list), size=n_sample, replace=False)
    drifts = []
    for i in idx:
        p = scene.photon_list[i]
        final_lmbda = 1.5 * scene.detector.D
        r_escape = 1.1 * scene.detector.D
        events = make_events(scene.blackhole,
                             acc_structure=scene.acc_structure,
                             r_escape=r_escape)
        res = integrate(common._rhs(scene.blackhole), p.iC,
                        (0.0, -final_lmbda), method=method,
                        events=events, **tol)
        if res.y.shape[0] < 3:
            continue
        H = Hamiltonian(res.y, scene.blackhole)
        drifts.append(np.abs(H - H[0]).max())
    return float(np.mean(drifts)) if drifts else float("nan")


def main():
    rows = []
    for a in SPINS:
        print(f"\n=== Kerr a={a} ===")
        scene = _build_scene(a)
        n_ph = len(scene.photon_list)

        # Ground truth with DOP853 tight tolerances
        print(f"  Computing reference (DOP853, rtol={REF_TOL['rtol']:.0e})...")
        I_ref, nfev_ref, wall_ref = _run_image(scene, "DOP853", REF_TOL)

        for m in METHODS:
            try:
                I, nfev, wall = _run_image(scene, m, TOL)
                diff = I - I_ref
                mask = np.isfinite(diff)
                rmse = (float(np.sqrt(np.mean(diff[mask] ** 2)))
                        if mask.any() else float("nan"))
                drift = _hamiltonian_drift(scene, m, TOL)
                rows.append((a, m, wall, wall/n_ph*1000,
                             nfev/n_ph, rmse, drift))
                print(f"  {m:7s}  wall={wall:7.2f}s  "
                      f"t/ph={wall/n_ph*1000:6.2f}ms  "
                      f"nfev/ph={nfev/n_ph:6.0f}  "
                      f"RMSE={rmse:.2e}  "
                      f"driftH={drift:.2e}")
            except Exception as e:
                print(f"  {m:7s}  FAILED: {e}")
                rows.append((a, m, float("nan"), float("nan"),
                             float("nan"), float("nan"), float("nan")))

    _summary(rows)
    _plot_pareto(rows)


def _summary(rows):
    print("\n\n========== SUMMARY ==========")
    header = ("a", "method", "wall[s]", "t/ph[ms]",
              "nfev/ph", "RMSE", "drift H")
    print("{:>4} {:>7} {:>9} {:>9} {:>9} {:>12} {:>12}".format(*header))
    for r in rows:
        print("{:>4.1f} {:>7s} {:>9.2f} {:>9.2f} {:>9.0f} {:>12.2e} {:>12.2e}"
              .format(*r))


def _plot_pareto(rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"LSODA": "C0", "DOP853": "C1", "RK45": "C2", "Verlet": "C3"}
    markers = {0.0: "o", 0.5: "s", 0.9: "^"}
    for a, m, wall, tph, nfev, rmse, drift in rows:
        if np.isnan(rmse) or rmse == 0:
            continue
        ax.scatter(tph, rmse, c=colors.get(m, "k"),
                   marker=markers.get(a, "x"), s=60,
                   edgecolor="black", linewidth=0.5,
                   label=f"{m} a={a}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time per photon [ms]")
    ax.set_ylabel("RMSE intensity vs DOP853@1e-13")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig("images/benchmark_integrators.png", dpi=120)
    print("\nPareto plot saved to images/benchmark_integrators.png")


if __name__ == "__main__":
    main()
