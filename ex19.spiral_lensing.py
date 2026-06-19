"""
===============================================================================
ex19 — Spiral-galaxy lensing with full cosmology
===============================================================================
Capstone for the realistic-lensing extension. Combines:

    * scr.common.cosmology.SceneGeometry  (z_L, z_S, M_lens) -> D_L,D_S,D_LS
    * scr.sources.light_profiles.LogSpiral (cosine-modulated logarithmic arms)
    * scr.lens_metrics.sie  (galaxy-scale SIE lens)
    * SourcePlane.from_redshifts                (cosmology-driven SourcePlane)
    * scr.common.lens_diagnostics  (magnification + critical/caustic curves)

The scene mimics a typical galaxy-galaxy strong-lensing system: an early-type
galaxy at z_L=0.5 lensing a face-on spiral disk at z_S=2.0. The Einstein-ring
angular radius is set by sigma_v of the SIE; the rendered ring should host
two distorted spiral arms.

Run from repo root:
    MPLCONFIGDIR=/tmp/mplconfig python ex19.spiral_lensing.py
===============================================================================
"""
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scr.common.common import set_ray_bounds
from scr.common.cosmology import LambdaCDM, SceneGeometry, C_SI
from scr.common.lens_diagnostics import (critical_and_caustic_curves,
                                         log_abs_magnification,
                                         magnification_map)
from scr.common.lens_image import LensImage, SourcePlane
from scr.detectors import image_plane
from scr.lens_metrics import sie
from scr.sources.light_profiles import LogSpiral


def _norm(img):
    peak = img.max()
    return img / peak if peak > 0.0 else img


def main():
    # ---------------------------------------------------------------- #
    # 1. Cosmology + geometry: galaxy-scale lens at z=0.5, source at z=2.
    # ---------------------------------------------------------------- #
    cosmo = LambdaCDM()                                # Planck18 default
    M_lens_Msun = 3.0e11
    scene = SceneGeometry(z_L=0.5, z_S=2.0,
                          M_lens_Msun=M_lens_Msun, cosmology=cosmo)
    print(f"Cosmology: H0={cosmo.H0}, Om={cosmo.Om}")
    print(f"Distances [Mpc]: D_L={scene.D_L_Mpc:.1f}, "
          f"D_S={scene.D_S_Mpc:.1f}, D_LS={scene.D_LS_Mpc:.1f}")
    print(f"Distances [M_lens]: D_L={scene.D_L:.3e}, "
          f"D_S={scene.D_S:.3e}, D_LS={scene.D_LS:.3e}")

    # ---------------------------------------------------------------- #
    # 2. SIE lens: sigma_v in units of c.
    # ---------------------------------------------------------------- #
    sigma_v_kms = 280.0
    sigma_v_c = (sigma_v_kms * 1000.0) / C_SI
    lens = sie.LensMetric(sigma_v=sigma_v_c, q_ax=0.7)

    # Predicted Einstein-ring impact parameter (in geometrized M_lens units).
    # Same formula as ex14: b_ring = 4 pi sigma_v^2 D_LS  for SIS-like behaviour.
    b_ring = 4.0 * pi * sigma_v_c**2 * scene.D_LS
    theta_E_arcsec = scene.theta_E_SIS(sigma_v_kms) * 206264.806
    print(f"Lens: sigma_v={sigma_v_kms} km/s, q_ax=0.7")
    print(f"b_ring (geometrized M)  = {b_ring:.3e}")
    print(f"theta_E (SIS analog)    = {theta_E_arcsec:.3f} arcsec")

    # ---------------------------------------------------------------- #
    # 3. Detector: square FOV ~ 3 b_ring centered on the lens.
    # ---------------------------------------------------------------- #
    detector = image_plane.detector(
        D=scene.D_L, iota=pi/2,
        x_pixels=384, x_side=2.5*b_ring, ratio="1:1")

    # ---------------------------------------------------------------- #
    # 4. Source: 2-arm spiral, slightly offset from optical axis to break
    #    the symmetry and produce visible arc distortion.
    # ---------------------------------------------------------------- #
    spiral = LogSpiral(
        x0=0.06*b_ring, y0=0.02*b_ring,
        Rd=0.10*b_ring, I0=1.0,
        A=0.7, m=2.0,
        pitch_angle=0.30,           # ~17 deg
        pa=0.4)

    # ---------------------------------------------------------------- #
    # 5. SourcePlane via cosmology helper.
    # ---------------------------------------------------------------- #
    sp = SourcePlane.from_redshifts(profile=spiral, scene=scene)

    set_ray_bounds(r_escape=0.5*scene.D_L,
                   final_lmbda=3.0*(scene.D_L + scene.D_LS))

    # ---------------------------------------------------------------- #
    # 6. Render and diagnostics.
    # ---------------------------------------------------------------- #
    img = LensImage(lens, sp, detector)
    img.create_photons()
    maps = img.create_diagnostics(n_workers=4)
    image_data = maps["image_data"]
    source_x = maps["source_x_map"]
    source_y = maps["source_y_map"]
    status_map = maps["status_map"]

    mag = magnification_map(detector, source_x, source_y, status_map,
                            det_floor=1e-10, clip_abs=80.0)
    mu_map = mag["mu"]
    mu_display = log_abs_magnification(mu_map, mag["valid_mu"])
    crit, caust = critical_and_caustic_curves(
        detector, mag["detA"], source_x, source_y,
        valid=mag["valid"], min_points=20)

    # ---------------------------------------------------------------- #
    # 7. Save outputs.
    # ---------------------------------------------------------------- #
    outdir = Path("images/lensing")
    datadir = Path("images_data/lensing")
    outdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    np.savez(datadir / "spiral_lensing.npz",
             image=image_data, source_x=source_x, source_y=source_y,
             status_map=status_map, mu_map=mu_map, detA_map=mag["detA"],
             critical_curves=np.array(crit, dtype=object),
             caustics=np.array(caust, dtype=object),
             D_L_Mpc=scene.D_L_Mpc, D_S_Mpc=scene.D_S_Mpc,
             D_LS_Mpc=scene.D_LS_Mpc, b_ring=b_ring,
             theta_E_arcsec=theta_E_arcsec,
             sigma_v_kms=sigma_v_kms,
             z_L=scene.z_lens, z_S=scene.z_source,
             M_lens_Msun=scene.M_lens_Msun)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.imshow(_norm(image_data).T, origin="lower", cmap="magma",
              extent=[detector.alphaRange[0], detector.alphaRange[-1],
                      detector.betaRange[0], detector.betaRange[-1]])
    ax.set_title(
        f"Spiral lensed by SIE\n"
        f"z_L={scene.z_lens}, z_S={scene.z_source}, "
        f"sigma_v={sigma_v_kms} km/s")
    ax.set_xlabel(r"$\alpha$ [M_lens]")
    ax.set_ylabel(r"$\beta$ [M_lens]")

    ax = axes[1]
    im = ax.imshow(mu_display.T, origin="lower", cmap="viridis",
                   extent=[detector.alphaRange[0], detector.alphaRange[-1],
                           detector.betaRange[0], detector.betaRange[-1]])
    for curve in crit:
        ax.plot(curve[:, 0], curve[:, 1], color="white", linewidth=0.7)
    ax.set_title("log |mu| with critical curves")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    # Scatter caustics in source-plane coords plus the unlensed source disk
    # in dashed for reference.
    if caust:
        for curve in caust:
            ax.plot(curve[:, 0], curve[:, 1], color="orange", linewidth=0.8)
    th = np.linspace(0, 2*pi, 200)
    ax.plot(spiral.x0 + 2*spiral.Rd*np.cos(th),
            spiral.y0 + 2*spiral.Rd*np.sin(th),
            color="cyan", linestyle="--", linewidth=0.7,
            label="2 R_d source")
    ax.set_aspect("equal")
    ax.set_title("Caustics in source plane")
    ax.set_xlabel("x_s [M_lens]")
    ax.set_ylabel("y_s [M_lens]")
    ax.legend(loc="upper right", fontsize=8)

    output = outdir / "spiral_lensing.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved {output}")
    plt.show()


if __name__ == "__main__":
    main()
