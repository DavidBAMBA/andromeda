"""
===============================================================================
Source-plane Doppler boosting in strong gravitational lensing
===============================================================================
Demonstrates the fast Numba path for source-plane Doppler correction:

    I_obs = g^3 I_emit

The same lensed source is rendered twice: first with a static source, then with
a rotating source-plane velocity field. A third panel shows Doppler/static.
===============================================================================
"""
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scr.common.common import set_ray_bounds
from scr.common.lens_diagnostics import (critical_and_caustic_curves,
                                         log_abs_magnification,
                                         magnification_map)
from scr.common.lens_image import LensImage, SourcePlane
from scr.detectors import image_plane
from scr.lens_metrics import sis
from scr.sources.light_profiles import Sersic
from scr.sources.velocity_models import RotatingDiskSource, StaticSource


def _norm(img):
    peak = img.max()
    return img / peak if peak > 0.0 else img


def main():
    D_L = 1.0e4
    D_LS = 1.0e4
    sigma_v = 0.025
    b_ring = 4.0 * pi * sigma_v**2 * D_LS

    lens = sis.LensMetric(sigma_v=sigma_v)
    detector = image_plane.detector(
        D=D_L, iota=pi/2, x_pixels=256,
        x_side=2.8 * b_ring, ratio="1:1")

    source = Sersic(
        x0=0.10 * b_ring, y0=0.02 * b_ring,
        R_e=0.13 * b_ring, n=1.0, I_e=1.0,
        ell=0.25, pa=pi/5)

    rotating = RotatingDiskSource(
        v_max=0.18, r_turn=0.08 * b_ring,
        inclination=pi/3, pa=pi/5,
        x0=0.10 * b_ring, y0=0.02 * b_ring)

    set_ray_bounds(r_escape=0.5 * D_L, final_lmbda=3.0 * D_L)

    static_plane = SourcePlane(
        D_LS=D_LS, profile=source, velocity_model=StaticSource())
    static_img = LensImage(lens, static_plane, detector)
    static_img.create_photons()
    static_img.create_image_doppler(n_workers=4)
    static = static_img.image_data.copy()

    doppler_plane = SourcePlane(
        D_LS=D_LS, profile=source, velocity_model=rotating)
    doppler_img = LensImage(lens, doppler_plane, detector)
    doppler_img.create_photons()
    maps = doppler_img.create_diagnostics(n_workers=4)
    doppler = maps["image_data"]
    g_map = maps["g_map"]
    source_x = maps["source_x_map"]
    source_y = maps["source_y_map"]
    status_map = maps["status_map"]
    mag = magnification_map(detector, source_x, source_y, status_map,
                            det_floor=1e-10, clip_abs=50.0)
    mu_map = mag["mu"]
    mu_display = log_abs_magnification(mu_map, mag["valid_mu"])
    critical_curves, caustics = critical_and_caustic_curves(
        detector, mag["detA"], source_x, source_y, valid=mag["valid"],
        min_points=20)

    ratio = np.divide(doppler, static, out=np.ones_like(doppler),
                      where=static > 1e-8)
    bright = static > 0.05 * static.max()
    print("Doppler/static ratio on bright pixels:")
    print("  min :", float(ratio[bright].min()))
    print("  mean:", float(ratio[bright].mean()))
    print("  max :", float(ratio[bright].max()))
    print("Magnification |mu| on valid pixels:")
    valid_mu = mag["valid_mu"]
    print("  median:", float(np.median(np.abs(mu_map[valid_mu]))))
    print("  p99   :", float(np.percentile(np.abs(mu_map[valid_mu]), 99.0)))
    print("Critical curves:", len(critical_curves))
    print("Caustics:", len(caustics))

    outdir = Path("images/lensing")
    datadir = Path("images_data/lensing")
    outdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    np.savez(datadir / "lensing_doppler_source_diagnostics.npz",
             static=static, doppler=doppler, ratio=ratio, g_map=g_map,
             source_x_map=source_x, source_y_map=source_y,
             status_map=status_map, mu_map=mu_map, detA_map=mag["detA"],
             critical_curves=np.array(critical_curves, dtype=object),
             caustics=np.array(caustics, dtype=object))

    fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    panels = [
        ("Static source", _norm(static), "magma"),
        ("Rotating source Doppler", _norm(doppler), "magma"),
        ("g factor", g_map, "coolwarm"),
        ("log |mu|", mu_display, "viridis"),
        ("Doppler / static", np.clip(ratio, 0.55, 1.55), "coolwarm"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        im = ax.imshow(data.T, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overlay critical curves on the magnification panel.
    mu_ax = axes[3]
    for curve in critical_curves:
        i = np.interp(curve[:, 0], detector.alphaRange,
                      np.arange(detector.x_pixels))
        j = np.interp(curve[:, 1], detector.betaRange,
                      np.arange(detector.y_pixels))
        mu_ax.plot(i, j, color="white", linewidth=0.8)

    output = outdir / "lensing_doppler_source.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved", output)


if __name__ == "__main__":
    main()
