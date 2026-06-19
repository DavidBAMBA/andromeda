"""
===============================================================================
ex22 — Elliptical NFW (eNFW) cluster lens with broken arcs
===============================================================================
Same NFW halo as ex20 but with q_ax = 0.65 (Golse & Kneib 2002 pseudo-
elliptical model). This single change is enough to fragment the smooth
Einstein ring into two main arcs plus counter-images, similar to JWST
SMACS J0723 / Abell 2218.

Run from repo root:
    MPLCONFIGDIR=/tmp/mplconfig python ex22.enfw_cluster.py
===============================================================================
"""
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from scr.common.common import set_ray_bounds
from scr.common.cosmology import LambdaCDM, SceneGeometry
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.visual import (add_photographic_noise, add_starfield,
                               asinh_stretch, project_profile_on_detector)
from scr.detectors import image_plane
from scr.lens_metrics import nfw
from scr.sources.light_profiles import LogSpiral, Sersic


def _norm_peak(img):
    peak = img.max()
    return img / peak if peak > 0.0 else img


def render_lensed(lens, detector, scene, source):
    sp = SourcePlane.from_redshifts(profile=source, scene=scene)
    img = LensImage(lens, sp, detector)
    img.create_photons()
    img.create_image(n_workers=4)
    return img.image_data.copy()


def main():
    cosmo = LambdaCDM()
    M_lens_Msun = 5.0e14
    scene = SceneGeometry(z_L=0.4, z_S=2.0, M_lens_Msun=M_lens_Msun,
                          cosmology=cosmo)

    # eNFW with the same M_s, r_s as ex20 but q_ax = 0.65 to break symmetry.
    M_s = 1.2
    r_s = 5.0e3
    q_ax = 0.65
    lens = nfw.LensMetric(M_s=M_s, r_s=r_s, q_ax=q_ax)

    # Use spherical-limit b_ring estimate from ex20 (numerical brentq).
    from scipy.optimize import brentq
    sph = nfw.LensMetric(M_s=M_s, r_s=r_s)
    target = scene.D_S / (scene.D_L * scene.D_LS)
    def lens_eq(b):
        return sph.deflection_angle(b) - b * target
    try:
        b_ring = brentq(lens_eq, 0.05*r_s, 30.0*r_s)
    except ValueError:
        b_ring = 5.0 * r_s

    print(f"eNFW: M_s={M_s}, r_s={r_s:.2e}, q_ax={q_ax}")
    print(f"|Phi(r_s)| eq: {abs(lens.metric([0,r_s,pi/2,0])[0]+1)/2:.3e}")
    print(f"b_ring (spherical estimate): {b_ring:.3e}")

    detector = image_plane.detector(
        D=scene.D_L, iota=pi/2, x_pixels=512,
        x_side=2.5*b_ring, ratio="1:1")

    set_ray_bounds(r_escape=0.5*scene.D_L,
                   final_lmbda=3.0*(scene.D_L + scene.D_LS))

    spiral_blue = LogSpiral(
        x0=0.30*b_ring, y0=0.10*b_ring,
        Rd=0.05*b_ring, I0=1.0,
        A=0.85, m=2.0, pitch_angle=0.30, pa=0.6)
    spiral_red = LogSpiral(
        x0=0.30*b_ring, y0=0.10*b_ring,
        Rd=0.10*b_ring, I0=0.4,
        A=0.30, m=2.0, pitch_angle=0.30, pa=0.6)

    print("\n[1/2] Lensing blue ...")
    img_blue = render_lensed(lens, detector, scene, spiral_blue)
    print("\n[2/2] Lensing red ...")
    img_red = render_lensed(lens, detector, scene, spiral_red)

    bcg = Sersic(x0=0.0, y0=0.0,
                 R_e=0.16*b_ring, n=4.0, I_e=0.20,
                 ell=1.0 - q_ax, pa=0.0)
    img_bcg = project_profile_on_detector(detector, bcg)

    rng = np.random.default_rng(seed=20260427)
    img_sat = np.zeros_like(img_bcg)
    for _ in range(8):
        sx = rng.uniform(-1.7*b_ring, 1.7*b_ring)
        sy = rng.uniform(-1.7*b_ring, 1.7*b_ring)
        if sx*sx + sy*sy < (0.4*b_ring)**2:
            continue
        sat = Sersic(x0=sx, y0=sy,
                     R_e=rng.uniform(0.03, 0.07)*b_ring,
                     n=rng.uniform(2.0, 4.0),
                     I_e=rng.uniform(0.04, 0.10),
                     ell=rng.uniform(0.1, 0.4),
                     pa=rng.uniform(0, pi))
        img_sat = img_sat + project_profile_on_detector(detector, sat)

    psf_sigma = 1.2
    img_blue = gaussian_filter(img_blue, psf_sigma)
    img_red = gaussian_filter(img_red, psf_sigma)
    img_bcg = gaussian_filter(img_bcg, psf_sigma)
    img_sat = gaussian_filter(img_sat, psf_sigma)

    Nx, Ny = img_blue.shape
    stars = add_starfield((Nx, Ny), n_stars=80,
                          flux_range=(0.02, 0.40),
                          sigma_range=(0.7, 1.7), rng=rng)

    arc_blue = _norm_peak(img_blue)
    arc_red = _norm_peak(img_red)
    bcg_norm = _norm_peak(img_bcg + 0.6*img_sat)
    stars_norm = _norm_peak(stars)

    bcg_rgb = np.array([1.00, 0.78, 0.42])
    arc_blue_rgb = np.array([0.30, 0.55, 1.00])
    arc_red_rgb = np.array([1.00, 0.55, 0.40])
    stars_rgb = np.array([0.92, 0.94, 1.00])

    rgb = (bcg_norm[..., None] * bcg_rgb
           + arc_blue[..., None] * arc_blue_rgb * 0.95
           + arc_red[..., None] * arc_red_rgb * 0.55
           + stars_norm[..., None] * stars_rgb * 0.7)

    rgb = rgb + 0.012
    rgb = add_photographic_noise(rgb, read_noise=0.003,
                                 poisson_scale=900.0, rng=rng)
    rgb_disp = asinh_stretch(rgb, softening=0.05, max_percentile=99.5)

    outdir = Path("images/lensing")
    datadir = Path("images_data/lensing")
    outdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    np.savez(datadir / "enfw_cluster.npz",
             rgb=rgb_disp, blue=img_blue, red=img_red,
             bcg=img_bcg, satellites=img_sat, stars=stars,
             M_s=M_s, r_s=r_s, q_ax=q_ax, b_ring=b_ring,
             z_L=scene.z_lens, z_S=scene.z_source)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax.imshow(np.transpose(rgb_disp, (1, 0, 2)), origin="lower")
    ax.set_title(
        f"Elliptical NFW cluster (q_ax={q_ax})\n"
        f"z_L={scene.z_lens}, z_S={scene.z_source}, "
        f"M_lens={M_lens_Msun:.0e} Msun, M_s={M_s:.1e}")
    ax.set_xticks([]); ax.set_yticks([])
    output = outdir / "enfw_cluster.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nSaved {output}")
    plt.show()


if __name__ == "__main__":
    main()
