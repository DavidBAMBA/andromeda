"""
===============================================================================
ex21 — PEMD (Power-law Elliptical Mass Distribution) galaxy lens
===============================================================================
Demonstrates the new PEMD/EPL metric, the elliptical generalization of SIE
with a free radial slope gamma. Typical SLACS galaxies have gamma~2.0-2.2;
gamma=2 is exactly isothermal (use scr.lens_metrics.sie). gamma>2 makes
the profile steeper than isothermal, producing more compact arcs.

Same JWST-style post-processing pipeline as ex19_realistic / ex20.

Run from repo root:
    MPLCONFIGDIR=/tmp/mplconfig python ex21.pemd_arcs.py
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
from scr.lens_metrics import pemd
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
    M_lens_Msun = 3.0e11        # galaxy-scale early-type
    scene = SceneGeometry(z_L=0.4, z_S=2.0, M_lens_Msun=M_lens_Msun,
                          cosmology=cosmo)

    # PEMD: pick gamma=2.10 (slightly steeper than isothermal, SLACS median),
    # q_ax=0.65 (moderately elongated). K tuned for galaxy-scale theta_E ~1".
    gamma = 2.10
    q_ax = 0.65
    K = 1.0e-5
    lens = pemd.LensMetric(K=K, gamma=gamma, q_ax=q_ax, r_min=1e-1)

    # Solve the lens equation: alpha(b) = b * D_S/(D_L * D_LS).
    from scipy.optimize import brentq
    target = scene.D_S / (scene.D_L * scene.D_LS)
    def lens_eq(b):
        return lens.deflection_angle_spherical(b) - b * target
    b_ring = brentq(lens_eq, 1e2, 1e9)
    print(f"PEMD: K={K:.2e}, gamma={gamma}, q_ax={q_ax}")
    print(f"|Phi(b_ring)|: {abs(K * b_ring**(2-gamma) / (2-gamma)):.3e}")
    print(f"Einstein impact parameter (spherical estimate): b_ring={b_ring:.3e}")

    detector = image_plane.detector(
        D=scene.D_L, iota=pi/2, x_pixels=512,
        x_side=2.5*b_ring, ratio="1:1")

    set_ray_bounds(r_escape=0.5*scene.D_L,
                   final_lmbda=3.0*(scene.D_L + scene.D_LS))

    # Source: small spiral offset to give pair of arcs (Einstein cross-like).
    spiral_blue = LogSpiral(
        x0=0.05*b_ring, y0=0.02*b_ring,
        Rd=0.06*b_ring, I0=1.0,
        A=0.85, m=2.0, pitch_angle=0.30, pa=0.5)
    spiral_red = LogSpiral(
        x0=0.05*b_ring, y0=0.02*b_ring,
        Rd=0.13*b_ring, I0=0.4,
        A=0.30, m=2.0, pitch_angle=0.30, pa=0.5)

    print("\n[1/2] Rendering blue (compact arms) ...")
    img_blue = render_lensed(lens, detector, scene, spiral_blue)
    print("\n[2/2] Rendering red (extended disk) ...")
    img_red = render_lensed(lens, detector, scene, spiral_red)

    galaxy = Sersic(x0=0.0, y0=0.0,
                    R_e=0.18*b_ring, n=4.0, I_e=0.18,
                    ell=1.0 - q_ax, pa=0.0)
    img_gal = project_profile_on_detector(detector, galaxy)

    psf_sigma = 1.1
    img_blue = gaussian_filter(img_blue, psf_sigma)
    img_red = gaussian_filter(img_red, psf_sigma)
    img_gal = gaussian_filter(img_gal, psf_sigma)

    rng = np.random.default_rng(seed=20260427)
    Nx, Ny = img_blue.shape
    stars = add_starfield((Nx, Ny), n_stars=70,
                          flux_range=(0.02, 0.35),
                          sigma_range=(0.7, 1.6), rng=rng)

    arc_blue = _norm_peak(img_blue)
    arc_red = _norm_peak(img_red)
    gal_norm = _norm_peak(img_gal)
    stars_norm = _norm_peak(stars)

    gal_rgb = np.array([1.00, 0.78, 0.42])
    arc_blue_rgb = np.array([0.30, 0.55, 1.00])
    arc_red_rgb = np.array([1.00, 0.55, 0.40])
    stars_rgb = np.array([0.92, 0.94, 1.00])

    rgb = (gal_norm[..., None] * gal_rgb
           + arc_blue[..., None] * arc_blue_rgb * 0.95
           + arc_red[..., None] * arc_red_rgb * 0.55
           + stars_norm[..., None] * stars_rgb * 0.7)

    rgb = rgb + 0.012
    rgb = add_photographic_noise(rgb, read_noise=0.003,
                                 poisson_scale=900.0, rng=rng)
    rgb_disp = asinh_stretch(rgb, softening=0.045, max_percentile=99.5)

    outdir = Path("images/lensing")
    datadir = Path("images_data/lensing")
    outdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    np.savez(datadir / "pemd_arcs.npz",
             rgb=rgb_disp, blue=img_blue, red=img_red,
             galaxy=img_gal, stars=stars,
             K=K, gamma=gamma, q_ax=q_ax, b_ring=b_ring,
             z_L=scene.z_lens, z_S=scene.z_source)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax.imshow(np.transpose(rgb_disp, (1, 0, 2)), origin="lower")
    ax.set_title(
        f"PEMD lens (gamma={gamma}, q_ax={q_ax})\n"
        f"z_L={scene.z_lens}, z_S={scene.z_source}, M_lens={M_lens_Msun:.0e} Msun")
    ax.set_xticks([]); ax.set_yticks([])
    output = outdir / "pemd_arcs.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nSaved {output}")
    plt.show()


if __name__ == "__main__":
    main()
