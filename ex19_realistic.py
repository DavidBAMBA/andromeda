"""
===============================================================================
ex19_realistic — JWST/HST-style render of a lensed spiral galaxy
===============================================================================
Same physics as ex19.spiral_lensing.py but with realistic post-processing:

    1. PSF convolution (Gaussian, FWHM ~ 2 px)
    2. Sky background + Poisson shot noise + Gaussian read noise
    3. Two-color composite (red disk extended + blue disk core)
    4. Lens-galaxy foreground emission (Sersic n=4, no lensing)
    5. Starfield in the background
    6. asinh stretch with percentile clipping

NO changes to the Numba kernel signature: each component is a separate render
or a numpy post-processing pass.

Run from repo root:
    MPLCONFIGDIR=/tmp/mplconfig python ex19_realistic.py
===============================================================================
"""
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from scr.common.common import set_ray_bounds
from scr.common.cosmology import LambdaCDM, SceneGeometry, C_SI
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.visual import (add_photographic_noise, add_starfield,
                               asinh_stretch, compose_rgb,
                               project_profile_on_detector)
from scr.detectors import image_plane
from scr.lens_metrics import sie
from scr.sources.light_profiles import LogSpiral, Sersic


def _norm_peak(img):
    peak = img.max()
    return img / peak if peak > 0.0 else img


def render_lensed_disk(lens, detector, scene, source):
    sp = SourcePlane.from_redshifts(profile=source, scene=scene)
    img = LensImage(lens, sp, detector)
    img.create_photons()
    img.create_image(n_workers=4)
    return img.image_data.copy()


def main():
    # ---------------------------------------------------------------- #
    # 1. Cosmology + lens geometry
    # ---------------------------------------------------------------- #
    cosmo = LambdaCDM()
    scene = SceneGeometry(z_L=0.5, z_S=2.0, M_lens_Msun=3.0e11, cosmology=cosmo)

    sigma_v_kms = 280.0
    sigma_v_c = (sigma_v_kms * 1000.0) / C_SI
    lens = sie.LensMetric(sigma_v=sigma_v_c, q_ax=0.7)
    b_ring = 4.0 * pi * sigma_v_c**2 * scene.D_LS

    detector = image_plane.detector(
        D=scene.D_L, iota=pi/2, x_pixels=512,
        x_side=2.5*b_ring, ratio="1:1")

    set_ray_bounds(r_escape=0.5*scene.D_L,
                   final_lmbda=3.0*(scene.D_L + scene.D_LS))

    # ---------------------------------------------------------------- #
    # 2. Two-channel source (mimic disk color: blue compact arms + red
    #    extended disk).
    # ---------------------------------------------------------------- #
    spiral_blue = LogSpiral(
        x0=0.06*b_ring, y0=0.02*b_ring,
        Rd=0.07*b_ring, I0=1.0,
        A=0.85, m=2.0, pitch_angle=0.30, pa=0.4)
    spiral_red = LogSpiral(
        x0=0.06*b_ring, y0=0.02*b_ring,
        Rd=0.16*b_ring, I0=0.45,
        A=0.35, m=2.0, pitch_angle=0.30, pa=0.4)

    # ---------------------------------------------------------------- #
    # 3. Render two lensed channels.
    # ---------------------------------------------------------------- #
    print("\n[1/2] Rendering blue (compact arms) channel ...")
    img_blue = render_lensed_disk(lens, detector, scene, spiral_blue)
    print("\n[2/2] Rendering red (extended disk) channel ...")
    img_red = render_lensed_disk(lens, detector, scene, spiral_red)

    # ---------------------------------------------------------------- #
    # 4. Foreground lens galaxy (no lensing — direct projection).
    #    Bright Sersic n=4 to mimic an early-type galaxy SED.
    # ---------------------------------------------------------------- #
    lens_galaxy = Sersic(
        x0=0.0, y0=0.0,
        R_e=0.18*b_ring, n=4.0, I_e=0.18,
        ell=0.30, pa=0.0)
    img_lens = project_profile_on_detector(detector, lens_galaxy)

    # ---------------------------------------------------------------- #
    # 5. PSF convolution (Gaussian, sigma ~ 1 px = FWHM ~2.35 px) on each
    #    layer separately.
    # ---------------------------------------------------------------- #
    psf_sigma_px = 1.1
    img_blue = gaussian_filter(img_blue, psf_sigma_px)
    img_red = gaussian_filter(img_red, psf_sigma_px)
    img_lens = gaussian_filter(img_lens, psf_sigma_px)

    # ---------------------------------------------------------------- #
    # 6. Background: starfield + faint distant galaxies (sub-threshold
    #    Sersic stamp not implemented here; we use the existing
    #    add_starfield helper).
    # ---------------------------------------------------------------- #
    rng = np.random.default_rng(seed=20260427)
    Nx, Ny = img_blue.shape
    stars = add_starfield((Nx, Ny), n_stars=60,
                          flux_range=(0.02, 0.35),
                          sigma_range=(0.7, 1.6), rng=rng)

    # ---------------------------------------------------------------- #
    # 7. RGB composition (lens = warm yellow, arcs = blue/cyan, stars = white).
    # ---------------------------------------------------------------- #
    # Combine red+blue arc channels into a 2-color arc layer that we then
    # tint with two slightly different RGB triplets.
    arc_layer_blue = _norm_peak(img_blue)
    arc_layer_red = _norm_peak(img_red)
    lens_layer = _norm_peak(img_lens)
    stars_layer = _norm_peak(stars)

    # Build the RGB by hand because we have two arc channels.
    lens_rgb = np.array([1.00, 0.80, 0.45])      # warm yellow (early-type)
    arc_blue_rgb = np.array([0.30, 0.55, 1.00])  # H-alpha-blue arms
    arc_red_rgb = np.array([1.00, 0.55, 0.40])   # red disk
    stars_rgb = np.array([0.92, 0.94, 1.00])

    rgb = (lens_layer[..., None] * lens_rgb
           + arc_layer_blue[..., None] * arc_blue_rgb * 0.9
           + arc_layer_red[..., None] * arc_red_rgb * 0.6
           + stars_layer[..., None] * stars_rgb * 0.7)

    # ---------------------------------------------------------------- #
    # 8. Sky background + Poisson + read noise.
    # ---------------------------------------------------------------- #
    sky = 0.012
    rgb = rgb + sky
    rgb = add_photographic_noise(rgb,
                                 read_noise=0.003,
                                 poisson_scale=900.0,
                                 rng=rng)

    # ---------------------------------------------------------------- #
    # 9. asinh stretch with percentile clip.
    # ---------------------------------------------------------------- #
    rgb_disp = asinh_stretch(rgb, softening=0.045, max_percentile=99.6)

    # ---------------------------------------------------------------- #
    # 10. Save.
    # ---------------------------------------------------------------- #
    outdir = Path("images/lensing")
    datadir = Path("images_data/lensing")
    outdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    np.savez(datadir / "spiral_lensing_realistic.npz",
             rgb=rgb_disp, blue=img_blue, red=img_red,
             lens_galaxy=img_lens, stars=stars,
             D_L_Mpc=scene.D_L_Mpc, D_S_Mpc=scene.D_S_Mpc,
             D_LS_Mpc=scene.D_LS_Mpc, b_ring=b_ring,
             sigma_v_kms=sigma_v_kms,
             z_L=scene.z_lens, z_S=scene.z_source)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax.imshow(np.transpose(rgb_disp, (1, 0, 2)), origin="lower")
    ax.set_title(
        f"Lensed spiral, JWST-style render\n"
        f"z_L={scene.z_lens}, z_S={scene.z_source}, "
        f"sigma_v={sigma_v_kms} km/s, M_lens={scene.M_lens_Msun:.1e} Msun")
    ax.set_xticks([]); ax.set_yticks([])
    output = outdir / "spiral_lensing_realistic.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nSaved {output}")
    plt.show()


if __name__ == "__main__":
    main()
