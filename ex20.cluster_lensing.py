"""
===============================================================================
ex20 — JWST-style cluster lensing through an NFW halo
===============================================================================
Demonstrates the new NFW (Navarro-Frenk-White) halo lens metric:

    rho(r) = rho_s / [(r/r_s) (1 + r/r_s)^2]

A spiral source at z_S=2.0 is lensed by a cluster-scale NFW halo at z_L=0.4.
NFW differs from SIS/SIE in two important ways:

    * Smooth core (no central singularity) -> finite Einstein-ring inner edge.
    * Logarithmic outer wings -> long arcs and giant tangential structures
      similar to those seen in JWST images of MACS J0723 / Abell 370.

Post-processing follows ex19_realistic: PSF, sky background, Poisson noise,
two-channel composite, foreground BCG light, asinh stretch.

Run from repo root:
    MPLCONFIGDIR=/tmp/mplconfig python ex20.cluster_lensing.py
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
    # ---------------------------------------------------------------- #
    # 1. Cosmology + cluster-scale lens
    # ---------------------------------------------------------------- #
    cosmo = LambdaCDM()
    # Cluster-scale: M_200 ~ 5e14 Msun. We use it as the geometric mass
    # scale; the NFW M_s and r_s are then dimensionless multiples of that.
    M_lens_Msun = 5.0e14
    scene = SceneGeometry(z_L=0.4, z_S=2.0, M_lens_Msun=M_lens_Msun,
                          cosmology=cosmo)

    # Halo parameters in geometrized M_lens units.
    # For M_lens=5e14 Msun, r_g ~ 24 kpc, so D_L ~ 5e7 r_g.
    # Cluster NFW r_s ~ 250 kpc -> r_s ~ 1e4 r_g.
    # Tune M_s so theta_E ~ 30 arcsec (typical massive cluster).
    M_s = 1.2
    r_s = 5.0e3
    lens = nfw.LensMetric(M_s=M_s, r_s=r_s)

    # Predicted Einstein-ring impact parameter from h(x_E)/x_E = D_S/(4 M_s D_LS)
    # solved numerically (we just bracket the analytical alpha):
    from scipy.optimize import brentq
    def lens_eq(b):
        # alpha(b) = b/(D_LS) * (D_S/D_L) for a thin lens at the ring radius.
        # Equivalently in this code's convention: alpha(b) = b * (D_S / (D_L * D_LS)).
        return lens.deflection_angle(b) - b * scene.D_S / (scene.D_L * scene.D_LS)
    try:
        b_ring = brentq(lens_eq, 0.05*r_s, 30.0*r_s)
    except ValueError:
        b_ring = 5.0 * r_s
    print(f"Cosmology: H0={cosmo.H0}, Om={cosmo.Om}")
    print(f"Distances [Mpc]: D_L={scene.D_L_Mpc:.1f}, "
          f"D_S={scene.D_S_Mpc:.1f}, D_LS={scene.D_LS_Mpc:.1f}")
    print(f"NFW halo: M_s={M_s:.2e}, r_s={r_s:.2e}, "
          f"|Phi(r_s)|={abs(lens.Phi(r_s)):.2e}")
    print(f"Einstein-ring impact parameter: b_ring={b_ring:.2e} M ({b_ring/r_s:.2f} r_s)")

    # ---------------------------------------------------------------- #
    # 2. Detector: FOV ~ 2.5 b_ring.
    # ---------------------------------------------------------------- #
    detector = image_plane.detector(
        D=scene.D_L, iota=pi/2, x_pixels=512,
        x_side=2.2*b_ring, ratio="1:1")

    set_ray_bounds(r_escape=0.5*scene.D_L,
                   final_lmbda=3.0*(scene.D_L + scene.D_LS))

    # ---------------------------------------------------------------- #
    # 3. Source: small spiral slightly offset from optical axis to break
    #    symmetry and produce arcs (not a perfect ring).
    # ---------------------------------------------------------------- #
    spiral_blue = LogSpiral(
        x0=0.30*b_ring, y0=0.10*b_ring,
        Rd=0.05*b_ring, I0=1.0,
        A=0.85, m=2.0, pitch_angle=0.30, pa=0.6)
    spiral_red = LogSpiral(
        x0=0.30*b_ring, y0=0.10*b_ring,
        Rd=0.10*b_ring, I0=0.4,
        A=0.30, m=2.0, pitch_angle=0.30, pa=0.6)

    # ---------------------------------------------------------------- #
    # 4. Render two channels.
    # ---------------------------------------------------------------- #
    print("\n[1/2] Lensing blue channel ...")
    img_blue = render_lensed(lens, detector, scene, spiral_blue)
    print("\n[2/2] Lensing red channel ...")
    img_red = render_lensed(lens, detector, scene, spiral_red)

    # ---------------------------------------------------------------- #
    # 5. Foreground BCG (Brightest Cluster Galaxy): Sersic n=4 elliptical.
    # ---------------------------------------------------------------- #
    bcg = Sersic(
        x0=0.0, y0=0.0,
        R_e=0.16*b_ring, n=4.0, I_e=0.20,
        ell=0.20, pa=0.1)
    img_bcg = project_profile_on_detector(detector, bcg)

    # Add a few satellite cluster galaxies to break the empty-foreground look.
    rng = np.random.default_rng(seed=20260427)
    n_sat = 6
    img_sat = np.zeros_like(img_bcg)
    for _ in range(n_sat):
        sx = rng.uniform(-1.7*b_ring, 1.7*b_ring)
        sy = rng.uniform(-1.7*b_ring, 1.7*b_ring)
        # Skip if too close to the BCG.
        if sx*sx + sy*sy < (0.4*b_ring)**2:
            continue
        sat = Sersic(
            x0=sx, y0=sy,
            R_e=rng.uniform(0.03, 0.07)*b_ring,
            n=rng.uniform(2.0, 4.0),
            I_e=rng.uniform(0.04, 0.10),
            ell=rng.uniform(0.1, 0.4),
            pa=rng.uniform(0, pi))
        img_sat = img_sat + project_profile_on_detector(detector, sat)

    # ---------------------------------------------------------------- #
    # 6. PSF convolution.
    # ---------------------------------------------------------------- #
    psf_sigma_px = 1.2
    img_blue = gaussian_filter(img_blue, psf_sigma_px)
    img_red = gaussian_filter(img_red, psf_sigma_px)
    img_bcg = gaussian_filter(img_bcg, psf_sigma_px)
    img_sat = gaussian_filter(img_sat, psf_sigma_px)

    # ---------------------------------------------------------------- #
    # 7. Background + RGB + noise + asinh stretch.
    # ---------------------------------------------------------------- #
    Nx, Ny = img_blue.shape
    stars = add_starfield((Nx, Ny), n_stars=80,
                          flux_range=(0.02, 0.40),
                          sigma_range=(0.7, 1.7), rng=rng)

    arc_blue = _norm_peak(img_blue)
    arc_red = _norm_peak(img_red)
    bcg_norm = _norm_peak(img_bcg + 0.6*img_sat)
    stars_norm = _norm_peak(stars)

    bcg_rgb = np.array([1.00, 0.78, 0.42])      # warm yellow BCG
    arc_blue_rgb = np.array([0.30, 0.55, 1.00]) # blue arms
    arc_red_rgb = np.array([1.00, 0.55, 0.40])  # extended red disk
    stars_rgb = np.array([0.92, 0.94, 1.00])

    rgb = (bcg_norm[..., None] * bcg_rgb
           + arc_blue[..., None] * arc_blue_rgb * 0.95
           + arc_red[..., None] * arc_red_rgb * 0.55
           + stars_norm[..., None] * stars_rgb * 0.7)

    sky = 0.012
    rgb = rgb + sky
    rgb = add_photographic_noise(rgb, read_noise=0.003,
                                 poisson_scale=900.0, rng=rng)
    rgb_disp = asinh_stretch(rgb, softening=0.05, max_percentile=99.5)

    # ---------------------------------------------------------------- #
    # 8. Save.
    # ---------------------------------------------------------------- #
    outdir = Path("images/lensing")
    datadir = Path("images_data/lensing")
    outdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    np.savez(datadir / "cluster_lensing.npz",
             rgb=rgb_disp, blue=img_blue, red=img_red,
             bcg=img_bcg, satellites=img_sat, stars=stars,
             D_L_Mpc=scene.D_L_Mpc, D_S_Mpc=scene.D_S_Mpc,
             D_LS_Mpc=scene.D_LS_Mpc, b_ring=b_ring,
             M_s=M_s, r_s=r_s,
             z_L=scene.z_lens, z_S=scene.z_source)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax.imshow(np.transpose(rgb_disp, (1, 0, 2)), origin="lower")
    ax.set_title(
        f"NFW cluster lensing (JWST-style)\n"
        f"z_L={scene.z_lens}, z_S={scene.z_source}, "
        f"M_lens={M_lens_Msun:.0e} Msun, M_s={M_s:.1e}")
    ax.set_xticks([]); ax.set_yticks([])
    output = outdir / "cluster_lensing.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nSaved {output}")
    plt.show()


if __name__ == "__main__":
    main()
