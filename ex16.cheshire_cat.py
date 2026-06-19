"""
===============================================================================
Cheshire Cat — binary-lens strong gravitational lensing (SDSS J1038+4849)
===============================================================================
Pipeline:
    1. BINARY SIE lens: two elliptical potentials ("eyes") superposed with
       offset centers in the image plane. Non-axisymmetric; k_phi not
       conserved.
    2. Lensed source: a disk Sersic + a couple of gaussian clumps, placed
       near a caustic so the lower arc fragments into the "smile".
    3. Lens light: two orange n=4 Sersics for the eyes, a small blue
       gaussian for the central "nose" galaxy, plus faint cluster members.
    4. Background field: unrelated galaxies of mixed type + dense
       starfield + one bright Milky-Way foreground star with 4-point
       diffraction spike.
    5. Hubble-style post-processing: Poisson + read noise, asinh stretch,
       RGB composition -> PNG + npy.
===============================================================================
"""

from math import pi, sqrt
import numpy as np
from numpy import save
import matplotlib.pyplot as plt

from scr.lens_metrics import binary_sie
from scr.detectors import image_plane
from scr.sources.light_profiles import Gaussian, Sersic
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.common import set_ray_bounds
from scr.common.visual import (project_profile_on_detector, add_starfield,
                                 add_photographic_noise, asinh_stretch)
import warnings
warnings.filterwarnings("ignore")




'''
===============================================================================
=============================== LENS DEFINITION ===============================
===============================================================================
'''
##### BINARY SIE: two elliptical "eyes" in the image plane.
# Effective isothermal sigma for the ring radius: sigma_eff^2 = s1^2 + s2^2.
sigma_v = (0.020, 0.020)                  # two equal-mass lenses
q_ax    = (0.85, 0.85)                    # mild flattening along z (vertical)
sigma_eff2 = sigma_v[0]**2 + sigma_v[1]**2



'''
===============================================================================
=========================== DETECTOR PARAMETERS ===============================
===============================================================================
'''
D_L = 1.0e4                               # Observer-to-lens distance (M)
D_LS = 1.0e4                              # Lens-to-source distance
iota = pi/2                               # Equatorial observer
b_ring = 4.0 * pi * sigma_eff2 * D_LS     # ~100 M for the chosen parameters
x_side = 3.5 * b_ring                     # wide FOV for deep-field context
x_pixels = 1920

detector = image_plane.detector(D=D_L, iota=iota, x_pixels=x_pixels,
                                 x_side=x_side, ratio='1:1')


##### Center-separation in the image plane.
# In this convention, alpha (horizontal image axis) <-> global Cartesian y,
# beta (vertical image axis) <-> global z. Put both lens centers on the y-axis
# (z = 0) so the two "eyes" are horizontally separated as in SDSS J1038+4849.
d_sep = 0.30 * b_ring                     # half-separation between the eyes

lens = binary_sie.LensMetric(
    sigma_v=sigma_v, q_ax=q_ax,
    centers=((0.0, +d_sep, 0.0), (0.0, -d_sep, 0.0)),
    r_ref=1.0, r_min=1e-2, r_min_center=0.5)



'''
===============================================================================
============================ LENSED SOURCES ===================================
===============================================================================
'''
##### PRIMARY SOURCE: small starburst disk slightly below the binary barycenter
# so the lower arc is brighter and fragmented (the "smile").
primary_source = Sersic(x0=+0.04*b_ring, y0=-0.06*b_ring,
                        R_e=0.06*b_ring, n=1.0, I_e=1.2,
                        ell=0.30, pa=-pi/8)

##### SECONDARY CLUMPS: 2-3 gaussian star-forming knots riding the disk,
# each slightly displaced to split into distinct sub-images along the smile.
secondary_sources = [
    Gaussian(x0=+0.09*b_ring, y0=-0.08*b_ring,
             sigma=0.018*b_ring, I0=0.85),
    Gaussian(x0=+0.02*b_ring, y0=-0.10*b_ring,
             sigma=0.016*b_ring, I0=0.70),
    Gaussian(x0=-0.04*b_ring, y0=-0.05*b_ring,
             sigma=0.014*b_ring, I0=0.55),
]



'''
===============================================================================
=========================== LENS GALAXY LIGHT =================================
===============================================================================
'''
##### Two massive orange ellipticals (the "eyes"): de Vaucouleurs profiles.
eye_right = Sersic(x0=+d_sep, y0=0.0,
                   R_e=0.13*b_ring, n=4.0, I_e=0.95,
                   ell=0.15, pa=pi/2)
eye_left  = Sersic(x0=-d_sep, y0=0.0,
                   R_e=0.13*b_ring, n=4.0, I_e=0.95,
                   ell=0.15, pa=pi/2)

##### Central "nose": small blue galaxy between the two eyes (not a lens).
nose_gal = Gaussian(x0=0.0, y0=0.0,
                    sigma=0.035*b_ring, I0=0.40)

##### Faint cluster members below the system (group halo companions).
cluster_members = [
    Sersic(x0=+0.18*b_ring, y0=-0.42*b_ring,
           R_e=0.035*b_ring, n=4.0, I_e=0.18, ell=0.20, pa=pi/7),
    Sersic(x0=-0.22*b_ring, y0=-0.48*b_ring,
           R_e=0.030*b_ring, n=4.0, I_e=0.15, ell=0.15, pa=-pi/5),
    Sersic(x0=+0.05*b_ring, y0=-0.55*b_ring,
           R_e=0.028*b_ring, n=4.0, I_e=0.13, ell=0.10, pa=0.0),
]



'''
===============================================================================
======================== BACKGROUND GALAXIES (NO LENSING) ====================
===============================================================================
'''
##### Deep-field ambience: resolved galaxies with morphology (not just
# dots). Sizes calibrated so R_e spans ~10-40 pixels at 1920x1920. Most
# galaxies have random orientations; only those closest to the lens
# receive a subtle tangential-shear hint (weak lensing by the cluster halo).

_rng_gal = np.random.default_rng(seed=1038)


def _tangential_pa(xg, yg, noise=0.25):
    """Position angle ~ perpendicular to the radius vector, with random
    scatter so the field does NOT look like a concentric ring."""
    return (np.arctan2(yg, xg) + pi / 2.0
            + _rng_gal.uniform(-noise, noise))


def _shear_ell(x, y, base_ell=0.20, amp=0.08, lens_scale=0.7*b_ring):
    """Subtle distance-dependent extra ellipticity (weak-lensing shear)."""
    r = sqrt(x*x + y*y)
    return min(0.85, base_ell + amp * np.exp(-(r/lens_scale)**2 * 0.9))


background_galaxies = [
    # --- Prominent face-on spiral (lower-left, Cheshire Cat hallmark) ---
    Sersic(x0=-2.10*b_ring, y0=-1.15*b_ring,
           R_e=0.22*b_ring, n=1.0, I_e=0.15, ell=0.12,
           pa=_rng_gal.uniform(0, pi)),
    # Bulge of the same spiral
    Gaussian(x0=-2.10*b_ring, y0=-1.15*b_ring,
             sigma=0.032*b_ring, I0=0.35),

    # --- Edge-on disks (thin elongated strips spread over the field) ---
    Sersic(x0=-1.95*b_ring, y0=+0.60*b_ring,
           R_e=0.14*b_ring, n=1.0, I_e=0.12, ell=0.80,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+0.90*b_ring, y0=+1.85*b_ring,
           R_e=0.13*b_ring, n=1.0, I_e=0.11, ell=0.78,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-0.70*b_ring, y0=+2.10*b_ring,
           R_e=0.11*b_ring, n=1.0, I_e=0.10, ell=0.75,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+2.20*b_ring, y0=+0.90*b_ring,
           R_e=0.10*b_ring, n=1.0, I_e=0.10, ell=0.72,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+1.80*b_ring, y0=-1.85*b_ring,
           R_e=0.12*b_ring, n=1.0, I_e=0.11, ell=0.76,
           pa=_rng_gal.uniform(0, pi)),

    # --- Large yellow ellipticals spread across the outer field ---
    Sersic(x0=+1.50*b_ring, y0=+1.35*b_ring,
           R_e=0.11*b_ring, n=4.0, I_e=0.16, ell=0.20,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+1.45*b_ring, y0=-1.70*b_ring,
           R_e=0.12*b_ring, n=4.0, I_e=0.16, ell=0.22,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-1.20*b_ring, y0=+1.80*b_ring,
           R_e=0.10*b_ring, n=4.0, I_e=0.14, ell=0.22,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+2.05*b_ring, y0=-0.45*b_ring,
           R_e=0.11*b_ring, n=4.0, I_e=0.15, ell=0.18,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-2.20*b_ring, y0=+1.10*b_ring,
           R_e=0.10*b_ring, n=4.0, I_e=0.14, ell=0.20,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+0.45*b_ring, y0=+2.10*b_ring,
           R_e=0.09*b_ring, n=4.0, I_e=0.13, ell=0.24,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-2.30*b_ring, y0=-0.55*b_ring,
           R_e=0.10*b_ring, n=4.0, I_e=0.14, ell=0.18,
           pa=_rng_gal.uniform(0, pi)),

    # --- Medium galaxies at intermediate radii with subtle weak-lensing
    #     shear (slight tangential hint) ---
    Sersic(x0=+1.10*b_ring, y0=+0.80*b_ring,
           R_e=0.070*b_ring, n=3.0, I_e=0.13,
           ell=_shear_ell(+1.10*b_ring, +0.80*b_ring, base_ell=0.22),
           pa=_tangential_pa(+1.10*b_ring, +0.80*b_ring)),
    Sersic(x0=-1.20*b_ring, y0=-0.70*b_ring,
           R_e=0.075*b_ring, n=3.0, I_e=0.12,
           ell=_shear_ell(-1.20*b_ring, -0.70*b_ring, base_ell=0.20),
           pa=_tangential_pa(-1.20*b_ring, -0.70*b_ring)),
    Sersic(x0=+1.70*b_ring, y0=+0.45*b_ring,
           R_e=0.060*b_ring, n=2.5, I_e=0.11,
           ell=0.28, pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-1.65*b_ring, y0=-0.50*b_ring,
           R_e=0.065*b_ring, n=2.5, I_e=0.11,
           ell=0.25, pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+0.70*b_ring, y0=-1.70*b_ring,
           R_e=0.060*b_ring, n=2.0, I_e=0.11,
           ell=0.22, pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-0.55*b_ring, y0=-1.75*b_ring,
           R_e=0.055*b_ring, n=2.0, I_e=0.10,
           ell=0.20, pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+1.90*b_ring, y0=+1.75*b_ring,
           R_e=0.060*b_ring, n=2.5, I_e=0.10,
           ell=0.28, pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-1.70*b_ring, y0=+1.95*b_ring,
           R_e=0.058*b_ring, n=2.0, I_e=0.10,
           ell=0.30, pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+0.00*b_ring, y0=-2.25*b_ring,
           R_e=0.055*b_ring, n=2.5, I_e=0.10,
           ell=0.20, pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+2.40*b_ring, y0=+0.20*b_ring,
           R_e=0.055*b_ring, n=2.5, I_e=0.10,
           ell=0.25, pa=_rng_gal.uniform(0, pi)),

    # --- Small compact galaxies sprinkled across the field ---
    Sersic(x0=+0.85*b_ring, y0=+1.55*b_ring,
           R_e=0.038*b_ring, n=2.0, I_e=0.12, ell=0.25,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+2.00*b_ring, y0=+0.65*b_ring,
           R_e=0.035*b_ring, n=2.0, I_e=0.11, ell=0.28,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-0.35*b_ring, y0=+1.70*b_ring,
           R_e=0.033*b_ring, n=2.0, I_e=0.11, ell=0.32,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+1.95*b_ring, y0=-1.00*b_ring,
           R_e=0.035*b_ring, n=2.0, I_e=0.11, ell=0.28,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-1.90*b_ring, y0=+1.40*b_ring,
           R_e=0.038*b_ring, n=2.0, I_e=0.12, ell=0.22,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+0.30*b_ring, y0=-2.00*b_ring,
           R_e=0.035*b_ring, n=2.0, I_e=0.11, ell=0.18,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-0.85*b_ring, y0=-1.35*b_ring,
           R_e=0.032*b_ring, n=2.0, I_e=0.10, ell=0.22,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+1.30*b_ring, y0=-0.60*b_ring,
           R_e=0.030*b_ring, n=2.0, I_e=0.10, ell=0.25,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-0.95*b_ring, y0=+0.95*b_ring,
           R_e=0.030*b_ring, n=2.0, I_e=0.10, ell=0.28,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+2.50*b_ring, y0=-1.35*b_ring,
           R_e=0.032*b_ring, n=2.0, I_e=0.10, ell=0.25,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=-2.50*b_ring, y0=-1.80*b_ring,
           R_e=0.030*b_ring, n=2.0, I_e=0.10, ell=0.30,
           pa=_rng_gal.uniform(0, pi)),
    Sersic(x0=+1.15*b_ring, y0=-2.40*b_ring,
           R_e=0.032*b_ring, n=2.0, I_e=0.10, ell=0.20,
           pa=_rng_gal.uniform(0, pi)),
]



'''
===============================================================================
================================= STARFIELD ===================================
===============================================================================
'''
##### Dense starfield for HST-like ambience.
n_stars = 220
rng = np.random.default_rng(seed=1038)    # "1038" -> SDSS J1038+4849

##### One bright foreground Milky-Way star with 4-point diffraction spike,
# near the bottom of the frame (cosmetic, not part of the lensing physics).
# Spike length scales with image size so the look is consistent at any x_pixels.
_spike_len = int(0.09 * x_pixels)
bright_star = {
    'alpha':  0.00 * b_ring,              # horizontal position in detector coords
    'beta':  -1.45 * b_ring,              # near the bottom edge
    'sigma_pix': max(2.0, x_pixels/640),  # core gaussian width (pixels)
    'flux':  1.2,                         # core peak flux
    'spike_length_pix': _spike_len,       # each spike arm
    'spike_width_pix':  max(1.0, x_pixels/1200),
    'spike_flux':       0.30,
}



'''
===============================================================================
================================== COLORS =====================================
===============================================================================
'''
lens_color     = (1.00, 0.72, 0.28)       # orange-yellow ellipticals (the eyes)
nose_color     = (0.60, 0.78, 1.00)       # bluish small galaxy (the nose)
arc_color      = (0.40, 0.80, 1.00)       # cyan (starburst lensed source)
gal_color      = (0.85, 0.65, 0.35)       # yellowish faint background galaxies
stars_color    = (0.95, 0.93, 1.00)       # blue-white stars
read_noise     = 0.002
poisson_scale  = 800.0



'''
===============================================================================
============================== RAY-TRACING BOUNDS =============================
===============================================================================
'''
set_ray_bounds(r_escape=0.5*D_L, final_lmbda=3.0*D_L)



'''
===============================================================================
============================ IMAGE FILENAME ===================================
===============================================================================
'''
filename = 'cheshire_cat'
savefig = True



###############################################################################
# Local helper: diffraction spike stamp for the bright MW foreground star.
###############################################################################
def add_bright_star_with_spikes(shape, star_params, detector):
    """Return a float image with a bright gaussian core plus a 4-pointed
    diffraction cross. `star_params` uses (alpha, beta) in detector coords."""
    Nx, Ny = shape
    img = np.zeros((Nx, Ny), dtype=np.float64)

    # Map detector coords (alpha, beta) to pixel indices.
    alpha = detector.alphaRange
    beta = detector.betaRange
    i_star = int(np.argmin(np.abs(alpha - star_params['alpha'])))
    j_star = int(np.argmin(np.abs(beta  - star_params['beta'])))

    # Gaussian core.
    sigma = star_params['sigma_pix']
    flux  = star_params['flux']
    rad = int(np.ceil(5.0 * sigma))
    i0 = max(0, i_star - rad); i1 = min(Nx, i_star + rad + 1)
    j0 = max(0, j_star - rad); j1 = min(Ny, j_star + rad + 1)
    xs = np.arange(i0, i1) - i_star
    ys = np.arange(j0, j1) - j_star
    gx = np.exp(-0.5 * (xs / sigma)**2)
    gy = np.exp(-0.5 * (ys / sigma)**2)
    img[i0:i1, j0:j1] += flux * gx[:, None] * gy[None, :]

    # 4-arm diffraction spike: horizontal + vertical thin gaussians with
    # a mild longitudinal falloff.
    L = star_params['spike_length_pix']
    w = star_params['spike_width_pix']
    F = star_params['spike_flux']

    # Horizontal arm: extends along i (alpha direction).
    ih0 = max(0, i_star - L); ih1 = min(Nx, i_star + L + 1)
    jh0 = max(0, j_star - int(6*w)); jh1 = min(Ny, j_star + int(6*w) + 1)
    xs = np.arange(ih0, ih1) - i_star
    ys = np.arange(jh0, jh1) - j_star
    falloff = np.exp(-np.abs(xs) / (L/2.5))    # smooth taper to tip
    prof_y = np.exp(-0.5 * (ys / w)**2)
    img[ih0:ih1, jh0:jh1] += F * falloff[:, None] * prof_y[None, :]

    # Vertical arm: extends along j (beta direction).
    iv0 = max(0, i_star - int(6*w)); iv1 = min(Nx, i_star + int(6*w) + 1)
    jv0 = max(0, j_star - L); jv1 = min(Ny, j_star + L + 1)
    xs = np.arange(iv0, iv1) - i_star
    ys = np.arange(jv0, jv1) - j_star
    prof_x = np.exp(-0.5 * (xs / w)**2)
    falloff = np.exp(-np.abs(ys) / (L/2.5))
    img[iv0:iv1, jv0:jv1] += F * prof_x[:, None] * falloff[None, :]

    return img




'''
===============================================================================
==================================== MAIN =====================================
===============================================================================
'''
print("Tracing primary lensed source (disk)...")
source_plane_primary = SourcePlane(D_LS=D_LS, profile=primary_source)
image_primary = LensImage(lens, source_plane_primary, detector)
image_primary.create_photons()
image_primary.create_image()
primary_arc = image_primary.image_data.copy()
if primary_arc.max() > 0:
    primary_arc /= primary_arc.max()


print("Tracing secondary lensed clumps...")
secondary_arcs = []
for idx, source in enumerate(secondary_sources):
    source_plane = SourcePlane(D_LS=D_LS, profile=source)
    image = LensImage(lens, source_plane, detector)
    image.create_photons()
    image.create_image()
    arc = image.image_data.copy()
    if arc.max() > 0:
        arc /= arc.max()
    secondary_arcs.append(arc)
    print(f"  Clump {idx+1}/{len(secondary_sources)} done")

# Composite all lensed layers (brightest-pixel combine).
arc_img = primary_arc.copy()
for secondary_arc in secondary_arcs:
    arc_img = np.maximum(arc_img, secondary_arc)


def _asinh_compress(img, scale=0.5):
    """Per-layer asinh compression: stretches faint halos while compressing
    sharp Sersic cores, then rescales so max = 1. This is what makes
    Sersic-n=4 galaxies look like galaxies (extended + core) instead of
    either tiny dots (max-normalize) or saturated blobs (percentile-clip)."""
    if not np.any(img > 0):
        return img
    y = np.arcsinh(img / scale)
    return y / y.max()


print("Projecting lens galaxy light (two eyes + cluster members)...")
eye_img = np.zeros((detector.x_pixels, detector.y_pixels), dtype=np.float64)
for profile in (eye_right, eye_left, *cluster_members):
    eye_img = np.maximum(eye_img, project_profile_on_detector(detector,
                                                                profile))
eye_img = _asinh_compress(eye_img, scale=0.3)


print("Projecting central blue 'nose' galaxy...")
nose_img = project_profile_on_detector(detector, nose_gal)
if nose_img.max() > 0:
    nose_img /= nose_img.max()


print("Projecting background galaxies (no lensing)...")
gal_img = np.zeros_like(eye_img)
for bg_gal in background_galaxies:
    gal_img = np.maximum(gal_img,
                         project_profile_on_detector(detector, bg_gal))
# asinh-compress so each galaxy's halo (out to R_e) stays visible
# while its sharp n=4 core doesn't dominate the layer.
gal_img = _asinh_compress(gal_img, scale=0.2)


print("Adding starfield + bright foreground star with diffraction spikes...")
stars_img = add_starfield(arc_img.shape, n_stars=n_stars, rng=rng)
stars_img += add_bright_star_with_spikes(arc_img.shape, bright_star, detector)


print("Composing RGB...")
rgb = (eye_img[..., None]   * np.array(lens_color) +
       nose_img[..., None]  * np.array(nose_color) * 0.8 +
       arc_img[..., None]   * np.array(arc_color) +
       0.55 * gal_img[..., None]   * np.array(gal_color) +
       stars_img[..., None] * np.array(stars_color))


print("Adding photographic noise (Poisson + read noise)...")
rgb = add_photographic_noise(rgb, read_noise=read_noise,
                              poisson_scale=poisson_scale, rng=rng)


print("Applying asinh stretch...")
rgb = asinh_stretch(rgb, softening=0.045, max_percentile=99.5)


print("Saving...")
save('images_data/'+filename+'.npy', rgb)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
ax.set_xticks([])
ax.set_yticks([])
if savefig:
    plt.savefig('images/'+filename+'.png', dpi=200, bbox_inches='tight')
plt.show()

print("Done! Image saved to images/"+filename+".png")
