"""
===============================================================================
Cheshire Cat lensing system (inspired by SDSS J1038+4849)
===============================================================================
Pipeline:
    1. Binary SIE lens (two SIE potentials superposed at offset centers).
       The asymmetry of the lensed arcs comes from the REAL geometry of the
       non-axisymmetric potential, not from cosmetic post-processing.
    2. Diffuse lensed source near the tangential caustic (small offset,
       large R_e) -> quasi-closed Einstein ring with natural asymmetry.
    3. Two cosmetic lens-galaxy light profiles (Sersic n=4) centered at the
       same (y, z) as the BinarySIE centers -> the visible "two eyes".
    4. Deep-field realistic background: 180 galaxies sampled uniformly over
       the detector with power-law luminosity, log-normal sizes, 20%
       edge-on population, and a 5-color palette simulating multiple
       redshifts.
    5. Starfield + photographic noise + asinh stretch.

Emulates the "Smiley" galaxy cluster lens (SDSS J1038+4849): two bright
nuclei + lensed starburst arc around them. The arc is not perfectly
symmetric because BinarySIE breaks axisymmetry in phi (k_phi is NOT
conserved), producing physically correct magnification asymmetry.
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
=========================== DETECTOR PARAMETERS ===============================
===============================================================================
'''
D_L = 1.0e4                                   # Observer-to-lens distance (M)
D_LS = 1.0e4                                  # Lens-to-source distance
iota = pi / 2                                 # Parallel projection
# Effective Einstein radius of the combined system:
# sigma_eff^2 = sigma_v1^2 + sigma_v2^2 (verified by test_binary_sie collapse)
sigma_v_1 = 0.022
sigma_v_2 = 0.022
sigma_eff_sq = sigma_v_1 ** 2 + sigma_v_2 ** 2
b_ring = 4.0 * pi * sigma_eff_sq * D_LS       # ~121 M
x_side = 3.5 * b_ring                         # large FOV for dense field
x_pixels = 1920

detector = image_plane.detector(D=D_L, iota=iota, x_pixels=x_pixels,
                                 x_side=x_side, ratio='1:1')



'''
===============================================================================
========================= BINARY SIE LENS DEFINITION ==========================
===============================================================================
'''
# Separation of the two lens centers (horizontal in the image, i.e. y-axis)
# Convention with iota=pi/2:
#   alpha (detector horizontal) <-> y  spatial
#   beta  (detector vertical)   <-> z  spatial
d = 0.18 * b_ring
centers = ((0.0, +d, 0.0),                    # eastern eye
           (0.0, -d, 0.0))                    # western eye
q_ax = (0.75, 0.75)

lens = binary_sie.LensMetric(sigma_v=(sigma_v_1, sigma_v_2),
                              centers=centers, q_ax=q_ax,
                              r_ref=1.0,
                              r_min=1.0e-2,
                              r_min_center=1.0)



'''
===============================================================================
============================ LENSED SOURCES ===================================
===============================================================================
'''
##### PRIMARY SOURCE: Near the tangential caustic + extended
# Small offset -> quasi-closed ring. Slight shift in y0 (= z spatial = down
# in the detector) produces the "lower arc brighter than upper arc"
# asymmetry characteristic of the Cheshire Cat.
primary_source = Sersic(x0=0.01 * b_ring, y0=-0.03 * b_ring,
                        R_e=0.11 * b_ring, n=1.0, I_e=1.2,
                        ell=0.25, pa=pi / 4)

##### SECONDARY SOURCES: small faint companions that light up localized
##### features of the arcs.
secondary_sources = [
    Sersic(x0=0.08 * b_ring, y0=0.04 * b_ring,
           R_e=0.035 * b_ring, n=1.5, I_e=0.45, ell=0.20, pa=-pi / 4),
    Sersic(x0=-0.06 * b_ring, y0=-0.02 * b_ring,
           R_e=0.030 * b_ring, n=1.0, I_e=0.40, ell=0.15, pa=0.0),
]



'''
===============================================================================
====================== LENS GALAXY LIGHT (cosmetic only) =====================
===============================================================================
Two Sersic n=4 profiles at the same (y, z) as the BinarySIE centers so
that visible light co-locates with the mass — as in real merging systems.
'''
lens_nucleus_1 = Sersic(x0=+d, y0=0.0,
                         R_e=0.12 * b_ring, n=4.0, I_e=0.9,
                         ell=0.18, pa=pi / 5)
lens_nucleus_2 = Sersic(x0=-d, y0=0.0,
                         R_e=0.10 * b_ring, n=4.0, I_e=0.75,
                         ell=0.20, pa=-pi / 6)



'''
===============================================================================
======================== DEEP FIELD BACKGROUND GALAXIES ======================
===============================================================================
Astrophysically-motivated sampling:
  - Uniform spatial distribution (no concentric artifact)
  - Power-law luminosity function (Schechter-like)
  - Log-normal size distribution
  - 20% edge-on population
  - 5-color palette simulating multiple redshifts
'''

def generate_background_galaxies(b_ring, rng, n_galaxies=180):
    """Realistic deep-field population of background galaxies."""
    # (R, G, B) palette with redshift-like color progression
    colors_palette = [
        (0.95, 0.45, 0.25),   # red ellipticals (z ~ 1)
        (1.00, 0.65, 0.30),   # orange (z ~ 0.8)
        (1.00, 0.85, 0.35),   # yellow (z ~ 0.6)
        (0.85, 0.85, 0.90),   # white/grey (z ~ 0.4)
        (0.40, 0.65, 1.00),   # blue starbursts (z ~ 0.3)
    ]
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]

    galaxies = []
    for _ in range(n_galaxies):
        # Uniform spatial sampling (no central exclusion zone)
        x0 = rng.uniform(-3.3 * b_ring, 3.3 * b_ring)
        y0 = rng.uniform(-3.3 * b_ring, 3.3 * b_ring)

        # Power-law luminosity function: many faint, few bright
        u = rng.uniform(0.01, 1.0)
        I_e = 0.015 * u ** (-1.5)
        I_e = min(I_e, 0.35)                  # clamp to avoid saturation

        # Log-normal size distribution
        R_e = float(np.exp(rng.normal(np.log(0.025 * b_ring), 0.6)))
        R_e = max(0.008 * b_ring, min(R_e, 0.07 * b_ring))

        # 20% edge-on disks, 80% moderately elliptical
        if rng.random() < 0.2:
            ell = rng.uniform(0.6, 0.85)
        else:
            ell = rng.uniform(0.0, 0.35)

        n_sersic = float(rng.choice([1.0, 1.5, 2.5, 4.0]))
        pa = rng.uniform(0.0, 2.0 * pi)
        color_idx = rng.choice(5, p=weights)
        color = colors_palette[color_idx]

        galaxies.append({
            'profile': Sersic(x0=x0, y0=y0, R_e=R_e, n=n_sersic,
                              I_e=I_e, ell=ell, pa=pa),
            'color': color,
        })
    return galaxies


rng_bg = np.random.default_rng(seed=7)
background_galaxies = generate_background_galaxies(b_ring, rng_bg,
                                                    n_galaxies=180)



'''
===============================================================================
=============================== STARFIELD =====================================
===============================================================================
'''
n_stars = 150
rng_stars = np.random.default_rng(seed=42)



'''
===============================================================================
================================ COLORS =======================================
===============================================================================
'''
arc_color      = (0.25, 0.55, 1.00)           # blue (high-z starburst)
nucleus1_color = (1.00, 0.75, 0.30)           # golden
nucleus2_color = (0.95, 0.70, 0.40)           # slightly redder
stars_color    = (0.95, 0.93, 1.00)
read_noise     = 0.002
poisson_scale  = 800.0



'''
===============================================================================
============================== RAY-TRACING BOUNDS =============================
===============================================================================
'''
set_ray_bounds(r_escape=0.5 * D_L, final_lmbda=3.0 * D_L)



'''
===============================================================================
============================ IMAGE FILENAME ===================================
===============================================================================
'''
filename = 'cheshire_cat'
savefig = True




'''
===============================================================================
==================================== MAIN =====================================
===============================================================================
'''
print("Tracing primary lensed source (BinarySIE)...")
source_plane_primary = SourcePlane(D_LS=D_LS, profile=primary_source)
image_primary = LensImage(lens, source_plane_primary, detector)
image_primary.create_photons()
image_primary.create_image()
primary_arc = image_primary.image_data.copy()
if primary_arc.max() > 0:
    primary_arc /= primary_arc.max()


print("Tracing secondary lensed sources...")
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
    print(f"  Secondary source {idx+1}/{len(secondary_sources)} done")

arc_img = primary_arc.copy()
for secondary_arc in secondary_arcs:
    arc_img = np.maximum(arc_img, secondary_arc)


print("Projecting lens nuclei...")
nucleus1_img = project_profile_on_detector(detector, lens_nucleus_1)
nucleus2_img = project_profile_on_detector(detector, lens_nucleus_2)
lens_img = nucleus1_img + nucleus2_img
if lens_img.max() > 0:
    lens_img /= lens_img.max()


print(f"Projecting {len(background_galaxies)} background galaxies...")
Nx, Ny = arc_img.shape
bg_img_red   = np.zeros((Nx, Ny), dtype=np.float64)
bg_img_green = np.zeros((Nx, Ny), dtype=np.float64)
bg_img_blue  = np.zeros((Nx, Ny), dtype=np.float64)

for idx, gal_dict in enumerate(background_galaxies):
    profile = gal_dict['profile']
    color = gal_dict['color']
    gal_img = project_profile_on_detector(detector, profile)
    bg_img_red   += color[0] * gal_img
    bg_img_green += color[1] * gal_img
    bg_img_blue  += color[2] * gal_img
    if (idx + 1) % 30 == 0:
        print(f"  {idx+1}/{len(background_galaxies)} galaxies done")

# Normalize jointly so the colour balance is preserved across channels
max_val = max(bg_img_red.max(), bg_img_green.max(), bg_img_blue.max())
if max_val > 0:
    bg_img_red   /= max_val
    bg_img_green /= max_val
    bg_img_blue  /= max_val


print("Adding starfield...")
stars_img = add_starfield(arc_img.shape, n_stars=n_stars,
                           flux_range=(0.03, 0.5),
                           sigma_range=(0.4, 1.2), rng=rng_stars)


print("Composing RGB...")
rgb = np.zeros((Nx, Ny, 3), dtype=np.float64)

# Lensed arcs (blue starburst)
rgb[..., 0] += 0.6 * arc_img * arc_color[0]
rgb[..., 1] += 0.6 * arc_img * arc_color[1]
rgb[..., 2] += 0.6 * arc_img * arc_color[2]

# Lens nuclei (golden)
rgb[..., 0] += lens_img * nucleus1_color[0] * 0.8
rgb[..., 1] += lens_img * nucleus1_color[1] * 0.8
rgb[..., 2] += lens_img * nucleus1_color[2] * 0.8

# Background galaxies
rgb[..., 0] += 0.35 * bg_img_red
rgb[..., 1] += 0.35 * bg_img_green
rgb[..., 2] += 0.35 * bg_img_blue

# Stars
rgb[..., 0] += stars_img * stars_color[0]
rgb[..., 1] += stars_img * stars_color[1]
rgb[..., 2] += stars_img * stars_color[2]


print("Adding photographic noise...")
rgb = add_photographic_noise(rgb, read_noise=read_noise,
                              poisson_scale=poisson_scale, rng=rng_stars)


print("Applying asinh stretch...")
rgb = asinh_stretch(rgb, softening=0.05, max_percentile=99.5)


print("Saving...")
save('images_data/lensing/'+filename+'.npy', rgb)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
ax.set_xticks([])
ax.set_yticks([])
if savefig:
    plt.savefig('images/lensing/'+filename+'.png', dpi=200,
                 bbox_inches='tight')
plt.show()

print("Done! Image saved to images/lensing/"+filename+".png")
