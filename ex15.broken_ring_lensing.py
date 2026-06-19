"""
===============================================================================
Broken Einstein Ring with Asymmetric Source
===============================================================================
Pipeline:
    Similar to ex14, but with:
    - Highly asymmetric source (strong ellipticity)
    - Off-center positioning to create "broken" sector
    - Natural magnification variation that darkens part of the ring
    - Creates realistic incomplete ring effect (like in observations)

Result: Einstein ring that appears to have a "missing" or "darkened" sector,
similar to real HST strong lensing systems.
===============================================================================
"""

from math import pi, sqrt
import numpy as np
from numpy import save
import matplotlib.pyplot as plt

from scr.lens_metrics import sie
from scr.detectors import image_plane
from scr.sources.light_profiles import Gaussian, Sersic
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.common import set_ray_bounds
from scr.common.visual import (project_profile_on_detector, add_starfield,
                                 add_photographic_noise, compose_rgb,
                                 asinh_stretch)
import warnings
warnings.filterwarnings("ignore")




'''
===============================================================================
=============================== LENS DEFINITION ===============================
===============================================================================
'''
##### SIE GALAXY LENS (elliptical, asymmetric)
sigma_v = 0.028                           # velocity dispersion (units of c)
q_ax = 0.70                               # axis ratio (0 < q < 1, 1=circular)
lens = sie.LensMetric(sigma_v=sigma_v, q_ax=q_ax,
                      r_ref=1.0, r_min=1e-2)



'''
===============================================================================
=========================== DETECTOR PARAMETERS ===============================
===============================================================================
'''
D_L = 1.0e4                               # Observer-to-lens distance (M)
D_LS = 1.0e4                              # Lens-to-source distance
iota = pi/2                               # Parallel projection
b_ring = 4.0 * pi * sigma_v**2 * D_LS     # SIS ring radius ~107 M
x_side = 3.2 * b_ring                     # FOV
x_pixels = 1920

detector = image_plane.detector(D=D_L, iota=iota, x_pixels=x_pixels,
                                 x_side=x_side, ratio='1:1')



'''
===============================================================================
============================ LENSED SOURCES ===================================
===============================================================================
'''
##### PRIMARY SOURCE: Highly asymmetric (strong ellipticity) creates multiple image configuration
# High ellipticity + offset creates distinct asymmetric arcs, not a continuous ring
primary_source = Sersic(x0=0.10*b_ring, y0=0.06*b_ring,
                        R_e=0.11*b_ring, n=1.0, I_e=1.4,
                        ell=0.45, pa=pi/3.5)  # highly elliptical for asymmetry!

##### SECONDARY SOURCES: Small companions
secondary_sources = [
    Sersic(x0=0.14*b_ring, y0=-0.07*b_ring,
           R_e=0.04*b_ring, n=1.5, I_e=0.5, ell=0.25, pa=-pi/3),
    Sersic(x0=0.06*b_ring, y0=0.13*b_ring,
           R_e=0.035*b_ring, n=1.0, I_e=0.35, ell=0.15, pa=0.0),
]



'''
===============================================================================
=========================== LENS GALAXY LIGHT =================================
===============================================================================
'''
##### SERSIC n=4 (de Vaucouleurs) profile for the elliptical lens galaxy
lens_light_profile = Sersic(x0=0.0, y0=0.0,
                             R_e=0.28*b_ring, n=4.0, I_e=0.8,
                             ell=0.20, pa=pi/6)



'''
===============================================================================
======================== BACKGROUND GALAXIES (NO LENSING) ====================
===============================================================================
'''
##### Multiple faint background galaxies
background_galaxies = [
    Sersic(x0=-0.50*b_ring, y0=0.55*b_ring,
           R_e=0.05*b_ring, n=1.0, I_e=0.15, ell=0.30, pa=pi/5),
    Sersic(x0=0.60*b_ring, y0=-0.45*b_ring,
           R_e=0.04*b_ring, n=4.0, I_e=0.12, ell=0.15, pa=-pi/7),
    Sersic(x0=-0.35*b_ring, y0=-0.38*b_ring,
           R_e=0.035*b_ring, n=1.0, I_e=0.10, ell=0.20, pa=pi/3),
    Gaussian(x0=0.48*b_ring, y0=0.42*b_ring, sigma=0.03*b_ring, I0=0.08),
    Sersic(x0=-0.58*b_ring, y0=0.08*b_ring,
           R_e=0.045*b_ring, n=1.5, I_e=0.11, ell=0.25, pa=-pi/6),
]



'''
===============================================================================
================================= STARFIELD ===================================
===============================================================================
'''
##### Dense starfield
n_stars = 150
rng = np.random.default_rng(seed=42)



'''
===============================================================================
================================== COLORS =====================================
===============================================================================
'''
lens_color     = (1.00, 0.70, 0.25)       # orange-red (old elliptical)
arc_color      = (0.25, 0.55, 1.00)       # blue (starburst background)
gal_color      = (0.85, 0.65, 0.35)       # yellowish (faint background galaxies)
stars_color    = (0.95, 0.93, 1.00)       # blue-white stars
read_noise     = 0.002
poisson_scale  = 800.0                    # reduced noise for clean image



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
filename = 'asymmetric_multi_image_lensing'
savefig = True




'''
===============================================================================
==================================== MAIN =====================================
===============================================================================
'''
print("Tracing primary lensed source...")
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

# Combine all lensed arcs
arc_img = primary_arc.copy()
for secondary_arc in secondary_arcs:
    arc_img = np.maximum(arc_img, secondary_arc)


print("Projecting lens galaxy light...")
lens_img = project_profile_on_detector(detector, lens_light_profile)
if lens_img.max() > 0:
    lens_img /= lens_img.max()


print("Projecting background galaxies (no lensing)...")
gal_img = np.zeros_like(lens_img)
for bg_gal in background_galaxies:
    gal_projected = project_profile_on_detector(detector, bg_gal)
    gal_img = np.maximum(gal_img, gal_projected)
if gal_img.max() > 0:
    gal_img /= gal_img.max()


print("Adding starfield...")
stars_img = add_starfield(arc_img.shape, n_stars=n_stars, rng=rng)


print("Composing RGB...")
rgb = (lens_img[..., None] * np.array(lens_color) +
       arc_img[..., None] * np.array(arc_color) +
       0.3 * gal_img[..., None] * np.array(gal_color) +
       stars_img[..., None] * np.array(stars_color))


print("Adding photographic noise (Poisson + read noise)...")
rgb = add_photographic_noise(rgb, read_noise=read_noise,
                              poisson_scale=poisson_scale, rng=rng)


print("Applying asinh stretch...")
rgb = asinh_stretch(rgb, softening=0.05, max_percentile=99.8)


print("Saving...")
save('images_data/lensing/'+filename+'.npy', rgb)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
ax.set_xticks([])
ax.set_yticks([])
if savefig:
    plt.savefig('images/lensing/'+filename+'.png', dpi=200, bbox_inches='tight')
plt.show()

print("Done! Image saved to images/lensing/"+filename+".png")
