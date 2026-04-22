"""
===============================================================================
Main script
Creates a Hubble-style composite image of a strong gravitational lens
(SIS galaxy lens + background source galaxy), in the spirit of the
"Cosmic Horseshoe" (LRG 3-757).
===============================================================================
Pipeline:
    1. Ray-traced lensed arc (LensImage, SIS + Sersic source offset)
    2. Direct projection of the LENS galaxy's own visible light (Sersic n=4)
    3. Background starfield (random Gaussian PSFs)
    4. RGB composition (lens color = orange/red, arc = blue, stars = white)
    5. Photographic noise (Poisson + Gaussian)
    6. asinh stretch -> PNG + npy
===============================================================================
"""

from math import pi
import numpy as np
from numpy import save
import matplotlib.pyplot as plt

from scr.lens_metrics import sis
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
##### SIS GALAXY LENS
sigma_v = 0.03                            # velocity dispersion (units of c)
lens = sis.LensMetric(sigma_v=sigma_v, r_ref=1.0, r_min=1e-2)



'''
===============================================================================
=========================== DETECTOR PARAMETERS ===============================
===============================================================================
'''
D_L = 1.0e4                               # Observer-to-lens distance (M)
D_LS = 1.0e4                              # Lens-to-source distance
iota = pi/2                               # Parallel projection
b_ring = 4.0 * pi * sigma_v**2 * D_LS     # SIS ring radius ~113 M
x_side = 2.8 * b_ring                     # half-width of the screen
x_pixels = 512

detector = image_plane.detector(D=D_L, iota=iota, x_pixels=x_pixels,
                                 x_side=x_side, ratio='1:1')



'''
===============================================================================
=============================== LENSED SOURCE =================================
===============================================================================
'''
##### BACKGROUND SOURCE GALAXY (offset from optical axis to break symmetry
##### and produce a "horseshoe" arc instead of a full Einstein ring)
source_offset = 0.09 * b_ring             # small offset -> near-complete ring
source_profile = Sersic(x0=source_offset, y0=0.03*b_ring,
                         R_e=0.10*b_ring, n=1.0, I_e=1.0,
                         ell=0.15, pa=pi/6)

source_plane = SourcePlane(D_LS=D_LS, profile=source_profile)



'''
===============================================================================
=========================== LENS GALAXY LIGHT =================================
===============================================================================
'''
##### SERSIC n=4 (de Vaucouleurs) profile representing the visible light of
##### the elliptical lens galaxy. Projected directly on the detector, no
##### ray tracing (this light does not itself get lensed in the standard
##### observational case).
lens_light_profile = Sersic(x0=0.0, y0=0.0,
                             R_e=0.30*b_ring, n=4.0, I_e=0.7,
                             ell=0.15, pa=pi/8)



'''
===============================================================================
================================= STARFIELD ===================================
===============================================================================
'''
n_stars = 220
rng = np.random.default_rng(seed=42)



'''
===============================================================================
============================ COLOR AND NOISE ==================================
===============================================================================
'''
lens_color  = (1.00, 0.70, 0.25)          # orange-red (old elliptical)
arc_color   = (0.25, 0.55, 1.00)          # blue (starburst background)
stars_color = (0.90, 0.92, 1.00)          # slight blue-white
read_noise   = 0.015
poisson_scale = 500.0                     # higher = cleaner image



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
filename = 'cosmic_horseshoe'
savefig = True




'''
===============================================================================
==================================== MAIN =====================================
===============================================================================
'''
# 1) Ray-traced lensed arc
image = LensImage(lens, source_plane, detector)
image.create_photons()
image.create_image()
arc_img = image.image_data.copy()
if arc_img.max() > 0:
    arc_img /= arc_img.max()

# 2) Direct projection of the lens galaxy's visible light
lens_img = project_profile_on_detector(detector, lens_light_profile)
if lens_img.max() > 0:
    lens_img /= lens_img.max()

# 3) Starfield
stars_img = add_starfield(arc_img.shape, n_stars=n_stars, rng=rng)

# 4) RGB composition
rgb = compose_rgb(lens_img, arc_img, stars_img,
                  lens_color=lens_color,
                  arc_color=arc_color,
                  stars_color=stars_color)

# 5) Photographic noise
rgb = add_photographic_noise(rgb, read_noise=read_noise,
                              poisson_scale=poisson_scale, rng=rng)

# 6) asinh stretch
rgb = asinh_stretch(rgb, softening=0.03, max_percentile=99.7)

# 7) Save & plot
save('images_data/'+filename+'.npy', rgb)

fig, ax = plt.subplots(figsize=(6, 6))
# Convention: imshow uses (row, col) with rows going down.
# Our rgb is (x_pixel, y_pixel); transpose so y is vertical.
ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
ax.set_xticks([]); ax.set_yticks([])
if savefig:
    plt.savefig('images/'+filename+'.png', dpi=200, bbox_inches='tight')
plt.show()
