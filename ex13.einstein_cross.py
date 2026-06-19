"""
===============================================================================
Main script
Singular Isothermal Ellipsoid (SIE) lensing compared to Singular Isothermal
Sphere (SIS) — demonstrates that an elliptical mass distribution plus a
source slightly off-axis produces multi-image configurations (the classical
"Einstein cross").

Observer at iota = pi/2 views the SIE with its symmetry axis (z) in the
plane of the sky, so the lens appears as an ellipse flattened along z.
===============================================================================
"""

from math import pi
import numpy as np
from numpy import save
import matplotlib.pyplot as plt

from scr.lens_metrics import sis, sie
from scr.detectors import image_plane
from scr.sources.light_profiles import Gaussian, Sersic
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.common import set_ray_bounds
from scr.common.visual import compose_rgb, asinh_stretch
import warnings
warnings.filterwarnings("ignore")




'''
===============================================================================
=========================== LENS METRIC PARAMETERS ============================
===============================================================================
'''
sigma_v = 0.03                           # velocity dispersion
q_ax = 0.55                              # SIE axis ratio (0.5-0.9 realistic)



'''
===============================================================================
================================= GEOMETRY ====================================
===============================================================================
'''
D_L = 1.0e4
D_LS = 1.0e4
iota = pi/2
b_ring = 4.0 * pi * sigma_v**2 * D_LS    # SIS Einstein radius ~113 M
x_pixels = 512
x_side = 2.3 * b_ring

detector = image_plane.detector(D=D_L, iota=iota, x_pixels=x_pixels,
                                 x_side=x_side, ratio='1:1')



'''
===============================================================================
=============================== LENSED SOURCE =================================
===============================================================================
'''
##### Very compact, nearly on-axis source. Small offset so the configuration
##### moves from "ring" to "multiple images" regime for the SIE.
source_profile = Gaussian(x0=0.04 * b_ring,
                           y0=0.015 * b_ring,
                           sigma=0.025 * b_ring,
                           I0=1.0)
source_plane = SourcePlane(D_LS=D_LS, profile=source_profile)



'''
===============================================================================
============================== RAY-TRACING BOUNDS =============================
===============================================================================
'''
set_ray_bounds(r_escape=0.5*D_L, final_lmbda=3.0*D_L)



'''
===============================================================================
================================= COLORS ======================================
===============================================================================
'''
arc_color = (0.25, 0.55, 1.00)



'''
===============================================================================
==================================== MAIN =====================================
===============================================================================
'''
def render_with_lens(lens, label):
    print(f"\n>>> Rendering {label}")
    image = LensImage(lens, source_plane, detector)
    image.create_photons()
    image.create_image()
    arc = image.image_data.copy()
    if arc.max() > 0:
        arc /= arc.max()
    return arc


def monochrome_to_rgb(arc):
    zeros = np.zeros_like(arc)
    rgb = compose_rgb(zeros, arc, None,
                      lens_color=(0.0, 0.0, 0.0),
                      arc_color=arc_color)
    return asinh_stretch(rgb, softening=0.03, max_percentile=99.7)


sis_arc = render_with_lens(sis.LensMetric(sigma_v=sigma_v),
                            f"SIS (sigma_v={sigma_v})")
sie_arc = render_with_lens(sie.LensMetric(sigma_v=sigma_v, q_ax=q_ax),
                            f"SIE (sigma_v={sigma_v}, q={q_ax})")

sis_rgb = monochrome_to_rgb(sis_arc)
sie_rgb = monochrome_to_rgb(sie_arc)

save('images_data/lensing/sis_vs_sie_lensing.npy',
     np.stack([sis_rgb, sie_rgb]))

# Side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), facecolor='black')
for ax, rgb, title in zip(
        axes,
        [sis_rgb, sie_rgb],
        [f'SIS  (spherical, q = 1)\nsigma_v = {sigma_v}',
         f'SIE  (ellipsoidal, q = {q_ax})\nsigma_v = {sigma_v}']):
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color='white', fontsize=11)

plt.tight_layout()
plt.savefig('images/lensing/sis_vs_sie_lensing.png',
            dpi=200, bbox_inches='tight', facecolor='black')
plt.show()
