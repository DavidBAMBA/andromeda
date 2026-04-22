"""
===============================================================================
Main script
Strong-field gravitational lensing by a Kerr black hole, compared against
Schwarzschild (a = 0) with identical source and detector geometry.

Physics
-------
- Observer in the equatorial plane (iota = pi/2) so the spin axis is
  perpendicular to the line of sight -> maximum azimuthal (frame-dragging)
  asymmetry visible in the image.
- D_L = D_LS ~ 100 M puts us in a regime where the Einstein radius
  (b ~ 20 M) is only a few times larger than the critical impact parameter
  (b_c ~ 5 M). Photons with b < b_c plunge into the horizon (shadow),
  photons with b close to b_Einstein form the ring.
- Kerr's spin adds a rotational drag: light rays prograde with the spin
  get dragged more than retrograde rays -> the ring and shadow are
  asymmetric in alpha (horizontal on the detector).
===============================================================================
"""

from math import pi, sqrt
import numpy as np
from numpy import save
import matplotlib.pyplot as plt

from scr.black_holes import schwarzschild, kerr
from scr.detectors import image_plane
from scr.sources.light_profiles import Gaussian, Sersic
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.common import set_ray_bounds
from scr.common.visual import compose_rgb, asinh_stretch
import warnings
warnings.filterwarnings("ignore")




'''
===============================================================================
================================= GEOMETRY ====================================
===============================================================================
'''
D_L = 50.0                               # Observer-to-lens distance (M)
D_LS = 50.0                              # Lens-to-source distance
iota = pi/2                              # Equatorial observer
M = 1.0
b_ring = 2.0 * sqrt(M * D_LS)            # Schwarzschild Einstein radius ~14 M
x_pixels = 512
x_side = 2.0 * b_ring                    # tight crop to see ring+shadow

detector = image_plane.detector(D=D_L, iota=iota, x_pixels=x_pixels,
                                 x_side=x_side, ratio='1:1')



'''
===============================================================================
=============================== LENSED SOURCE =================================
===============================================================================
'''
##### Extended background ("diffuse galaxy disk") so the BH shadow shows up
##### as the dark hole in the middle and the Einstein ring is a brighter
##### distortion on top of the background brightness. A broad Gaussian
##### + small offset produces a near-complete asymmetric ring.
source_offset = 0.08 * b_ring
source_profile = Gaussian(x0=source_offset, y0=0.02*b_ring,
                           sigma=0.45*b_ring, I0=1.0)
source_plane = SourcePlane(D_LS=D_LS, profile=source_profile)



'''
===============================================================================
============================== RAY-TRACING BOUNDS =============================
===============================================================================
'''
set_ray_bounds(r_escape=0.5*D_L, final_lmbda=4.0*D_L)



'''
===============================================================================
================================= COLORS ======================================
===============================================================================
'''
arc_color = (0.25, 0.55, 1.00)           # blue background galaxy



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


# --- Schwarzschild (a = 0) ---
sch_arc = render_with_lens(schwarzschild.BlackHole(), "Schwarzschild (a=0)")

# --- Kerr (a = 0.9) ---
kerr_arc = render_with_lens(kerr.BlackHole(a=0.9), "Kerr (a=0.9)")

sch_rgb = monochrome_to_rgb(sch_arc)
kerr_rgb = monochrome_to_rgb(kerr_arc)

# Save raw npy for later inspection
save('images_data/kerr_vs_schwarzschild_lensing.npy',
     np.stack([sch_rgb, kerr_rgb]))

# Difference map (Kerr - Schwarzschild) reveals frame-dragging asymmetry
diff = kerr_arc - sch_arc
diff_rgb = np.zeros((*diff.shape, 3))
scale = max(abs(diff.min()), abs(diff.max()), 1e-12)
# Red where Kerr > Sch, blue where Sch > Kerr
diff_rgb[..., 0] = np.clip(diff / scale, 0, 1)                    # Kerr excess
diff_rgb[..., 2] = np.clip(-diff / scale, 0, 1)                   # Sch excess
diff_rgb[..., 1] = 0.25 * (diff_rgb[..., 0] + diff_rgb[..., 2])

# 3-panel plot
fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                          facecolor='black')

for ax, rgb, title in zip(
        axes,
        [sch_rgb, kerr_rgb, diff_rgb],
        [f'Schwarzschild  (a = 0)\nb_E ≈ {b_ring:.1f} M,  D_L = {D_L:.0f} M',
         f'Kerr  (a = 0.9)\nequatorial view (iota = pi/2)',
         f'Difference  (Kerr - Schwarzschild)\nred: Kerr brighter,  blue: Sch brighter']):
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color='white', fontsize=10.5)

plt.tight_layout()
plt.savefig('images/kerr_vs_schwarzschild_lensing.png',
            dpi=200, bbox_inches='tight', facecolor='black')
plt.show()
