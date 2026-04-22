"""
===============================================================================
Main script
Creates the Einstein ring image (gravitational lensing)
===============================================================================
Geometry: iota = pi/2 puts the detector in parallel-projection mode (observer
at infinity limit). Each pixel backward-traces a photon along -x with impact
parameter b = sqrt(alpha^2 + beta^2) at the lens. The source plane is at
x = -D_LS. The Einstein-ring condition alpha_deflection * D_LS = b gives

    Schwarzschild (alpha = 4 M / b):    b_ring = 2 sqrt(M D_LS)
    SIS (alpha = 4 pi sigma_v^2):       b_ring = 4 pi sigma_v^2 D_LS
===============================================================================
"""

from math import pi, sqrt
import numpy as np
from numpy import save

from scr.black_holes import schwarzschild
from scr.lens_metrics import sis
from scr.detectors import image_plane
from scr.sources.light_profiles import Gaussian, Sersic
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.common import set_ray_bounds
import warnings
warnings.filterwarnings("ignore")




'''
===============================================================================
=============================== LENS DEFINITION ===============================
===============================================================================
'''
##### SCHWARZSCHILD POINT-MASS LENS (M = 1)
lens = schwarzschild.BlackHole()


##### SINGULAR ISOTHERMAL SPHERE (SIS) GALAXY LENS
#sigma_v = 0.03   # velocity dispersion in units of c
#lens = sis.LensMetric(sigma_v=sigma_v, r_ref=1.0, r_min=1e-2)



'''
===============================================================================
=========================== DETECTOR PARAMETERS ===============================
===============================================================================
'''
D_L = 1.0e4               # Observer-to-lens distance (M units)
D_LS = 1.0e4              # Lens-to-source distance
iota = pi/2               # Equatorial view: parallel rays along -x
x_side = 500              # Half-width of the screen in M
x_pixels = 1000

detector = image_plane.detector(D=D_L, iota=iota, x_pixels=x_pixels,
                                 x_side=x_side, ratio='1:1')



'''
===============================================================================
================================ SOURCE PLANE =================================
===============================================================================
'''

########### GAUSSIAN BACKGROUND SOURCE
#profile = Sersic(x0=0.0, y0=0.0, R_e=20.0, n=1.0, I_e=1.0, ell=0.0, pa=0.0)
profile = Gaussian(x0=0.0, y0=0.0, sigma=20.0, I0=1.0)

source_plane = SourcePlane(D_LS=D_LS, profile=profile)



'''
===============================================================================
============================== RAY-TRACING BOUNDS =============================
===============================================================================
'''
# Lensing needs r_escape < D_L so the escape event fires on the photon's
# way back out of the lens region, and final_lmbda large enough to cover
# the round-trip through the lens.
set_ray_bounds(r_escape=0.5*D_L, final_lmbda=3.0*D_L)



'''
===============================================================================
============================ IMAGE FILENAME ===================================
===============================================================================
'''
filename = 'einstein_ring_Schwarzschild'
savefig = True




'''
===============================================================================
==================================== MAIN =====================================
===============================================================================
'''
image = LensImage(lens, source_plane, detector)

# Photons creation
image.create_photons()

# Create the image data (mode = lensing)
image.create_image()

# Save the image data
save('images_data/'+filename+'.npy', image.image_data)

# Plot the image
image.plot(savefig=savefig, filename=filename)
