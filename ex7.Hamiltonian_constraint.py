"""
===============================================================================
Main script 
Creates the Black Hole image
===============================================================================
@author: Alexis Larrañga - 2023
===============================================================================
"""

from numpy import pi, save
from scr.black_holes import schwarzschild
from scr.black_holes import kerr
from scr.black_holes import num_schwarzschild
from scr.accretion_structures import simple_disk 
from scr.accretion_structures import thin_disk 
from scr.detectors import image_plane 
from scr.common.common import Image
import warnings
warnings.filterwarnings("ignore")




'''
===============================================================================
============================ BLACK HOLE DEFINITION ============================
===============================================================================
'''
##### SCHWARZSCHILD BH
#blackhole = schwarzschild.BlackHole()


##### KERR BH
a = 0. # Angular Monmentum
blackhole = kerr.BlackHole(a)


##### NUMERICAL SCHWARZSCHILD BH
#blackhole = num_schwarzschild.BlackHole()



'''
===============================================================================
=========================== DETECTOR PARAMETERS ===============================
===============================================================================
'''
D = 100              # Distance to the BH
iota = (pi/180)*(85)    # Inclination Angle
x_side = 25
x_pixels = 150

detector = image_plane.detector(D=D, iota=iota, x_pixels=x_pixels, x_side=x_side, ratio='16:9')


'''
===============================================================================
============================ ACCRETION STRUCTURE ==============================
===============================================================================
'''

#acc_structure = simple_disk.structure(blackhole, 6*M, 20*M)


########### NOVIKOV-THORNE THIN DISK
#R_min = blackhole.ISCOco 
#R_max = 20*M
acc_structure = thin_disk.structure(blackhole)

'''
===============================================================================
==================================== MAIN =====================================
===============================================================================
'''
image = Image(blackhole, acc_structure, detector)

# Photons creation
image.create_photons()

# Create the image data
image.verify_Hamiltonian(n=10)


