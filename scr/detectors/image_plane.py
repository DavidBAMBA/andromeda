"""
===============================================================================
Image plane to generate the image from the traced photons trajectories in a 
curved spacetime
===============================================================================
@author: Alexis Larrañaga - 2023
===============================================================================
"""

from numpy import sqrt, sin, cos, arccos, arctan, linspace


class detector:
    def __init__(self, D, iota, x_side, x_pixels=25, ratio = '16:9'):
        '''
        =======================================================================
        Defines a screen with sides of size x_s_side and y_s_side, located 
        at a distance D and with an inclination iota. 
        The number of pixels in each direction is given by the variables
        x_pixels and y_pixels
        =======================================================================
        '''
        self.D = D 
        self.sin_iota = sin(iota)
        self.cos_iota = cos(iota)
        if x_pixels & 1:
            # If x_pixels is odd, we add 1 to make it even
            self.x_pixels = x_pixels + 1
        else:
            self.x_pixels = x_pixels 

        if ratio == '16:9':
            self.y_pixels = int(x_pixels*9/16)
            y_side = x_side*9/16

        if ratio == '4:3':
            self.y_pixels = int(x_pixels*3/4)
            y_side = x_side*3/4
        
        if ratio == '1:1':
            self.y_pixels = x_pixels
            y_side = x_side

        self.alphaRange = linspace(-x_side, x_side, self.x_pixels)
        self.betaRange = linspace(-y_side, y_side, self.y_pixels)
        print()
        print ("Size of the screen in Pixels: ", self.x_pixels, "X", self.y_pixels)
        print ("Total Number of Photons: ", self.x_pixels*self.y_pixels)
        print("Expected time of integration : %4.2f seconds \n" % (self.x_pixels*self.y_pixels*0.004))
        print()


    def photon_coords(self, blackhole, alpha, beta, freq=1): 
        '''
        ===========================================================================
        Given the initial cartesian coordinates in the image plane (alpha,beta),
        the distance D to the force center and the inclination angle i, 
        this function calculates the initial spherical coordinates (r, theta, phi) 
        and the initial components of the momentum (k_t, k_r, k_theta, k_phi)
        ===========================================================================
        '''
        # Transformation from (Alpha, Beta, D) to (r, theta, phi) 
        r = sqrt(alpha**2 + beta**2 + self.D**2)
        theta = arccos((beta*self.sin_iota + self.D*self.cos_iota)/r)
        phi = arctan(alpha/(self.D*self.sin_iota - beta*self.cos_iota))

        # Initial position of the photon in spherical coordinates 
        # (t=0, r, theta, phi)
        xin = [0., r, theta, phi]

        # Metric components
        g_tt, g_rr, g_thth, g_phph, g_tph = blackhole.metric(xin)

        # Given a frequency value w0=1, calculates the initial 
        # 4-momentum of the photon  
        #w0 =  freq    # Frequency of the photon at infinity
        k_th = sqrt(g_thth)*beta/self.D
        k_ph = - sqrt(g_phph)*alpha/(self.D)
        k_t = - sqrt(g_phph/(g_tph**2 - g_tt*g_phph)) + alpha*g_tph/(self.D*sqrt(g_phph))
        k_r =  sqrt(g_rr*(1 - (k_th**2)/g_thth - (k_ph**2)/g_phph ))
            
        # Initial 4-momentum in spherical coordinates (kt, kr, ktheta, kphi)
        k_in = [k_t, k_r, k_th, k_ph]

        return xin + k_in 
 


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')
