"""
===============================================================================
Image plane to generate the image from the traced photons trajectories in a 
curved spacetime
===============================================================================
@author: AshCat
===============================================================================
"""

from numpy import sqrt, sin, cos, arccos, arctan, linspace

class detector:
    def __init__(self, D, iota , x_s_side, y_s_side , n_pixels):
        '''
        =======================================================================
        Defines a screen with sides of size x_s_side and y_s_side, located 
        at a distance D and with an inclination iota. 
        The number of pixels in each direction is given by the variables
        x_pixels and y_pixels
        =======================================================================
        '''
        
        self.D = D 
        self.iota = iota  
        if n_pixels & 1:
            self.numPixels = n_pixels + 1
        else:
            self.numPixels = n_pixels 
        
        self.x_pixels = self.numPixels*x_s_side
        self.y_pixels = self.numPixels*y_s_side
        self.alphaRange = linspace(-x_s_side, x_s_side, self.x_pixels)
        self.betaRange = linspace(-y_s_side, y_s_side, self.y_pixels)
        print()
        print ("Size of the screen in Pixels: ", self.x_pixels, "X", self.y_pixels)
        print ("Total Number of Photons: ", self.x_pixels*self.y_pixels)
        print()


    def photon_coords(self, alpha, beta, freq=1): 
        '''
        ===========================================================================
        Given the initial cartesian coordinates in the image plane (alpha,beta),
        the distance D to the force center and the inclination angle i, 
        this function calculates the initial spherical coordinates (r, theta, phi) 
        and the initial components of the momentum (kt, kr, ktheta, kphi)
        ===========================================================================
        '''
        # Transformation from (Alpha, Beta, D) to (r, theta, phi) 
        r = sqrt(alpha**2 + beta**2 + self.D**2)
        theta = arccos((beta*sin(self.iota) + self.D*cos(self.iota))/r)
        phi = arctan(alpha/(self.D*sin(self.iota) - beta*cos(self.iota)))

        # Initial position of the photon in spherical coordinates 
        # (t=0, r, theta, phi)
        xin = [0., r, theta, phi]
                       
        # Given a frequency value w0=1, this calculates the initial 
        # 4-momentum of the photon  
        w0 =  freq    
        aux = alpha**2 + (-beta*cos(self.iota) + self.D*sin(self.iota))**2 
        kr =  (self.D/r)*w0   
        ktheta = (w0/sqrt(aux))*(-cos(self.iota) 
                    + (beta*sin(self.iota) + self.D*cos(self.iota))*(self.D/(r**2)))  
        kphi = - alpha*sin(self.iota)*w0/aux     
        kt = sqrt(kr**2 + r**2 * ktheta**2 + r**2*(sin(theta))**2 *kphi**2)
            
        # Initial 4-momentum in spherical coordinates  
        # (kt, kr, ktheta, kphi)
        kin = [kt, kr, ktheta, kphi]
        return xin, kin
 


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')
