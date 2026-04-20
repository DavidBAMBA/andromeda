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
        Builds a NULL 4-momentum at the screen pixel (alpha, beta) for any
        stationary, axisymmetric metric (Kerr, Kerr-MOG, Kerr-MOG+quintessence).

        Angular components k_θ, k_φ come from pixel geometry. k_r is taken from
        the spatial "unit 3-momentum" condition. k_t is then solved so that
        g^μν k_μ k_ν = 0 exactly (quadratic in k_t), picking the past-pointing
        root (k_t < 0) appropriate for backward ray tracing.
        '''
        # (alpha, beta, D) → (r, theta, phi) in BL coordinates
        r = sqrt(alpha**2 + beta**2 + self.D**2)
        theta = arccos((beta*self.sin_iota + self.D*self.cos_iota)/r)
        phi = arctan(alpha/(self.D*self.sin_iota - beta*self.cos_iota))

        xin = [0., r, theta, phi]

        # Metric components
        g_tt, g_rr, g_thth, g_phph, g_tph = blackhole.metric(xin)

        # Angular components from pixel geometry (same as original)
        k_th = sqrt(g_thth)*beta/self.D
        k_ph = -sqrt(g_phph)*alpha/self.D

        # Radial component: spatial-unit-norm convention
        k_r = sqrt(g_rr*(1 - (k_th**2)/g_thth - (k_ph**2)/g_phph))

        # Inverse metric for the null-condition quadratic
        if hasattr(blackhole, 'inverse_metric'):
            gtt_, grr_, gthth_, gphph_, gtph_ = blackhole.inverse_metric(xin)
        else:
            det_tp = g_tt*g_phph - g_tph**2
            gtt_   =  g_phph/det_tp
            gphph_ =  g_tt/det_tp
            gtph_  = -g_tph/det_tp
            grr_   = 1.0/g_rr
            gthth_ = 1.0/g_thth

        # Solve null condition: g^tt k_t² + 2 g^tφ k_φ k_t + spatial_part = 0
        a_c = gtt_
        b_c = 2.0 * gtph_ * k_ph
        c_c = grr_*k_r*k_r + gthth_*k_th*k_th + gphph_*k_ph*k_ph
        disc = b_c*b_c - 4.0*a_c*c_c
        if disc < 0:
            disc = 0.0
        sq = sqrt(disc)
        # Pick past-pointing root (k_t < 0). With a_c < 0, the "+ sq" branch
        # is typically negative; fall back to the other if needed.
        k_t = (-b_c + sq) / (2.0 * a_c)
        if k_t > 0:
            k_t = (-b_c - sq) / (2.0 * a_c)

        k_in = [k_t, k_r, k_th, k_ph]
        return xin + k_in
 


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')
