"""
===============================================================================
Common functions for ray tracing in a curved spacetime
===============================================================================
@author: Eduard Larrañga - 2023
===============================================================================
"""
from scipy.integrate import odeint
from numpy import linspace, cos, zeros, argmax
import matplotlib.pyplot as plt
import sys


def initCond(x, k, blackhole):
    '''
    Given the initial conditions (x,k)
    this function returns the list
    [t, r, theta, phi, k_t, k_r, k_theta, k_phi] 
    with the initial conditions needed to solve 
    the geodesic equations 
    (with the covariant components of the momentum vector)    
    # Coordinates
    t = x[0]
    r = x[1]
    theta = x[2]
    phi = x[3]
    kt = k[0]
    kr = k[1]
    ktheta = k[2]
    kphi = k[3]
    '''
    # Metric components
    g_tt, g_rr, g_thth, g_phph, g_tph= blackhole.metric(x)
    
    # Lower k-indices
    k_t = g_tt*k[0] + g_tph*k[3]
    k_r = g_rr*k[1]
    k_th = g_thth*k[2]
    k_phi = g_phph*k[3] + g_tph*k[0]
    
    return [x[0], x[1], x[2], x[3], k_t, k_r, k_th, k_phi]



class Photon:
    def __init__(self, alpha, beta, freq=1.):
        '''
        Given the initial coordinates in the image plane (X,Y), the distance D 
        to the force center and inclination angle i, this calculates the 
        initial coordinates in spherical coordinates (r, theta, phi)
        ''' 
        # Initial Cartesian Coordinates in the Image Plane
        self.alpha = alpha
        self.beta = beta
        
        # Pixel coordinates
        self.i = None
        self.j = None

        #Initial position and momentum in spherical coordinates 
        self.xin = None 
        self.kin = None 
        
        # Stores the final values of coordinates and momentum 
        self.fP = None
    
    def initial_conditions(self, blackhole):
        '''
        Calcualtes and stores the initial conditions of coordinates 
        and momentum to solve the geodesic equations.
        '''
        self.iC = initCond(self.xin, self.kin, blackhole)


def geo_integ(p, blackhole, acc_structure, detector):
    '''
    Integrates the motion equations of the photon 
    '''
    in_edge, out_edge = acc_structure.in_edge, acc_structure.out_edge
    final_lmbda = 2*detector.D
    lmbda = linspace(0, -final_lmbda,8*final_lmbda)
    sol = odeint(blackhole.geodesics, p.iC, lmbda)
    p.fP = [0,0,0,0,0,0,0,0]

    #Put True in the index that met the condition
    eh_condition   = (sol[:,1] < (blackhole.EH + 1e-5))

    edge_condition = (abs(sol[:, 1]*cos(sol[:, 2])) < 1e-1) & \
                     ( sol[:, 1] > in_edge) & (sol[:, 1] < out_edge)

    # Find the first index where the photon intersects the accretion structure.
    indx_eh   = argmax(eh_condition)
    indx_edge = argmax(edge_condition)


    if edge_condition[indx_edge]:  
        p.fP = sol[indx_edge]
    elif eh_condition[indx_eh]:  
        p.fP = [0, 0, 0, 0, 0, 0, 0, 0]


    
class Image:
    '''
    Image class
    Creates the photon list and generates the image
    '''
    def __init__(self,start_alpha,start_beta, end_alpha, end_beta):

        self.start_alpha = start_alpha
        self.start_beta  = start_beta
        self.end_alpha   = end_alpha
        self.end_beta    = end_beta

    def create_photons(self, blackhole, detector):
        '''
        Creates the photon list
        '''
        print('Creating photons ...')
        self.detector = detector
        self.photon_list = []
        
        for i in range(self.start_alpha, self.end_alpha +1):

            if i == self.start_alpha:
                beta_start = self.start_beta
            else:
                beta_start = 0

            if i == self.end_alpha:
                beta_end = self.end_beta
            else:
                beta_end = len(self.detector.betaRange)

            for j in range(beta_start, beta_end):
                a = self.detector.alphaRange[i]
                b = self.detector.betaRange[j]

                p = Photon(alpha=a, beta=b)
                p.xin, p.kin = self.detector.photon_coords(a, b) 
                p.i, p.j = i, j
                p.initial_conditions(blackhole)
                self.photon_list.append(p)

    
    def create_image(self, blackhole, acc_structure):
        '''
        Creates the image data 
        '''
        self.image_data = zeros([self.detector.x_pixels, self.detector.y_pixels])
        photon=1
        for p in self.photon_list:
            geo_integ(p, blackhole, acc_structure, self.detector)
            self.image_data[p.i, p.j] = acc_structure.energy_flux(p.fP[1])
            sys.stdout.write("\rPhoton in thread 0 # %d" %photon)
            sys.stdout.flush()
            photon +=1

    def plot(self, savefig=False, filename=None, cmap='inferno'):
        '''
        Plots the image of the BH 
        '''
        ax = plt.figure().add_subplot(aspect='equal')
        ax.imshow(self.image_data.T, cmap = cmap , origin='lower')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        if savefig:
            plt.savefig('images/'+filename+'.png')
        plt.show()
    

    def plotContours(self, savefig=False, filename=None, cmap='gray'):
        '''
        Contour plots in the image of the BH 
        '''
        ax = plt.figure().add_subplot(aspect='equal')
        ax.contour(self.image_data.T, cmap = cmap)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        if savefig:
            plt.savefig('images/'+filename+'.png')
        plt.show()




###############################################################################

if __name__ == "__main__":
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')