"""
===============================================================================
Common functions for ray tracing in a curved spacetime
===============================================================================
@author: Alexis Larrañaga - 2023
===============================================================================
"""
from scipy.integrate import odeint
from numpy import linspace, cos, sqrt, zeros, where, roll, save
from numpy.random import randint
import matplotlib.pyplot as plt
import sys
import time


class Photon:
    def __init__(self, alpha, beta, freq=1.):
        '''
        Photon class
        ========================================================================
        This class stores the information of each photon
        Initial coordinates (alpha,beta) in the image plane  
        Initial coordinates in spherical coordinates (r, theta, phi)
        Final coordinates and momentum after integration
        ========================================================================
        ''' 
        
        # Pixel coordinates
        self.i = None
        self.j = None

        # Initial Cartesian Coordinates in the Image Plane
        self.iC = None
        
        # Stores the final values of coordinates and momentum 
        self.fP = None


def geodesic_integrate(p, blackhole, acc_structure, detector):
    '''
    Integrates the motion equations of the photon 
    '''
    final_lmbda = 1.5*detector.D
    lmbda = linspace(0, -final_lmbda, int(7*final_lmbda))
    sol = odeint(blackhole.geodesics, p.iC, lmbda)
    p.fP = [0.,0.,0.,0.,0.,0.,0.,0.]
    I_f = 0.
    zi = cos(sol[:,2])
    zi1 = roll(zi,-1)
    zi1[-1] = 0.
    indxs = where(zi*zi1 < 0)[0]
    for i in indxs: 
        if sol[i,1] < acc_structure.out_edge and sol[i,1] > acc_structure.in_edge:
            p.fP = sol[i]
            I_0 = acc_structure.intensity(p.fP[1])
            I_f = doppler_shift(p, I_0, blackhole)
            break
    return I_f 

def geo_integ_no_Doppler(p, blackhole, acc_structure, detector):
    '''
    Integrates the motion equations of the photon whitout 
    Doppler shift
    '''
    final_lmbda = 1.5*detector.D
    lmbda = linspace(0, -final_lmbda, int(7*final_lmbda))
    sol = odeint(blackhole.geodesics, p.iC, lmbda)
    
    p.fP = [0.,0.,0.,0.,0.,0.,0.,0.]
    zi = cos(sol[:,2])
    zi1 = roll(zi,-1)
    zi1[-1] = 0.
    indxs = where(zi*zi1 < 0)[0]
    for i in indxs: 
        if sol[i,1] < acc_structure.out_edge and sol[i,1] > acc_structure.in_edge:
            p.fP = sol[i]
            break
    return acc_structure.intensity(p.fP[1])

def shadow_integ(p, blackhole, detector):
    '''
    Integrates the motion equations of the photon to plot the shadow
    of the black hole
    '''
    final_lmbda = 1.5*detector.D
    lmbda = linspace(0, -final_lmbda, int(7*final_lmbda))
    sol = odeint(blackhole.geodesics, p.iC, lmbda)
    indxs = where(sol[:,1] < blackhole.EH + 1e-7)[0]
    if len(indxs) == 0:
        return 100
    else:
        return 0

def doppler_shift(p, I0, blackhole):
    '''
    ===========================================================================
    Applies the Doppler shift to the image data
    ===========================================================================
    Coordinates and momentum components of the photon at the accretion disk
    t = fP[0]
    r = fP[1]
    theta = fP[2]
    phi = fP[3]
    k_t = fP[4]
    k_r = fP[5]
    k_th = fP[6]
    k_phi = fP[7]
    ===========================================================================
    '''
    # Metric components
    g_tt, _, _, g_phph, g_tph = blackhole.metric(p.fP[:4])
    Omega = blackhole.Omega(p.fP[1])
    g = sqrt(- g_tt - 2*g_tph*Omega - g_phph*Omega**2)/(1 + p.fP[7]*Omega/p.fP[4])
    return I0 * g**3

def integrate_for_H(p, blackhole, acc_structure, detector):
    '''
    Integrates the motion equations of the photon to verify
    the Hamiltonian constraint 
    '''
    final_lmbda = 1.5*detector.D
    lmbda = linspace(0, -final_lmbda, int(7*final_lmbda))
    sol = odeint(blackhole.geodesics, p.iC, lmbda)
    solution = sol
    # Find the point where the photon crosses the accretion structure
    zi = cos(sol[:,2])
    zi1 = roll(zi,-1)
    zi1[-1] = 0.
    indxs = where(zi*zi1 < 0)[0]
    for i in indxs: 
        if sol[i,1] < acc_structure.out_edge and sol[i,1] > acc_structure.in_edge:
            solution = sol[:i]
            break
    # Find the point where the photon crosses the event horizon
    indxsEH = where(sol[:,1] < blackhole.EH + 0.1)[0]
    for i in indxsEH:
        solution = sol[:i]
    # Calculate the Hamiltonian
    H = Hamiltonian(solution, blackhole)
    print('Hamiltonian constraint verified: |H_max - H_0 | = ', abs(H.max() - H[0]))
    return H

def Hamiltonian(sol, blackhole):
    H = zeros(len(sol))
    for i in range(len(sol)):
        x = sol[i,0:4]
        p = sol[i,4:]
        gtt, grr, gthth, gphph, gtph = blackhole.inverse_metric(x)
        H[i] = 0.5*(gtt*p[0]*p[0] + grr*p[1]*p[1] + gthth*p[2]*p[2] + gphph*p[3]*p[3] + 2*gtph*p[0]*p[3])
    return H


class Image:
    '''
    ===========================================================================
    Image class
    Creates the photon list and generates the image
    ===========================================================================
    '''
    def __init__(self, blackhole, acc_structure, detector):
        self.blackhole = blackhole
        self.acc_structure = acc_structure
        self.detector = detector

    def create_photons(self):
        '''
        Creates the photon array
        ========================================================================
        This function creates the photon array with the initial coordinates
        (alpha, beta) in the image plane. The photons are stored in a list
        of Photon objects. The i and j coordinates are also stored in the
        Photon object, which correspond to the pixel coordinates in the image.
        ========================================================================
        '''
        print('Creating photons ...')
        self.photon_list = []
        i=0
        for a in self.detector.alphaRange:
            j = 0
            for b in self.detector.betaRange:
                p = Photon(alpha=a, beta=b)
                p.iC = self.detector.photon_coords(self.blackhole, a, b)
                p.i, p.j = i, j
                self.photon_list.append(p)
                j += 1
            i += 1
    
    def create_image(self):
        '''
        Creates the image data 
        '''
        self.image_data = zeros([self.detector.x_pixels, self.detector.y_pixels])
        photon=1
        print('Integrating trajectories ...')
        start_time = time.time()
        for p in self.photon_list:
            self.image_data[p.i, p.j] = geodesic_integrate(p, self.blackhole, self.acc_structure, self.detector)
            sys.stdout.write("\rPhoton # %d" %photon)
            sys.stdout.flush()
            photon +=1
        total_time= time.time() - start_time
        print("\n\n--- Total time of integration : %s seconds ---" % total_time)
        print("\n--- Time of integration : %s seconds/photon ---\n" % (total_time/len(self.photon_list)))
        
    def create_image_no_Doppler(self):
        '''
        Creates the image data with no Doppler shift 
        '''
        self.image_data = zeros([self.detector.x_pixels, self.detector.y_pixels])
        photon=1
        print('Integrating trajectories ...')
        start_time = time.time()
        for p in self.photon_list:
            self.image_data[p.i, p.j] = geo_integ_no_Doppler(p, self.blackhole, self.acc_structure, self.detector)
            sys.stdout.write("\rPhoton # %d" %photon)
            sys.stdout.flush()
            photon +=1
        total_time= time.time() - start_time
        print("\n\n--- Total time of integration : %s seconds ---" % total_time)
        print("\n--- Time of integration : %s seconds/photon ---\n" % (total_time/len(self.photon_list)))

    def create_shadow(self):
        '''
        Creates the image data 
        '''
        self.image_data = zeros([self.detector.x_pixels, self.detector.y_pixels])
        photon=1
        print('Integrating trajectories ...')
        start_time = time.time()
        for p in self.photon_list:
            self.image_data[p.i, p.j] = shadow_integ(p, self.blackhole, self.detector)
            sys.stdout.write("\rPhoton # %d" %photon)
            sys.stdout.flush()
            photon +=1
        total_time= time.time() - start_time
        print("\n\nEH radius %s  ---" % self.blackhole.EH)
        print("\n\n--- Total time of integration : %s seconds ---" % total_time)
        print("\n--- Time of integration : %s seconds/photon ---\n" % (total_time/len(self.photon_list)))

    def save_data(self, filename):
        save(filename+'.npy', self.image_data)

    def plot(self, savefig=False, filename=None, cmap='afmhot'):
        '''
        Plots the image of the Black Hole 
        '''
        self.image_data = self.image_data/self.image_data.max()
        ax = plt.figure().add_subplot(aspect='equal')
        ax.imshow(self.image_data.T, cmap = cmap , origin='lower')
        #ax.set_xlabel(r'$\alpha$')
        #ax.set_ylabel(r'$\beta$', rotation=0)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        
        if savefig:
            plt.savefig('images/'+filename+'.png')
        plt.show()
    
    def plot_shadow(self, savefig=False, filename=None, cmap='gray'):
        '''
        Plots the image of the BH 
        '''
        self.image_data = self.image_data/self.image_data.max()
        ax = plt.figure().add_subplot(aspect='equal')
        ax.imshow(self.image_data.T, cmap = cmap , origin='lower')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$', rotation=0)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        #plt.axhline(40, color='black', linewidth=0.5)
        #plt.axvline(70, color='black', linewidth=0.5)
        plt.grid(which='both')
        if savefig:
            plt.savefig('images/'+filename+'.png')
        plt.show()
    
    def plot_contours(self, savefig=False, filename=None, cmap='gray'):
        '''
        Plots the contours in the image of the Black Hole 
        '''
        ax = plt.figure().add_subplot(aspect='equal')
        ax.contour(self.image_data.T, cmap=cmap)
        ax.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$', rotation=0)
        ax.grid(alpha=0.25)
        if savefig:
            plt.savefig('images/'+filename+'.png')
        plt.show()
    
    def verify_Hamiltonian(self, n=10):
        '''
        Verifies the Hamiltonian constrain for a specific number of photons 
        randomly chosen
        '''
        # Check if the inverse metric is defined
        if not hasattr(self.blackhole, 'inverse_metric'):
            print('The inverse metric is not defined for this black hole.')
            print('Please, check the black hole definition.')
            return
        
        photon=0
        print('Integrating trajectories ...\n')
        ax = plt.figure(figsize=(10,7)).add_subplot()
        while photon < n:
            i = randint(1,len(self.photon_list))
            p = self.photon_list[i]
            H = integrate_for_H(p, self.blackhole, self.acc_structure, self.detector)
            ax.plot(H, label='Photon # %d' %i)
            photon +=1
        
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$H$', rotation=0)
        ax.set_ylim(-2,2)
        ax.grid(alpha=0.25)
        ax.legend()
        plt.show()
        print('\n')
        


###############################################################################

if __name__ == "__main__":
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')