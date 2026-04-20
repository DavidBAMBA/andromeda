"""
===============================================================================
Common functions for ray tracing in a curved spacetime
===============================================================================
@author: Alexis Larrañaga - 2023
===============================================================================
"""
from numpy import linspace, cos, sqrt, zeros, where, roll, save
from numpy.random import randint
import matplotlib.pyplot as plt
import sys
import time

from scr.common.integrator import integrate, make_events

# Default integrator configuration. Override via set_integrator(...).
# "auto" picks the numba RK45 when the BH exposes _rhs_nb, else DOP853 (scipy).
_INTEGRATOR_METHOD = "auto"
_INTEGRATOR_RTOL = 1e-9
_INTEGRATOR_ATOL = 1e-11

# Ray-tracing bounds. None → use Kerr defaults (1.1*D, 1.5*D).
# Override via set_ray_bounds() for spacetimes with cosmological horizons.
_R_ESCAPE = None
_FINAL_LMBDA = None


def set_integrator(method="DOP853", rtol=1e-9, atol=1e-11):
    """Change the default integrator used by geodesic_* functions."""
    global _INTEGRATOR_METHOD, _INTEGRATOR_RTOL, _INTEGRATOR_ATOL
    _INTEGRATOR_METHOD = method
    _INTEGRATOR_RTOL = rtol
    _INTEGRATOR_ATOL = atol


def set_ray_bounds(r_escape=None, final_lmbda=None):
    """Override the ray-tracing bounds.

    r_escape : radius at which a photon is declared escaped. Must be
               inside any cosmological horizon.
    final_lmbda : max affine-parameter length per trajectory. Should be
                  large enough for photons to complete their paths.
    Pass None to restore per-detector defaults (1.1*D, 1.5*D).
    """
    global _R_ESCAPE, _FINAL_LMBDA
    _R_ESCAPE = r_escape
    _FINAL_LMBDA = final_lmbda


def _ray_bounds(detector):
    """Resolve effective (r_escape, final_lmbda) using overrides or defaults."""
    r_esc = _R_ESCAPE if _R_ESCAPE is not None else 1.1 * detector.D
    fl    = _FINAL_LMBDA if _FINAL_LMBDA is not None else 1.5 * detector.D
    return r_esc, fl


def _rhs(blackhole):
    """Wrap odeint-style geodesics(q, lmbda) into solve_ivp-style f(lmbda, q)."""
    geo = blackhole.geodesics
    def f(lmbda, q):
        return geo(q, lmbda)
    return f


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


def _first_disk_hit(res, acc_structure):
    '''Scan recorded equator crossings for the first one inside the annulus.

    The disk event is non-terminal, so every equator crossing is recorded in
    res.y_events[1]. Entries appear in integration order (backward from
    observer), which matches the physical emission order: the first in-annulus
    crossing is the actual disk emission point.
    '''
    y_ev = res.y_events[1] if len(res.y_events) > 1 else None
    if y_ev is None or len(y_ev) == 0:
        return None
    in_edge = acc_structure.in_edge
    out_edge = acc_structure.out_edge
    for yev in y_ev:
        r_hit = float(yev[1])
        if in_edge <= r_hit <= out_edge:
            return list(yev)
    return None


def geodesic_integrate(p, blackhole, acc_structure, detector):
    '''
    Integrates the motion equations of the photon, stopping on the first
    horizon/escape event. Every equator crossing is recorded; the caller
    picks the first in-annulus crossing.
    '''
    r_escape, final_lmbda = _ray_bounds(detector)
    events = make_events(blackhole, acc_structure=acc_structure,
                         r_escape=r_escape)
    res = integrate(_rhs(blackhole), p.iC, (0.0, -final_lmbda),
                    method=_INTEGRATOR_METHOD,
                    events=events, rtol=_INTEGRATOR_RTOL,
                    atol=_INTEGRATOR_ATOL)

    p.fP = [0., 0., 0., 0., 0., 0., 0., 0.]
    hit = _first_disk_hit(res, acc_structure)
    if hit is None:
        return 0.
    p.fP = hit
    I_0 = acc_structure.intensity(p.fP[1])
    return doppler_shift(p, I_0, blackhole)

def geo_integ_no_Doppler(p, blackhole, acc_structure, detector):
    '''
    Same as geodesic_integrate but without Doppler shift.
    '''
    r_escape, final_lmbda = _ray_bounds(detector)
    events = make_events(blackhole, acc_structure=acc_structure,
                         r_escape=r_escape)
    res = integrate(_rhs(blackhole), p.iC, (0.0, -final_lmbda),
                    method=_INTEGRATOR_METHOD,
                    events=events, rtol=_INTEGRATOR_RTOL,
                    atol=_INTEGRATOR_ATOL)

    p.fP = [0., 0., 0., 0., 0., 0., 0., 0.]
    hit = _first_disk_hit(res, acc_structure)
    if hit is None:
        return 0.
    p.fP = hit
    return acc_structure.intensity(p.fP[1])

def shadow_integ(p, blackhole, detector):
    '''
    Integrates to determine whether the photon falls into the BH shadow.
    Returns 0 if the photon crosses the horizon, 100 otherwise.
    '''
    r_escape, final_lmbda = _ray_bounds(detector)
    events = make_events(blackhole, acc_structure=None, r_escape=r_escape)
    res = integrate(_rhs(blackhole), p.iC, (0.0, -final_lmbda),
                    method=_INTEGRATOR_METHOD,
                    events=events, rtol=_INTEGRATOR_RTOL,
                    atol=_INTEGRATOR_ATOL)
    return 0 if res.status == "horizon" else 100

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
    Integrates the motion equations of the photon to verify the
    Hamiltonian constraint along the accepted-step trajectory.
    '''
    r_escape, final_lmbda = _ray_bounds(detector)
    events = make_events(blackhole, acc_structure=acc_structure,
                         r_escape=r_escape)
    res = integrate(_rhs(blackhole), p.iC, (0.0, -final_lmbda),
                    method=_INTEGRATOR_METHOD,
                    events=events, rtol=_INTEGRATOR_RTOL,
                    atol=_INTEGRATOR_ATOL, dense=True)
    H = Hamiltonian(res.y, blackhole)
    print('Hamiltonian constraint verified: |H_max - H_0 | = ',
          abs(H.max() - H[0]))
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
    
    def _trace(self, mode, n_workers, chunksize):
        '''Dispatch to parallel or serial tracer depending on n_workers.'''
        from scr.common.parallel import trace_parallel, trace_serial
        tasks = [(p.i, p.j, p.iC) for p in self.photon_list]
        r_esc, fl = _ray_bounds(self.detector)
        print('Integrating trajectories ...')
        if n_workers == 1:
            img, stats = trace_serial(
                tasks, self.blackhole, self.acc_structure, self.detector,
                method=_INTEGRATOR_METHOD, rtol=_INTEGRATOR_RTOL,
                atol=_INTEGRATOR_ATOL, mode=mode,
                r_escape=r_esc, final_lmbda=fl)
        else:
            img, stats = trace_parallel(
                tasks, self.blackhole, self.acc_structure, self.detector,
                n_workers=n_workers, chunksize=chunksize,
                method=_INTEGRATOR_METHOD, rtol=_INTEGRATOR_RTOL,
                atol=_INTEGRATOR_ATOL, mode=mode,
                r_escape=r_esc, final_lmbda=fl)
        self.image_data = img
        self._stats = stats
        n_ph = len(self.photon_list)
        wall = stats["wall_total"]
        print(f"\n--- Total time of integration : {wall:.3f} seconds ---")
        print(f"--- Time of integration : {wall/n_ph*1000:.3f} ms/photon ---")
        print(f"--- Workers: {stats['n_workers']}, chunksize: {stats['chunksize']} ---\n")

    def create_image(self, n_workers=None, chunksize=None):
        '''Create image data in parallel (Doppler shift applied).'''
        self._trace(mode="doppler", n_workers=n_workers, chunksize=chunksize)

    def create_image_no_Doppler(self, n_workers=None, chunksize=None):
        '''Create image data in parallel without Doppler shift.'''
        self._trace(mode="no_doppler", n_workers=n_workers, chunksize=chunksize)

    def create_shadow(self, n_workers=None, chunksize=None):
        '''Create shadow image in parallel.'''
        print(f"EH radius: {self.blackhole.EH}")
        self._trace(mode="shadow", n_workers=n_workers, chunksize=chunksize)
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