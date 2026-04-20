"""
===============================================================================
Schwarzschild metric

ds^2 = -(1-2M/r)dt^2 + dr^2 /(1-2M/r) + r^2 dtheta^2 + r^2 sin^2 (theta) dphi^2

===============================================================================
- Event horizon at r=2M
- ISCO at r = 6M (co- and counter-rotation)
- We use units where G = c = 1, and M = 1
===============================================================================
@author: Alexis Larrañaga - 2023
===============================================================================
"""

from math import sin, cos
import numpy as np
from numba import njit


@njit(cache=True)
def _geodesics_nb(q):
    '''Schwarzschild geodesic RHS. q is float64[8]; returns float64[8].'''
    sin_theta = sin(q[2])
    sin_theta2 = sin_theta * sin_theta
    r = q[1]
    r2 = r * r
    r3 = r2 * r
    f = 1.0 - 2.0 / r

    out = np.empty(8)
    out[0] = -q[4] / f
    out[1] = f * q[5]
    out[2] = q[6] / r2
    out[3] = q[7] / (r2 * sin_theta2)
    out[4] = 0.0
    out[5] = (-(q[4] / (r - 2.0)) ** 2
              - (q[5] / r) ** 2
              + q[6] ** 2 / r3
              + q[7] ** 2 / (r3 * sin_theta2))
    out[6] = (cos(q[2]) / (sin_theta2 * sin_theta)) * (q[7] / r) ** 2
    out[7] = 0.0
    return out


@njit(cache=True)
def _metric_nb(x):
    '''Schwarzschild metric components (g_tt, g_rr, g_thth, g_phph, g_tph).'''
    r = x[1]
    g_tt = -(1.0 - 2.0 / r)
    return g_tt, -1.0 / g_tt, r * r, (r * sin(x[2])) ** 2, 0.0


@njit(cache=True)
def _omega_nb(r):
    return 1.0 / (r ** 1.5)


class BlackHole:
    '''
    Definition of the Black Hole described by Schwarzschild metric
    '''
    def __init__(self):
        self.a = 0
        self.EH = 2
        self.ISCOco = 6
        self.ISCOcounter = 6
        # Numba hot-path hooks (detected by parallel/common dispatchers)
        self._rhs_nb = _geodesics_nb
        self._metric_nb = _metric_nb
        self._omega_nb = _omega_nb

    def Omega(self, r, corotating=True):
            '''
            Returns the angular velocity of a particle at radius r
            '''
            return 1/(r**(3/2))
    
    def metric(self,x):
        '''
        =======================================================================
        This procedure contains the Schwarzschild metric non-zero components 
        in spherical coordinates
        =======================================================================
        Coordinates 
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        =======================================================================
        '''
        # Metric components
        g_tt = -(1 - 2/x[1])
        g_rr = -1/g_tt
        g_thth = x[1]**2
        g_phph = (x[1]*sin(x[2]))**2
        g_tph = 0.
        
        return [g_tt, g_rr, g_thth, g_phph, g_tph]
    
    def inverse_metric(self,x):
        '''
        =======================================================================
        This procedure contains the inverse Schwarzschild metric non-zero 
        components in spherical coordinates
        =======================================================================
        Coordinates 
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        =======================================================================
        '''
        # Metric components
        gtt = -1/(1 - 2/x[1])
        grr = -1/gtt
        gthth = 1/x[1]**2
        gphph = 1/(x[1]*sin(x[2]))**2
        gtph = 0.
        
        return [gtt, grr, gthth, gphph, gtph]

    def geodesics(self, q, lmbda):
        '''
        This function contains the geodesic equations in Hamiltonian form for 
        the Schwarzschild metric
        ===========================================================================
        Coordinates and momentum components
        t = q[0]
        r = q[1]
        theta = q[2]
        phi = q[3]
        k_t = q[4]
        k_r = q[5]
        k_th = q[6]
        k_phi = q[7]
        ===========================================================================
        Conserved Quantities
        E = - k_t = -q[4]
        L = k_phi = q[7]
        ===========================================================================
        '''
        # Auxiliar functions
        sin_theta = sin(q[2])
        f = 1 - 2/q[1]
        # Geodesics differential equations 
        dtdlmbda =  -q[4]/f  #q[4]*q[1]**2/(q[1]**2 - 2*q[1])
        drdlmbda =  f*q[5]         #(1 - 2/q[1])*q[5]
        dthdlmbda = q[6]/q[1]**2
        dphidlmbda = q[7]/((q[1]*sin_theta)**2)
        
        dk_tdlmbda = 0.
        dk_rdlmbda = -(q[4]/(q[1]-2))**2 - (q[5]/q[1])**2 + q[6]**2/q[1]**3 \
                     + q[7]**2/((q[1]**3)*sin_theta**2)
        dk_thdlmbda = (cos(q[2])/sin_theta**3)*(q[7]/q[1])**2
        dk_phidlmbda = 0.
        
        return [dtdlmbda, drdlmbda, dthdlmbda, dphidlmbda, 
                dk_tdlmbda, dk_rdlmbda, dk_thdlmbda, dk_phidlmbda]





###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')