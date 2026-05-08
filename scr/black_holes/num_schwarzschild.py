"""
===============================================================================
Schwarzschild metric

ds^2 = -(1-2M/r)dt^2 + dr^2 /(1-2M/r) + r^2 dtheta^2 + r^2 sin^2 (theta) dphi^2

===============================================================================
- Event horizon at r=2M
- ISCO at r = 6M

@author: Alexis Larrañaga - 2023
===============================================================================
"""

import numpy as np
from numpy import sin, cos, loadtxt, linspace, asarray
from scipy.interpolate import interp1d
from numba import njit


# Load numerical metric tables once at module import (used by numba hot path).
_data_N    = loadtxt('scr/black_holes/numerical_data/schwarzschild_data/N.txt')
_data_dNdr = loadtxt('scr/black_holes/numerical_data/schwarzschild_data/derN.txt')
_R_TBL  = np.ascontiguousarray(_data_N[:, 0], dtype=np.float64)
_N_TBL  = np.ascontiguousarray(_data_N[:, 1], dtype=np.float64)
_DR_TBL = np.ascontiguousarray(_data_dNdr[:, 0], dtype=np.float64)
_DN_TBL = np.ascontiguousarray(_data_dNdr[:, 1], dtype=np.float64)


@njit(cache=True)
def _geodesics_nb(q):
    '''Numerical-Schwarzschild geodesic RHS (array form).'''
    r = q[1]
    sin_theta = sin(q[2])
    sin_theta2 = sin_theta * sin_theta
    r2 = r * r
    r3 = r2 * r

    N_r   = np.interp(r, _R_TBL, _N_TBL)
    dN_r  = np.interp(r, _DR_TBL, _DN_TBL)

    gtt   = -1.0 / N_r
    grr   =  N_r
    gthth =  1.0 / r2
    gphph =  1.0 / (r2 * sin_theta2)

    drgtt   =  dN_r / (N_r * N_r)
    drgrr   =  dN_r
    drgthth = -2.0 / r3
    drgphph = -2.0 / (r3 * sin_theta2)

    out = np.empty(8)
    out[0] = gtt * q[4]
    out[1] = grr * q[5]
    out[2] = gthth * q[6]
    out[3] = gphph * q[7]
    out[4] = 0.0
    out[5] = -(drgtt * q[4]**2) * 0.5 \
             -(drgrr * q[5]**2) * 0.5 \
             -(drgthth * q[6]**2) * 0.5 \
             -(drgphph * q[7]**2) * 0.5
    out[6] = (cos(q[2]) / (sin_theta2 * sin_theta)) * (q[7] / r) ** 2
    out[7] = 0.0
    return out


@njit(cache=True)
def _metric_nb(x):
    r = x[1]
    N_r = np.interp(r, _R_TBL, _N_TBL)
    g_tt = -N_r
    g_rr = 1.0 / N_r
    g_thth = r * r
    g_phph = (r * sin(x[2])) ** 2
    return g_tt, g_rr, g_thth, g_phph, 0.0


class BlackHole:
    '''
    Definition of the Black Hole described by Schwarzschild metric
    '''
    def __init__(self):
        self.N = interp1d(_R_TBL, _N_TBL)
        self.dNdr = interp1d(_DR_TBL, _DN_TBL)
        self.a = 0.
        self.EH = 2
        self.ISCOco = 6
        self.ISCOcounter = 6
        # Numba hot-path hooks
        self._rhs_nb = _geodesics_nb
        self._metric_nb = _metric_nb
        self._omega_nb = None  # no Doppler shift for this BH

    def metric(self,x):
        '''
        This procedure contains the Schwarzschild metric non-zero components in 
        spherical coordinates
        ===========================================================================
        Coordinates 
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        ===========================================================================
        '''
        # Metric components
        g_tt = - self.N(x[1])
        g_rr = 1/self.N(x[1])
        g_thth = x[1]**2
        g_phph = (x[1]*sin(x[2]))**2
        g_tph = 0.
        
        return [g_tt, g_rr, g_thth, g_phph, g_tph]
    
    def inverse_metric(self,x):
        '''
        This procedure contains the Schwarzschild inverse metric non-zero 
        components in spherical coordinates
        ===========================================================================
        Coordinates 
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        ===========================================================================
        '''
        # Metric components
        gtt = - 1/self.N(x[1])
        grr = self.N(x[1])
        gthth = 1/x[1]**2
        gphph = 1/(x[1]*sin(x[2]))**2
        gtph = 0.
        
        return [gtt, grr, gthth, gphph, gtph]
    
    def dr_inverse_metric(self,x):
        '''
        This procedure returns the derivative of the Schwarzschild inverse metric 
        w.r.t the coordinate r
        ===========================================================================
        Coordinates 
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        ===========================================================================
        '''
        # Derivative of the metric components
        drgtt =  self.dNdr(x[1])/(self.N(x[1])**2)
        drgrr = self.dNdr(x[1])
        drgthth = -2/x[1]**3
        drgphph = -2/(x[1]**3*sin(x[2])**2)
        drgtph = 0.
        
        return [drgtt, drgrr, drgthth, drgphph, drgtph]


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
        E = - k_t
        L = k_phi
        ===========================================================================
        '''
        # Metric and its numerical derivative
        gtt, grr, gthth, gphph, gtph = self.inverse_metric(q[0:4])
        drgtt, drgrr, drgthth, drgphph, drgtph = self.dr_inverse_metric(q[0:4])
        
        # Geodesics differential equations 
        dtdlmbda = gtt*q[4]
        drdlmbda = grr*q[5]
        dthdlmbda = gthth*q[6]
        dphidlmbda = gphph*q[7]
        
        dk_tdlmbda = 0.
        dk_rdlmbda = - (drgtt*q[4]**2)/2 - (drgrr*q[5]**2)/2 \
                     - (drgthth*q[6]**2)/2 - (drgphph*q[7]**2)/2
        dk_thdlmbda = (cos(q[2])/sin(q[2])**3)*(q[7]/q[1])**2
        dk_phidlmbda = 0.
        
        return [dtdlmbda, drdlmbda, dthdlmbda, dphidlmbda, 
                dk_tdlmbda, dk_rdlmbda, dk_thdlmbda, dk_phidlmbda]





###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')


