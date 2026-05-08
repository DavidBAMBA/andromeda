"""
===============================================================================
Scalar Hair Black Hole 

ds^2 = g_tt dt^2 + g_rr dr^2  + r^2 dtheta^2 + r^2 sin^2 (theta) dphi^2

===============================================================================
- Event horizon at r = 1

@author: Eduard Larrañaga - 2023
===============================================================================
"""

import numpy as np
from numpy import sin, cos, loadtxt, linspace, asarray
from scipy.interpolate import interp1d
from numba import njit


# Load numerical metric table once at module import (9 columns).
_DATA = loadtxt('scr/black_holes/numerical_data/scalarBH/phi1=5.0/metricpp0=1.6.txt')
_R_TBL     = np.ascontiguousarray(_DATA[:, 0], dtype=np.float64)
_GTT_COV   = np.ascontiguousarray(_DATA[:, 1], dtype=np.float64)
_GRR_COV   = np.ascontiguousarray(_DATA[:, 2], dtype=np.float64)
_GTT_INV   = np.ascontiguousarray(_DATA[:, 3], dtype=np.float64)
_GRR_INV   = np.ascontiguousarray(_DATA[:, 4], dtype=np.float64)
_DRGTT     = np.ascontiguousarray(_DATA[:, 5], dtype=np.float64)
_DRGRR     = np.ascontiguousarray(_DATA[:, 6], dtype=np.float64)


@njit(cache=True)
def _geodesics_nb(q):
    '''Scalar-hair BH geodesic RHS (array form).'''
    r = q[1]
    sin_theta = sin(q[2])
    sin_theta2 = sin_theta * sin_theta
    r2 = r * r
    r3 = r2 * r

    gtt   = np.interp(r, _R_TBL, _GTT_INV)
    grr   = np.interp(r, _R_TBL, _GRR_INV)
    drgtt = np.interp(r, _R_TBL, _DRGTT)
    drgrr = np.interp(r, _R_TBL, _DRGRR)

    out = np.empty(8)
    out[0] = gtt * q[4]
    out[1] = grr * q[5]
    out[2] = (1.0 / r2) * q[6]
    out[3] = (1.0 / (r2 * sin_theta2)) * q[7]
    out[4] = 0.0
    out[5] = -(drgtt * q[4]**2) * 0.5 \
             -(drgrr * q[5]**2) * 0.5 \
             -((-2.0 / r3) * q[6]**2) * 0.5 \
             -((-2.0 / (r3 * sin_theta2)) * q[7]**2) * 0.5
    out[6] = (cos(q[2]) / (sin_theta2 * sin_theta)) * (q[7] / r) ** 2
    out[7] = 0.0
    return out


@njit(cache=True)
def _metric_nb(x):
    r = x[1]
    g_tt = np.interp(r, _R_TBL, _GTT_COV)
    g_rr = np.interp(r, _R_TBL, _GRR_COV)
    g_thth = r * r
    g_phph = (r * sin(x[2])) ** 2
    return g_tt, g_rr, g_thth, g_phph, 0.0


class BlackHole:
    '''
    Definition of the Black Hole described by Schwarzschild metric
    '''
    def __init__(self):
        self.M = 1
        self.a = 0.
        self.EH = 1*self.M
        self.ISCOco = 3*self.M
        self.ISCOcounter = 3*self.M
        # Legacy scipy interpolators for callers that use the Python path.
        self.g_tt  = interp1d(_R_TBL, _GTT_COV, bounds_error=False, fill_value=0)
        self.g_rr  = interp1d(_R_TBL, _GRR_COV, bounds_error=False, fill_value=0)
        self.gtt   = interp1d(_R_TBL, _GTT_INV, bounds_error=False, fill_value=0)
        self.grr   = interp1d(_R_TBL, _GRR_INV, bounds_error=False, fill_value=0)
        self.drgtt = interp1d(_R_TBL, _DRGTT,   bounds_error=False, fill_value=0)
        self.drgrr = interp1d(_R_TBL, _DRGRR,   bounds_error=False, fill_value=0)
        # Numba hot-path hooks
        self._rhs_nb = _geodesics_nb
        self._metric_nb = _metric_nb
        self._omega_nb = None

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
        g_tt = self.g_tt(x[1])
        g_rr = self.g_rr(x[1])
        g_thth = x[1]**2
        g_phph = (x[1]*sin(x[2]))**2
        g_tph = 0.
        
        return [g_tt, g_rr, g_thth, g_phph, g_tph]
    
    
    def inverse_metric(self, x):
        gtt   = self.gtt(x[1])
        grr   = self.grr(x[1])
        gthth = 1.0 / x[1]**2
        gphph = 1.0 / (x[1] * sin(x[2]))**2
        gtph  = 0.
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
        E = - k_t
        L = k_phi
        ===========================================================================
        '''
        # Metric and its numerical derivative
        #gtt, grr, gthth, gphph, gtph = self.inverse_metric(q[0:4])
        #drgtt, drgrr, drgthth, drgphph, drgtph = self.dr_inverse_metric(q[0:4])
        
        # Geodesics differential equations 
        dtdlmbda = self.gtt(q[1])*q[4]
        drdlmbda = self.grr(q[1])*q[5]
        dthdlmbda = (1./q[1]**2)*q[6]
        dphidlmbda = (1./(q[1]*sin(q[2]))**2)*q[7]
        
        dk_tdlmbda = 0.
        dk_rdlmbda = - (self.drgtt(q[1])*q[4]**2)/2 - (self.drgrr(q[1])*q[5]**2)/2 \
                     - ((-2/q[1]**3)*q[6]**2)/2 - ((-2/(q[1]**3*sin(q[2])**2))*q[7]**2)/2
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

    

