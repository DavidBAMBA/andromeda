"""
===============================================================================
Kerr metric
===============================================================================
- Event horizon at r = M + sqrt(M^2 - a^2)
- ISCO at r = 3M + Z2 -+ sqrt((3M - Z1)(3M + Z1 + 2Z2))
- We use units where G = c = 1, and M = 1
===============================================================================
@author: Alexis Larrañaga - 2023
===============================================================================
"""

from math import sin, cos, sqrt
import numpy as np
from numba import njit


@njit(cache=True)
def _geodesics_nb_array(q, a):
    '''Kerr geodesic RHS, array form (signature (float64[:], float64) -> float64[:]).'''
    r2 = q[1]*q[1]
    a2 = a*a
    sin_th = sin(q[2])
    cos_th = cos(q[2])
    sin_th2 = sin_th*sin_th
    cos_th2 = cos_th*cos_th
    Sigma = r2 + a2*cos_th2
    Sigma2 = Sigma*Sigma
    Delta = r2 - 2*q[1] + a2

    W = -q[4]*(r2 + a2) - a*q[7]
    partXi = r2 + (q[7] + a*q[4])**2 + a2*(1 + q[4]*q[4])*cos_th2 + q[7]*q[7]*cos_th2/sin_th2
    Xi = W*W - Delta*partXi

    dXidE = 2*W*(r2 + a2) + 2.*a*Delta*(q[7] + a*q[4]*sin_th2)
    dXidL = -2*a*W - 2*a*q[4]*Delta - 2*q[7]*Delta/sin_th2
    dXidr = -4*q[1]*q[4]*W - 2*(q[1] - 1)*partXi - 2*q[1]*Delta

    dAdr = (q[1] - 1)/Sigma - (q[1]*Delta)/Sigma2
    dBdr = -q[1]/Sigma2
    dCdr = dXidr/(2*Delta*Sigma) - (Xi*(q[1]-1))/(Sigma*Delta*Delta) - q[1]*Xi/(Delta*Sigma2)

    auxth = a2*cos_th*sin_th
    dAdth = Delta*auxth/Sigma2
    dBdth = auxth/Sigma2
    dCdth = ((1+q[4]*q[4])*auxth + q[7]*q[7]*cos_th/(sin_th2*sin_th))/Sigma + (Xi/(Delta*Sigma2))*auxth

    out = np.empty(8)
    out[0] = dXidE/(2.*Delta*Sigma)
    out[1] = (Delta/Sigma)*q[5]
    out[2] = q[6]/Sigma
    out[3] = -dXidL/(2.*Delta*Sigma)
    out[4] = 0.0
    out[5] = -dAdr*q[5]*q[5] - dBdr*q[6]*q[6] + dCdr
    out[6] = -dAdth*q[5]*q[5] - dBdth*q[6]*q[6] + dCdth
    out[7] = 0.0
    return out


@njit(cache=True)
def _metric_nb(x, a):
    '''Kerr metric components (g_tt, g_rr, g_thth, g_phph, g_tph).'''
    r = x[1]
    r2 = r * r
    a2 = a * a
    sin_theta2 = sin(x[2]) ** 2
    Delta = r2 - 2.0 * r + a2
    Sigma = r2 + a2 * cos(x[2]) ** 2
    g_tt = -(1.0 - 2.0 * r / Sigma)
    g_rr = Sigma / Delta
    g_thth = Sigma
    g_phph = (r2 + a2 + 2.0 * a2 * r * sin_theta2 / Sigma) * sin_theta2
    g_tph = -2.0 * a * r * sin_theta2 / Sigma
    return g_tt, g_rr, g_thth, g_phph, g_tph


@njit(cache=True)
def _omega_nb(r, a):
    return 1.0 / (r ** 1.5 + a)


@njit(cache=True)
def _geodesics_nb(q0, q1, q2, q3, q4, q5, q6, q7, a):
    r2 = q1*q1
    a2 = a*a
    sin_th = sin(q2)
    cos_th = cos(q2)
    sin_th2 = sin_th*sin_th
    cos_th2 = cos_th*cos_th
    Sigma = r2 + a2*cos_th2
    Sigma2 = Sigma*Sigma
    Delta = r2 - 2*q1 + a2

    W = -q4*(r2 + a2) - a*q7
    partXi = r2 + (q7 + a*q4)**2 + a2*(1 + q4*q4)*cos_th2 + q7*q7*cos_th2/sin_th2
    Xi = W**2 - Delta*partXi

    dXidE = 2*W*(r2 + a2) + 2.*a*Delta*(q7 + a*q4*sin_th2)
    dXidL = -2*a*W - 2*a*q4*Delta - 2*q7*Delta/sin_th2
    dXidr = -4*q1*q4*W - 2*(q1 - 1)*partXi - 2*q1*Delta

    dAdr = (q1 - 1)/Sigma - (q1*Delta)/Sigma2
    dBdr = -q1/Sigma2
    dCdr = dXidr/(2*Delta*Sigma) - (Xi*(q1-1))/(Sigma*Delta*Delta) - q1*Xi/(Delta*Sigma2)

    auxth = a2*cos_th*sin_th
    dAdth = Delta*auxth/Sigma2
    dBdth = auxth/Sigma2
    dCdth = ((1+q4**2)*auxth + q7*q7*cos_th/(sin_th2*sin_th))/Sigma + (Xi/(Delta*Sigma2))*auxth

    dtdlmbda    = dXidE/(2.*Delta*Sigma)
    drdlmbda    = (Delta/Sigma)*q5
    dthdlmbda   = q6/Sigma
    dphidlmbda  = -dXidL/(2.*Delta*Sigma)
    dk_rdlmbda  = -dAdr*q5*q5 - dBdr*q6*q6 + dCdr
    dk_thdlmbda = -dAdth*q5*q5 - dBdth*q6*q6 + dCdth

    return (dtdlmbda, drdlmbda, dthdlmbda, dphidlmbda,
            0.0, dk_rdlmbda, dk_thdlmbda, 0.0)


class BlackHole:
    '''
    Definition of the Black Hole described by Kerr metric
    '''
    def __init__(self, a):
        self.a = a
        self.EH = 1 + sqrt(1 - self.a**2)
        Z1 = 1 + (1 - self.a**2)**(1/3)*((1 + self.a)**(1/3) + (1 - self.a)**(1/3))
        Z2 = sqrt(3*self.a**2 + Z1**2)
        self.ISCOco = 3 + Z2 - sqrt((3 - Z1)*(3 + Z1 + 2*Z2))
        self.ISCOcounter = 3 + Z2 + sqrt((3 - Z1)*(3 + Z1 + 2*Z2))
        # Numba hot-path hooks (closures specialize on `a`).
        _a = float(a)
        @njit
        def _rhs(q):
            return _geodesics_nb_array(q, _a)
        @njit
        def _metric(x):
            return _metric_nb(x, _a)
        @njit
        def _omega(r):
            return _omega_nb(r, _a)
        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _omega

    
    def Omega(self, r, corotating=True):
        '''
        Returns the angular velocity of a particle at radius r
        '''
        if corotating:
            return 1/(r**(3/2) + self.a)
        else:
            return -1/(r**(3/2) - self.a)

    def metric(self,x):
        '''
        This procedure contains the Kerr metric non-zero components in 
        spherical coordinates
        ===========================================================================
        Coordinates 
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        ===========================================================================
        '''
        # Auxiliary functions
        r2 = x[1]*x[1]
        a2 = self.a*self.a
        sin_theta2 = sin(x[2])**2
        Delta = r2 - 2*x[1] + a2
        Sigma = r2 + a2*cos(x[2])**2
        
        # Metric components
        g_tt = -(1 - 2*x[1]/Sigma)
        g_rr = Sigma/Delta
        g_thth = Sigma
        g_phph = (r2 + a2 + 2*a2*x[1]*sin_theta2/Sigma)*sin_theta2
        g_tph = -2*self.a*x[1]*sin_theta2/Sigma
        
        return [g_tt, g_rr, g_thth, g_phph, g_tph]
    
    def inverse_metric(self,x):
        '''
        This procedure contains the inverse Kerr metric non-zero components in 
        spherical coordinates
        ===========================================================================
        Coordinates 
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        ===========================================================================
        '''
        # Auxiliary functions
        r2 = x[1]*x[1]
        a2 = self.a*self.a
        sin_theta2 = sin(x[2])**2
        Delta = r2 - 2*x[1] + a2
        Sigma = r2 + a2*cos(x[2])**2
        A = (r2 + a2)**2 - Delta*a2*sin_theta2
        
        # Metric components
        gtt = - A/(Delta*Sigma)
        grr = Delta/Sigma
        gthth = 1/Sigma
        gphph = (Delta - a2*sin_theta2)/(Delta*Sigma*sin_theta2)
        gtph = - 2*self.a*x[1]/(Delta*Sigma)
        
        return [gtt, grr, gthth, gphph, gtph]

    def geodesics(self, q, lmbda):
        return list(_geodesics_nb(q[0], q[1], q[2], q[3],
                                  q[4], q[5], q[6], q[7], self.a))





###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')