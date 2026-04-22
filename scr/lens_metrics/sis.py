"""
===============================================================================
Singular Isothermal Sphere (weak-field, isotropic coordinates)
===============================================================================
Line element:

    ds^2 = -A(r)^2 dt^2 + B(r)^2 (dr^2 + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2)

with

    A(r) = 1 + Phi(r)
    B(r) = 1 - Phi(r)
    Phi(r) = 2 sigma_v^2 ln(r / r_ref)

so that A' = dPhi/dr = 2 sigma_v^2 / r and B' = -A'.

- Weak-field form: valid while |Phi| << 1. Reproduces the classical thin-lens
  deflection alpha = 4 pi sigma_v^2 exactly (sigma_v in units of c = 1).
- No event horizon. A small r_min cutoff is exposed as self.EH so the
  integrator's horizon event terminates photons that plunge toward r = 0.
- Units: G = c = 1. sigma_v is the 1-D velocity dispersion in units of c.
===============================================================================
"""

from math import sin, cos, log
import numpy as np
from numba import njit


@njit(cache=True)
def _phi_nb(r, sv2, r_ref):
    return 2.0 * sv2 * log(r / r_ref)


@njit(cache=True)
def _geodesics_nb_array(q, sv2, r_ref):
    r = q[1]
    th = q[2]
    sin_th = sin(th)
    cos_th = cos(th)
    sin_th2 = sin_th * sin_th
    sin_th3 = sin_th2 * sin_th

    Phi = _phi_nb(r, sv2, r_ref)
    A = 1.0 + Phi
    B = 1.0 - Phi
    A2 = A * A
    B2 = B * B
    A3 = A2 * A
    B3 = B2 * B
    r2 = r * r
    r3 = r2 * r

    Aprime = 2.0 * sv2 / r          # dPhi/dr
    Bprime = -Aprime                # -dPhi/dr
    BrB = Bprime * r + B            # = B - 2*sv2

    kt = q[4]; kr = q[5]; kth = q[6]; kph = q[7]

    out = np.empty(8)
    # Positions: dx^mu/dlambda = g^{mu nu} k_nu
    out[0] = -kt / A2
    out[1] = kr / B2
    out[2] = kth / (B2 * r2)
    out[3] = kph / (B2 * r2 * sin_th2)

    # Conserved: k_t (static) and k_phi (axisymmetric)
    out[4] = 0.0
    out[7] = 0.0

    # dk_r/dlambda = -(1/2) d_r g^{alpha beta} k_alpha k_beta
    out[5] = (-(Aprime / A3) * kt * kt
              + (Bprime / B3) * kr * kr
              + (BrB / (B3 * r3)) * kth * kth
              + (BrB / (B3 * r3 * sin_th2)) * kph * kph)

    # dk_theta/dlambda = -(1/2) d_theta g^{phi phi} k_phi^2
    out[6] = (cos_th / (B2 * r2 * sin_th3)) * kph * kph

    return out


@njit(cache=True)
def _metric_nb(x, sv2, r_ref):
    """Covariant metric components (g_tt, g_rr, g_thth, g_phph, g_tph)."""
    r = x[1]
    th = x[2]
    Phi = _phi_nb(r, sv2, r_ref)
    A = 1.0 + Phi
    B = 1.0 - Phi
    A2 = A * A
    B2 = B * B
    r2 = r * r
    sin_th2 = sin(th) ** 2
    g_tt = -A2
    g_rr = B2
    g_thth = B2 * r2
    g_phph = B2 * r2 * sin_th2
    g_tph = 0.0
    return g_tt, g_rr, g_thth, g_phph, g_tph


@njit(cache=True)
def _null_omega_nb(r):
    """Unused for lens metrics; kept to satisfy kernel typing if ever needed."""
    return 0.0


class LensMetric:
    """Singular Isothermal Sphere lens (weak-field, isotropic).

    Parameters
    ----------
    sigma_v : float
        1-D velocity dispersion in units of c. For a real galaxy
        sigma_v ~ 1e-3 (v_1d ~ 300 km/s). Validation scripts use ~ 1e-2
        to produce a resolvable ring in geometrized units.
    r_ref : float, optional
        Reference radius for the logarithmic potential (default 1 M).
        Only shifts the zero of Phi; geodesics are ~invariant to leading
        order in sigma_v^2.
    r_min : float, optional
        Numerical cutoff acting as 'horizon' for the integrator (default
        1e-2 M). Photons reaching this radius are terminated.
    """

    def __init__(self, sigma_v, r_ref=1.0, r_min=1e-2):
        self.sigma_v = float(sigma_v)
        self._sv2 = self.sigma_v ** 2
        self._r_ref = float(r_ref)
        self._r_min = float(r_min)

        # Interface compatibility with the BlackHole duck-type used elsewhere.
        self.a = 0.0
        self.EH = self._r_min          # integrator horizon event fires at r <= EH + 1e-3
        self.ISCOco = 0.0              # unused for lensing
        self.ISCOcounter = 0.0

        _sv2 = float(self._sv2)
        _rr = float(self._r_ref)

        @njit
        def _rhs(q):
            return _geodesics_nb_array(q, _sv2, _rr)

        @njit
        def _metric(x):
            return _metric_nb(x, _sv2, _rr)

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _null_omega_nb

    # -- Python-level API ----------------------------------------------------
    def Phi(self, r):
        return 2.0 * self._sv2 * np.log(r / self._r_ref)

    def metric(self, x):
        g = _metric_nb(np.asarray(x, dtype=np.float64), self._sv2, self._r_ref)
        return [g[0], g[1], g[2], g[3], g[4]]

    def inverse_metric(self, x):
        r = x[1]
        th = x[2]
        Phi = 2.0 * self._sv2 * np.log(r / self._r_ref)
        A = 1.0 + Phi
        B = 1.0 - Phi
        A2 = A * A
        B2 = B * B
        r2 = r * r
        sin_th2 = sin(th) ** 2
        return [-1.0 / A2, 1.0 / B2, 1.0 / (B2 * r2),
                1.0 / (B2 * r2 * sin_th2), 0.0]

    def geodesics(self, q, lmbda):
        return list(_geodesics_nb_array(
            np.asarray(q, dtype=np.float64), self._sv2, self._r_ref))

    def Omega(self, r, corotating=True):
        return 0.0


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('')
