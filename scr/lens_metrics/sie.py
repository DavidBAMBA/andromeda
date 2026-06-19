"""
===============================================================================
Singular Isothermal Ellipsoid (SIE): weak-field, isotropic, axisymmetric
around the z-axis.
===============================================================================
Line element:

    ds^2 = -A(r,theta)^2 dt^2
           + B(r,theta)^2 (dr^2 + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2)

with A = 1 + Phi, B = 1 - Phi, and the ellipsoidal potential

    Phi(r, theta) = 2 sigma_v^2 log(xi(r, theta) / r_ref)

    xi(r, theta) = r * sqrt( sin^2(theta) + cos^2(theta) / q_ax^2 )

- q_ax = 1 recovers the SIS (spherical) case exactly.
- q_ax < 1 is an oblate ellipsoid, flattened along the z-axis. Equipotential
  surfaces are ellipsoids of revolution with semi-axes (xi, xi, q_ax * xi).
- The metric is still axisymmetric (phi-independent), so k_phi is conserved.
- The ellipticity breaks the equatorial-reflection symmetry only for
  off-equatorial rays: photons stay in the equatorial plane if they start
  there with k_theta = 0, so the view from an equatorial observer shows
  the ring squashed along the axis perpendicular to z (see ex13).
- Units: G = c = 1. No horizon; r_min cutoff acts as the "EH" event for
  the integrator.
===============================================================================
"""
from math import sin, cos, log, sqrt
import numpy as np
from numba import njit


@njit(cache=True)
def _aux_nb(r, th, sv2, r_ref, q_ax):
    """Return Phi(r,th), its radial and angular derivatives, and helper H.

    H(theta) = sin(th)*cos(th) * (1 - 1/q_ax^2) / f(theta)^2, where
    f^2 = sin^2 + cos^2/q_ax^2. Vanishes at theta=0, pi/2, pi.
    """
    sin_th = sin(th); cos_th = cos(th)
    q2 = q_ax * q_ax
    f2 = sin_th*sin_th + cos_th*cos_th / q2
    # xi = r * sqrt(f2)
    Phi = 2.0 * sv2 * (log(r / r_ref) + 0.5 * log(f2))
    dPhi_dr = 2.0 * sv2 / r
    H = sin_th * cos_th * (1.0 - 1.0 / q2) / f2
    dPhi_dth = 2.0 * sv2 * H
    return Phi, dPhi_dr, dPhi_dth, sin_th, cos_th


@njit(cache=True)
def _geodesics_nb_array(q, sv2, r_ref, q_ax):
    """SIE null-geodesic RHS in Hamiltonian form. q = [t, r, th, ph, k_t,
    k_r, k_th, k_phi]; returns dq/dlambda."""
    r = q[1]; th = q[2]
    kt = q[4]; kr = q[5]; kth = q[6]; kph = q[7]

    Phi, dPhi_dr, dPhi_dth, sin_th, cos_th = _aux_nb(r, th, sv2, r_ref, q_ax)
    sin_th2 = sin_th * sin_th
    sin_th3 = sin_th2 * sin_th

    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    A3 = A2 * A; B3 = B2 * B
    r2 = r * r; r3 = r2 * r

    # A'_r = dPhi/dr, B'_r = -dPhi/dr, similarly for theta.
    dA_dr = dPhi_dr;  dB_dr = -dPhi_dr
    dA_dth = dPhi_dth; dB_dth = -dPhi_dth

    # Radial derivatives of the contravariant metric components.
    dgtt_dr   =  2.0 * dA_dr / A3
    dgrr_dr   = -2.0 * dB_dr / B3
    dgthth_dr = -2.0 * (dB_dr * r + B) / (B3 * r3)
    dgphph_dr = dgthth_dr / sin_th2

    # Angular derivatives.
    dgtt_dth   =  2.0 * dA_dth / A3
    dgrr_dth   = -2.0 * dB_dth / B3
    dgthth_dth = -2.0 * dB_dth / (B3 * r2)
    dgphph_dth = -2.0 * (dB_dth * sin_th + B * cos_th) / (B3 * r2 * sin_th3)

    out = np.empty(8)
    # Positions
    out[0] = -kt / A2
    out[1] = kr / B2
    out[2] = kth / (B2 * r2)
    out[3] = kph / (B2 * r2 * sin_th2)
    # Killing directions (t, phi)
    out[4] = 0.0
    out[7] = 0.0
    # Radial and angular momenta
    out[5] = -0.5 * (dgtt_dr   * kt*kt
                     + dgrr_dr   * kr*kr
                     + dgthth_dr * kth*kth
                     + dgphph_dr * kph*kph)
    out[6] = -0.5 * (dgtt_dth   * kt*kt
                     + dgrr_dth   * kr*kr
                     + dgthth_dth * kth*kth
                     + dgphph_dth * kph*kph)
    return out


@njit(cache=True)
def _metric_nb(x, sv2, r_ref, q_ax):
    r = x[1]; th = x[2]
    Phi, _, _, sin_th, _ = _aux_nb(r, th, sv2, r_ref, q_ax)
    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    r2 = r * r
    return (-A2, B2, B2 * r2, B2 * r2 * sin_th * sin_th, 0.0)


@njit(cache=True)
def _null_omega_nb(r):
    return 0.0


class LensMetric:
    """Singular Isothermal Ellipsoid, axisymmetric around z.

    Parameters
    ----------
    sigma_v : float
        1-D velocity dispersion (units of c). Same as in SIS.
    q_ax : float, optional (default 0.7)
        Axis ratio q_ax = (minor / major). q_ax = 1 recovers SIS.
        Typical observed SLACS lenses have q_ax ~ 0.6-0.9.
    r_ref : float, optional (default 1.0)
        Reference radius for the logarithmic potential.
    r_min : float, optional (default 1e-2)
        Numerical cutoff acting as "horizon" for the integrator.
    """

    def __init__(self, sigma_v, q_ax=0.7, r_ref=1.0, r_min=1e-2):
        self.sigma_v = float(sigma_v)
        self.q_ax = float(q_ax)
        self._sv2 = self.sigma_v ** 2
        self._r_ref = float(r_ref)
        self._r_min = float(r_min)

        # Interface compatibility.
        self.a = 0.0
        self.EH = self._r_min
        self.ISCOco = 0.0
        self.ISCOcounter = 0.0

        _sv2 = float(self._sv2)
        _rr  = float(self._r_ref)
        _q   = float(self.q_ax)

        @njit
        def _rhs(q):
            return _geodesics_nb_array(q, _sv2, _rr, _q)

        @njit
        def _metric(x):
            return _metric_nb(x, _sv2, _rr, _q)

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _null_omega_nb

    def metric(self, x):
        g = _metric_nb(np.asarray(x, dtype=np.float64),
                        self._sv2, self._r_ref, self.q_ax)
        return [g[0], g[1], g[2], g[3], g[4]]

    def inverse_metric(self, x):
        r = x[1]; th = x[2]
        Phi, _, _, sin_th, _ = _aux_nb(r, th, self._sv2, self._r_ref,
                                        self.q_ax)
        A = 1.0 + Phi; B = 1.0 - Phi
        A2 = A * A; B2 = B * B
        r2 = r * r
        return [-1.0 / A2, 1.0 / B2, 1.0 / (B2 * r2),
                1.0 / (B2 * r2 * sin_th * sin_th), 0.0]

    def geodesics(self, q, lmbda):
        return list(_geodesics_nb_array(
            np.asarray(q, dtype=np.float64),
            self._sv2, self._r_ref, self.q_ax))

    def Omega(self, r, corotating=True):
        return 0.0


###############################################################################

if __name__ == '__main__':
    print("SIE module: scr.lens_metrics.sie")
