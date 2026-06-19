"""
===============================================================================
Power-law Elliptical Mass Distribution (PEMD / EPL): weak-field, axisymmetric
===============================================================================
Generalises SIE (gamma = 2) to an arbitrary radial slope. The same
isotropic line element is used:

    ds^2 = -A^2 dt^2 + B^2 (dr^2 + r^2 dtheta^2 + r^2 sin^2 theta dphi^2)
    A = 1 + Phi(r, theta),  B = 1 - Phi(r, theta)

with the *power-law* ellipsoidal potential

    Phi(r, theta) = (K / (2 - gamma)) * xi^(2 - gamma) ,    gamma != 2
    xi(r, theta)  = r * sqrt( sin^2(theta) + cos^2(theta) / q_ax^2 )

where K (a positive amplitude) and gamma (the radial slope of the surface
density) are free parameters.  q_ax in (0, 1] controls ellipticity around
the z-axis (q_ax = 1 reduces to a spherical power-law lens).

The convergence kappa(R) of the projected surface density goes as
R^{-(gamma - 1)}; gamma = 2 is the isothermal case (use scr.lens_metrics.sie
for that), gamma > 2 makes the profile steeper, gamma < 2 makes it shallower.

Derivatives entering the geodesic RHS:

    dxi/dr = f(theta) = sqrt(sin^2 + cos^2/q^2)
    dxi/dtheta = xi * H(theta),     H = sin*cos*(1 - 1/q^2)/f^2
    dPhi/dr     = K * xi^(1 - gamma) * f
    dPhi/dtheta = K * xi^(2 - gamma) * H

Validity is the standard weak-field condition |Phi| << 1; choose
(K, gamma) such that |Phi(r_min)| stays below ~1e-3.

Units: G = c = 1; K has units of [length]^{gamma - 2}.

Usage
-----
    >>> lens = pemd.LensMetric(K=4e-4, gamma=2.05, q_ax=0.7)
    >>> lens.deflection_angle(b=1.0e3)        # analytic spherical limit

For gamma exactly 2 the formula has a removable singularity; we raise an
error and direct the user to scr.lens_metrics.sie.
===============================================================================
"""
from math import sin, cos, sqrt
import numpy as np
from numba import njit


@njit(cache=True)
def _aux_nb(r, th, K, gamma, q_ax):
    """Returns (Phi, dPhi/dr, dPhi/dth, sin_th, cos_th) for the PEMD."""
    sin_th = sin(th); cos_th = cos(th)
    q2 = q_ax * q_ax
    f2 = sin_th*sin_th + cos_th*cos_th / q2
    f = sqrt(f2)
    xi = r * f
    # Phi = K * xi^(2-gamma) / (2 - gamma)
    expo = 2.0 - gamma
    xi_pow = xi ** expo                # xi^(2-gamma)
    Phi = (K / expo) * xi_pow
    # dPhi/dr = K * xi^(1-gamma) * f
    xi_pow_m1 = xi_pow / xi            # xi^(1-gamma)
    dPhi_dr = K * xi_pow_m1 * f
    # dPhi/dth = K * xi^(2-gamma) * H, H = sin*cos*(1 - 1/q^2)/f^2
    H = sin_th * cos_th * (1.0 - 1.0 / q2) / f2
    dPhi_dth = K * xi_pow * H
    return Phi, dPhi_dr, dPhi_dth, sin_th, cos_th


@njit(cache=True)
def _geodesics_nb_array(q, K, gamma, q_ax):
    r = q[1]; th = q[2]
    kt = q[4]; kr = q[5]; kth = q[6]; kph = q[7]

    Phi, dPhi_dr, dPhi_dth, sin_th, cos_th = _aux_nb(r, th, K, gamma, q_ax)
    sin_th2 = sin_th * sin_th
    sin_th3 = sin_th2 * sin_th

    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    A3 = A2 * A; B3 = B2 * B
    r2 = r * r; r3 = r2 * r

    dA_dr = dPhi_dr;  dB_dr = -dPhi_dr
    dA_dth = dPhi_dth; dB_dth = -dPhi_dth

    dgtt_dr   =  2.0 * dA_dr / A3
    dgrr_dr   = -2.0 * dB_dr / B3
    dgthth_dr = -2.0 * (dB_dr * r + B) / (B3 * r3)
    dgphph_dr = dgthth_dr / sin_th2

    dgtt_dth   =  2.0 * dA_dth / A3
    dgrr_dth   = -2.0 * dB_dth / B3
    dgthth_dth = -2.0 * dB_dth / (B3 * r2)
    dgphph_dth = -2.0 * (dB_dth * sin_th + B * cos_th) / (B3 * r2 * sin_th3)

    out = np.empty(8)
    out[0] = -kt / A2
    out[1] = kr / B2
    out[2] = kth / (B2 * r2)
    out[3] = kph / (B2 * r2 * sin_th2)
    out[4] = 0.0
    out[7] = 0.0
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
def _metric_nb(x, K, gamma, q_ax):
    r = x[1]; th = x[2]
    Phi, _, _, sin_th, _ = _aux_nb(r, th, K, gamma, q_ax)
    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    r2 = r * r
    return (-A2, B2, B2 * r2, B2 * r2 * sin_th * sin_th, 0.0)


@njit(cache=True)
def _null_omega_nb(r):
    return 0.0


class LensMetric:
    """Power-law elliptical lens.

    Parameters
    ----------
    K : float
        Potential amplitude (positive). Sets the deflection scale.
        For physical SLACS-like systems use a value such that
        |Phi(r_ref)| ~ 1e-4.
    gamma : float
        Radial slope. gamma = 2 is isothermal (use SIE module instead);
        typical SLACS measurements give gamma ~ 1.9-2.2. Must satisfy
        |gamma - 2| > 1e-3 to keep the formula stable.
    q_ax : float, optional (default 0.7)
        Axis ratio b/a in (0, 1].
    r_min : float, optional (default 1e-2)
        Numerical horizon cutoff for the integrator.
    """

    def __init__(self, K, gamma, q_ax=0.7, r_min=1e-2):
        if abs(gamma - 2.0) < 1.0e-3:
            raise ValueError(
                "PEMD with gamma=2 is the isothermal case; use "
                "scr.lens_metrics.sie.LensMetric instead.")
        self.K = float(K)
        self.gamma = float(gamma)
        self.q_ax = float(q_ax)
        self._r_min = float(r_min)

        self.a = 0.0
        self.EH = self._r_min
        self.ISCOco = 0.0
        self.ISCOcounter = 0.0

        _K = float(self.K)
        _g = float(self.gamma)
        _q = float(self.q_ax)

        @njit
        def _rhs(q):
            return _geodesics_nb_array(q, _K, _g, _q)

        @njit
        def _metric(x):
            return _metric_nb(x, _K, _g, _q)

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _null_omega_nb

    # -- Python-level API ---------------------------------------------------
    def Phi(self, x):
        Phi, _, _, _, _ = _aux_nb(float(x[1]), float(x[2]),
                                   self.K, self.gamma, self.q_ax)
        return Phi

    def metric(self, x):
        g = _metric_nb(np.asarray(x, dtype=np.float64),
                        self.K, self.gamma, self.q_ax)
        return [g[0], g[1], g[2], g[3], g[4]]

    def inverse_metric(self, x):
        r = x[1]; th = x[2]
        Phi, _, _, sin_th, _ = _aux_nb(r, th, self.K, self.gamma, self.q_ax)
        A = 1.0 + Phi; B = 1.0 - Phi
        A2 = A * A; B2 = B * B
        r2 = r * r
        return [-1.0 / A2, 1.0 / B2, 1.0 / (B2 * r2),
                1.0 / (B2 * r2 * sin_th * sin_th), 0.0]

    def geodesics(self, q, lmbda):
        return list(_geodesics_nb_array(
            np.asarray(q, dtype=np.float64),
            self.K, self.gamma, self.q_ax))

    def Omega(self, r, corotating=True):
        return 0.0

    def deflection_angle_spherical(self, b):
        """Analytical spherical-limit deflection (q_ax = 1) at impact b.

        For Phi(r) = K r^{2-gamma}/(2-gamma) with q_ax=1, integrating
        2 dPhi/dR along the line of sight gives the thin-lens deflection

            alpha(b) = 2 K * sqrt(pi) * Gamma((gamma-1)/2) /
                       Gamma(gamma/2) * b^{2-gamma}

        (line-of-sight integral 2 int_{-inf}^{inf} dPhi/dR dz; matches
        Tessore & Metcalf 2015 in the spherical limit). Used by the
        unit test as an analytic reference.
        """
        from math import gamma as Gamma_func, pi
        coeff = 2.0 * self.K * sqrt(pi) * (
            Gamma_func((self.gamma - 1.0) / 2.0)
            / Gamma_func(self.gamma / 2.0))
        return coeff * (b ** (2.0 - self.gamma))


###############################################################################

if __name__ == '__main__':
    lens = LensMetric(K=4.0e-4, gamma=2.10, q_ax=0.7, r_min=1e-2)
    print(f"PEMD: K={lens.K:.2e}, gamma={lens.gamma}, q_ax={lens.q_ax}")
    print(f"|Phi| at xi=1   : {abs(lens.Phi([0, 1.0, 1.5708, 0])):.3e}")
    print(f"|Phi| at xi=100 : {abs(lens.Phi([0, 100.0, 1.5708, 0])):.3e}")
