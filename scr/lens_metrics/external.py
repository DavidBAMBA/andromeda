"""
===============================================================================
External convergence + shear (kappa_ext, gamma_+_ext, gamma_x_ext)
===============================================================================
Constant background tidal field representing the cumulative effect of
nearby structures (cluster environment, line-of-sight halos). In the
thin-lens / weak-field limit the lensing potential is

    psi_ext(theta_1, theta_2) = (kappa/2) (theta_1^2 + theta_2^2)
                              + (gamma_+/2) (theta_1^2 - theta_2^2)
                              + gamma_x * theta_1 * theta_2

We promote this to a 3-D Newtonian potential acting on (y, z) — the two
directions perpendicular to the line of sight (taken along +x in this
code's iota = pi/2 convention):

    Phi_ext(y, z) = (kappa/2) (y^2 + z^2)
                  + (gamma_+/2) (y^2 - z^2)
                  + gamma_x * y * z

In Boyer-Lindquist:  y = r sin(theta) sin(phi),  z = r cos(theta).
The potential thus depends on (r, theta, phi) — the metric is no longer
axisymmetric and k_phi is NOT conserved. The geodesic RHS picks up an
extra term out[7] = -(1/2) d_phi g^{ab} k_a k_b.

Used standalone (rare; nothing in the universe is pure shear) or as the
external piece of a composite lens — see scr.lens_metrics.composite.

Units: G = c = 1; kappa, gamma_+, gamma_x are dimensionless. The induced
deflection scales linearly with these.
===============================================================================
"""
from math import sin, cos, exp
import numpy as np
from numba import njit


@njit(cache=True)
def _ext_aux_nb(r, th, ph, kappa, gp, gx, r_window):
    """Return (Phi, dPhi/dr, dPhi/dth, dPhi/dph) for the external field.

    A Gaussian window W(r) = exp(-r^2/(2 r_window^2)) localizes the
    quadratic potential near the lens so that asymptotic flatness is
    preserved (otherwise Phi ~ r^2 violates |Phi| << 1 at large r).
    Effective external (kappa, gamma) gets renormalized by the integral
    of W along the line of sight; r_window should be of order the
    impact-parameter scale of interest.
    """
    sin_th = sin(th); cos_th = cos(th)
    sin_ph = sin(ph); cos_ph = cos(ph)
    y = r * sin_th * sin_ph
    z = r * cos_th
    Q = 0.5*kappa*(y*y + z*z) + 0.5*gp*(y*y - z*z) + gx*y*z
    inv_r_window2 = 1.0 / (r_window * r_window)
    W = exp(-0.5 * r * r * inv_r_window2)
    Phi = Q * W
    # dQ/dy, dQ/dz
    dQ_dy = (kappa + gp) * y + gx * z
    dQ_dz = (kappa - gp) * z + gx * y
    # Chain rule pieces: y, z depend on (r, th, ph)
    dy_dr = sin_th * sin_ph
    dz_dr = cos_th
    dy_dth = r * cos_th * sin_ph
    dz_dth = -r * sin_th
    dy_dph = r * sin_th * cos_ph
    # Q has no explicit dependence on r/th/ph other than through (y,z):
    dQ_dr  = dQ_dy * dy_dr  + dQ_dz * dz_dr
    dQ_dth = dQ_dy * dy_dth + dQ_dz * dz_dth
    dQ_dph = dQ_dy * dy_dph
    # W depends only on r:
    dW_dr = -r * inv_r_window2 * W
    dPhi_dr  = dQ_dr  * W + Q * dW_dr
    dPhi_dth = dQ_dth * W
    dPhi_dph = dQ_dph * W
    return Phi, dPhi_dr, dPhi_dth, dPhi_dph


@njit(cache=True)
def _geodesics_nonaxi_nb(q, Phi, dPhi_dr, dPhi_dth, dPhi_dph):
    """Generic non-axisymmetric weak-field RHS given pre-computed Phi and
    its three partial derivatives. Used by external.py (alone) and by
    composite.py (sum of two Phi contributions).

    A = 1 + Phi,  B = 1 - Phi.  Metric is the same isotropic line element
    as SIS/SIE/NFW.
    """
    r = q[1]; th = q[2]
    kt = q[4]; kr = q[5]; kth = q[6]; kph = q[7]
    sin_th = sin(th); cos_th = cos(th)
    sin_th2 = sin_th * sin_th
    sin_th3 = sin_th2 * sin_th

    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    A3 = A2 * A; B3 = B2 * B
    r2 = r * r; r3 = r2 * r

    dA_dr = dPhi_dr;   dB_dr  = -dPhi_dr
    dA_dth = dPhi_dth; dB_dth = -dPhi_dth
    dA_dph = dPhi_dph; dB_dph = -dPhi_dph

    # Radial derivatives of inverse metric.
    dgtt_dr   =  2.0 * dA_dr / A3
    dgrr_dr   = -2.0 * dB_dr / B3
    dgthth_dr = -2.0 * (dB_dr * r + B) / (B3 * r3)
    dgphph_dr = dgthth_dr / sin_th2

    # Angular (theta) derivatives.
    dgtt_dth   =  2.0 * dA_dth / A3
    dgrr_dth   = -2.0 * dB_dth / B3
    dgthth_dth = -2.0 * dB_dth / (B3 * r2)
    dgphph_dth = -2.0 * (dB_dth * sin_th + B * cos_th) / (B3 * r2 * sin_th3)

    # Phi (azimuthal) derivatives — only the diagonal pieces; sin_th does
    # not depend on phi.
    dgtt_dph   =  2.0 * dA_dph / A3
    dgrr_dph   = -2.0 * dB_dph / B3
    dgthth_dph = -2.0 * dB_dph / (B3 * r2)
    dgphph_dph = -2.0 * dB_dph / (B3 * r2 * sin_th2)

    out = np.empty(8)
    out[0] = -kt / A2
    out[1] = kr / B2
    out[2] = kth / (B2 * r2)
    out[3] = kph / (B2 * r2 * sin_th2)
    # k_t still conserved (no time dependence).
    out[4] = 0.0
    # k_phi NOT conserved any more — pick up the d_phi term.
    out[7] = -0.5 * (dgtt_dph   * kt*kt
                     + dgrr_dph   * kr*kr
                     + dgthth_dph * kth*kth
                     + dgphph_dph * kph*kph)
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
def _geodesics_external_nb(q, kappa, gp, gx, r_window):
    Phi, dPhi_dr, dPhi_dth, dPhi_dph = _ext_aux_nb(
        q[1], q[2], q[3], kappa, gp, gx, r_window)
    return _geodesics_nonaxi_nb(q, Phi, dPhi_dr, dPhi_dth, dPhi_dph)


@njit(cache=True)
def _metric_external_nb(x, kappa, gp, gx, r_window):
    Phi, _, _, _ = _ext_aux_nb(x[1], x[2], x[3], kappa, gp, gx, r_window)
    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    r = x[1]; th = x[2]
    r2 = r * r
    sin_th2 = sin(th) ** 2
    return (-A2, B2, B2 * r2, B2 * r2 * sin_th2, 0.0)


@njit(cache=True)
def _null_omega_nb(r):
    return 0.0


class LensMetric:
    """Pure external convergence + shear.

    Parameters
    ----------
    kappa : float
        External convergence (0 = no extra mass sheet).
    gamma_plus : float
        Plus-mode shear (stretches along x relative to y).
    gamma_cross : float, optional
        Cross-mode shear (stretches at 45 deg). Default 0.
    r_min : float, optional
        Numerical horizon cutoff (default 1e-2).

    Notes
    -----
    A pure shear is unphysical alone (it grows with r^2, breaking
    asymptotic flatness); use scr.lens_metrics.composite to add external
    shear on top of a galaxy-scale lens.
    """

    def __init__(self, kappa=0.0, gamma_plus=0.0, gamma_cross=0.0,
                 r_window=1.0e6, r_min=1e-2):
        self.kappa = float(kappa)
        self.gamma_plus = float(gamma_plus)
        self.gamma_cross = float(gamma_cross)
        self.r_window = float(r_window)
        self._r_min = float(r_min)

        self.a = 0.0
        self.EH = self._r_min
        self.ISCOco = 0.0
        self.ISCOcounter = 0.0

        _k = self.kappa
        _gp = self.gamma_plus
        _gx = self.gamma_cross
        _rw = self.r_window

        @njit
        def _rhs(q):
            return _geodesics_external_nb(q, _k, _gp, _gx, _rw)

        @njit
        def _metric(x):
            return _metric_external_nb(x, _k, _gp, _gx, _rw)

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _null_omega_nb

    def Phi(self, x):
        Phi, _, _, _ = _ext_aux_nb(float(x[1]), float(x[2]), float(x[3]),
                                    self.kappa, self.gamma_plus,
                                    self.gamma_cross, self.r_window)
        return Phi

    def metric(self, x):
        g = _metric_external_nb(np.asarray(x, dtype=np.float64),
                                 self.kappa, self.gamma_plus,
                                 self.gamma_cross, self.r_window)
        return [g[0], g[1], g[2], g[3], g[4]]

    def inverse_metric(self, x):
        Phi, _, _, _ = _ext_aux_nb(float(x[1]), float(x[2]), float(x[3]),
                                    self.kappa, self.gamma_plus,
                                    self.gamma_cross, self.r_window)
        A = 1.0 + Phi; B = 1.0 - Phi
        A2 = A * A; B2 = B * B
        r = x[1]; th = x[2]
        r2 = r * r
        sin_th2 = sin(th) ** 2
        return [-1.0/A2, 1.0/B2, 1.0/(B2*r2), 1.0/(B2*r2*sin_th2), 0.0]

    def geodesics(self, q, lmbda):
        return list(_geodesics_external_nb(
            np.asarray(q, dtype=np.float64),
            self.kappa, self.gamma_plus, self.gamma_cross, self.r_window))

    def Omega(self, r, corotating=True):
        return 0.0


###############################################################################

if __name__ == '__main__':
    ext = LensMetric(kappa=0.05, gamma_plus=0.10, gamma_cross=0.0,
                     r_window=1e6)
    print(f"Phi at axis (r=1,th=pi/2,ph=0):  {ext.Phi([0,1,1.5708,0]):.3e}")
    print(f"Phi at off (r=1,th=pi/2,ph=pi/2): {ext.Phi([0,1,1.5708,1.5708]):.3e}")
    print(f"Phi at far (r=1e8,th=pi/2,ph=pi/2): "
          f"{ext.Phi([0,1e8,1.5708,1.5708]):.3e} (windowed: ~0)")
