"""
===============================================================================
Binary Singular Isothermal Ellipsoid (BinarySIE): weak-field superposition of
two SIE potentials centered at arbitrary Cartesian positions.
===============================================================================
Line element (isotropic, same as SIE):

    ds^2 = -A^2 dt^2 + B^2 (dr^2 + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2)
    A = 1 + Phi_total,   B = 1 - Phi_total
    Phi_total(x,y,z) = Phi_1(x,y,z) + Phi_2(x,y,z)

with, for i = 1, 2,

    Phi_i = 2 sigma_v_i^2 log(xi_i / r_ref)
    xi_i  = sqrt((x - x_ci)^2 + (y - y_ci)^2 + (z - z_ci)^2 / q_ax_i^2)

- The potential is evaluated by converting the spherical coordinates of the
  photon to Cartesian, offsetting to each center, computing the gradient
  analytically in Cartesian, and chain-ruling back to (r, theta, phi).
- Because the two offset potentials break axial symmetry, k_phi is NOT
  conserved: out[7] in the geodesic RHS is nonzero (unlike SIE/SIS).
- k_t is still conserved (static spacetime), out[4] = 0.
- Units: G = c = 1.
===============================================================================
"""
from math import sin, cos, log, sqrt
import numpy as np
from numba import njit


@njit(cache=True)
def _phi_and_grad_nb(r, th, ph, sv2_arr, q_ax_arr, centers, r_ref, r_min_c):
    """Return (Phi_total, dPhi/dr, dPhi/dth, dPhi/dph) at spherical (r,th,ph).

    Sums two SIE potentials whose Cartesian centers are rows of `centers`
    (shape (2,3)). r_min_c clamps xi_i from below to avoid divergence near a
    lens center.
    """
    sin_th = sin(th); cos_th = cos(th)
    sin_ph = sin(ph); cos_ph = cos(ph)

    # Global Cartesian position of the photon.
    x = r * sin_th * cos_ph
    y = r * sin_th * sin_ph
    z = r * cos_th

    Phi = 0.0
    dPhi_dx = 0.0
    dPhi_dy = 0.0
    dPhi_dz = 0.0
    for i in range(2):
        sv2 = sv2_arr[i]
        q   = q_ax_arr[i]
        q2  = q * q
        dx  = x - centers[i, 0]
        dy  = y - centers[i, 1]
        dz  = z - centers[i, 2]
        xi2 = dx*dx + dy*dy + dz*dz / q2
        # Guard near center-i singularity.
        if xi2 < r_min_c * r_min_c:
            xi2 = r_min_c * r_min_c
        Phi += sv2 * log(xi2 / (r_ref * r_ref))     # = 2 sv2 log(xi/r_ref)
        two_sv2_over_xi2 = 2.0 * sv2 / xi2
        dPhi_dx += two_sv2_over_xi2 * dx
        dPhi_dy += two_sv2_over_xi2 * dy
        dPhi_dz += two_sv2_over_xi2 * dz / q2

    # Chain rule spherical partials.
    # x = r sinθ cosφ, y = r sinθ sinφ, z = r cosθ
    dPhi_dr  = ( dPhi_dx * sin_th * cos_ph
                +dPhi_dy * sin_th * sin_ph
                +dPhi_dz * cos_th)
    dPhi_dth = ( dPhi_dx * r * cos_th * cos_ph
                +dPhi_dy * r * cos_th * sin_ph
                -dPhi_dz * r * sin_th)
    dPhi_dph = (-dPhi_dx * r * sin_th * sin_ph
                +dPhi_dy * r * sin_th * cos_ph)
    return Phi, dPhi_dr, dPhi_dth, dPhi_dph, sin_th, cos_th


@njit(cache=True)
def _geodesics_nb_array(q, sv2_arr, q_ax_arr, centers, r_ref, r_min_c):
    """BinarySIE null-geodesic RHS in Hamiltonian form. q = [t, r, th, ph,
    k_t, k_r, k_th, k_phi]. Non-axisymmetric: out[7] != 0 in general."""
    r  = q[1]; th = q[2]; ph = q[3]
    kt = q[4]; kr = q[5]; kth = q[6]; kph = q[7]

    Phi, dPhi_dr, dPhi_dth, dPhi_dph, sin_th, cos_th = _phi_and_grad_nb(
        r, th, ph, sv2_arr, q_ax_arr, centers, r_ref, r_min_c)

    sin_th2 = sin_th * sin_th
    sin_th3 = sin_th2 * sin_th

    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A;    B2 = B * B
    A3 = A2 * A;   B3 = B2 * B
    r2 = r * r;    r3 = r2 * r

    dA_dr  =  dPhi_dr;   dB_dr  = -dPhi_dr
    dA_dth =  dPhi_dth;  dB_dth = -dPhi_dth
    dA_dph =  dPhi_dph;  dB_dph = -dPhi_dph

    # Radial derivatives of g^{μν} (identical form to SIE).
    dgtt_dr   =  2.0 * dA_dr / A3
    dgrr_dr   = -2.0 * dB_dr / B3
    dgthth_dr = -2.0 * (dB_dr * r + B) / (B3 * r3)
    dgphph_dr = dgthth_dr / sin_th2

    # Angular-theta derivatives (same form as SIE).
    dgtt_dth   =  2.0 * dA_dth / A3
    dgrr_dth   = -2.0 * dB_dth / B3
    dgthth_dth = -2.0 * dB_dth / (B3 * r2)
    dgphph_dth = -2.0 * (dB_dth * sin_th + B * cos_th) / (B3 * r2 * sin_th3)

    # Angular-phi derivatives (NEW; zero for axisymmetric SIE).
    dgtt_dph   =  2.0 * dA_dph / A3
    dgrr_dph   = -2.0 * dB_dph / B3
    dgthth_dph = -2.0 * dB_dph / (B3 * r2)
    dgphph_dph = dgthth_dph / sin_th2

    out = np.empty(8)
    # Positions: dx^mu/dlambda = g^{mu nu} k_nu
    out[0] = -kt / A2
    out[1] =  kr / B2
    out[2] =  kth / (B2 * r2)
    out[3] =  kph / (B2 * r2 * sin_th2)
    # k_t conserved (static spacetime); k_phi is NOT.
    out[4] = 0.0
    # dk_r/dlambda
    out[5] = -0.5 * (dgtt_dr   * kt*kt
                     + dgrr_dr   * kr*kr
                     + dgthth_dr * kth*kth
                     + dgphph_dr * kph*kph)
    # dk_theta/dlambda
    out[6] = -0.5 * (dgtt_dth   * kt*kt
                     + dgrr_dth   * kr*kr
                     + dgthth_dth * kth*kth
                     + dgphph_dth * kph*kph)
    # dk_phi/dlambda — NEW term, vanishes only if dPhi/dph = 0 everywhere.
    out[7] = -0.5 * (dgtt_dph   * kt*kt
                     + dgrr_dph   * kr*kr
                     + dgthth_dph * kth*kth
                     + dgphph_dph * kph*kph)
    return out


@njit(cache=True)
def _metric_nb(x, sv2_arr, q_ax_arr, centers, r_ref, r_min_c):
    r = x[1]; th = x[2]; ph = x[3]
    Phi, _, _, _, sin_th, _ = _phi_and_grad_nb(
        r, th, ph, sv2_arr, q_ax_arr, centers, r_ref, r_min_c)
    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A;    B2 = B * B
    r2 = r * r
    return (-A2, B2, B2 * r2, B2 * r2 * sin_th * sin_th, 0.0)


@njit(cache=True)
def _null_omega_nb(r):
    return 0.0


class LensMetric:
    """Binary SIE lens: superposition of two SIE potentials with offset
    Cartesian centers.

    Parameters
    ----------
    sigma_v : tuple of 2 floats
        Velocity dispersions (units of c) of the two lenses.
    centers : tuple of 2 triplets
        Cartesian coordinates (x, y, z) of each lens center, in geometrized
        M. For a horizontal "two-eye" configuration in the image plane put
        both centers on the y-axis at z = 0: ((0, +d, 0), (0, -d, 0)).
    q_ax : tuple of 2 floats, optional
        Axis ratios of each lens (flattening along z). q=1 recovers a
        spherical SIS for that component.
    r_ref : float, optional
        Reference radius for the logarithmic potential (same for both).
    r_min : float, optional
        Global numerical cutoff for the integrator horizon event (on the
        global r). Analogous to sie.LensMetric.r_min.
    r_min_center : float, optional
        Per-center xi_i clamp to avoid divergence when a photon passes very
        close to one of the lens centers. Default matches r_min.
    """

    def __init__(self, sigma_v=(0.02, 0.02),
                 centers=((0.0, +1.0, 0.0), (0.0, -1.0, 0.0)),
                 q_ax=(0.8, 0.8),
                 r_ref=1.0, r_min=1e-2, r_min_center=None):
        sv = np.asarray(sigma_v, dtype=np.float64)
        qx = np.asarray(q_ax, dtype=np.float64)
        ct = np.asarray(centers, dtype=np.float64)
        if sv.shape != (2,) or qx.shape != (2,) or ct.shape != (2, 3):
            raise ValueError(
                "sigma_v, q_ax must be length-2 and centers shape (2,3).")

        self.sigma_v = sv
        self.q_ax = qx
        self.centers = ct
        self._sv2 = sv * sv
        self._r_ref = float(r_ref)
        self._r_min = float(r_min)
        self._r_min_c = float(r_min if r_min_center is None else r_min_center)

        # Duck-type compatibility with the integrator and BlackHole API.
        self.a = 0.0
        self.EH = self._r_min
        self.ISCOco = 0.0
        self.ISCOcounter = 0.0

        _sv2_arr = self._sv2.copy()
        _qx_arr  = self.q_ax.copy()
        _ct_arr  = self.centers.copy()
        _rr   = float(self._r_ref)
        _rmc  = float(self._r_min_c)

        @njit
        def _rhs(q):
            return _geodesics_nb_array(q, _sv2_arr, _qx_arr, _ct_arr,
                                       _rr, _rmc)

        @njit
        def _metric(x):
            return _metric_nb(x, _sv2_arr, _qx_arr, _ct_arr, _rr, _rmc)

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _null_omega_nb

    # -- Python-level API ----------------------------------------------------
    def metric(self, x):
        g = _metric_nb(np.asarray(x, dtype=np.float64),
                       self._sv2, self.q_ax, self.centers,
                       self._r_ref, self._r_min_c)
        return [g[0], g[1], g[2], g[3], g[4]]

    def inverse_metric(self, x):
        r = x[1]; th = x[2]; ph = x[3]
        Phi, _, _, _, sin_th, _ = _phi_and_grad_nb(
            r, th, ph, self._sv2, self.q_ax, self.centers,
            self._r_ref, self._r_min_c)
        A = 1.0 + Phi; B = 1.0 - Phi
        A2 = A * A;    B2 = B * B
        r2 = r * r
        return [-1.0 / A2, 1.0 / B2, 1.0 / (B2 * r2),
                1.0 / (B2 * r2 * sin_th * sin_th), 0.0]

    def geodesics(self, q, lmbda):
        return list(_geodesics_nb_array(
            np.asarray(q, dtype=np.float64),
            self._sv2, self.q_ax, self.centers,
            self._r_ref, self._r_min_c))

    def Omega(self, r, corotating=True):
        return 0.0


###############################################################################

if __name__ == '__main__':
    print("BinarySIE module: scr.lens_metrics.binary_sie")
