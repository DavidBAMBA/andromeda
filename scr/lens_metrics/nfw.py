"""
===============================================================================
Navarro-Frenk-White (NFW) halo lens (weak-field, isotropic coordinates)
===============================================================================
Cold-dark-matter halo profile with smooth core and r^-3 outer wings:

    rho(r) = rho_s / [(r/r_s) * (1 + r/r_s)^2]

Newtonian potential (G = c = 1, in units where the unit length is r_g of the
fiducial M_lens used by SceneGeometry):

    Phi(r) = -(M_s / r_s) * ln(1 + x) / x ,    x = r / r_s
    M_s    = 4 pi rho_s r_s^3       (characteristic mass scale)

Derivative used by the geodesic RHS:

    dPhi/dr = -(M_s / r_s^2) * [ 1/((1+x)*x) - ln(1+x)/x^2 ]

Both expressions stay finite as x -> 0 (Phi -> -M_s/r_s, gradient -> -M_s/(2 r_s^2)).

Same isotropic line element as SIS:

    ds^2 = -A(r)^2 dt^2 + B(r)^2 (dr^2 + r^2 dtheta^2 + r^2 sin^2 theta dphi^2)
    A = 1 + Phi,  B = 1 - Phi.

Valid in the weak-field regime |Phi| << 1. NFW is numerically benign
(no horizon, finite gradient at the origin); we still expose a small r_min
cutoff as self.EH so the integrator's horizon event terminates plunging
photons.

Usage from physical halo parameters
-----------------------------------
    M_200, c_NFW -> r_200, r_s, M_s with the standard Wright-Brainerd
    relations (left to the caller; see helpers in scr.common.cosmology).

Units: G = c = 1; M_s and r_s in geometrized M_lens units.
===============================================================================
"""

from math import sin, cos, log, sqrt
import numpy as np
from numba import njit


@njit(cache=True)
def _phi_nfw_nb(r, M_s, r_s):
    """NFW Newtonian potential Phi(r) = -(M_s/r_s) * ln(1 + r/r_s) / (r/r_s)."""
    x = r / r_s
    if x < 1.0e-8:
        # Series expansion: ln(1+x)/x ~ 1 - x/2 + x^2/3 ... avoids 0/0.
        return -(M_s / r_s) * (1.0 - 0.5*x + (1.0/3.0)*x*x)
    return -(M_s / r_s) * log(1.0 + x) / x


@njit(cache=True)
def _dphi_nfw_nb(r, M_s, r_s):
    """dPhi/dr for the spherical NFW potential."""
    x = r / r_s
    if x < 1.0e-8:
        # Series: 1/((1+x)x) - ln(1+x)/x^2 -> 1/2 - 2x/3 + 3x^2/4 ...
        bracket = 0.5 - (2.0/3.0)*x + (3.0/4.0)*x*x
    else:
        bracket = 1.0 / ((1.0 + x) * x) - log(1.0 + x) / (x * x)
    return -(M_s / (r_s * r_s)) * bracket


@njit(cache=True)
def _aux_nb(r, th, M_s, r_s, q_ax):
    """Returns (Phi, dPhi/dr, dPhi/dth, sin_th, cos_th) for the elliptical NFW.

    Pseudo-elliptical model (Golse & Kneib 2002): replace r by an
    ellipsoidal radius xi = r * f(theta) where
    f(theta) = sqrt(sin^2 + cos^2/q^2). q_ax = 1 reduces to spherical.
    """
    sin_th = sin(th); cos_th = cos(th)
    q2 = q_ax * q_ax
    f2 = sin_th*sin_th + cos_th*cos_th / q2
    f = sqrt(f2)
    xi = r * f
    Phi = _phi_nfw_nb(xi, M_s, r_s)
    dPhi_dxi = _dphi_nfw_nb(xi, M_s, r_s)
    dPhi_dr = dPhi_dxi * f
    H = sin_th * cos_th * (1.0 - 1.0 / q2) / f2
    dPhi_dth = dPhi_dxi * xi * H
    return Phi, dPhi_dr, dPhi_dth, sin_th, cos_th


@njit(cache=True)
def _geodesics_nb_spherical(q, M_s, r_s):
    """Spherical NFW geodesic RHS (q_ax = 1)."""
    r = q[1]
    th = q[2]
    sin_th = sin(th)
    cos_th = cos(th)
    sin_th2 = sin_th * sin_th
    sin_th3 = sin_th2 * sin_th

    Phi = _phi_nfw_nb(r, M_s, r_s)
    A = 1.0 + Phi
    B = 1.0 - Phi
    A2 = A * A
    B2 = B * B
    A3 = A2 * A
    B3 = B2 * B
    r2 = r * r
    r3 = r2 * r

    Aprime = _dphi_nfw_nb(r, M_s, r_s)   # dPhi/dr
    Bprime = -Aprime                     # -dPhi/dr
    BrB = Bprime * r + B

    kt = q[4]; kr = q[5]; kth = q[6]; kph = q[7]

    out = np.empty(8)
    out[0] = -kt / A2
    out[1] = kr / B2
    out[2] = kth / (B2 * r2)
    out[3] = kph / (B2 * r2 * sin_th2)

    out[4] = 0.0
    out[7] = 0.0

    out[5] = (-(Aprime / A3) * kt * kt
              + (Bprime / B3) * kr * kr
              + (BrB / (B3 * r3)) * kth * kth
              + (BrB / (B3 * r3 * sin_th2)) * kph * kph)

    out[6] = (cos_th / (B2 * r2 * sin_th3)) * kph * kph
    return out


@njit(cache=True)
def _geodesics_nb_elliptical(q, M_s, r_s, q_ax):
    """Pseudo-elliptical NFW geodesic RHS (q_ax < 1)."""
    r = q[1]; th = q[2]
    kt = q[4]; kr = q[5]; kth = q[6]; kph = q[7]

    Phi, dPhi_dr, dPhi_dth, sin_th, cos_th = _aux_nb(
        r, th, M_s, r_s, q_ax)
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
def _metric_nb(x, M_s, r_s):
    """Covariant metric components (g_tt, g_rr, g_thth, g_phph, g_tph)."""
    r = x[1]
    th = x[2]
    Phi = _phi_nfw_nb(r, M_s, r_s)
    A = 1.0 + Phi
    B = 1.0 - Phi
    A2 = A * A
    B2 = B * B
    r2 = r * r
    sin_th2 = sin(th) ** 2
    return -A2, B2, B2 * r2, B2 * r2 * sin_th2, 0.0


@njit(cache=True)
def _metric_nb_ell(x, M_s, r_s, q_ax):
    """Elliptical NFW covariant metric (xi = r f(theta))."""
    r = x[1]; th = x[2]
    Phi, _, _, sin_th, _ = _aux_nb(r, th, M_s, r_s, q_ax)
    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    r2 = r * r
    return (-A2, B2, B2 * r2, B2 * r2 * sin_th * sin_th, 0.0)


@njit(cache=True)
def _null_omega_nb(r):
    return 0.0


# Backward-compatible alias for the spherical RHS (used by existing tests).
_geodesics_nb_array = _geodesics_nb_spherical


class LensMetric:
    """NFW halo lens, weak-field isotropic. Optional pseudo-elliptical
    extension via q_ax < 1 (Golse & Kneib 2002).

    Parameters
    ----------
    M_s : float
        Characteristic NFW mass M_s = 4 pi rho_s r_s^3, in geometrized
        M_lens units.
    r_s : float
        NFW scale radius in geometrized M_lens units.
    q_ax : float, optional (default 1.0 = spherical)
        Axis ratio in (0, 1]. Replaces r by xi = r * sqrt(sin^2 +
        cos^2/q^2) inside the NFW potential. Typical cluster halos:
        q_ax ~ 0.5-0.8 (substantial triaxiality).
    r_min : float, optional
        Numerical horizon cutoff (default 1e-2 r_s).

    Notes
    -----
    Validity: |Phi(r_min)| << 1. Choose M_s/r_s such that this holds
    (typically M_s/r_s < 1e-2 in geometrized units). The pseudo-elliptical
    pseudo-potential does not correspond to the projection of an exact
    triaxial NFW; use it for visualization rather than precise fits.
    """

    def __init__(self, M_s, r_s, q_ax=1.0, r_min=None):
        if not (0.0 < q_ax <= 1.0 + 1e-9):
            raise ValueError(f"q_ax must be in (0, 1], got {q_ax}")
        self.M_s = float(M_s)
        self.r_s = float(r_s)
        self.q_ax = float(q_ax)
        self._r_min = float(r_min) if r_min is not None else 1.0e-2 * self.r_s

        self.a = 0.0
        self.EH = self._r_min
        self.ISCOco = 0.0
        self.ISCOcounter = 0.0

        _Ms = float(self.M_s)
        _rs = float(self.r_s)
        _q  = float(self.q_ax)

        if abs(_q - 1.0) < 1.0e-9:
            @njit
            def _rhs(q):
                return _geodesics_nb_spherical(q, _Ms, _rs)

            @njit
            def _metric(x):
                return _metric_nb(x, _Ms, _rs)
        else:
            @njit
            def _rhs(q):
                return _geodesics_nb_elliptical(q, _Ms, _rs, _q)

            @njit
            def _metric(x):
                return _metric_nb_ell(x, _Ms, _rs, _q)

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _null_omega_nb

    # -- Python-level API ---------------------------------------------------
    def Phi(self, r):
        # Spherical convenience: takes r as a float; for the elliptical
        # case use Phi_xy or pass a 4-vector (x[1] = r, x[2] = theta).
        return _phi_nfw_nb(float(r), self.M_s, self.r_s)

    def dPhi_dr(self, r):
        return _dphi_nfw_nb(float(r), self.M_s, self.r_s)

    def metric(self, x):
        if abs(self.q_ax - 1.0) < 1e-9:
            g = _metric_nb(np.asarray(x, dtype=np.float64),
                            self.M_s, self.r_s)
        else:
            g = _metric_nb_ell(np.asarray(x, dtype=np.float64),
                                self.M_s, self.r_s, self.q_ax)
        return [g[0], g[1], g[2], g[3], g[4]]

    def inverse_metric(self, x):
        r = x[1]; th = x[2]
        if abs(self.q_ax - 1.0) < 1e-9:
            Phi = _phi_nfw_nb(float(r), self.M_s, self.r_s)
            sin_th = sin(th)
        else:
            Phi, _, _, sin_th, _ = _aux_nb(r, th, self.M_s, self.r_s,
                                            self.q_ax)
        A = 1.0 + Phi
        B = 1.0 - Phi
        A2 = A * A
        B2 = B * B
        r2 = r * r
        sin_th2 = sin_th * sin_th
        return [-1.0 / A2, 1.0 / B2, 1.0 / (B2 * r2),
                1.0 / (B2 * r2 * sin_th2), 0.0]

    def geodesics(self, q, lmbda):
        if abs(self.q_ax - 1.0) < 1e-9:
            return list(_geodesics_nb_spherical(
                np.asarray(q, dtype=np.float64), self.M_s, self.r_s))
        return list(_geodesics_nb_elliptical(
            np.asarray(q, dtype=np.float64),
            self.M_s, self.r_s, self.q_ax))

    def Omega(self, r, corotating=True):
        return 0.0

    def deflection_angle(self, b):
        """Thin-lens analytical deflection at impact parameter b (radians).

        Bartelmann 1996 / Wright & Brainerd 2000 closed form:

            alpha(b) = (4 G M_s / c^2 b) * h(b/r_s)

        with

            h(x) = ln(x/2) + F(x)

                       { arccosh(1/x) / sqrt(1 - x^2)     x < 1
            F(x)    =  { 1                                 x = 1
                       { arccos (1/x) / sqrt(x^2 - 1)     x > 1

        c = 1, M_s in geometrized M_lens units. Used by test_nfw.py and
        ex20 to validate the ray-traced deflection.
        """
        from math import sqrt, log, acos, acosh
        x = b / self.r_s
        if abs(x - 1.0) < 1.0e-6:
            F = 1.0
        elif x < 1.0:
            F = acosh(1.0 / x) / sqrt(1.0 - x*x)
        else:
            F = acos(1.0 / x) / sqrt(x*x - 1.0)
        h = log(0.5 * x) + F
        return (4.0 * self.M_s / b) * h


###############################################################################

if __name__ == '__main__':
    # Galaxy-cluster-scale halo: r_s ~ 1e6 M_lens (~ 30 kpc for the
    # M_lens chosen in ex20), M_s tuned so |Phi(r_s)| ~ 1e-5 (weak field).
    nfw = LensMetric(M_s=2.0, r_s=1.0e6)
    print(f"NFW: M_s={nfw.M_s:.2e}, r_s={nfw.r_s:.2e}")
    print(f"Phi(r_s) = {nfw.Phi(nfw.r_s):.3e}")
    print(f"dPhi/dr at r_s = {nfw.dPhi_dr(nfw.r_s):.3e}")
    print(f"Deflection at b=2 r_s = {nfw.deflection_angle(2.0*nfw.r_s):.3e} rad")
