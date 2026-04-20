"""
===============================================================================
Rotating MOG black hole with optional quintessence (Kerr-MOG+Quintessence)
===============================================================================
Boyer-Lindquist-like line element (eq. 46 of the accompanying note):

    ds^2 = -(1 - 2M(r) r / Sigma) dt^2
           - (4 a M(r) r sin^2(theta) / Sigma) dt dphi
           + (Sigma / Delta) dr^2
           + Sigma dtheta^2
           + (r^2 + a^2 + 2 M(r) r a^2 sin^2(theta) / Sigma) sin^2(theta) dphi^2

with

    Sigma = r^2 + a^2 cos^2(theta)
    Delta = r^2 - 2 M(r) r + a^2
    2 M(r) = 2(1+alpha) M r^3 / (r^2 + K)^(3/2)
             - K r^3 / (r^2 + K)^2
             + c(1+alpha) / r^(3 w_q)
    K = alpha (1 + alpha) M^2

- Event horizon at the outermost root of Delta(r) = 0 (found via brentq).
- ISCO: Kerr analytic formula (exact at alpha = c = 0, a small deviation
  otherwise — override with R_min in the thin_disk if needed).
- Units: G = c_light = 1, M = 1.
- alpha = 0 recovers Kerr exactly; c = 0 removes the quintessence term.
===============================================================================
@author: Benavides-Gallego, Bambague, Larrañaga (2026) — implementation
         based on the note `scr/black_holes/kerr_mog.pdf`.
===============================================================================
"""

from math import sin, cos, sqrt
import numpy as np
from numba import njit
from scipy.optimize import brentq


# ----------------------------------------------------------------------------
# Generalized mass function and its first derivative (Numba hot path).
# M = 1 is absorbed into the prefactors. K = alpha(1+alpha).
# ----------------------------------------------------------------------------
@njit(cache=True)
def _M_nb(r, alpha, K, c, w_q):
    """Mass function M(r) in M=1 units (eq. 60 of the note, divided by 2)."""
    r2 = r * r
    r3 = r2 * r
    s = r2 + K
    sqrt_s = sqrt(s)
    s_1p5 = s * sqrt_s
    term_mog  = (1.0 + alpha) * r3 / s_1p5
    term_mog2 = -0.5 * K * r3 / (s * s)
    if c != 0.0:
        term_q = 0.5 * c * (1.0 + alpha) / (r ** (3.0 * w_q))
    else:
        term_q = 0.0
    return term_mog + term_mog2 + term_q


@njit(cache=True)
def _dMdr_nb(r, alpha, K, c, w_q):
    """First radial derivative of M(r) (eq. 63 of the note, divided by 2)."""
    r2 = r * r
    r3 = r2 * r
    r4 = r3 * r
    s = r2 + K
    s2 = s * s
    sqrt_s = sqrt(s)
    s_2p5 = s2 * sqrt_s
    d_mog  = (1.0 + alpha) * 3.0 * K * r2 / s_2p5
    d_mog2 = -0.5 * K * (3.0 * K * r2 - r4) / (s2 * s)
    if c != 0.0:
        d_q = -0.5 * 3.0 * w_q * c * (1.0 + alpha) / (r ** (3.0 * w_q + 1.0))
    else:
        d_q = 0.0
    return d_mog + d_mog2 + d_q


# ----------------------------------------------------------------------------
# Geodesic RHS in Hamiltonian form. Generalizes kerr.py (which is this same
# structure at M=1, dM/dr=0). Only Delta and its r-derivative change.
# ----------------------------------------------------------------------------
@njit(cache=True)
def _geodesics_nb_array(q, a, alpha, K, c, w_q):
    r = q[1]
    r2 = r * r
    a2 = a * a
    sin_th = sin(q[2])
    cos_th = cos(q[2])
    sin_th2 = sin_th * sin_th
    cos_th2 = cos_th * cos_th
    Sigma = r2 + a2 * cos_th2
    Sigma2 = Sigma * Sigma

    M_r  = _M_nb(r, alpha, K, c, w_q)
    dMdr = _dMdr_nb(r, alpha, K, c, w_q)
    Delta = r2 - 2.0 * M_r * r + a2
    # d(Delta)/dr = 2r - 2 M_r - 2 r M'(r).  half_dDelta = (d Delta / dr) / 2.
    half_dDelta = r - M_r - r * dMdr

    W = -q[4] * (r2 + a2) - a * q[7]
    partXi = r2 + (q[7] + a*q[4])**2 \
             + a2 * (1.0 + q[4]*q[4]) * cos_th2 \
             + q[7]*q[7] * cos_th2 / sin_th2
    Xi = W * W - Delta * partXi

    dXidE = 2.0 * W * (r2 + a2) + 2.0 * a * Delta * (q[7] + a * q[4] * sin_th2)
    dXidL = -2.0 * a * W - 2.0 * a * q[4] * Delta - 2.0 * q[7] * Delta / sin_th2
    # dXi/dr = 2 W dW/dr  -  (dDelta/dr) partXi  -  Delta (d partXi / dr)
    # with dW/dr = -2 r q[4]  and  d partXi / dr = 2 r  (only the r^2 term).
    dXidr = -4.0 * r * q[4] * W - 2.0 * half_dDelta * partXi - 2.0 * r * Delta

    dAdr = half_dDelta / Sigma - (r * Delta) / Sigma2
    dBdr = -r / Sigma2
    dCdr = (dXidr / (2.0 * Delta * Sigma)
            - (Xi * half_dDelta) / (Sigma * Delta * Delta)
            - r * Xi / (Delta * Sigma2))

    auxth = a2 * cos_th * sin_th
    dAdth = Delta * auxth / Sigma2
    dBdth = auxth / Sigma2
    dCdth = (((1.0 + q[4]*q[4]) * auxth
              + q[7]*q[7] * cos_th / (sin_th2 * sin_th)) / Sigma
             + (Xi / (Delta * Sigma2)) * auxth)

    out = np.empty(8)
    out[0] = dXidE / (2.0 * Delta * Sigma)
    out[1] = (Delta / Sigma) * q[5]
    out[2] = q[6] / Sigma
    out[3] = -dXidL / (2.0 * Delta * Sigma)
    out[4] = 0.0
    out[5] = -dAdr * q[5] * q[5] - dBdr * q[6] * q[6] + dCdr
    out[6] = -dAdth * q[5] * q[5] - dBdth * q[6] * q[6] + dCdth
    out[7] = 0.0
    return out


@njit(cache=True)
def _metric_nb(x, a, alpha, K, c, w_q):
    """Covariant metric components (g_tt, g_rr, g_thth, g_phph, g_tph)."""
    r = x[1]
    r2 = r * r
    a2 = a * a
    sin_theta2 = sin(x[2]) ** 2
    cos_theta2 = cos(x[2]) ** 2
    M_r = _M_nb(r, alpha, K, c, w_q)
    Delta = r2 - 2.0 * M_r * r + a2
    Sigma = r2 + a2 * cos_theta2
    g_tt = -(1.0 - 2.0 * M_r * r / Sigma)
    g_rr = Sigma / Delta
    g_thth = Sigma
    g_phph = (r2 + a2 + 2.0 * M_r * r * a2 * sin_theta2 / Sigma) * sin_theta2
    g_tph = -2.0 * a * M_r * r * sin_theta2 / Sigma
    return g_tt, g_rr, g_thth, g_phph, g_tph


@njit(cache=True)
def _omega_nb(r, a):
    """Kerr-like approximation for the orbital frequency at radius r.

    The exact Kerr-MOG expression requires solving for circular orbits with
    M(r); for small alpha and c this Kerr form is within a few percent. For
    no-Doppler renders it is unused. Prograde corotating convention.
    """
    return 1.0 / (r ** 1.5 + a)


def _Delta_py(r, a, alpha, K, c, w_q):
    """Pure-Python Delta(r); for brentq in __init__ and for external scripts."""
    r2 = r * r
    s = r2 + K
    m_r = ((1.0 + alpha) * r ** 3 / (s ** 1.5)
           - 0.5 * K * r ** 3 / (s * s))
    if c != 0.0:
        m_r += 0.5 * c * (1.0 + alpha) / (r ** (3.0 * w_q))
    return r2 - 2.0 * m_r * r + a * a


# ----------------------------------------------------------------------------
# Public class.
# ----------------------------------------------------------------------------
class BlackHole:
    """Rotating MOG black hole with optional quintessence.

    Parameters
    ----------
    a : float
        Spin parameter in units of M = 1. Must satisfy |a| < a_E(alpha, c).
    alpha : float, default 0.0
        MOG parameter. alpha = 0 recovers Kerr exactly.
    c : float, default 0.0
        Quintessence normalization. c = 0 removes the quintessence term.
    w_q : float, default -2/3
        Quintessence equation-of-state parameter (typically -1 < w_q < -1/3).
        Irrelevant when c = 0.
    """

    def __init__(self, a, alpha=0.0, c=0.0, w_q=-2.0/3.0):
        self.a = a
        self.alpha = alpha
        self.c = c
        self.w_q = w_q
        self.M = 1.0
        self.K = alpha * (1.0 + alpha)  # K = alpha (1+alpha) M^2, M=1

        # Outer event horizon: largest root of Delta(r) = 0 in a reasonable
        # bracket. For alpha = c = 0 this matches Kerr's analytic r_+.
        self.EH = self._find_event_horizon()

        # ISCO: Kerr analytic formula. Exact at alpha = c = 0; small error
        # otherwise. Users can override the disk in_edge via R_min.
        if a * a < 1.0:
            Z1 = 1.0 + (1.0 - a*a)**(1.0/3.0) \
                 * ((1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0))
            Z2 = sqrt(3.0 * a*a + Z1*Z1)
            self.ISCOco      = 3.0 + Z2 - sqrt((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))
            self.ISCOcounter = 3.0 + Z2 + sqrt((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))
        else:
            self.ISCOco = self.ISCOcounter = self.EH

        # Numba hot-path hooks — closures specialize on the BH parameters
        # so the generic kernels (_solve_photon_nb, _render_image_nb) can
        # consume them as first-class functions.
        _a = float(a)
        _alpha = float(alpha)
        _K = float(self.K)
        _c = float(c)
        _w = float(w_q)

        @njit
        def _rhs(q):
            return _geodesics_nb_array(q, _a, _alpha, _K, _c, _w)

        @njit
        def _metric(x):
            return _metric_nb(x, _a, _alpha, _K, _c, _w)

        @njit
        def _omega(r):
            return _omega_nb(r, _a)

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _omega

    # -- horizon / Delta ----------------------------------------------------
    def _Delta(self, r):
        """Delta(r) = r^2 - 2 M(r) r + a^2 (for root-finding)."""
        return _Delta_py(r, self.a, self.alpha, self.K, self.c, self.w_q)

    def _find_event_horizon(self):
        """Outer event horizon: the largest r for which Delta(r) crosses
        from negative to positive as r increases, while staying below any
        cosmological horizon (Delta can have a second positive->negative
        crossing at very large r when c > 0)."""
        a2 = self.a * self.a
        # Kerr analytic upper bound, stretched for MOG.
        r_kerr = 1.0 + sqrt(max(1.0 - a2, 0.0))
        r_max = max(10.0, 4.0 * r_kerr * (1.0 + abs(self.alpha)) + 2.0)
        # Sample Delta and look for the rightmost neg->pos transition in the
        # "inner" region (stop as soon as Delta stays positive for a while —
        # beyond that any further sign change would be the cosmological
        # horizon, which is not what we want here).
        n = 2000
        rs = np.linspace(1e-3, r_max, n)
        deltas = np.array([self._Delta(r) for r in rs])
        r_plus = None
        for i in range(1, n):
            if deltas[i-1] < 0.0 and deltas[i] >= 0.0:
                try:
                    r_plus = brentq(self._Delta, rs[i-1], rs[i], xtol=1e-10)
                except (ValueError, RuntimeError):
                    pass
        if r_plus is not None:
            return r_plus
        # Fallback: Kerr-like estimate (valid in the alpha, c -> 0 limit).
        return r_kerr

    # -- Python-level API (used by scipy fallback paths and by verifiers) ---
    def Omega(self, r, corotating=True):
        """Kerr-like orbital frequency (approximation, see _omega_nb)."""
        if corotating:
            return 1.0 / (r ** 1.5 + self.a)
        else:
            return -1.0 / (r ** 1.5 - self.a)

    def metric(self, x):
        g_tt, g_rr, g_thth, g_phph, g_tph = _metric_nb(
            np.asarray(x, dtype=np.float64),
            self.a, self.alpha, self.K, self.c, self.w_q)
        return [g_tt, g_rr, g_thth, g_phph, g_tph]

    def inverse_metric(self, x):
        r = x[1]; r2 = r * r; a2 = self.a * self.a
        sin2 = sin(x[2]) ** 2
        s = r2 + self.K
        m_r = ((1.0 + self.alpha) * r ** 3 / (s ** 1.5)
               - 0.5 * self.K * r ** 3 / (s * s))
        if self.c != 0.0:
            m_r += 0.5 * self.c * (1.0 + self.alpha) / (r ** (3.0 * self.w_q))
        Delta = r2 - 2.0 * m_r * r + a2
        Sigma = r2 + a2 * cos(x[2]) ** 2
        A = (r2 + a2) ** 2 - Delta * a2 * sin2
        gtt = -A / (Delta * Sigma)
        grr = Delta / Sigma
        gthth = 1.0 / Sigma
        gphph = (Delta - a2 * sin2) / (Delta * Sigma * sin2)
        gtph = -2.0 * self.a * m_r * r / (Delta * Sigma)
        return [gtt, grr, gthth, gphph, gtph]

    def geodesics(self, q, lmbda):
        return list(_geodesics_nb_array(
            np.asarray(q, dtype=np.float64),
            self.a, self.alpha, self.K, self.c, self.w_q))


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')
