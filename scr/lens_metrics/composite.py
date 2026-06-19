"""
===============================================================================
Composite weak-field lens: galaxy + external convergence/shear
===============================================================================
Adds an external (kappa, gamma_+, gamma_x) tidal field on top of an
axisymmetric base lens (PEMD, SIE, SIS, NFW). In the weak-field limit the
gravitational potentials add:

    Phi_total(r, theta, phi) = Phi_base(r, theta) + Phi_ext(r, theta, phi)

The resulting space-time is non-axisymmetric (k_phi is not conserved); we
therefore use the generic non-axisymmetric RHS implemented in
scr.lens_metrics.external (`_geodesics_nonaxi_nb`).

Currently the wired bases are:
    * 'pemd'  -> scr.lens_metrics.pemd._aux_nb         (gives Phi, dr, dth)
    * 'sie'   -> scr.lens_metrics.sie._aux_nb
    * 'nfw'   -> scr.lens_metrics.nfw._aux_nb (uses xi-substitution; q_ax param)

For SLACS-style systems use 'pemd' or 'sie' as the galaxy base.

Usage
-----
    from scr.lens_metrics import pemd, composite
    galaxy = pemd.LensMetric(K=1e-5, gamma=2.05, q_ax=0.7)
    lens = composite.with_external(galaxy, kappa=0.02,
                                    gamma_plus=0.05, gamma_cross=0.01)
    # lens behaves like any other LensMetric — same _rhs_nb, _metric_nb hooks.
===============================================================================
"""
from math import sin
import numpy as np
from numba import njit

from scr.lens_metrics.external import (_ext_aux_nb, _geodesics_nonaxi_nb,
                                        _null_omega_nb)
from scr.lens_metrics import pemd as _pemd_mod
from scr.lens_metrics import sie as _sie_mod
from scr.lens_metrics import nfw as _nfw_mod


# ----------------------------------------------------------------------------
# Hard-wired composite RHS for each supported base.
# ----------------------------------------------------------------------------

@njit(cache=True)
def _rhs_pemd_ext(q, K, gamma, q_ax, kappa, gp, gx, r_window):
    Phi_b, dr_b, dth_b, _, _ = _pemd_mod._aux_nb(q[1], q[2], K, gamma, q_ax)
    Phi_e, dr_e, dth_e, dph_e = _ext_aux_nb(
        q[1], q[2], q[3], kappa, gp, gx, r_window)
    return _geodesics_nonaxi_nb(
        q, Phi_b + Phi_e, dr_b + dr_e, dth_b + dth_e, dph_e)


@njit(cache=True)
def _rhs_sie_ext(q, sv2, r_ref, q_ax, kappa, gp, gx, r_window):
    Phi_b, dr_b, dth_b, _, _ = _sie_mod._aux_nb(q[1], q[2], sv2, r_ref, q_ax)
    Phi_e, dr_e, dth_e, dph_e = _ext_aux_nb(
        q[1], q[2], q[3], kappa, gp, gx, r_window)
    return _geodesics_nonaxi_nb(
        q, Phi_b + Phi_e, dr_b + dr_e, dth_b + dth_e, dph_e)


@njit(cache=True)
def _rhs_nfw_ext(q, M_s, r_s, q_ax, kappa, gp, gx, r_window):
    Phi_b, dr_b, dth_b, _, _ = _nfw_mod._aux_nb(q[1], q[2], M_s, r_s, q_ax)
    Phi_e, dr_e, dth_e, dph_e = _ext_aux_nb(
        q[1], q[2], q[3], kappa, gp, gx, r_window)
    return _geodesics_nonaxi_nb(
        q, Phi_b + Phi_e, dr_b + dr_e, dth_b + dth_e, dph_e)


# ----------------------------------------------------------------------------
# Composite metric (covariant g_mu_nu) built from total Phi.
# ----------------------------------------------------------------------------

@njit(cache=True)
def _metric_from_total_phi(r, th, Phi):
    A = 1.0 + Phi; B = 1.0 - Phi
    A2 = A * A; B2 = B * B
    r2 = r * r
    sin_th2 = sin(th) ** 2
    return (-A2, B2, B2 * r2, B2 * r2 * sin_th2, 0.0)


@njit(cache=True)
def _metric_pemd_ext(x, K, gamma, q_ax, kappa, gp, gx, r_window):
    Phi_b, _, _, _, _ = _pemd_mod._aux_nb(x[1], x[2], K, gamma, q_ax)
    Phi_e, _, _, _ = _ext_aux_nb(x[1], x[2], x[3], kappa, gp, gx, r_window)
    return _metric_from_total_phi(x[1], x[2], Phi_b + Phi_e)


@njit(cache=True)
def _metric_sie_ext(x, sv2, r_ref, q_ax, kappa, gp, gx, r_window):
    Phi_b, _, _, _, _ = _sie_mod._aux_nb(x[1], x[2], sv2, r_ref, q_ax)
    Phi_e, _, _, _ = _ext_aux_nb(x[1], x[2], x[3], kappa, gp, gx, r_window)
    return _metric_from_total_phi(x[1], x[2], Phi_b + Phi_e)


@njit(cache=True)
def _metric_nfw_ext(x, M_s, r_s, q_ax, kappa, gp, gx, r_window):
    Phi_b, _, _, _, _ = _nfw_mod._aux_nb(x[1], x[2], M_s, r_s, q_ax)
    Phi_e, _, _, _ = _ext_aux_nb(x[1], x[2], x[3], kappa, gp, gx, r_window)
    return _metric_from_total_phi(x[1], x[2], Phi_b + Phi_e)


# ----------------------------------------------------------------------------
# Public class
# ----------------------------------------------------------------------------

class CompositeLensMetric:
    """Galaxy lens + external (kappa, gamma) tidal field.

    Use the helper :func:`with_external` to construct one from any
    supported base lens.
    """

    def __init__(self, base_kind, base_params, kappa, gamma_plus,
                 gamma_cross, r_window=1.0e6, r_min=1e-2):
        self.base_kind = str(base_kind)
        self.base_params = tuple(float(p) for p in base_params)
        self.kappa = float(kappa)
        self.gamma_plus = float(gamma_plus)
        self.gamma_cross = float(gamma_cross)
        self.r_window = float(r_window)
        self._r_min = float(r_min)

        self.a = 0.0
        self.EH = self._r_min
        self.ISCOco = 0.0
        self.ISCOcounter = 0.0

        bp = self.base_params
        _k  = self.kappa
        _gp = self.gamma_plus
        _gx = self.gamma_cross
        _rw = self.r_window

        if self.base_kind == "pemd":
            _K, _g, _q = bp
            @njit
            def _rhs(q):
                return _rhs_pemd_ext(q, _K, _g, _q, _k, _gp, _gx, _rw)
            @njit
            def _metric(x):
                return _metric_pemd_ext(x, _K, _g, _q, _k, _gp, _gx, _rw)
        elif self.base_kind == "sie":
            _sv2, _rr, _q = bp
            @njit
            def _rhs(q):
                return _rhs_sie_ext(q, _sv2, _rr, _q, _k, _gp, _gx, _rw)
            @njit
            def _metric(x):
                return _metric_sie_ext(x, _sv2, _rr, _q, _k, _gp, _gx, _rw)
        elif self.base_kind == "nfw":
            _Ms, _rs, _q = bp
            @njit
            def _rhs(q):
                return _rhs_nfw_ext(q, _Ms, _rs, _q, _k, _gp, _gx, _rw)
            @njit
            def _metric(x):
                return _metric_nfw_ext(x, _Ms, _rs, _q, _k, _gp, _gx, _rw)
        else:
            raise ValueError(
                f"Unsupported base_kind '{self.base_kind}'. "
                "Use 'pemd', 'sie', or 'nfw'.")

        self._rhs_nb = _rhs
        self._metric_nb = _metric
        self._omega_nb = _null_omega_nb

    def metric(self, x):
        g = self._metric_nb(np.asarray(x, dtype=np.float64))
        return [g[0], g[1], g[2], g[3], g[4]]

    def inverse_metric(self, x):
        g = self.metric(x)
        return [1.0/g[0], 1.0/g[1], 1.0/g[2], 1.0/g[3], 0.0]

    def geodesics(self, q, lmbda):
        return list(self._rhs_nb(np.asarray(q, dtype=np.float64)))

    def Omega(self, r, corotating=True):
        return 0.0


def with_external(base_lens, kappa=0.0, gamma_plus=0.0, gamma_cross=0.0,
                  r_window=None):
    """Wrap any supported base lens with an external (kappa, gamma) field.

    Parameters
    ----------
    base_lens : LensMetric
        Instance of pemd.LensMetric, sie.LensMetric, or nfw.LensMetric.
    kappa, gamma_plus, gamma_cross : float
        External tidal field parameters in the dimensionless Bartelmann
        convention.
    r_window : float, optional
        Localization scale for the external potential (Gaussian window
        in r). The shear is felt only within ~r_window of the lens
        center; this preserves asymptotic flatness. If None, set
        automatically to 10 * (r_s if NFW else 1e5) — choose explicitly
        for tight calibration.

    Returns
    -------
    CompositeLensMetric
        Has the same `_rhs_nb`, `_metric_nb`, `_omega_nb` interface as
        any other LensMetric and can be used directly with
        scr.common.lens_image.LensImage.
    """
    cls_name = type(base_lens).__module__.split('.')[-1]
    if cls_name == "pemd":
        params = (base_lens.K, base_lens.gamma, base_lens.q_ax)
        kind = "pemd"
        rw_default = 1.0e5
    elif cls_name == "sie":
        params = (base_lens._sv2, base_lens._r_ref, base_lens.q_ax)
        kind = "sie"
        rw_default = 1.0e5
    elif cls_name == "nfw":
        params = (base_lens.M_s, base_lens.r_s, base_lens.q_ax)
        kind = "nfw"
        rw_default = 10.0 * base_lens.r_s
    else:
        raise ValueError(
            f"Unsupported base lens type {cls_name}; use PEMD/SIE/NFW.")
    if r_window is None:
        r_window = rw_default
    return CompositeLensMetric(
        base_kind=kind, base_params=params,
        kappa=kappa, gamma_plus=gamma_plus, gamma_cross=gamma_cross,
        r_window=r_window,
        r_min=getattr(base_lens, "_r_min", 1e-2))


###############################################################################

if __name__ == '__main__':
    from scr.lens_metrics import pemd
    base = pemd.LensMetric(K=1e-5, gamma=2.05, q_ax=0.7, r_min=1e-1)
    lens = with_external(base, kappa=0.02, gamma_plus=0.05, gamma_cross=0.01)
    print(f"Composite type: {type(lens).__name__}")
    print(f"Metric at (r=100, th=pi/2, ph=0): {lens.metric([0,100,1.5708,0])}")
    print(f"Geodesic RHS at simple point: ", lens.geodesics(
        np.array([0, 100, 1.5708, 0, -1.0, -1.0, 0.0, 0.0]), 0.0))
