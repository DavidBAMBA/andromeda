"""
===============================================================================
2-D luminosity profiles for a lensing source plane
===============================================================================
Each profile packs its parameters into a float64[8] array and exposes an
integer `kind` tag. The numba dispatcher `_eval_profile_nb(xs, ys, kind,
params)` evaluates the brightness at (xs, ys) on the source plane without
Python overhead in the render hot path.

Units: (xs, ys) in geometrized M, matching the rest of the pipeline.
===============================================================================
"""
from math import sin, cos, exp, sqrt, log
import numpy as np
from numba import njit


# Profile kind tags — kept in sync with _eval_profile_nb.
KIND_GAUSSIAN = 0
KIND_SERSIC = 1


class Gaussian:
    """Circular Gaussian brightness profile.

        I(x, y) = I0 * exp[-0.5 * ((x-x0)^2 + (y-y0)^2) / sigma^2]
    """

    def __init__(self, x0=0.0, y0=0.0, sigma=1.0, I0=1.0):
        self.x0 = float(x0); self.y0 = float(y0)
        self.sigma = float(sigma); self.I0 = float(I0)
        self._kind = KIND_GAUSSIAN
        self._params = np.array(
            [self.x0, self.y0, self.sigma, self.I0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64)


class Sersic:
    """Elliptical Sérsic profile (used for galaxies).

        I(x, y) = I_e * exp[-bn * ((R/R_e)^(1/n) - 1)]
        R = sqrt(x_rot^2 + (y_rot/q)^2),    q = 1 - ell
        (x_rot, y_rot) = rotation by position-angle pa around (x0, y0)

    n=1 gives an exponential disk; n=4 gives a de Vaucouleurs profile.
    ell = 0, pa = 0 reduces to a circular profile.

    bn is approximated by the Ciotti & Bertin (1999) expansion.
    """

    def __init__(self, x0=0.0, y0=0.0, R_e=1.0, n=1.0, I_e=1.0,
                 ell=0.0, pa=0.0):
        self.x0 = float(x0); self.y0 = float(y0)
        self.R_e = float(R_e); self.n = float(n); self.I_e = float(I_e)
        self.ell = float(ell); self.pa = float(pa)
        # Ciotti & Bertin 1999 bn approximation
        n_ = self.n
        bn = 2.0*n_ - 1.0/3.0 + 4.0/(405.0*n_) + 46.0/(25515.0*n_*n_)
        self._bn = bn
        self._kind = KIND_SERSIC
        self._params = np.array(
            [self.x0, self.y0, self.R_e, self.n, self.I_e,
             bn, self.ell, self.pa],
            dtype=np.float64)


@njit(cache=True)
def _eval_profile_nb(xs, ys, kind, params):
    """Evaluate 2-D brightness at (xs, ys). Numba-callable dispatcher."""
    if kind == 0:       # Gaussian
        x0 = params[0]; y0 = params[1]
        sigma = params[2]; I0 = params[3]
        dx = xs - x0; dy = ys - y0
        return I0 * exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))
    elif kind == 1:     # Sersic
        x0 = params[0]; y0 = params[1]
        Re = params[2]; n = params[3]
        Ie = params[4]; bn = params[5]
        ell = params[6]; pa = params[7]
        c = cos(pa); s = sin(pa)
        dx = xs - x0; dy = ys - y0
        xr =  c*dx + s*dy
        yr = -s*dx + c*dy
        q = 1.0 - ell
        R = sqrt(xr*xr + (yr/q)*(yr/q))
        if R == 0.0:
            return Ie * exp(bn)
        return Ie * exp(-bn * ((R/Re) ** (1.0/n) - 1.0))
    return 0.0


###############################################################################

if __name__ == '__main__':
    # Quick self-test.
    g = Gaussian(x0=0.0, y0=0.0, sigma=1.0, I0=1.0)
    print("Gaussian at origin:", _eval_profile_nb(0.0, 0.0, g._kind, g._params))
    print("Gaussian at (1,0) :", _eval_profile_nb(1.0, 0.0, g._kind, g._params))

    s = Sersic(x0=0.0, y0=0.0, R_e=1.0, n=1.0, I_e=1.0)
    print("Sersic(n=1) at origin:", _eval_profile_nb(0.0, 0.0, s._kind, s._params))
    print("Sersic(n=1) at (1,0):", _eval_profile_nb(1.0, 0.0, s._kind, s._params))
