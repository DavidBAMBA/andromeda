"""
===============================================================================
Strong gravitational lensing: Source plane + LensImage wrapper
===============================================================================
SourcePlane carries a 2-D light profile (from scr.sources.light_profiles) at
axial distance D_LS behind the lens. LensImage drives the ray tracer in
mode="lensing": photons are integrated through the lens spacetime, escape to
the asymptotic region, and their straight-line extrapolation is intersected
with the source plane to fetch the brightness value.

Observer geometry: iota = 0 puts the observer on the Boyer-Lindquist z-axis,
aligned with the lens axis — the natural setup for a symmetric Einstein ring.

Example
-------
    lens = schwarzschild.BlackHole()                  # or sis.LensMetric(...)
    det  = image_plane.detector(D=1e4, iota=0.0,
                                x_pixels=512, x_side=350, ratio='1:1')
    src  = SourcePlane(D_LS=1e4,
                       profile=Gaussian(x0=0, y0=0, sigma=15, I0=1.0))
    set_ray_bounds(r_escape=0.5*det.D, final_lmbda=2.5*(det.D + src.D_LS))
    img  = LensImage(lens, src, det)
    img.create_photons()
    img.create_image()
    img.plot()
===============================================================================
"""
import numpy as np

from scr.common.common import Image


class SourcePlane:
    """Flat source plane at z = -D_LS with a 2-D light profile.

    Parameters
    ----------
    D_LS : float
        Lens-to-source distance in geometrized M units. The source plane
        is perpendicular to the observer-lens axis.
    profile : object
        Light profile from scr.sources.light_profiles (Gaussian, Sersic, ...).
        Must expose integer `_kind` and `_params: float64[8]`.

    Notes
    -----
    Also duck-types the thin_disk.structure attributes (_r_tbl, _I_tbl,
    in_edge, out_edge) so parallel.py's routing code can carry the object
    through the non-lensing code path unchanged.
    """

    def __init__(self, D_LS, profile):
        self.D_LS = float(D_LS)
        self.profile = profile
        # Duck-type compatibility (unused when mode == 'lensing')
        self._r_tbl = np.zeros(2, dtype=np.float64)
        self._I_tbl = np.zeros(2, dtype=np.float64)
        self.in_edge = 0.0
        self.out_edge = 0.0

    def intensity(self, r):
        """Unused for lensing; kept for Image API compatibility."""
        return 0.0


class LensImage(Image):
    """Image specialization for strong gravitational lensing.

    Same construction as Image(blackhole, acc_structure, detector), but
    acc_structure is a SourcePlane and create_image() runs mode='lensing'.
    """

    def create_image(self, n_workers=None, chunksize=None):
        '''Create lensed image data via backward null ray tracing.'''
        self._trace(mode="lensing", n_workers=n_workers, chunksize=chunksize)
