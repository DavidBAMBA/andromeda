"""
===============================================================================
Post-processing helpers for lensing images: direct profile projection, star
fields, photographic noise, RGB composition, and asinh stretch.
===============================================================================
All functions are pure numpy and operate on float arrays; no ray tracing.
Reused by ex10.cosmic_horseshoe.py and future Hubble-style composites.
===============================================================================
"""
import numpy as np

from scr.sources.light_profiles import _eval_profile_nb


def project_profile_on_detector(detector, profile):
    """Evaluate a 2-D light profile at every pixel of the detector,
    WITHOUT ray tracing.

    Used for the lens galaxy's own visible light: the photons come from the
    lens region itself and don't undergo lensing in the usual sense. The
    profile is treated as living on the image plane directly.

    Parameters
    ----------
    detector : detectors.image_plane.detector
    profile  : object with `_kind: int` and `_params: float64[8]`
               (Gaussian, Sersic from scr.sources.light_profiles).

    Returns
    -------
    img : float64[Nx, Ny]
    """
    alpha = detector.alphaRange
    beta = detector.betaRange
    Nx = detector.x_pixels
    Ny = detector.y_pixels
    img = np.zeros((Nx, Ny), dtype=np.float64)
    kind = int(profile._kind)
    params = profile._params
    for i in range(Nx):
        a = alpha[i]
        for j in range(Ny):
            img[i, j] = _eval_profile_nb(a, beta[j], kind, params)
    return img


def add_starfield(shape, n_stars, flux_range=(0.05, 0.8),
                  sigma_range=(0.5, 1.4), rng=None):
    """Render a starfield of Gaussian point sources with random positions,
    log-uniform fluxes, and uniform Gaussian sigmas (pixel units).

    Parameters
    ----------
    shape : (Nx, Ny)
    n_stars : int
    flux_range : (lo, hi) log-uniform peak flux
    sigma_range : (lo, hi) PSF width in pixels
    rng : np.random.Generator or None

    Returns
    -------
    img : float64[Nx, Ny]
    """
    if rng is None:
        rng = np.random.default_rng()
    Nx, Ny = shape
    img = np.zeros((Nx, Ny), dtype=np.float64)

    xs = rng.uniform(0, Nx - 1, size=n_stars)
    ys = rng.uniform(0, Ny - 1, size=n_stars)
    log_lo, log_hi = np.log(flux_range[0]), np.log(flux_range[1])
    fluxes = np.exp(rng.uniform(log_lo, log_hi, size=n_stars))
    sigmas = rng.uniform(sigma_range[0], sigma_range[1], size=n_stars)

    # Stamp each star on a local patch (radius = 4 sigma) for speed.
    for x0, y0, flux, sigma in zip(xs, ys, fluxes, sigmas):
        rad = int(np.ceil(4.0 * sigma))
        i0 = max(0, int(x0) - rad); i1 = min(Nx, int(x0) + rad + 1)
        j0 = max(0, int(y0) - rad); j1 = min(Ny, int(y0) + rad + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        xs_grid = np.arange(i0, i1) - x0
        ys_grid = np.arange(j0, j1) - y0
        gx = np.exp(-0.5 * (xs_grid / sigma) ** 2)
        gy = np.exp(-0.5 * (ys_grid / sigma) ** 2)
        img[i0:i1, j0:j1] += flux * gx[:, None] * gy[None, :]
    return img


def add_photographic_noise(img, read_noise=0.02, poisson_scale=200.0,
                           rng=None):
    """Apply Poisson shot noise (scaled so typical pixel has ~poisson_scale
    counts at peak brightness) plus Gaussian read noise.

    Parameters
    ----------
    img : float[..., channels] or float[H, W]
    read_noise : Gaussian stdev added to every pixel
    poisson_scale : multiplier before np.random.poisson; larger = less noise
    rng : np.random.Generator or None

    Returns
    -------
    noisy image, same shape
    """
    if rng is None:
        rng = np.random.default_rng()
    # Poisson requires non-negative
    lam = np.maximum(img, 0.0) * poisson_scale
    shot = rng.poisson(lam).astype(np.float64) / poisson_scale
    read = rng.normal(0.0, read_noise, size=img.shape)
    return shot + read


def compose_rgb(lens_img, arc_img, stars_img=None,
                lens_color=(1.0, 0.7, 0.25),
                arc_color=(0.25, 0.55, 1.0),
                stars_color=(0.9, 0.92, 1.0)):
    """Compose monochromatic layers into an RGB image.

    rgb = lens_img[...,None] * lens_color
        + arc_img[...,None]  * arc_color
        + stars_img[...,None] * stars_color   (if provided)

    Each layer should already be normalized to a reasonable range
    (the helper does NOT rescale; that's the caller's job if desired).

    Returns
    -------
    rgb : float64[Nx, Ny, 3]
    """
    Nx, Ny = lens_img.shape
    rgb = (lens_img[..., None] * np.array(lens_color)
           + arc_img[..., None] * np.array(arc_color))
    if stars_img is not None:
        rgb = rgb + stars_img[..., None] * np.array(stars_color)
    return rgb


def asinh_stretch(rgb, softening=0.02, max_percentile=99.7):
    """asinh stretch in the style of Hubble press-release images.

    I_stretched = asinh(I / softening) / asinh(I_max / softening)

    with I_max taken from the max_percentile of the luminance so a few
    saturated pixels do not compress the rest of the image.

    Parameters
    ----------
    rgb : float[Nx, Ny, 3]
    softening : lower = more aggressive stretch of faint pixels
    max_percentile : 0 to 100

    Returns
    -------
    float[Nx, Ny, 3] clipped to [0, 1]
    """
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lum = np.maximum(lum, 0.0)
    vmax = np.percentile(lum, max_percentile)
    if vmax <= 0.0:
        vmax = 1.0
    scale = np.arcsinh(vmax / softening)
    out = np.arcsinh(np.maximum(rgb, 0.0) / softening) / scale
    return np.clip(out, 0.0, 1.0)
