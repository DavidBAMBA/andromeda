"""
Diagnostic maps for gravitational lensing renders.

These helpers operate on the image-plane to source-plane maps produced by
LensImage.create_diagnostics(). They do not ray trace; they only differentiate
the already computed source coordinates.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from skimage import measure


def jacobian_map(detector, source_x_map, source_y_map, status_map=None):
    """Compute A = d(source_x, source_y) / d(alpha, beta).

    Parameters
    ----------
    detector : scr.detectors.image_plane.detector
        Provides alphaRange and betaRange.
    source_x_map, source_y_map : ndarray [Nx, Ny]
        Source-plane coordinates for each image pixel.
    status_map : ndarray [Nx, Ny], optional
        If provided, only pixels with status == 2 are considered valid.

    Returns
    -------
    dict with keys dxdalpha, dxdbeta, dydalpha, dydbeta, detA, valid
    """
    sx = np.asarray(source_x_map, dtype=np.float64)
    sy = np.asarray(source_y_map, dtype=np.float64)
    if sx.shape != sy.shape:
        raise ValueError("source_x_map and source_y_map must have same shape.")

    alpha = np.asarray(detector.alphaRange, dtype=np.float64)
    beta = np.asarray(detector.betaRange, dtype=np.float64)
    d_sx_dalpha, d_sx_dbeta = np.gradient(sx, alpha, beta, edge_order=2)
    d_sy_dalpha, d_sy_dbeta = np.gradient(sy, alpha, beta, edge_order=2)
    detA = d_sx_dalpha * d_sy_dbeta - d_sx_dbeta * d_sy_dalpha

    finite = (np.isfinite(d_sx_dalpha) & np.isfinite(d_sx_dbeta)
              & np.isfinite(d_sy_dalpha) & np.isfinite(d_sy_dbeta)
              & np.isfinite(detA))
    if status_map is None:
        valid = finite
    else:
        valid = finite & (np.asarray(status_map) == 2.0)

    return {
        "dxdalpha": d_sx_dalpha,
        "dxdbeta": d_sx_dbeta,
        "dydalpha": d_sy_dalpha,
        "dydbeta": d_sy_dbeta,
        "detA": detA,
        "valid": valid,
    }


def magnification_map(detector, source_x_map, source_y_map, status_map=None,
                      det_floor=1e-12, clip_abs=None):
    """Compute signed magnification mu = 1 / det(A).

    Invalid pixels are set to 0. If clip_abs is provided, the signed
    magnification is clipped to [-clip_abs, clip_abs] for visualization.
    """
    jac = jacobian_map(detector, source_x_map, source_y_map, status_map)
    detA = jac["detA"]
    valid = jac["valid"] & (np.abs(detA) > det_floor)
    mu = np.zeros_like(detA)
    mu[valid] = 1.0 / detA[valid]
    if clip_abs is not None:
        mu = np.clip(mu, -float(clip_abs), float(clip_abs))
    jac["mu"] = mu
    jac["valid_mu"] = valid
    return jac


def log_abs_magnification(mu, valid=None, clip_percentile=99.0):
    """Visualization helper for |mu| using log10(1 + |mu|)."""
    out = np.zeros_like(mu, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(mu)
    vals = np.log10(1.0 + np.abs(mu[valid]))
    out[valid] = vals
    if vals.size > 0 and clip_percentile is not None:
        vmax = np.percentile(vals, clip_percentile)
        if vmax > 0.0:
            out = np.clip(out / vmax, 0.0, 1.0)
    return out


def _pixel_contour_to_image_coords(detector, contour):
    """Convert skimage contour coordinates to (alpha, beta).

    skimage returns coordinates as (row, col), which correspond to array
    indices (i, j) for our [Nx, Ny] maps.
    """
    i = contour[:, 0]
    j = contour[:, 1]
    alpha = np.interp(i, np.arange(len(detector.alphaRange)),
                      detector.alphaRange)
    beta = np.interp(j, np.arange(len(detector.betaRange)),
                     detector.betaRange)
    return np.column_stack([alpha, beta])


def critical_curves(detector, detA_map, valid=None, min_points=8):
    """Extract critical curves from det(A)=0.

    Returns a list of arrays with shape [N, 2], columns (alpha, beta).
    """
    detA = np.asarray(detA_map, dtype=np.float64)
    field = detA.copy()
    if valid is not None:
        field[~np.asarray(valid)] = np.nan
    contours = measure.find_contours(field, level=0.0)
    curves = []
    for contour in contours:
        if len(contour) >= min_points:
            curves.append(_pixel_contour_to_image_coords(detector, contour))
    return curves


def caustics_from_critical_curves(critical_curves_px, source_x_map,
                                  source_y_map):
    """Map critical curves from image plane to source plane.

    Parameters
    ----------
    critical_curves_px : list of ndarray [N, 2]
        Contours in pixel-index coordinates (i, j), not alpha/beta.
        Use critical_and_caustic_curves() for the usual public interface.
    source_x_map, source_y_map : ndarray
        Source-plane coordinate maps.
    """
    sx = np.asarray(source_x_map, dtype=np.float64)
    sy = np.asarray(source_y_map, dtype=np.float64)
    caustics = []
    for contour in critical_curves_px:
        coords = np.vstack([contour[:, 0], contour[:, 1]])
        xs = map_coordinates(sx, coords, order=1, mode="nearest")
        ys = map_coordinates(sy, coords, order=1, mode="nearest")
        caustics.append(np.column_stack([xs, ys]))
    return caustics


def critical_and_caustic_curves(detector, detA_map, source_x_map,
                                source_y_map, valid=None, min_points=8):
    """Return critical curves in image coords and caustics in source coords."""
    detA = np.asarray(detA_map, dtype=np.float64)
    field = detA.copy()
    if valid is not None:
        field[~np.asarray(valid)] = np.nan
    contours_px = [c for c in measure.find_contours(field, level=0.0)
                   if len(c) >= min_points]
    critical = [_pixel_contour_to_image_coords(detector, c)
                for c in contours_px]
    caustics = caustics_from_critical_curves(
        contours_px, source_x_map, source_y_map)
    return critical, caustics
