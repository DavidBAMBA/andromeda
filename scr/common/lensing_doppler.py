"""
Relativistic Doppler factors for source-plane gravitational lensing.

This module implements the radiative-transfer piece independently from the
ray tracer. In vacuum, Liouville's theorem gives I_nu / nu^3 = const along a
null geodesic, so a moving source contributes

    I_obs_nu = g^3 I_emit_nu,
    g = nu_obs / nu_emit.

For the current lensing source plane we evaluate this in the asymptotically
flat lens-frame region:

    g = 1 / (gamma * (1 - beta . n_emit_to_obs)) / (1 + z_cosmo),

where n_emit_to_obs is the photon propagation direction from the source toward
the observer and beta is the source 3-velocity in units of c.
"""
from math import sqrt

import numpy as np


def normalize(v):
    """Return a unit vector; raise for zero-length input."""
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return arr / norm


def doppler_factor_from_beta(beta, n_emit_to_obs, z_cosmo=0.0):
    """Special-relativistic Doppler factor for source-plane emission.

    Parameters
    ----------
    beta : array-like, shape (3,)
        Source velocity v/c in the asymptotic lens-frame Cartesian basis.
    n_emit_to_obs : array-like, shape (3,)
        Unit photon propagation direction at emission, pointing from source
        toward observer.
    z_cosmo : float
        Optional cosmological/source redshift applied as an extra global
        factor. Leave at 0 for the existing geometrized local simulations.
    """
    beta = np.asarray(beta, dtype=np.float64)
    n = normalize(n_emit_to_obs)
    beta2 = float(np.dot(beta, beta))
    if beta2 >= 1.0:
        raise ValueError("Velocity magnitude must be < c.")
    gamma = 1.0 / sqrt(1.0 - beta2)
    local_g = 1.0 / (gamma * (1.0 - float(np.dot(beta, n))))
    return local_g / (1.0 + float(z_cosmo))


def apply_lensing_doppler(I_emit, n_emit_to_obs, xs, ys, velocity_model,
                          z_cosmo=0.0):
    """Apply Doppler boosting to source-plane specific intensity."""
    if velocity_model is None:
        beta = np.zeros(3, dtype=np.float64)
    else:
        beta = velocity_model.velocity(xs, ys)
    g = doppler_factor_from_beta(beta, n_emit_to_obs, z_cosmo=z_cosmo)
    return float(I_emit) * g * g * g, g


def source_direction_from_escape(rhs, y_final):
    """Direction helpers at the escape point.

    The ray tracer integrates backward from observer to source. At escape, the
    negative of the RHS spatial velocity continues the backward ray toward the
    source plane. The physical photon emitted by the source travels in the
    opposite direction, toward the observer.

    Returns
    -------
    n_to_source, n_emit_to_obs : ndarray shape (3,), ndarray shape (3,)
    """
    r = float(y_final[1])
    th = float(y_final[2])
    ph = float(y_final[3])
    sth = np.sin(th)
    cth = np.cos(th)
    sph = np.sin(ph)
    cph = np.cos(ph)

    dq = rhs(y_final)
    dr_ = float(dq[1])
    dth_ = float(dq[2])
    dph_ = float(dq[3])

    vx_back = -(dr_ * sth * cph + r * cth * cph * dth_
                - r * sth * sph * dph_)
    vy_back = -(dr_ * sth * sph + r * cth * sph * dth_
                + r * sth * cph * dph_)
    vz_back = -(dr_ * cth - r * sth * dth_)
    n_to_source = normalize([vx_back, vy_back, vz_back])
    return n_to_source, -n_to_source


def extrapolate_to_source_plane_debug(rhs, y_final, D_LS):
    """Python equivalent of the Numba source-plane extrapolator.

    Returns ok, xs, ys, n_emit_to_obs. The source-plane coordinates are
    (xs, ys) = (Cartesian y, Cartesian z), matching the existing kernel.
    """
    r = float(y_final[1])
    th = float(y_final[2])
    ph = float(y_final[3])
    sth = np.sin(th)
    cth = np.cos(th)
    sph = np.sin(ph)
    cph = np.cos(ph)

    xf = r * sth * cph
    yf = r * sth * sph
    zf = r * cth
    n_to_source, n_emit_to_obs = source_direction_from_escape(rhs, y_final)

    vx = n_to_source[0]
    if vx >= 0.0:
        return False, 0.0, 0.0, n_emit_to_obs
    s = (-float(D_LS) - xf) / vx
    if s < 0.0:
        return False, 0.0, 0.0, n_emit_to_obs
    return True, yf + s * n_to_source[1], zf + s * n_to_source[2], n_emit_to_obs
