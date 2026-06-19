"""
Tests for source-plane Doppler boosting in gravitational lensing.

Run from repo root:
    python -m scr.common.tests.test_lensing_doppler
"""
from math import pi

import numpy as np

from scr.common.common import set_integrator, set_ray_bounds
from scr.common.lensing_doppler import (apply_lensing_doppler,
                                         doppler_factor_from_beta)
from scr.common.lens_image import LensImage, SourcePlane
from scr.detectors import image_plane
from scr.lens_metrics import sis
from scr.sources.light_profiles import Gaussian
from scr.sources.velocity_models import (RotatingDiskSource, StaticSource,
                                          UniformVelocitySource)


def assert_close(value, expected, tol, label):
    err = abs(value - expected)
    ok = err <= tol
    print(f"    {label}: {value:.12e} vs {expected:.12e} "
          f"err={err:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def test_doppler_factor_signs():
    print("\n[1] Doppler factor signs")
    n = np.array([1.0, 0.0, 0.0])
    beta = 0.1
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    g_toward = doppler_factor_from_beta([beta, 0.0, 0.0], n)
    g_away = doppler_factor_from_beta([-beta, 0.0, 0.0], n)
    ok1 = assert_close(g_toward, 1.0 / (gamma * (1.0 - beta)),
                       1e-14, "approaching source blueshifts")
    ok2 = assert_close(g_away, 1.0 / (gamma * (1.0 + beta)),
                       1e-14, "receding source redshifts")
    return ok1 and ok2 and g_toward > 1.0 and g_away < 1.0


def test_apply_static_source():
    print("\n[2] Static source leaves intensity unchanged")
    I, g = apply_lensing_doppler(
        2.5, [1.0, 0.0, 0.0], 0.0, 0.0, StaticSource())
    return (assert_close(g, 1.0, 1e-14, "g static")
            and assert_close(I, 2.5, 1e-14, "I static"))


def test_small_image_static_matches_lensing():
    print("\n[3] Static Doppler debug image matches plain lensing")
    # Force both images through the same Python/SciPy path. This keeps the
    # comparison focused on the Doppler factor, not backend differences.
    set_integrator("DOP853", rtol=1e-8, atol=1e-10)

    D_L = 1.0e4
    D_LS = 1.0e4
    sigma_v = 0.02
    b_ring = 4.0 * pi * sigma_v * sigma_v * D_LS
    set_ray_bounds(r_escape=0.5 * D_L, final_lmbda=3.0 * D_L)

    lens = sis.LensMetric(sigma_v=sigma_v)
    det = image_plane.detector(D=D_L, iota=pi/2, x_pixels=12,
                                x_side=2.5 * b_ring, ratio='1:1')
    profile = Gaussian(0.0, 0.0, 0.2 * b_ring, 1.0)

    plain = LensImage(lens, SourcePlane(D_LS, profile), det)
    plain.create_photons()
    plain.create_image(n_workers=1)

    shifted = LensImage(
        lens, SourcePlane(D_LS, profile, velocity_model=StaticSource()), det)
    shifted.create_photons()
    shifted.create_image_doppler_debug()

    diff = float(np.max(np.abs(plain.image_data - shifted.image_data)))
    ok = diff < 1e-10
    print(f"    max image diff={diff:.3e} {'PASS' if ok else 'FAIL'}")
    set_integrator("auto", rtol=1e-9, atol=1e-11)
    set_ray_bounds(None, None)
    return ok


def test_uniform_velocity_changes_intensity():
    print("\n[4] Uniform velocity changes intensity by g^3")
    I0 = 1.7
    beta = 0.03
    model = UniformVelocitySource(vx=beta)
    I, g = apply_lensing_doppler(I0, [1.0, 0.0, 0.0], 0.0, 0.0, model)
    ok = assert_close(I, I0 * g ** 3, 1e-14, "I = I0 g^3")
    return ok and g > 1.0


def test_numba_matches_debug_rotating_source():
    print("\n[5] Numba Doppler image matches Python debug image")
    set_integrator("auto", rtol=1e-8, atol=1e-10)

    D_L = 1.0e4
    D_LS = 1.0e4
    sigma_v = 0.02
    b_ring = 4.0 * pi * sigma_v * sigma_v * D_LS
    set_ray_bounds(r_escape=0.5 * D_L, final_lmbda=3.0 * D_L)

    lens = sis.LensMetric(sigma_v=sigma_v)
    det = image_plane.detector(D=D_L, iota=pi/2, x_pixels=10,
                                x_side=2.5 * b_ring, ratio='1:1')
    profile = Gaussian(0.05 * b_ring, 0.0, 0.2 * b_ring, 1.0)
    velocity = RotatingDiskSource(v_max=0.08, r_turn=0.1 * b_ring,
                                  inclination=pi/4, pa=pi/6,
                                  x0=0.05 * b_ring, y0=0.0)

    fast = LensImage(lens, SourcePlane(D_LS, profile,
                                       velocity_model=velocity), det)
    fast.create_photons()
    fast.create_image_doppler(n_workers=1)

    debug = LensImage(lens, SourcePlane(D_LS, profile,
                                        velocity_model=velocity), det)
    debug.create_photons()
    debug.create_image_doppler_debug()

    diff = float(np.max(np.abs(fast.image_data - debug.image_data)))
    scale = max(1.0, float(np.max(np.abs(debug.image_data))))
    rel = diff / scale
    # The fast path uses the in-house Numba RK45 integrator while debug uses
    # the Python/SciPy path, so this is a cross-backend regression tolerance.
    ok = rel < 2e-3
    print(f"    max abs diff={diff:.3e}, scaled={rel:.3e} "
          f"{'PASS' if ok else 'FAIL'}")
    set_integrator("auto", rtol=1e-9, atol=1e-11)
    set_ray_bounds(None, None)
    return ok


def test_diagnostics_match_fast_image():
    print("\n[6] Diagnostic image matches fast Doppler image")
    set_integrator("auto", rtol=1e-8, atol=1e-10)

    D_L = 1.0e4
    D_LS = 1.0e4
    sigma_v = 0.02
    b_ring = 4.0 * pi * sigma_v * sigma_v * D_LS
    set_ray_bounds(r_escape=0.5 * D_L, final_lmbda=3.0 * D_L)

    lens = sis.LensMetric(sigma_v=sigma_v)
    det = image_plane.detector(D=D_L, iota=pi/2, x_pixels=10,
                                x_side=2.5 * b_ring, ratio='1:1')
    profile = Gaussian(0.05 * b_ring, 0.0, 0.2 * b_ring, 1.0)
    velocity = UniformVelocitySource(vx=0.05)

    fast = LensImage(lens, SourcePlane(D_LS, profile,
                                       velocity_model=velocity), det)
    fast.create_photons()
    fast.create_image_doppler(n_workers=1)

    diag = LensImage(lens, SourcePlane(D_LS, profile,
                                       velocity_model=velocity), det)
    diag.create_photons()
    maps = diag.create_diagnostics(n_workers=1)

    diff = float(np.max(np.abs(fast.image_data - maps["image_data"])))
    escaped = maps["status_map"] == 2.0
    ok_shape = maps["g_map"].shape == fast.image_data.shape
    ok_g = bool(np.all(maps["g_map"][escaped] > 0.0))
    ok = diff < 1e-12 and ok_shape and ok_g
    print(f"    max image diff={diff:.3e}, escaped={int(escaped.sum())} "
          f"{'PASS' if ok else 'FAIL'}")
    set_integrator("auto", rtol=1e-9, atol=1e-11)
    set_ray_bounds(None, None)
    return ok


def main():
    results = [
        test_doppler_factor_signs(),
        test_apply_static_source(),
        test_small_image_static_matches_lensing(),
        test_uniform_velocity_changes_intensity(),
        test_numba_matches_debug_rotating_source(),
        test_diagnostics_match_fast_image(),
    ]
    print("\n" + "=" * 60)
    if all(results):
        print("ALL LENSING DOPPLER TESTS PASS")
        return 0
    print(f"FAILED: {sum(not r for r in results)}/{len(results)} tests")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
