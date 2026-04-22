"""
Integration tests for the gravitational-lensing extension.

Coverage:
    1. SIS deflection: trace a single photon past an SIS lens and verify
       alpha ~ 4 pi sigma_v^2 to ~1%.
    2. Light profile dispatcher: values at known points.
    3. Kernel extrapolator: straight-line intersection with source plane.
    4. End-to-end Einstein ring: rendered ring radius matches the
       parallel-projection formula to ~5% (Schwarzschild and SIS).

Run from repo root:
    conda run -n engrenage python -m scr.common.tests.test_lensing
"""
from math import pi, sqrt
import numpy as np

from scr.black_holes import schwarzschild
from scr.lens_metrics import sis
from scr.detectors import image_plane
from scr.sources.light_profiles import Gaussian, Sersic, _eval_profile_nb
from scr.common.lens_image import LensImage, SourcePlane
from scr.common.common import set_ray_bounds
from scr.common._numba_kernels import _extrapolate_to_source_plane_nb
from scr.common.integrator import integrate


def assert_rel(x, x_ref, tol, label):
    rel = abs(x - x_ref) / abs(x_ref)
    ok = rel < tol
    print(f"    {label}: {x:.6e} vs {x_ref:.6e}  rel={100*rel:6.3f}%  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 1. SIS deflection test
# ============================================================================
def test_sis_deflection():
    print("\n[1] SIS deflection angle (alpha = 4 pi sigma_v^2)")
    all_ok = True
    for sigma_v, b in [(0.01, 100.0), (0.03, 200.0)]:
        lens = sis.LensMetric(sigma_v=sigma_v, r_ref=1.0, r_min=1e-2)
        L_start = 5e4
        r0 = sqrt(L_start**2 + b**2)
        phi0 = np.arctan2(b, L_start)
        Phi0 = 2.0 * sigma_v**2 * np.log(r0 / 1.0)
        A0 = 1.0 + Phi0; B0 = 1.0 - Phi0
        kt = -1.0; kphi = b
        kr = -sqrt(max((B0*B0/(A0*A0))*kt*kt - kphi*kphi/(r0*r0), 0.0))
        q0 = np.array([0.0, r0, pi/2, phi0, kt, kr, 0.0, kphi],
                      dtype=np.float64)

        def rhs(lmbda, y):
            return lens.geodesics(y, lmbda)

        res = integrate(rhs, q0, (0.0, 1.5e5),
                        method="DOP853", rtol=1e-11, atol=1e-13)
        r_e = res.y[-1, 1]; ph_e = res.y[-1, 3]
        kr_e = res.y[-1, 5]; kphi_e = res.y[-1, 7]
        g = lens.metric([0.0, r_e, pi/2, ph_e])
        vr = kr_e / g[1]; vphi = kphi_e / g[3]
        vx = vr*np.cos(ph_e) - r_e*vphi*np.sin(ph_e)
        vy = vr*np.sin(ph_e) + r_e*vphi*np.cos(ph_e)
        alpha_meas = float(np.arctan2(-vy, -vx))
        alpha_th = 4.0 * pi * sigma_v * sigma_v
        all_ok &= assert_rel(alpha_meas, alpha_th, 0.02,
                             f"sigma_v={sigma_v} b={b}")
    return all_ok


# ============================================================================
# 2. Light profile dispatcher
# ============================================================================
def test_light_profiles():
    print("\n[2] Light profile dispatcher")
    g = Gaussian(x0=0.0, y0=0.0, sigma=1.0, I0=1.0)
    ok1 = assert_rel(_eval_profile_nb(0.0, 0.0, g._kind, g._params),
                     1.0, 1e-10, "Gaussian(0,0)")
    ok2 = assert_rel(_eval_profile_nb(1.0, 0.0, g._kind, g._params),
                     np.exp(-0.5), 1e-10, "Gaussian(1,0)")
    s = Sersic(x0=0.0, y0=0.0, R_e=1.0, n=1.0, I_e=1.0)
    # Sersic at R=R_e returns I_e
    ok3 = assert_rel(_eval_profile_nb(1.0, 0.0, s._kind, s._params),
                     1.0, 1e-10, "Sersic(R=R_e)")
    return ok1 and ok2 and ok3


# ============================================================================
# 3. Kernel extrapolator straight-line test
# ============================================================================
def test_extrapolator():
    print("\n[3] Source-plane extrapolator")
    # Fake rhs that gives a straight-line trajectory in flat space
    # y_final at (r=5000, theta=pi/2, phi=0) moving in +x direction forward
    # => backward -lambda direction is -x, which takes us to x = -D_LS.
    lens = schwarzschild.BlackHole()
    # Place y_final in the far asymptotic region, moving in +x forward.
    # In BL equatorial (theta=pi/2), +x is the r-hat direction when phi=0.
    r0 = 5000.0
    y_final = np.array([0.0, r0, pi/2, 0.0, -1.0, 1.0, 0.0, 0.0],
                       dtype=np.float64)
    D_LS = 1.0e4
    ok, xs, ys = _extrapolate_to_source_plane_nb(
        lens._rhs_nb, y_final, D_LS)
    # Expected: backward-trajectory from (5000, 0, 0) in -x direction
    # reaches x = -1e4 at (y, z) = (0, 0).
    ok_flag = bool(ok)
    print(f"    ok={ok_flag}, xs={xs:.3f}, ys={ys:.3f}  "
          f"(expected (0, 0))  {'PASS' if ok_flag and abs(xs) < 1e-6 and abs(ys) < 1e-6 else 'FAIL'}")
    return ok_flag and abs(xs) < 1e-6 and abs(ys) < 1e-6


# ============================================================================
# 4. End-to-end Einstein ring
# ============================================================================
def _measure_ring_radius(data, det):
    Nx, Ny = data.shape
    ic, jc = (Nx - 1) / 2.0, (Ny - 1) / 2.0
    X, Y = np.meshgrid(np.arange(Nx) - ic, np.arange(Ny) - jc, indexing='ij')
    R = np.sqrt(X * X + Y * Y)
    Rmax = min(ic, jc)
    bins = np.linspace(0, Rmax, int(Rmax) + 1)
    radial = np.zeros(len(bins) - 1)
    for k in range(len(bins) - 1):
        mask = (R >= bins[k]) & (R < bins[k + 1])
        if mask.sum() > 0:
            radial[k] = np.median(data[mask])
    k_peak = int(np.argmax(radial))
    if 0 < k_peak < len(radial) - 1:
        y0, y1, y2 = radial[k_peak - 1], radial[k_peak], radial[k_peak + 1]
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
        r_pix = bins[k_peak] + delta * (bins[1] - bins[0])
    else:
        r_pix = bins[k_peak]
    dx_per_px = (det.alphaRange[-1] - det.alphaRange[0]) / (det.x_pixels - 1)
    return float(r_pix * dx_per_px)


def test_einstein_ring():
    print("\n[4] End-to-end Einstein ring (Schwarzschild + SIS, 96x96)")
    D_L = 1.0e4; D_LS = 1.0e4
    set_ray_bounds(r_escape=0.5 * D_L, final_lmbda=3.0 * D_L)
    all_ok = True

    # Schwarzschild
    b_th = 2.0 * sqrt(1.0 * D_LS)
    det = image_plane.detector(D=D_L, iota=pi/2, x_pixels=96,
                                x_side=2.5 * b_th, ratio='1:1')
    src = SourcePlane(D_LS=D_LS,
                      profile=Gaussian(0, 0, 0.1 * b_th, 1.0))
    img = LensImage(schwarzschild.BlackHole(), src, det)
    img.create_photons()
    img.create_image(n_workers=4)
    b_meas = _measure_ring_radius(img.image_data, det)
    all_ok &= assert_rel(b_meas, b_th, 0.05, "Schwarzschild b_ring")

    # SIS
    sigma_v = 0.03
    b_th = 4.0 * pi * sigma_v**2 * D_LS
    det = image_plane.detector(D=D_L, iota=pi/2, x_pixels=96,
                                x_side=2.5 * b_th, ratio='1:1')
    src = SourcePlane(D_LS=D_LS,
                      profile=Gaussian(0, 0, 0.1 * b_th, 1.0))
    img = LensImage(sis.LensMetric(sigma_v=sigma_v), src, det)
    img.create_photons()
    img.create_image(n_workers=4)
    b_meas = _measure_ring_radius(img.image_data, det)
    all_ok &= assert_rel(b_meas, b_th, 0.05, "SIS b_ring")
    return all_ok


def main():
    import sys
    results = [
        test_sis_deflection(),
        test_light_profiles(),
        test_extrapolator(),
        test_einstein_ring(),
    ]
    print("\n" + "=" * 60)
    if all(results):
        print("ALL LENSING INTEGRATION TESTS PASS")
        return 0
    print(f"FAILED: {sum(not r for r in results)}/{len(results)} tests")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
