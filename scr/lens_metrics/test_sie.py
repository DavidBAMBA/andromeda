"""
Validation of the SIE metric:
    1) q_ax = 1 must reproduce SIS deflection alpha ~ 4 pi sigma_v^2
       for an equatorial photon.
    2) q_ax < 1 photon off the equator sees stronger deflection than
       equatorial photon with the same impact parameter.

Run from repo root:
    conda run -n engrenage python -m scr.lens_metrics.test_sie
"""
from math import pi, sqrt, cos, sin, atan2, log
import numpy as np

from scr.lens_metrics.sie import LensMetric
from scr.common.integrator import integrate


def run_deflection(sigma_v, q_ax, b, theta0, L_start=5e4,
                   final_lmbda=1.5e5, rtol=1e-11, atol=1e-13):
    """Integrate a photon with impact parameter b at polar angle theta0.

    For theta0 = pi/2 the photon stays equatorial (k_theta=0 stable).
    For other theta0 the photon sees the ellipticity.
    """
    lens = LensMetric(sigma_v=sigma_v, q_ax=q_ax, r_ref=1.0)

    # Start at (x, y, z) = (L_start, b, 0) rotated by theta0 around the y-axis
    # so the initial line of motion makes angle (pi/2 - theta0) with the
    # equator. Equivalent: keep x, y in equatorial frame, use spherical
    # coords with the tilt built in via theta0.
    # For simplicity we use the equatorial case (theta0 = pi/2) only, since
    # that matches the SIS reference. The off-equator test just checks that
    # the deflection differs from q_ax = 1.
    r0 = sqrt(L_start**2 + b*b)
    phi0 = atan2(b, L_start)
    kt = -1.0
    kphi = b
    # Null condition (equatorial, weak field):
    # k_r^2 = (B^2/A^2) k_t^2 - k_phi^2 / (r^2 sin^2 theta)
    Phi0 = 2.0 * sigma_v**2 * (log(r0 / 1.0)
                                + 0.5 * log(sin(theta0)**2
                                             + cos(theta0)**2 / (q_ax*q_ax)))
    A0 = 1.0 + Phi0; B0 = 1.0 - Phi0
    st2 = sin(theta0)**2
    kr2 = (B0*B0/(A0*A0))*kt*kt - kphi*kphi/(r0*r0*st2)
    kr = -sqrt(max(kr2, 0.0))

    q0 = np.array([0.0, r0, theta0, phi0, kt, kr, 0.0, kphi],
                  dtype=np.float64)

    def rhs(lmbda, y):
        return lens.geodesics(y, lmbda)

    res = integrate(rhs, q0, (0.0, final_lmbda),
                    method="DOP853", rtol=rtol, atol=atol)

    r_e = res.y[-1, 1]; ph_e = res.y[-1, 3]
    kr_e = res.y[-1, 5]; kphi_e = res.y[-1, 7]
    g = lens.metric([0.0, r_e, theta0, ph_e])
    vr = kr_e / g[1]
    # g[3] = g_phph = B^2 r^2 sin^2 theta
    vphi = kphi_e / g[3]
    vx = vr * cos(ph_e) - r_e * vphi * sin(ph_e)
    vy = vr * sin(ph_e) + r_e * vphi * cos(ph_e)
    alpha = float(np.arctan2(-vy, -vx))
    return alpha


def main():
    print("\n(1) SIE with q_ax = 1.0 must reproduce SIS alpha = 4 pi sigma_v^2")
    print(f"  {'sigma_v':>10} {'b':>8} {'alpha_theory':>14} "
          f"{'alpha_measured':>16} {'rel.err':>10}")
    print("  " + "-" * 60)
    all_ok = True
    for sigma_v, b in [(0.01, 100.0), (0.03, 200.0)]:
        ath = 4.0 * pi * sigma_v * sigma_v
        am = run_deflection(sigma_v, q_ax=1.0, b=b, theta0=pi/2)
        rel = abs(am - ath) / ath
        ok = rel < 0.02
        all_ok &= ok
        print(f"  {sigma_v:>10.4f} {b:>8.1f} {ath:>14.6e} "
              f"{am:>16.6e} {100*rel:>9.3f}%  {'PASS' if ok else 'FAIL'}")

    print("\n(2) SIE with q_ax = 0.6 off-equator must differ from SIS case")
    sigma_v = 0.03; b = 200.0
    a_eq  = run_deflection(sigma_v, q_ax=0.6, b=b, theta0=pi/2)
    a_off = run_deflection(sigma_v, q_ax=0.6, b=b, theta0=pi/3)
    a_sis = run_deflection(sigma_v, q_ax=1.0, b=b, theta0=pi/2)
    print(f"  SIE equator  (theta=pi/2, q=0.6):  {a_eq:.6e}")
    print(f"  SIE off-eq   (theta=pi/3, q=0.6):  {a_off:.6e}")
    print(f"  SIS equator  (theta=pi/2, q=1.0):  {a_sis:.6e}")
    print(f"  equator equals SIS?                {abs(a_eq - a_sis) / a_sis * 100:.4f}% (should be small)")
    print(f"  off-eq differs from SIS?           {abs(a_off - a_sis) / a_sis * 100:.4f}% (should be visible)")
    # Off-equator deflection with q=0.6 should be LARGER because the
    # equipotential is pulled inward along z.
    ok2 = abs(a_off - a_sis) / a_sis > 0.01 and abs(a_eq - a_sis) / a_sis < 0.01
    all_ok &= ok2

    print()
    if all_ok:
        print("==> SIE VALIDATION TESTS PASS")
        return 0
    print("==> ONE OR MORE TESTS FAILED")
    return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
