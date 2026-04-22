"""
Validation of the SIS metric: measure the deflection angle of a photon
passing the lens at impact parameter b, compare to alpha_theory = 4 pi sigma_v^2.

The photon is set up in the equatorial plane (theta = pi/2, k_theta = 0),
incoming along the -x direction at y = b, integrated through the lens
with the DOP853 scipy backend, and the total change in phi is measured.
The deflection alpha equals (Delta phi - pi).

Run from repo root:
    python -m scr.lens_metrics.test_sis_deflection
"""
import numpy as np

from scr.lens_metrics.sis import LensMetric
from scr.common.integrator import integrate


def run_deflection(sigma_v, L_start=5e4, b=100.0, final_lmbda=1.5e5,
                   r_ref=1.0, rtol=1e-11, atol=1e-13):
    lens = LensMetric(sigma_v=sigma_v, r_ref=r_ref, r_min=1e-2)

    r0 = float(np.sqrt(L_start**2 + b**2))
    phi0 = float(np.arctan2(b, L_start))   # small positive angle
    theta0 = np.pi / 2.0

    # Impact parameter b at asymptotic infinity: L = b * E. Choose E = 1 so
    # k_t = -1 to match (g_tt -> -1 as r -> inf). k_phi = b, k_theta = 0.
    # k_r is fixed by the null condition at r0:
    #   g^tt k_t^2 + g^rr k_r^2 + g^phiphi k_phi^2 = 0
    # => k_r^2 = (B^2 / A^2) k_t^2 - k_phi^2 / r0^2  (since sin(theta0)=1)
    kt = -1.0
    kphi = float(b)
    Phi0 = 2.0 * sigma_v**2 * np.log(r0 / r_ref)
    A0 = 1.0 + Phi0; B0 = 1.0 - Phi0
    kr2 = (B0*B0 / (A0*A0)) * kt*kt - kphi*kphi / (r0*r0)
    kr = -float(np.sqrt(max(kr2, 0.0)))  # incoming (negative)

    q0 = np.array([0.0, r0, theta0, phi0, kt, kr, 0.0, kphi],
                  dtype=np.float64)

    def rhs(lmbda, y):
        return lens.geodesics(y, lmbda)

    res = integrate(rhs, q0, (0.0, final_lmbda),
                    method="DOP853", events=None,
                    rtol=rtol, atol=atol)

    # Asymptotic outgoing velocity in Cartesian (equatorial plane: y = r sin(phi),
    # x = r cos(phi); derivatives give v^x, v^y).
    r_end = res.y[-1, 1]
    phi_end = res.y[-1, 3]
    kr_end = res.y[-1, 5]
    kphi_end = res.y[-1, 7]

    g_tt, g_rr, g_thth, g_phph, g_tph = lens.metric(
        [0.0, r_end, theta0, phi_end])
    vr = kr_end / g_rr
    vphi = kphi_end / g_phph

    cos_p = float(np.cos(phi_end))
    sin_p = float(np.sin(phi_end))
    vx = vr * cos_p - r_end * vphi * sin_p
    vy = vr * sin_p + r_end * vphi * cos_p

    # Deflection: angle between incoming (-1, 0) and outgoing (vx, vy).
    # For attractive lens with photon at +y, vy < 0 and vx < 0.
    # Angle of outgoing measured from -x axis (positive toward -y) is the
    # classical deflection alpha.
    alpha_measured = float(np.arctan2(-vy, -vx))
    return alpha_measured, res


def main():
    cases = [
        dict(sigma_v=0.01, b=100.0),
        dict(sigma_v=0.03, b=200.0),
        dict(sigma_v=0.05, b=300.0),
    ]

    print(f"\n{'sigma_v':>10} {'b [M]':>8} {'alpha_theory':>14} "
          f"{'alpha_measured':>16} {'rel.err':>10}")
    print("-" * 65)

    all_pass = True
    for case in cases:
        sv = case['sigma_v']; b = case['b']
        alpha_th = 4.0 * np.pi * sv * sv
        alpha_me, res = run_deflection(sigma_v=sv, b=b)
        rel = abs(alpha_me - alpha_th) / alpha_th
        ok = rel < 0.02
        all_pass &= ok
        print(f"{sv:>10.4f} {b:>8.1f} {alpha_th:>14.6e} "
              f"{alpha_me:>16.6e} {rel*100:>9.3f}%  {'PASS' if ok else 'FAIL'}")

    print()
    if all_pass:
        print("==> ALL SIS DEFLECTION TESTS PASS (<2% tolerance)")
    else:
        print("==> SIS DEFLECTION TESTS FAILED — check metric sign / formula")
    return 0 if all_pass else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
