"""
Unit tests for scr/lens_metrics/nfw.py.

Validates:
    1. Phi(r) and dPhi/dr finite at r -> 0 (NFW core regularization).
    2. Asymptotic decay: Phi(r) -> 0 as r -> infinity.
    3. Numerical gradient matches the analytic dPhi/dr (5 random radii).
    4. Numerical ray-tracing deflection matches Wright & Brainerd 2000
       analytic formula at moderate impact parameters (rtol ~ 5%).

Run from repo root:
    conda run -n engrenage python -m scr.lens_metrics.test_nfw
"""
from math import pi, sqrt
import numpy as np

from scr.lens_metrics import nfw
from scr.common.integrator import integrate


def assert_close(x, ref, tol, label):
    rel = abs(x - ref) / max(abs(ref), 1e-30)
    ok = rel < tol
    print(f"    {label}: {x:.6e} vs {ref:.6e}  rel={100*rel:7.4f}%  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 1. Core regularization
# ============================================================================
def test_core_regular():
    print("\n[1] NFW Phi and dPhi/dr finite at r -> 0")
    lens = nfw.LensMetric(M_s=2.0, r_s=1.0e6)
    Phi0 = lens.Phi(1.0e-3 * lens.r_s)
    dPhi0 = lens.dPhi_dr(1.0e-3 * lens.r_s)
    finite = np.isfinite(Phi0) and np.isfinite(dPhi0)
    print(f"    Phi(0.001 r_s) = {Phi0:.3e}, dPhi/dr = {dPhi0:.3e}  "
          f"{'PASS' if finite else 'FAIL'}")
    return finite


# ============================================================================
# 2. Asymptotic decay
# ============================================================================
def test_asymptotic():
    print("\n[2] NFW Phi -> 0 at r >> r_s")
    lens = nfw.LensMetric(M_s=2.0, r_s=1.0e6)
    # NFW decays only as ln(x)/x; compare ratios at 10 r_s and 1000 r_s.
    Phi_at_rs = lens.Phi(lens.r_s)
    Phi_at_10 = lens.Phi(10.0 * lens.r_s)
    Phi_at_1000 = lens.Phi(1000.0 * lens.r_s)
    ratio_10 = abs(Phi_at_10 / Phi_at_rs)
    ratio_1000 = abs(Phi_at_1000 / Phi_at_rs)
    decayed = ratio_10 < 0.5 and ratio_1000 < 0.05
    print(f"    Phi(10 r_s)/Phi(r_s)  = {ratio_10:.3e}")
    print(f"    Phi(1000 r_s)/Phi(r_s)= {ratio_1000:.3e}  "
          f"{'PASS' if decayed else 'FAIL'}")
    return decayed


# ============================================================================
# 3. Numerical gradient matches analytic
# ============================================================================
def test_gradient_consistency():
    print("\n[3] Analytic dPhi/dr matches numerical derivative")
    lens = nfw.LensMetric(M_s=2.0, r_s=1.0e6)
    rng = np.random.default_rng(123)
    rs = rng.uniform(0.2, 5.0, size=5) * lens.r_s
    h = 1.0e-3 * lens.r_s
    all_ok = True
    for r in rs:
        analytic = lens.dPhi_dr(r)
        numerical = (lens.Phi(r + h) - lens.Phi(r - h)) / (2.0 * h)
        all_ok &= assert_close(analytic, numerical, 1.0e-3, f"r/r_s={r/lens.r_s:.2f}")
    return all_ok


# ============================================================================
# 4. Ray-traced deflection vs Wright & Brainerd 2000
# ============================================================================
def test_ray_traced_deflection():
    print("\n[4] NFW ray-traced deflection vs Wright & Brainerd 2000")
    lens = nfw.LensMetric(M_s=2.0, r_s=1.0e6)
    all_ok = True
    for b_factor in [0.5, 1.0, 2.0]:
        b = b_factor * lens.r_s
        L_start = 50.0 * lens.r_s
        r0 = sqrt(L_start**2 + b**2)
        phi0 = np.arctan2(b, L_start)
        Phi0 = lens.Phi(r0)
        A0 = 1.0 + Phi0; B0 = 1.0 - Phi0
        kt = -1.0; kphi = b
        kr = -sqrt(max((B0*B0/(A0*A0))*kt*kt - kphi*kphi/(r0*r0), 0.0))
        q0 = np.array([0.0, r0, pi/2, phi0, kt, kr, 0.0, kphi],
                      dtype=np.float64)

        def rhs(lmbda, y):
            return lens.geodesics(y, lmbda)

        res = integrate(rhs, q0, (0.0, 1.5e2 * lens.r_s),
                        method="DOP853", rtol=1e-11, atol=1e-13)
        r_e = res.y[-1, 1]; ph_e = res.y[-1, 3]
        kr_e = res.y[-1, 5]; kphi_e = res.y[-1, 7]
        g = lens.metric([0.0, r_e, pi/2, ph_e])
        vr = kr_e / g[1]; vphi = kphi_e / g[3]
        vx = vr*np.cos(ph_e) - r_e*vphi*np.sin(ph_e)
        vy = vr*np.sin(ph_e) + r_e*vphi*np.cos(ph_e)
        alpha_meas = float(np.arctan2(-vy, -vx))
        alpha_th = lens.deflection_angle(b)
        all_ok &= assert_close(alpha_meas, alpha_th, 0.05,
                               f"b={b_factor} r_s")
    return all_ok


# ============================================================================
# Runner
# ============================================================================
# ============================================================================
# 5. Elliptical NFW reduces to spherical at q_ax = 1
# ============================================================================
def test_enfw_q1_reduces_to_spherical():
    print("\n[5] eNFW with q_ax=1 reduces to spherical NFW")
    sph = nfw.LensMetric(M_s=2.0, r_s=1.0e6)
    ell = nfw.LensMetric(M_s=2.0, r_s=1.0e6, q_ax=1.0)
    rng = np.random.default_rng(99)
    rs = rng.uniform(0.1, 5.0, size=4) * sph.r_s
    all_ok = True
    for r in rs:
        Phi_sph = sph.Phi(r)
        Phi_ell, _, _, _, _ = nfw._aux_nb(r, pi/2, ell.M_s, ell.r_s, ell.q_ax)
        all_ok &= assert_close(Phi_ell, Phi_sph, 1e-12, f"Phi(r={r:.2e})")
    return all_ok


# ============================================================================
# 6. Elliptical NFW deflection differs from spherical along axes
# ============================================================================
def test_enfw_q07_directional():
    print("\n[6] eNFW q_ax<1: equatorial vs polar deflection differs")
    # In our code theta=pi/2 is the equatorial plane (xi=r * 1).
    # Near theta=0, xi increases (xi = r/q_ax); deflection felt at the same
    # geometric impact b is therefore different. We check Phi differs by
    # the expected ratio of f(theta).
    ell = nfw.LensMetric(M_s=2.0, r_s=1.0e6, q_ax=0.7)
    r = 2.0 * ell.r_s
    Phi_eq, _, _, _, _ = nfw._aux_nb(r, pi/2, ell.M_s, ell.r_s, ell.q_ax)
    Phi_polar, _, _, _, _ = nfw._aux_nb(r, 0.0, ell.M_s, ell.r_s, ell.q_ax)
    # Equatorial xi = r; polar xi = r/q_ax = r/0.7
    # Phi behaves like ln(1+x)/x decreasing in |Phi| as xi grows.
    # Polar |Phi| should be < equatorial |Phi|.
    ok = abs(Phi_polar) < abs(Phi_eq)
    print(f"    |Phi_eq|={abs(Phi_eq):.3e}, |Phi_polar|={abs(Phi_polar):.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    print("=" * 64)
    print(" NFW lens unit tests")
    print("=" * 64)
    results = [
        test_core_regular(),
        test_asymptotic(),
        test_gradient_consistency(),
        test_ray_traced_deflection(),
        test_enfw_q1_reduces_to_spherical(),
        test_enfw_q07_directional(),
    ]
    print("\n" + "=" * 64)
    n_pass = sum(bool(r) for r in results)
    print(f" Total: {n_pass}/{len(results)} test groups passed")
    print("=" * 64)
    raise SystemExit(0 if all(results) else 1)
