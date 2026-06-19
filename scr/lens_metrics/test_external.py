"""
Unit tests for scr/lens_metrics/external.py and scr/lens_metrics/composite.py.

Validates:
    1. External Phi vanishes on the optical axis (y = z = 0).
    2. External Phi is symmetric under (y -> -y, z -> -z) [pure quadratic].
    3. Composite reduces to the base lens when kappa = gamma_+ = gamma_x = 0
       (geodesic RHS bit-for-bit identical).
    4. Composite RHS picks up a non-zero d k_phi / d lambda when external
       shear is present (axisymmetry breaking, sanity check).

Run from repo root:
    conda run -n engrenage python -m scr.lens_metrics.test_external
"""
from math import pi
import numpy as np

from scr.lens_metrics import external, composite, pemd, sie


def assert_close(x, ref, tol, label):
    rel = abs(x - ref) / max(abs(ref), 1e-30)
    ok = rel < tol
    print(f"    {label}: {x:.6e} vs {ref:.6e}  rel={100*rel:7.4f}%  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 1. External Phi vanishes on optical axis
# ============================================================================
def test_external_axis_zero():
    print("\n[1] External Phi = 0 on optical axis (y=z=0)")
    ext = external.LensMetric(kappa=0.05, gamma_plus=0.10, gamma_cross=0.02)
    # Optical axis: theta=pi/2, phi=0  =>  y=0, z=0  =>  Phi=0.
    Phi = ext.Phi([0, 1.0, pi/2, 0.0])
    ok = abs(Phi) < 1e-10
    print(f"    Phi(r=1,th=pi/2,ph=0) = {Phi:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 2. External Phi parity under (y,z) -> (-y,-z)
# ============================================================================
def test_external_parity():
    print("\n[2] External Phi(y,z) = Phi(-y,-z) (quadratic form)")
    ext = external.LensMetric(kappa=0.05, gamma_plus=0.10, gamma_cross=0.02)
    rng = np.random.default_rng(2)
    all_ok = True
    for _ in range(4):
        r = rng.uniform(0.5, 5.0)
        th = rng.uniform(0.3, pi - 0.3)
        ph = rng.uniform(0.0, 2*pi)
        Phi_pos = ext.Phi([0, r, th, ph])
        # (y, z) -> (-y, -z) is the antipodal point: theta -> pi - theta,
        # phi -> phi + pi.
        Phi_neg = ext.Phi([0, r, pi - th, ph + pi])
        all_ok &= assert_close(Phi_pos, Phi_neg, 1e-10,
                               f"r={r:.2f}, th={th:.2f}, ph={ph:.2f}")
    return all_ok


# ============================================================================
# 3. Composite reduces to base when external is zero
# ============================================================================
def test_composite_zero_external_recovers_base():
    print("\n[3] Composite with kappa=gp=gx=0 reduces to base lens RHS")
    base = pemd.LensMetric(K=1e-5, gamma=2.05, q_ax=0.7, r_min=1e-1)
    comp = composite.with_external(base, kappa=0.0,
                                    gamma_plus=0.0, gamma_cross=0.0)
    rng = np.random.default_rng(11)
    all_ok = True
    for _ in range(4):
        q = np.array([0.0,
                      rng.uniform(50.0, 500.0),
                      rng.uniform(0.4, pi - 0.4),
                      rng.uniform(0.0, 2*pi),
                      -1.0,
                      rng.uniform(-1.0, 1.0),
                      rng.uniform(-0.5, 0.5),
                      rng.uniform(-0.5, 0.5)])
        rhs_b = base.geodesics(q, 0.0)
        rhs_c = comp.geodesics(q, 0.0)
        diff = max(abs(a - b) for a, b in zip(rhs_b, rhs_c))
        ok = diff < 1e-12
        print(f"    max |rhs_base - rhs_comp| = {diff:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    return all_ok


# ============================================================================
# 4. Composite breaks k_phi conservation when external shear is on
# ============================================================================
def test_composite_breaks_axisymmetry():
    print("\n[4] Composite with non-zero external shear: dk_phi/dlambda != 0")
    base = pemd.LensMetric(K=1e-5, gamma=2.05, q_ax=0.7, r_min=1e-1)
    comp = composite.with_external(base, kappa=0.0,
                                    gamma_plus=0.05, gamma_cross=0.0)
    q = np.array([0.0, 200.0, pi/2, 0.7, -1.0, -1.0, 0.0, 50.0])
    rhs = comp.geodesics(q, 0.0)
    out_kphi_dot = rhs[7]
    ok = abs(out_kphi_dot) > 1e-10
    print(f"    dk_phi/dlambda = {out_kphi_dot:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    # And: with same setup but base alone, dk_phi/dlambda = 0.
    rhs_base = base.geodesics(q, 0.0)
    ok &= abs(rhs_base[7]) < 1e-12
    print(f"    base alone dk_phi/dlambda = {rhs_base[7]:.3e}  "
          f"(should be 0 because PEMD is axisymmetric)")
    return ok


# ============================================================================
# 5. Composite supports SIE base
# ============================================================================
def test_composite_sie_base():
    print("\n[5] Composite supports SIE base")
    base = sie.LensMetric(sigma_v=0.025, q_ax=0.7)
    comp = composite.with_external(base, kappa=0.01,
                                    gamma_plus=0.03, gamma_cross=0.0)
    q = np.array([0.0, 200.0, pi/2, 0.0, -1.0, -1.0, 0.0, 0.0])
    rhs = comp.geodesics(q, 0.0)
    finite = all(np.isfinite(x) for x in rhs)
    print(f"    finite RHS: {'PASS' if finite else 'FAIL'}")
    return finite


# ============================================================================
# Runner
# ============================================================================
if __name__ == '__main__':
    print("=" * 64)
    print(" External + Composite lens unit tests")
    print("=" * 64)
    results = [
        test_external_axis_zero(),
        test_external_parity(),
        test_composite_zero_external_recovers_base(),
        test_composite_breaks_axisymmetry(),
        test_composite_sie_base(),
    ]
    print("\n" + "=" * 64)
    n_pass = sum(bool(r) for r in results)
    print(f" Total: {n_pass}/{len(results)} test groups passed")
    print("=" * 64)
    raise SystemExit(0 if all(results) else 1)
