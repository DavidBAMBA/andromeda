"""
Tests for lensing diagnostic maps.

Run from repo root:
    python -m scr.common.tests.test_lens_diagnostics
"""
import numpy as np

from scr.common.lens_diagnostics import (critical_and_caustic_curves,
                                         jacobian_map, magnification_map)


class DummyDetector:
    def __init__(self, alpha, beta):
        self.alphaRange = np.asarray(alpha, dtype=np.float64)
        self.betaRange = np.asarray(beta, dtype=np.float64)


def assert_close(value, expected, tol, label):
    err = abs(float(value) - float(expected))
    ok = err <= tol
    print(f"    {label}: {value:.12e} vs {expected:.12e} "
          f"err={err:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def test_identity_mapping():
    print("\n[1] Identity mapping has detA=1 and mu=1")
    alpha = np.linspace(-2.0, 2.0, 21)
    beta = np.linspace(-3.0, 3.0, 25)
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    det = DummyDetector(alpha, beta)
    status = np.full(aa.shape, 2.0)

    jac = magnification_map(det, aa, bb, status)
    core = np.s_[2:-2, 2:-2]
    ok1 = assert_close(np.max(np.abs(jac["detA"][core] - 1.0)), 0.0,
                       1e-12, "max |detA-1|")
    ok2 = assert_close(np.max(np.abs(jac["mu"][core] - 1.0)), 0.0,
                       1e-12, "max |mu-1|")
    return ok1 and ok2


def test_linear_scaling():
    print("\n[2] Linear scaling has expected magnification")
    alpha = np.linspace(-1.0, 1.0, 31)
    beta = np.linspace(-1.0, 1.0, 31)
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    sx = 2.0 * aa
    sy = 0.5 * bb
    det = DummyDetector(alpha, beta)

    jac = jacobian_map(det, sx, sy)
    detA_expected = 1.0
    core = np.s_[2:-2, 2:-2]
    return assert_close(np.max(np.abs(jac["detA"][core] - detA_expected)),
                        0.0, 1e-12, "max |detA-expected|")


def test_critical_curve_circle():
    print("\n[3] Critical curve extraction finds a circle")
    alpha = np.linspace(-2.0, 2.0, 101)
    beta = np.linspace(-2.0, 2.0, 101)
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    radius = 1.1
    detA = aa * aa + bb * bb - radius * radius
    det = DummyDetector(alpha, beta)

    critical, caustics = critical_and_caustic_curves(
        det, detA, aa, bb, min_points=20)
    if len(critical) != 1 or len(caustics) != 1:
        print(f"    expected 1 curve, got {len(critical)} FAIL")
        return False
    r = np.sqrt(critical[0][:, 0] ** 2 + critical[0][:, 1] ** 2)
    return assert_close(np.median(r), radius, 2e-3,
                        "median critical radius")


def main():
    results = [test_identity_mapping(), test_linear_scaling(),
               test_critical_curve_circle()]
    print("\n" + "=" * 60)
    if all(results):
        print("ALL LENS DIAGNOSTIC TESTS PASS")
        return 0
    print(f"FAILED: {sum(not r for r in results)}/{len(results)} tests")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
