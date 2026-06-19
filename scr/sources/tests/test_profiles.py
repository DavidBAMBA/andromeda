"""
Unit tests for scr/sources/light_profiles.py.

Validates:
    1. Central-value sanity for each profile.
    2. BulgeDisk = bulge + disk (sum decomposition).
    3. DoubleSersic at r=0 equals analytical sum I_b*exp(b_n_b) + I_d*exp(b_n_d).
    4. LogSpiral core regularization (r->0 returns I0).
    5. LogSpiral arm count matches m on a sampling circle.
    6. Numba dispatcher (_eval_profile_nb in light_profiles.py) matches the
       in-kernel copy (_eval_profile_nb_kernel in _numba_kernels.py)
       bit-for-bit for 100 random points across each KIND.

Run from repo root:
    conda run -n engrenage python -m scr.sources.tests.test_profiles
"""
from math import pi, exp
import numpy as np

from scr.sources.light_profiles import (
    Gaussian, Sersic, BulgeDisk, DoubleSersic, LogSpiral,
    _eval_profile_nb, _bn_ciotti_bertin,
)
from scr.common._numba_kernels import _eval_profile_nb_kernel


def assert_close(x, ref, tol, label):
    rel = abs(x - ref) / max(abs(ref), 1e-30)
    ok = rel < tol
    print(f"    {label}: {x:.6e} vs {ref:.6e}  rel={100*rel:7.4f}%  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 1. Central-value sanity
# ============================================================================
def test_central_values():
    print("\n[1] Central values")
    ok = True
    g = Gaussian(I0=2.0, sigma=1.0)
    ok &= assert_close(_eval_profile_nb(0.0, 0.0, g._kind, g._params),
                       2.0, 1e-12, "Gaussian(I0=2) at origin")
    s = Sersic(I_e=1.0, R_e=1.0, n=1.0)
    expected = 1.0 * exp(_bn_ciotti_bertin(1.0))
    ok &= assert_close(_eval_profile_nb(0.0, 0.0, s._kind, s._params),
                       expected, 1e-12, "Sersic(n=1) at origin = I_e*exp(b_1)")
    return ok


# ============================================================================
# 2. BulgeDisk decomposition
# ============================================================================
def test_bulge_disk_decomposition():
    print("\n[2] BulgeDisk = bulge + disk")
    bd = BulgeDisk(R_e_bulge=0.3, R_e_disk=1.5,
                   I_e_bulge=2.0, I_e_disk=0.5,
                   ell=0.0, pa=0.0)
    bn4 = _bn_ciotti_bertin(4.0)
    bn1 = _bn_ciotti_bertin(1.0)

    # At origin: I_e_bulge*exp(b4) + I_e_disk*exp(b1)
    expected_origin = 2.0 * exp(bn4) + 0.5 * exp(bn1)
    val = _eval_profile_nb(0.0, 0.0, bd._kind, bd._params)
    ok = assert_close(val, expected_origin, 1e-12, "origin = I_b*exp(b4)+I_d*exp(b1)")

    # At a chosen offset (1.0, 0.0): bulge Sersic + disk exponential
    r = 1.0
    bulge_at_1 = 2.0 * exp(-bn4 * ((r/0.3) ** 0.25 - 1.0))
    disk_at_1 = 0.5 * exp(-bn1 * (r/1.5 - 1.0))
    expected = bulge_at_1 + disk_at_1
    val = _eval_profile_nb(1.0, 0.0, bd._kind, bd._params)
    ok &= assert_close(val, expected, 1e-12, "(1,0) = bulge + disk decomposition")
    return ok


# ============================================================================
# 3. DoubleSersic at origin
# ============================================================================
def test_double_sersic_origin():
    print("\n[3] DoubleSersic at origin")
    ds = DoubleSersic(R_b=0.3, n_b=4.0, I_b=2.0,
                      R_d=1.5, n_d=1.0, I_d=0.5)
    expected = 2.0 * exp(_bn_ciotti_bertin(4.0)) + 0.5 * exp(_bn_ciotti_bertin(1.0))
    val = _eval_profile_nb(0.0, 0.0, ds._kind, ds._params)
    return assert_close(val, expected, 1e-12,
                        "origin = I_b*exp(b4) + I_d*exp(b1)")


# ============================================================================
# 4. LogSpiral core regularization
# ============================================================================
def test_logspiral_core():
    print("\n[4] LogSpiral core regularization")
    ls = LogSpiral(Rd=1.0, I0=3.0, A=0.7, m=2.0, pitch_angle=0.26)
    # Inside eps_core (1e-3 * Rd) the modulation is dropped: returns I0.
    val_origin = _eval_profile_nb(0.0, 0.0, ls._kind, ls._params)
    ok = assert_close(val_origin, 3.0, 1e-12, "core return value = I0")
    # Outside eps_core, modulation is active.
    val_far = _eval_profile_nb(0.5, 0.0, ls._kind, ls._params)
    return ok and (val_far != 3.0)


# ============================================================================
# 5. LogSpiral arm count on a sampling circle
# ============================================================================
def test_logspiral_arm_count():
    print("\n[5] LogSpiral arm count from zero crossings on a circle")
    m_arms = 2
    ls = LogSpiral(Rd=1.0, I0=1.0, A=1.0, m=float(m_arms),
                   pitch_angle=0.26, pa=0.0)
    r_circ = 0.5  # well inside the disk
    n_samp = 720
    phis = np.linspace(0.0, 2.0*pi, n_samp, endpoint=False)
    vals = np.array([_eval_profile_nb(r_circ*np.cos(p), r_circ*np.sin(p),
                                       ls._kind, ls._params)
                     for p in phis])
    # Count zero crossings of (val - mean) on the circle. With A=1 and a
    # cosine modulator of frequency m_arms, the brightness oscillates with
    # 2*m_arms zero crossings per full revolution about the mean.
    mean = vals.mean()
    diffs = vals - mean
    crossings = np.sum(np.diff(np.sign(diffs)) != 0)
    expected = 2 * m_arms
    print(f"    crossings={crossings} expected={expected}")
    return crossings == expected


# ============================================================================
# 6. Python dispatcher matches kernel dispatcher
# ============================================================================
def test_dispatcher_consistency():
    print("\n[6] _eval_profile_nb (Python copy) == _eval_profile_nb_kernel (in-kernel copy)")
    rng = np.random.default_rng(42)
    pts = rng.uniform(-2.0, 2.0, size=(100, 2))
    profiles = [
        ("Gaussian",     Gaussian(x0=0.1, y0=-0.2, sigma=0.7, I0=1.5)),
        ("Sersic",       Sersic(x0=0.0, y0=0.0, R_e=1.0, n=4.0, I_e=1.0,
                                ell=0.3, pa=0.5)),
        ("BulgeDisk",    BulgeDisk(R_e_bulge=0.3, R_e_disk=1.5,
                                   I_e_bulge=2.0, I_e_disk=0.5,
                                   ell=0.4, pa=0.7)),
        ("DoubleSersic", DoubleSersic(R_b=0.3, n_b=4.0, I_b=2.0,
                                      R_d=1.5, n_d=1.0, I_d=0.5)),
        ("LogSpiral",    LogSpiral(Rd=1.0, I0=1.0, A=0.6, m=2.0,
                                   pitch_angle=0.26, pa=0.3)),
    ]
    all_ok = True
    for name, prof in profiles:
        max_diff = 0.0
        for x, y in pts:
            v_py = _eval_profile_nb(x, y, prof._kind, prof._params)
            v_kn = _eval_profile_nb_kernel(x, y, prof._kind, prof._params)
            d = abs(v_py - v_kn)
            if d > max_diff:
                max_diff = d
        ok = max_diff < 1e-12
        print(f"    {name:14s}: max |py - kernel| = {max_diff:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    return all_ok


# ============================================================================
# Runner
# ============================================================================
if __name__ == '__main__':
    print("=" * 64)
    print(" Light-profile unit tests")
    print("=" * 64)
    results = [
        test_central_values(),
        test_bulge_disk_decomposition(),
        test_double_sersic_origin(),
        test_logspiral_core(),
        test_logspiral_arm_count(),
        test_dispatcher_consistency(),
    ]
    print("\n" + "=" * 64)
    n_pass = sum(bool(r) for r in results)
    print(f" Total: {n_pass}/{len(results)} test groups passed")
    print("=" * 64)
    raise SystemExit(0 if all(results) else 1)
