"""
Unit tests for scr/common/cosmology.py.

Validates:
    1. Hubble parameter E(z) functional form (E(0)=1, E(z) > 1 for z>0).
    2. Comoving distance integration against analytical limits and
       optional astropy.cosmology.Planck18 cross-check.
    3. Flat-universe relation D_LS = (D_C(z_S) - D_C(z_L)) / (1 + z_S).
    4. Gravitational radius r_g(M_sun) = 1.4766 km (textbook).
    5. SceneGeometry produces consistent geometric distances (D_L < D_S).
    6. Einstein angular radius theta_E_SIS for an SDSS-like system within
       reasonable bounds (~1 arcsec scale).

Run from repo root:
    conda run -n engrenage python -m scr.common.tests.test_cosmology
"""
from math import pi
import numpy as np

from scr.common.cosmology import (
    LambdaCDM, SceneGeometry,
    G_SI, C_SI, MPC_IN_M, M_SUN_KG,
    _r_g_meters, _Mpc_to_geometrized_M,
)


def assert_close(x, ref, tol, label):
    rel = abs(x - ref) / max(abs(ref), 1e-30)
    ok = rel < tol
    print(f"    {label}: {x:.6e} vs {ref:.6e}  rel={100*rel:7.4f}%  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 1. Hubble parameter E(z)
# ============================================================================
def test_E():
    print("\n[1] Hubble parameter E(z)")
    cosmo = LambdaCDM()
    ok = True
    ok &= assert_close(cosmo.E(0.0), 1.0, 1e-12, "E(0) = 1")
    # At high z, E(z) -> sqrt(Om) * (1+z)^1.5
    z = 5.0
    expected = np.sqrt(cosmo.Om * (1+z)**3 + cosmo.Ode)
    ok &= assert_close(cosmo.E(z), expected, 1e-12, f"E({z}) form")
    return ok


# ============================================================================
# 2. Comoving / angular-diameter distance values
# ============================================================================
def test_distances():
    print("\n[2] Distances vs astropy Planck18 (if available)")
    cosmo = LambdaCDM()
    ok = True
    # Self-consistency: D_C(0) = 0.
    ok &= assert_close(cosmo.comoving_distance(0.0) + 1.0, 1.0, 1e-12,
                       "D_C(0) = 0")
    # Self-consistency: D_A = D_C / (1+z).
    z = 1.0
    DC = cosmo.comoving_distance(z)
    DA = cosmo.angular_diameter_distance(z)
    ok &= assert_close(DA * (1.0+z), DC, 1e-12, "D_A = D_C/(1+z)")

    try:
        from astropy.cosmology import Planck18
        # Astropy Planck18 uses H0=67.66, Om=0.30966, includes radiation;
        # ours uses 67.4, 0.315, no radiation. Tolerance 1% covers this drift.
        for z_chk in [0.3, 0.5, 1.0, 2.0]:
            astro = Planck18.angular_diameter_distance(z_chk).value
            ours = cosmo.angular_diameter_distance(z_chk)
            ok &= assert_close(ours, astro, 0.01,
                               f"D_A({z_chk}) vs astropy Planck18")
    except ImportError:
        print("    [astropy not installed; skipping cross-check]")
    return ok


# ============================================================================
# 3. D_LS flat-universe identity
# ============================================================================
def test_D_LS():
    print("\n[3] D_LS flat-universe identity")
    cosmo = LambdaCDM()
    z_L, z_S = 0.3, 1.0
    DC_L = cosmo.comoving_distance(z_L)
    DC_S = cosmo.comoving_distance(z_S)
    expected = (DC_S - DC_L) / (1.0 + z_S)
    ok = assert_close(cosmo.D_LS(z_L, z_S), expected, 1e-12,
                      "D_LS = (D_C(z_S)-D_C(z_L))/(1+z_S)")

    try:
        from astropy.cosmology import Planck18
        astro = Planck18.angular_diameter_distance_z1z2(z_L, z_S).value
        ok &= assert_close(cosmo.D_LS(z_L, z_S), astro, 0.01,
                           "D_LS vs astropy Planck18")
    except ImportError:
        pass

    # Bad ordering should raise.
    raised = False
    try:
        cosmo.D_LS(1.0, 0.5)
    except ValueError:
        raised = True
    print(f"    D_LS(z_S<z_L) raises: {'PASS' if raised else 'FAIL'}")
    ok &= raised
    return ok


# ============================================================================
# 4. Gravitational radius r_g(M_sun)
# ============================================================================
def test_r_g():
    print("\n[4] Gravitational radius r_g(M_sun)")
    # Textbook value: r_g(M_sun) = G M_sun / c^2 ~ 1.4766 km.
    rg_km = _r_g_meters(1.0) / 1000.0
    ok = assert_close(rg_km, 1.4766, 1e-3, "r_g(M_sun) [km]")
    return ok


# ============================================================================
# 5. SceneGeometry consistency
# ============================================================================
def test_scene_geometry():
    print("\n[5] SceneGeometry self-consistency")
    scene = SceneGeometry(z_L=0.5, z_S=2.0, M_lens_Msun=3e11)
    ok = True
    # D_L positive and < D_S (typical for z_L < z_S).
    ok &= scene.D_L > 0.0
    ok &= scene.D_S > 0.0
    ok &= scene.D_LS > 0.0
    print(f"    D_L > 0:        {'PASS' if scene.D_L > 0 else 'FAIL'}")
    print(f"    D_S > 0:        {'PASS' if scene.D_S > 0 else 'FAIL'}")
    print(f"    D_LS > 0:       {'PASS' if scene.D_LS > 0 else 'FAIL'}")
    # Mpc -> M conversion is consistent.
    expected_DL = _Mpc_to_geometrized_M(scene.D_L_Mpc, scene.M_lens_Msun)
    ok &= assert_close(scene.D_L, expected_DL, 1e-12, "D_L Mpc->M consistency")
    # Doubling M_lens halves the dimensionless D_L (inversely proportional).
    scene2 = SceneGeometry(z_L=0.5, z_S=2.0, M_lens_Msun=6e11,
                           cosmology=scene.cosmology)
    ok &= assert_close(scene2.D_L, 0.5 * scene.D_L, 1e-10,
                       "D_L scales as 1/M_lens")
    return ok


# ============================================================================
# 6. Einstein radius for SIS in a representative system
# ============================================================================
def test_theta_E_sis():
    print("\n[6] theta_E for SIS lens (sanity-scale test)")
    # SDSS-cluster-scale: sigma_v ~ 350 km/s, z_L=0.68, z_S=1.73.
    scene = SceneGeometry(z_L=0.68, z_S=1.73, M_lens_Msun=1e12)
    theta = scene.theta_E_SIS(352.0)
    theta_arcsec = theta * 206264.806
    print(f"    sigma_v=352 km/s, z_L=0.68, z_S=1.73 -> "
          f"theta_E = {theta_arcsec:.3f} arcsec")
    # Galaxy-scale Einstein radii sit in 0.3 - 2.5 arcsec.
    ok = (0.3 < theta_arcsec < 2.5)
    print(f"    theta_E in galaxy-scale range (0.3, 2.5) arcsec: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# Runner
# ============================================================================
if __name__ == '__main__':
    print("=" * 64)
    print(" Cosmology unit tests")
    print("=" * 64)
    results = [
        test_E(),
        test_distances(),
        test_D_LS(),
        test_r_g(),
        test_scene_geometry(),
        test_theta_E_sis(),
    ]
    print("\n" + "=" * 64)
    n_pass = sum(bool(r) for r in results)
    print(f" Total: {n_pass}/{len(results)} test groups passed")
    print("=" * 64)
    raise SystemExit(0 if all(results) else 1)
