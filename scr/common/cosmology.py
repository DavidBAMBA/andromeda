"""
===============================================================================
Flat Lambda-CDM cosmology and scene geometry for gravitational lensing
===============================================================================
Provides:
    - LambdaCDM: Hubble parameter, comoving and angular-diameter distances
      via numerical integration of c/H(z). Defaults to Planck18.
    - SceneGeometry: bundles (z_lens, z_source, M_lens) into a self-contained
      scene whose distances (D_L, D_S, D_LS) are expressed in geometrized
      M_lens units, matching the rest of the ray-tracing pipeline.

Units
-----
    Public cosmology API uses Mpc for distances, km/s/Mpc for H_0.
    SceneGeometry returns distances in *geometrized* M_lens (G = c = 1, M = M_lens).
    Internally we convert via r_g = G * M_lens / c^2 [m] and
    1 Mpc = Mpc_m / r_g  units of M_lens.

Constants
---------
    CODATA 2018 values for G, c. IAU 2015 nominal solar mass parameter.
===============================================================================
"""
from math import sqrt
import numpy as np
from scipy.integrate import quad


# Physical constants (SI)
G_SI = 6.67430e-11           # m^3 kg^-1 s^-2  (CODATA 2018)
C_SI = 299792458.0           # m/s            (exact)
MPC_IN_M = 3.0856775814913673e22   # m   (1 Mpc, IAU 2015 nominal)
M_SUN_KG = 1.98892e30        # kg  (IAU 2015 nominal solar mass)


# Speed of light in km/s used with H0 [km/s/Mpc] gives Hubble distance in Mpc
C_KM_S = C_SI / 1000.0


class LambdaCDM:
    """Flat Lambda-CDM cosmology.

    Parameters
    ----------
    H0 : float, default 67.4
        Hubble constant in km/s/Mpc. Default is Planck18.
    Om : float, default 0.315
        Matter density parameter today. Default is Planck18.
    Ode : float, optional
        Dark-energy density parameter. If None, uses 1 - Om (flat universe).

    Notes
    -----
    Curvature is set so that Om + Ode = 1 (flat). Radiation is neglected
    (valid for z << 3000). Distance integrals use scipy.integrate.quad with
    default tolerance (~1e-8 relative).
    """

    def __init__(self, H0=67.4, Om=0.315, Ode=None):
        self.H0 = float(H0)
        self.Om = float(Om)
        self.Ode = float(1.0 - Om) if Ode is None else float(Ode)
        # Hubble distance in Mpc: D_H = c / H0
        self._D_H_Mpc = C_KM_S / self.H0

    def E(self, z):
        """Dimensionless Hubble parameter H(z)/H0 for flat LambdaCDM."""
        zp1 = 1.0 + z
        return sqrt(self.Om * zp1 * zp1 * zp1 + self.Ode)

    def comoving_distance(self, z):
        """Line-of-sight comoving distance D_C(z) in Mpc."""
        if z <= 0.0:
            return 0.0
        integrand = lambda zp: 1.0 / self.E(zp)
        val, _ = quad(integrand, 0.0, z)
        return self._D_H_Mpc * val

    def angular_diameter_distance(self, z):
        """Angular-diameter distance D_A(z) in Mpc.

        Flat universe: D_A = D_C / (1 + z).
        """
        return self.comoving_distance(z) / (1.0 + z)

    def D_LS(self, z_L, z_S):
        """Angular-diameter distance from lens to source, in Mpc.

        Flat-universe formula: D_LS = (D_C(z_S) - D_C(z_L)) / (1 + z_S).
        Valid only for z_S > z_L.
        """
        if z_S <= z_L:
            raise ValueError(
                f"z_S={z_S} must be greater than z_L={z_L} for a real lens.")
        DC_S = self.comoving_distance(z_S)
        DC_L = self.comoving_distance(z_L)
        return (DC_S - DC_L) / (1.0 + z_S)


# ---------------------------------------------------------------------------
# Scene geometry: redshifts + lens mass -> distances in geometrized M_lens
# ---------------------------------------------------------------------------

def _r_g_meters(M_lens_Msun):
    """Gravitational radius r_g = G * M_lens / c^2 in meters."""
    M_kg = M_lens_Msun * M_SUN_KG
    return G_SI * M_kg / (C_SI * C_SI)


def _Mpc_to_geometrized_M(distance_Mpc, M_lens_Msun):
    """Convert a distance in Mpc to multiples of M_lens (geometrized G=c=1)."""
    return distance_Mpc * MPC_IN_M / _r_g_meters(M_lens_Msun)


class SceneGeometry:
    """Bundle a lensing scene's redshifts and lens mass into geometric distances.

    The pipeline works internally in geometrized units where G = c = 1 and the
    unit of length is the gravitational radius r_g = G * M_lens / c^2. This
    class translates physical (z_L, z_S, M_lens) into (D_L, D_S, D_LS) in
    those units so callers can stay in the existing API.

    Parameters
    ----------
    z_L : float
        Lens redshift.
    z_S : float
        Source redshift. Must satisfy z_S > z_L.
    M_lens_Msun : float
        Total mass scale of the lens in solar masses. Sets the geometric
        unit. For SIS/SIE this is an effective mass; for a Schwarzschild
        BH it is the BH mass; for an NFW halo it is M_200.
    cosmology : LambdaCDM, optional
        If None, uses the default Planck18 LambdaCDM().

    Attributes
    ----------
    z_lens, z_source : float
    M_lens_Msun : float
    cosmology : LambdaCDM
    D_L, D_S, D_LS : float
        Angular-diameter distances in units of M_lens (geometrized).
    D_L_Mpc, D_S_Mpc, D_LS_Mpc : float
        Same distances in Mpc, kept for diagnostics and unit tests.

    Examples
    --------
    >>> scene = SceneGeometry(z_L=0.5, z_S=2.0, M_lens_Msun=3e11)
    >>> scene.D_L > 0
    True
    """

    def __init__(self, z_L, z_S, M_lens_Msun, cosmology=None):
        self.z_lens = float(z_L)
        self.z_source = float(z_S)
        self.M_lens_Msun = float(M_lens_Msun)
        self.cosmology = cosmology if cosmology is not None else LambdaCDM()

        D_L_Mpc = self.cosmology.angular_diameter_distance(self.z_lens)
        D_S_Mpc = self.cosmology.angular_diameter_distance(self.z_source)
        D_LS_Mpc = self.cosmology.D_LS(self.z_lens, self.z_source)
        self.D_L_Mpc = D_L_Mpc
        self.D_S_Mpc = D_S_Mpc
        self.D_LS_Mpc = D_LS_Mpc

        self.D_L = _Mpc_to_geometrized_M(D_L_Mpc, self.M_lens_Msun)
        self.D_S = _Mpc_to_geometrized_M(D_S_Mpc, self.M_lens_Msun)
        self.D_LS = _Mpc_to_geometrized_M(D_LS_Mpc, self.M_lens_Msun)

    def theta_E_SIS(self, sigma_v_kms):
        """Einstein-ring angular radius (radians) for an SIS lens.

        Uses theta_E = 4 * pi * (sigma_v/c)^2 * D_LS / D_S (dimensionless distances).
        """
        beta = (sigma_v_kms * 1000.0) / C_SI
        return 4.0 * np.pi * beta * beta * (self.D_LS_Mpc / self.D_S_Mpc)

    def __repr__(self):
        return (f"SceneGeometry(z_L={self.z_lens}, z_S={self.z_source}, "
                f"M_lens_Msun={self.M_lens_Msun:.3e}, "
                f"D_L={self.D_L:.3e} M, D_S={self.D_S:.3e} M, "
                f"D_LS={self.D_LS:.3e} M)")


###############################################################################

if __name__ == '__main__':
    # Quick smoke check.
    cosmo = LambdaCDM()
    print(f"Planck18: H0={cosmo.H0}, Om={cosmo.Om}, Ode={cosmo.Ode}")
    print(f"D_A(z=0.5) = {cosmo.angular_diameter_distance(0.5):.3f} Mpc")
    print(f"D_A(z=2.0) = {cosmo.angular_diameter_distance(2.0):.3f} Mpc")
    print(f"D_LS(0.3, 1.0) = {cosmo.D_LS(0.3, 1.0):.3f} Mpc")

    scene = SceneGeometry(z_L=0.5, z_S=2.0, M_lens_Msun=3e11)
    print(scene)
    print(f"theta_E(sigma_v=280 km/s) = "
          f"{scene.theta_E_SIS(280.0)*206264.8:.3f} arcsec")
