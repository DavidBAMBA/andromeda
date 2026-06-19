"""
Velocity models for source-plane emission in gravitational lensing.

The lensing renderer uses these models to assign a local 3-velocity to the
emitting material on the source plane. Velocities are dimensionless fractions
of c in the asymptotic lens-frame Cartesian basis:

    x : optical axis, positive from lens toward observer
    y : source-plane horizontal coordinate
    z : source-plane vertical coordinate

The returned velocity is the emitter velocity relative to the lens/observer
rest frame. The Doppler module then converts it into a redshift factor.
"""
from math import cos, sin, sqrt

import numpy as np


KIND_STATIC = 0
KIND_UNIFORM = 1
KIND_ROTATING_DISK = 2


class StaticSource:
    """Source material at rest in the asymptotic lens frame."""

    def __init__(self):
        self._kind = KIND_STATIC
        self._params = np.zeros(8, dtype=np.float64)

    def velocity(self, xs, ys):
        return np.zeros(3, dtype=np.float64)


class UniformVelocitySource:
    """Source plane with a constant 3-velocity beta = v/c."""

    def __init__(self, vx=0.0, vy=0.0, vz=0.0):
        self.beta = np.array([vx, vy, vz], dtype=np.float64)
        b2 = float(np.dot(self.beta, self.beta))
        if b2 >= 1.0:
            raise ValueError("Velocity magnitude must be < c.")
        self._kind = KIND_UNIFORM
        self._params = np.array(
            [self.beta[0], self.beta[1], self.beta[2], 0.0,
             0.0, 0.0, 0.0, 0.0],
            dtype=np.float64)

    def velocity(self, xs, ys):
        return self.beta.copy()


class RotatingDiskSource:
    """Simple rotating disk projected on the source plane.

    Parameters
    ----------
    v_max : float
        Asymptotic circular speed in units of c.
    r_turn : float
        Turnover radius for v(r) = v_max * r / sqrt(r^2 + r_turn^2).
    inclination : float
        Disk inclination in radians. 0 is face-on, pi/2 is edge-on.
    pa : float
        Position angle of the projected major axis in the source plane.
    x0, y0 : float
        Disk center in source-plane coordinates.

    Notes
    -----
    This is a kinematic model for Doppler debugging, not a full galaxy
    dynamics model. The source plane coordinates are (xs, ys) = (lens-frame
    y, z); the line of sight is lens-frame x.
    """

    def __init__(self, v_max=0.002, r_turn=1.0, inclination=0.0, pa=0.0,
                 x0=0.0, y0=0.0):
        if abs(v_max) >= 1.0:
            raise ValueError("v_max magnitude must be < c.")
        self.v_max = float(v_max)
        self.r_turn = float(r_turn)
        self.inclination = float(inclination)
        self.pa = float(pa)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self._kind = KIND_ROTATING_DISK
        self._params = np.array(
            [self.v_max, self.r_turn, self.inclination, self.pa,
             self.x0, self.y0, 0.0, 0.0],
            dtype=np.float64)

    def velocity(self, xs, ys):
        cpa = cos(self.pa)
        spa = sin(self.pa)
        dx = float(xs) - self.x0
        dy = float(ys) - self.y0

        # Projected disk axes on the source plane.
        x_major = cpa * dx + spa * dy
        y_minor_proj = -spa * dx + cpa * dy

        ci = cos(self.inclination)
        si = sin(self.inclination)
        if abs(ci) < 1e-8:
            ci = 1e-8 if ci >= 0.0 else -1e-8
        y_disk = y_minor_proj / ci
        r = sqrt(x_major * x_major + y_disk * y_disk)
        if r == 0.0:
            return np.zeros(3, dtype=np.float64)

        v_circ = self.v_max * r / sqrt(r * r + self.r_turn * self.r_turn)

        # Tangential velocity in the disk frame, then project into observer
        # Cartesian axes. The line-of-sight component is along lens-frame x.
        v_major = -v_circ * y_disk / r
        v_minor_disk = v_circ * x_major / r
        vx = v_minor_disk * si
        v_minor_proj = v_minor_disk * ci
        vy = cpa * v_major - spa * v_minor_proj
        vz = spa * v_major + cpa * v_minor_proj

        return np.array([vx, vy, vz], dtype=np.float64)
