"""
===============================================================================
Horizontal 2-D diagnostic: trace individual null geodesics through the SIS
lens, in the physical direction (source -> lens -> observer), and plot
the photon paths bending as they pass near the lens.

Straight dashed grey lines show the "no-lensing" counterfactual trajectories
so the deflection is visually obvious. Same SIS parameters as ex10.
===============================================================================
"""

from math import pi, cos, sin, sqrt, atan2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scr.lens_metrics import sis
from scr.common.integrator import integrate




'''
===============================================================================
=============================== LENS DEFINITION ===============================
===============================================================================
'''
sigma_v = 0.03
lens = sis.LensMetric(sigma_v=sigma_v, r_ref=1.0, r_min=1e-2)



'''
===============================================================================
================================= GEOMETRY ====================================
===============================================================================
'''
D_L = 1.0e4
D_LS = 1.0e4
b_ring = 4.0 * pi * sigma_v**2 * D_LS     # ~113 M



'''
===============================================================================
========================== PHOTON IMPACT PARAMETERS ===========================
===============================================================================
'''
# Symmetric set; include the Einstein radius for pedagogy
impact_params = [-300, -200, -b_ring, -60, -20, 20, 60, b_ring, 200, 300]



'''
===============================================================================
=============================== INTEGRATION ===================================
===============================================================================
'''
def trace_photon_source_to_observer(b, final_lmbda=3.0 * (D_L + D_LS)):
    """Integrate a null geodesic in theta = pi/2 equatorial plane, from
    the source side (x = -D_LS) toward the observer (x = +D_L), with
    impact parameter b at large distance.

    Null IC in weak field:
        k_t = -1  (k^t = 1 choice)
        k_r = cos(phi0)    (v^r = dr/dlmbda = cos(phi0) for velocity +x)
        k_phi = -b         (L_z = -b for (x, y) = (-D_LS, b), v = (+1, 0, 0))
    """
    r0 = float(sqrt(D_LS * D_LS + b * b))
    phi0 = float(atan2(b, -D_LS))       # in Q2 for b>0, Q3 for b<0
    theta0 = pi / 2
    q0 = np.array([0.0, r0, theta0, phi0,
                   -1.0,
                   cos(phi0),
                   0.0,
                   -float(b)], dtype=np.float64)

    def rhs(lmbda, y):
        return lens.geodesics(y, lmbda)

    res = integrate(rhs, q0, (0.0, final_lmbda),
                    method="DOP853", rtol=1e-9, atol=1e-11)
    r = res.y[:, 1]; phi = res.y[:, 3]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    # Stop at the observer plane (x >= D_L)
    mask = x <= D_L + 10
    return x[mask], y[mask]


trajectories = [trace_photon_source_to_observer(b) for b in impact_params]



'''
===============================================================================
=================================== PLOT ======================================
===============================================================================
'''
fig, ax = plt.subplots(figsize=(13, 5.8))

# Straight-line (no-lensing) trajectories for comparison: light grey dashed
for b in impact_params:
    ax.plot([-D_LS, D_L], [b, b], color='lightgrey', lw=0.8, ls='--',
            zorder=1)

# Actual GR trajectories
for b, (x, y) in zip(impact_params, trajectories):
    color = 'tab:blue' if b > 0 else 'tab:red'
    ax.plot(x, y, color=color, lw=1.4, alpha=0.9, zorder=3)

# Einstein radius guides
ax.axhline(+b_ring, color='green', lw=0.7, ls=':', alpha=0.6, zorder=2)
ax.axhline(-b_ring, color='green', lw=0.7, ls=':', alpha=0.6, zorder=2)
ax.text(D_L*0.55, +b_ring + 18, f'+b_Einstein = {b_ring:.1f} M',
        color='green', fontsize=8, alpha=0.85)
ax.text(D_L*0.55, -b_ring - 35, f'-b_Einstein = {-b_ring:.1f} M',
        color='green', fontsize=8, alpha=0.85)

# Source plane + source marker at (0, 0)
ax.axvline(-D_LS, color='gray', lw=1.5, ls='-', alpha=0.5, zorder=2)
ax.plot(-D_LS, 0, 'o', color='royalblue', markersize=14,
        markeredgecolor='white', markeredgewidth=0.8, zorder=6)
ax.text(-D_LS, 420, 'SOURCE PLANE\n(x = -D_LS)',
        ha='center', fontsize=9.5, color='dimgray')
ax.text(-D_LS - 700, 0, 'source', ha='right', va='center',
        fontsize=9, color='royalblue')

# Lens at origin
ax.plot(0, 0, 'o', color='orange', markersize=22,
        markeredgecolor='black', markeredgewidth=0.8, zorder=10)
ax.text(0, -140, 'LENS\n(SIS galaxy)', ha='center', fontsize=9.5,
        color='black')

# Observer plane + schematic detector pixels
ax.axvline(+D_L, color='gray', lw=1.5, ls='-', alpha=0.5, zorder=2)
for yy in np.linspace(-350, 350, 15):
    ax.plot([D_L, D_L], [yy - 9, yy + 9], 'k-', lw=1.8, zorder=5)
ax.text(+D_L, 420, 'OBSERVER / DETECTOR\n(x = +D_L)',
        ha='center', fontsize=9.5, color='dimgray')

# Arrows at the observer end showing propagation direction
for b, (x, y) in zip(impact_params, trajectories):
    if len(x) < 3: continue
    dx = x[-1] - x[-2]; dy = y[-1] - y[-2]
    norm = sqrt(dx*dx + dy*dy)
    if norm > 0:
        color = 'tab:blue' if b > 0 else 'tab:red'
        ax.annotate('', xy=(x[-1], y[-1]),
                    xytext=(x[-1] - 400*dx/norm, y[-1] - 400*dy/norm),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

ax.set_xlabel('x  [M]')
ax.set_ylabel('y  [M]   (vertical scale magnified)')
ax.set_xlim(-1.12 * D_LS, 1.12 * D_L)
ax.set_ylim(-500, 500)
ax.grid(alpha=0.15)
ax.set_title(f'SIS lensing — null geodesics, source → observer  '
             f'(σ_v = {sigma_v},  b_Einstein = {b_ring:.1f} M)')

blue_line = mlines.Line2D([], [], color='tab:blue', lw=1.4,
                           label='GR trajectory (b > 0)')
red_line  = mlines.Line2D([], [], color='tab:red',  lw=1.4,
                           label='GR trajectory (b < 0)')
grey_line = mlines.Line2D([], [], color='lightgrey', lw=0.8, ls='--',
                           label='Straight line (no lensing)')
ein_line  = mlines.Line2D([], [], color='green', lw=0.7, ls=':',
                           label='Einstein radius')
ax.legend(handles=[blue_line, red_line, grey_line, ein_line],
          loc='lower right', fontsize=8.5, framealpha=0.9)

plt.tight_layout()
plt.savefig('images/lensing_trajectory.png', dpi=200, bbox_inches='tight')
plt.show()
