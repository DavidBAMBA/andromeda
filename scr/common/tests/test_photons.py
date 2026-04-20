"""
Test físico: 5 fotones hacia Kerr a=0.9, un integrador por fotón.
Muestra destino, r_final, pasos, tiempo, y grafica r(λ) para cada uno.

Run from repo root:
    conda run -n engrenage python -m scr.common.test_photons
"""
from math import pi
import numpy as np
import matplotlib.pyplot as plt

from scr.black_holes.kerr import BlackHole
from scr.accretion_structures.thin_disk import structure as ThinDisk
from scr.detectors.image_plane import detector as Detector
from scr.common.integrator import integrate, make_events

# ── Escena ──────────────────────────────────────────────────────────────────
a = 0.9
blackhole = BlackHole(a)
acc      = ThinDisk(blackhole)
det      = Detector(D=100, iota=pi/180*80, x_pixels=16, x_side=25, ratio="1:1")

# Función RHS compatible con solve_ivp / integrate (lmbda, q)
def rhs(lmbda, q):
    return blackhole.geodesics(q, lmbda)

# Eventos: horizonte + disco + escape
events = make_events(blackhole, acc_structure=acc, r_escape=1.1*det.D)

# ── 5 fotones con condiciones iniciales distintas ────────────────────────────
# Tomamos alpha, beta del plano imagen y construimos iC con el detector
alphas = [-15., -5., 0., 5., 15.]
betas  = [  0.,  0., 0., 0.,  0.]

photons = [det.photon_coords(blackhole, a_px, b_px)
           for a_px, b_px in zip(alphas, betas)]

METHODS = ["LSODA", "DOP853", "RK45", "Verlet"]
LABELS  = [f"α={ap}" for ap in alphas]

# Referencia: DOP853 con tolerancias muy tight
REF_METHOD = "DOP853"
REF_TOL    = dict(rtol=1e-13, atol=1e-13)
TOL        = dict(rtol=1e-9,  atol=1e-11)

# ── Integración: cada método corre los 5 fotones ─────────────────────────────
final_lmbda = 1.5 * det.D

# results[method][photon]
all_results = {}
for method in METHODS + [REF_METHOD + "_ref"]:
    m = REF_METHOD if method.endswith("_ref") else method
    tol = REF_TOL if method.endswith("_ref") else (
          dict(first_step=0.05) if m == "Verlet" else TOL)
    all_results[method] = []
    for iC in photons:
        res = integrate(rhs, iC, (0.0, -final_lmbda),
                        method=m, events=events, **tol)
        all_results[method].append(res)

# ── Tabla comparativa ────────────────────────────────────────────────────────
ref_key = REF_METHOD + "_ref"
print(f"\n{'Fotón':>6} {'α':>6} {'Método':>7} {'status':>10} "
      f"{'r_final':>8} {'|Δr| vs ref':>12} {'nfev':>6} {'t [ms]':>8}")
print("─" * 75)
for i, label in enumerate(LABELS):
    r_ref = all_results[ref_key][i].y[-1, 1]
    for method in METHODS:
        res = all_results[method][i]
        r_f = res.y[-1, 1]
        dr  = abs(r_f - r_ref)
        print(f"{i+1:>6} {label:>6} {method:>7} {res.status:>10} "
              f"{r_f:>8.4f} {dr:>12.2e} {res.nfev:>6} "
              f"{1000*res.wall_time:>8.2f}")
    print()

# ── Plot: r(λ) por fotón, una línea por método ───────────────────────────────
method_styles = {
    "LSODA":  ("C0", "-",  1.5),
    "DOP853": ("C1", "--", 1.5),
    "RK45":   ("C2", "-.", 1.5),
    "Verlet": ("C3", ":",  1.8),
}

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for i, (ax, label) in enumerate(zip(axes, LABELS)):
    for method in METHODS:
        res = all_results[method][i]
        color, ls, lw = method_styles[method]
        ax.plot(-res.t, res.y[:, 1], color=color, ls=ls, lw=lw,
                label=f"{method} ({res.status})", alpha=0.85)

    ax.axhline(blackhole.EH,  color="red",    ls="--", lw=0.8,
               label=f"r+ ={blackhole.EH:.2f}")
    ax.axhline(acc.in_edge,   color="orange", ls="--", lw=0.8,
               label=f"ISCO={acc.in_edge:.2f}")
    ax.axhline(acc.out_edge,  color="green",  ls=":",  lw=0.8,
               label=f"R_out={acc.out_edge:.0f}")

    ax.set_title(label, fontsize=9)
    ax.set_xlabel(r"$-\lambda$")
    if i == 0:
        ax.set_ylabel(r"$r / M$")
    ax.set_ylim(0, 35)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, loc="upper right")

plt.suptitle(f"Kerr a={a}  —  5 fotones × 4 integradores", fontsize=11)
plt.tight_layout()
plt.savefig("images/test_5photons.png", dpi=130)
print("Plot guardado en images/test_5photons.png")
plt.show()
