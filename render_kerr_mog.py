"""
Render a Kerr-MOG+quintessence black hole image using 16 parallel workers.
Run: conda run -n engrenage python render_kerr_mog.py
"""
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be set before pyplot is imported

from math import pi
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from scr.black_holes.kerr_mog import BlackHole
from scr.accretion_structures.thin_disk import structure as ThinDisk
from scr.detectors.image_plane import detector as Detector
from scr.common.common import Image, set_ray_bounds

import warnings
warnings.filterwarnings("ignore")

# ── Black hole parameters ─────────────────────────────────────────────────────
a     = 0.3        # spin
alpha = 0.3        # MOG parameter (α=0 recovers Kerr)
c     = 0.005       # quintessence normalization (paper Table I)
w_q   = -2/3       # quintessence equation of state

# ── Detector parameters ───────────────────────────────────────────────────────
# Must have D < r_∞ (cosmological horizon) when c > 0.
# For (a=0.3, α=0.3, c=0.01, w_q=-2/3) → r_∞ ≈ 74, so D=40 fits comfortably.
D        = 70
iota     = (pi/180) * 85    # high inclination → top disk + lensed bottom ring
x_pixels = 1980               # 1980x1110 = 2.2M photons → ~1 min on 32 workers
x_side   = 25               # x_side/D preserves ~same angular field of view
ratio    = "16:9"

# ── Run config ────────────────────────────────────────────────────────────────
n_workers = 32
OUT_DIR   = "images"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Build scene ───────────────────────────────────────────────────────────────
bh  = BlackHole(a=a, alpha=alpha, c=c, w_q=w_q)

# Locate cosmological horizon (if any) and set ray-tracing bounds accordingly.
r_cosmo = None
if c > 0:
    try:
        r_cosmo = brentq(bh._Delta, bh.EH * 2, 500.0)
        print(f"Cosmological horizon r_∞ = {r_cosmo:.2f}")
        if D >= r_cosmo:
            raise ValueError(f"D={D} must be < r_∞={r_cosmo:.2f}. Reduce D or c.")
        # Push r_escape well beyond D but safely inside r_∞,
        # and lengthen final_lmbda so photons have runway to bend around the BH.
        r_escape_override = 0.9 * r_cosmo
        final_lmbda_override = 3.0 * r_escape_override
        set_ray_bounds(r_escape=r_escape_override,
                       final_lmbda=final_lmbda_override)
        print(f"Ray bounds: r_escape={r_escape_override:.2f}   "
              f"final_lmbda={final_lmbda_override:.2f}")
    except ValueError as e:
        if "sign" in str(e):
            print("No cosmological horizon found — spacetime is asymptotically flat.")
        else:
            raise

det = Detector(D=D, iota=iota, x_pixels=x_pixels, x_side=x_side, ratio=ratio)
acc = ThinDisk(bh)

print(f"\nKerr-MOG+Quint  a={a}  α={alpha}  c={c}  w_q={w_q}")
print(f"EH = {bh.EH:.4f}   ISCO_co = {bh.ISCOco:.4f}")
print(f"Disk: [{acc.in_edge:.2f}, {acc.out_edge:.2f}]")
print(f"Detector: {det.x_pixels}x{det.y_pixels} = "
      f"{det.x_pixels*det.y_pixels} photons   workers={n_workers}\n")

# ── Render ────────────────────────────────────────────────────────────────────
img = Image(bh, acc, det)
img.create_photons()
t0 = time.perf_counter()
img.create_image_no_Doppler(n_workers=n_workers)
wall = time.perf_counter() - t0

# ── Save data + figure ────────────────────────────────────────────────────────
tag = f"KerrMOG_a{a}_alpha{alpha}_c{c}_{x_pixels}x{det.y_pixels}_NoDoppler"
np.save(f"{OUT_DIR}/{tag}.npy", img.image_data)

data = img.image_data
norm = data / data.max() if data.max() > 0 else data

fig, ax = plt.subplots(figsize=(12, 12 * det.y_pixels / det.x_pixels))
ax.imshow(norm.T, cmap="magma", origin="lower")
ax.set_title(f"Kerr-MOG+Quint  a={a}, α={alpha}, c={c}   "
             f"({det.x_pixels}×{det.y_pixels}, {wall:.1f}s, {n_workers} workers)",
             fontsize=10)
ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/{tag}.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nTotal wall time : {wall:.2f} s")
print(f"Data saved to   : {OUT_DIR}/{tag}.npy")
print(f"Image saved to  : {OUT_DIR}/{tag}.png")
print(f"Non-zero pixels : {(data > 0).sum()} / {data.size}")
