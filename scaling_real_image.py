"""
Scaling test: genera imagen Kerr 200x200 con 1,2,4,8,16 procesos.
Run: conda run -n engrenage python scaling_real_image.py
"""
from math import pi
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scr.black_holes.kerr import BlackHole
from scr.accretion_structures.thin_disk import structure as ThinDisk
from scr.detectors.image_plane import detector as Detector
from scr.common.common import Image

import warnings
warnings.filterwarnings("ignore")

# ── Escena ────────────────────────────────────────────────────────────────────
a         = 0.7
D         = 100
iota      = pi/180 * 85.
x_pixels  = 1980
x_side    = 25
workers   = [1, 2, 4, 8, 16, 24, 32]

blackhole = BlackHole(a)
det       = Detector(D=D, iota=iota, x_pixels=x_pixels,
                     x_side=x_side, ratio="16:9")
acc       = ThinDisk(blackhole)

OUT_DIR = "images/tests_parallel"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\nKerr a={a}  |  {det.x_pixels}x{det.y_pixels} = "
      f"{det.x_pixels*det.y_pixels} fotones\n")

# ── Scaling loop ──────────────────────────────────────────────────────────────
results = []
ref_image = None

for nw in workers:
    img = Image(blackhole, acc, det)
    img.create_photons()
    t0 = time.perf_counter()
    img.create_image_no_Doppler(n_workers=nw)
    wall = time.perf_counter() - t0
    results.append((nw, wall))
    # Guardar imagen BH de cada run
    data = img.image_data
    norm = data / data.max() if data.max() > 0 else data
    fig_bh, ax_bh = plt.subplots(figsize=(5, 5))
    ax_bh.imshow(norm.T, cmap="afmhot", origin="lower")
    ax_bh.set_title(f"Kerr a={a} — {nw} worker{'s' if nw > 1 else ''} ({wall:.1f}s)",
                    fontsize=9)
    ax_bh.axis("off")
    fig_bh.tight_layout()
    fig_bh.savefig(f"{OUT_DIR}/bh_{nw}workers.png", dpi=120)
    plt.close(fig_bh)

    if ref_image is None:
        ref_image = img.image_data.copy()
    print(f"  n_workers={nw:2d}  wall={wall:7.2f}s  "
          f"t/ph={wall/img._stats['n_workers']/1000*1e6:.0f}µs")   # approx

t1 = results[0][1]
print("\n  n_workers | wall [s] | speedup | efficiency")
print("  " + "-"*42)
for nw, t in results:
    sp  = t1 / t
    eff = sp / nw * 100
    print(f"  {nw:9d} | {t:8.2f} | {sp:7.2f}x | {eff:9.1f}%")


# ── Plot scaling ──────────────────────────────────────────────────────────────
nw_arr   = np.array([r[0] for r in results])
wall_arr = np.array([r[1] for r in results])
sp_arr   = t1 / wall_arr
eff_arr  = sp_arr / nw_arr * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ── Izquierda: speedup ────────────────────────────────────────────────────────
ax1.plot(nw_arr, sp_arr, "o-", lw=2, ms=8, color="C0", label="medido")
ax1.plot(nw_arr, nw_arr, "k--", alpha=0.4, label="ideal")
for nw, sp, eff in zip(nw_arr, sp_arr, eff_arr):
    ax1.annotate(f"{sp:.2f}×\n({eff:.0f}%)",
                 xy=(nw, sp), xytext=(4, 6), textcoords="offset points",
                 fontsize=8, color="C0")
ax1.set_xlabel("n_workers")
ax1.set_ylabel("speedup")
ax1.set_title(f"Speedup — Kerr a={a}, {det.x_pixels}×{det.y_pixels}")
ax1.legend()
ax1.grid(alpha=0.3)

# ── Derecha: tiempo total por configuración ───────────────────────────────────
colors = [f"C{i}" for i in range(len(workers))]
bars = ax2.bar([str(nw) for nw in nw_arr], wall_arr,
               color=colors, edgecolor="black", linewidth=0.6)
for bar, t in zip(bars, wall_arr):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{t:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xlabel("n_workers")
ax2.set_ylabel("tiempo total [s]")
ax2.set_title("Tiempo total por configuración")
ax2.grid(axis="y", alpha=0.3)

plt.suptitle(f"Paralelización — Kerr a={a},  {det.x_pixels}×{det.y_pixels} pixeles",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/scaling_speedup_time.png", dpi=120)
print(f"\nTodos los plots → {OUT_DIR}/")
