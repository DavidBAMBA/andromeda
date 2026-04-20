"""
Render images for all black holes in scr/black_holes/ (except Kerr-MOG).
Run: conda run -n engrenage python render_all_blackholes.py
"""
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be set before pyplot is imported

from math import pi
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scr.accretion_structures.thin_disk import structure as ThinDisk
from scr.detectors.image_plane import detector as Detector
from scr.common.common import Image

import warnings
warnings.filterwarnings("ignore")

# ── Common detector config ────────────────────────────────────────────────────
D        = 100
iota     = (pi / 180) * 85
x_pixels = 200
x_side   = 25
ratio    = "16:9"
N_WORKERS = 16

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR  = "images/all_blackholes"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Black hole registry ───────────────────────────────────────────────────────
# Each entry: (label, factory, mode)
#   mode = "doppler"    → BH has Omega(), full Doppler shift
#   mode = "no_doppler" → BH has no Omega(), intensity only
def _make_cases():
    from scr.black_holes.schwarzschild import BlackHole as SchwBH
    from scr.black_holes.num_schwarzschild import BlackHole as NumSchwBH
    from scr.black_holes.kerr import BlackHole as KerrBH
    from scr.black_holes.scalar_hair_BH import BlackHole as ScalarBH

    cases = [
        ("Schwarzschild",        SchwBH(),        "no_doppler"),
        ("NumSchwarzschild",     NumSchwBH(),     "no_doppler"),
        ("Kerr_a0.3",            KerrBH(a=0.3),   "doppler"),
        ("Kerr_a0.7",            KerrBH(a=0.7),   "no_doppler"),
        ("Kerr_a0.99",           KerrBH(a=0.99),  "doppler"),
        ("ScalarHair_phi5_pp1.6", ScalarBH(),     "no_doppler"),
    ]
    return cases

# ── Render loop ───────────────────────────────────────────────────────────────
cases = _make_cases()
results = []

for label, bh, mode in cases:
    print(f"\n{'='*60}")
    print(f"  {label}   EH={bh.EH:.4f}   ISCO={bh.ISCOco:.4f}   mode={mode}")
    print(f"{'='*60}")

    det = Detector(D=D, iota=iota, x_pixels=x_pixels, x_side=x_side, ratio=ratio)
    acc = ThinDisk(bh)

    print(f"Disk: [{acc.in_edge:.2f}, {acc.out_edge:.2f}]")
    print(f"Pixels: {det.x_pixels}x{det.y_pixels} = {det.x_pixels*det.y_pixels}   workers={N_WORKERS}")

    img = Image(bh, acc, det)
    img.create_photons()

    t0 = time.perf_counter()
    if mode == "doppler":
        img.create_image(n_workers=N_WORKERS)
    else:
        img.create_image_no_Doppler(n_workers=N_WORKERS)
    wall = time.perf_counter() - t0

    # ── Save raw data ─────────────────────────────────────────────────────────
    tag = f"{label}_{x_pixels}x{det.y_pixels}"
    np.save(f"{OUT_DIR}/{tag}.npy", img.image_data)

    # ── Save figure ───────────────────────────────────────────────────────────
    data = img.image_data
    norm = data / data.max() if data.max() > 0 else data
    fig, ax = plt.subplots(figsize=(10, 10 * det.y_pixels / det.x_pixels))
    ax.imshow(norm.T, cmap="afmhot", origin="lower")
    ax.set_title(
        f"{label}   EH={bh.EH:.3f}   ({det.x_pixels}×{det.y_pixels}, "
        f"{wall:.1f}s, {N_WORKERS} workers)",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    nonzero = (data > 0).sum()
    print(f"Wall: {wall:.2f}s   Non-zero pixels: {nonzero}/{data.size}")
    results.append((label, wall, nonzero, data.size))

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {'Black hole':<28} {'time (s)':>9}  {'nonzero':>8}  {'fill%':>6}")
print(f"  {'-'*56}")
for label, wall, nz, total in results:
    print(f"  {label:<28} {wall:>9.2f}  {nz:>8}  {100*nz/total:>5.1f}%")
print(f"\nAll images saved to {OUT_DIR}/")
