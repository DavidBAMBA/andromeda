"""
Test de paridad: correr la misma imagen 32x18 en serial y paralelo,
comparar pixel a pixel.

Run from repo root:
    conda run -n engrenage python -m scr.common.test_parallel
"""
from math import pi
import numpy as np

from scr.black_holes import kerr
from scr.accretion_structures import thin_disk
from scr.detectors import image_plane
from scr.common.common import Image


def build_image():
    blackhole = kerr.BlackHole(0.9)
    det = image_plane.detector(D=100, iota=pi/180*80,
                               x_pixels=32, x_side=25, ratio="16:9")
    acc = thin_disk.structure(blackhole)
    img = Image(blackhole, acc, det)
    img.create_photons()
    return img


def main():
    print("== Modo serial (n_workers=1) ==")
    img_s = build_image()
    img_s.create_image(n_workers=1)
    data_s = img_s.image_data.copy()
    t_s = img_s._stats["wall_total"]

    print("\n== Modo paralelo (n_workers=4) ==")
    img_p = build_image()
    img_p.create_image(n_workers=4)
    data_p = img_p.image_data.copy()
    t_p = img_p._stats["wall_total"]

    # NaN-safe comparison: a pixel is equal if both NaN or both finite-with-diff
    nan_s = np.isnan(data_s)
    nan_p = np.isnan(data_p)
    nan_match = np.array_equal(nan_s, nan_p)
    diff = np.abs(np.nan_to_num(data_s) - np.nan_to_num(data_p))
    max_diff = float(diff.max())
    denom = max(1e-30, float(np.nanmax(np.abs(data_s))))
    rel = max_diff / denom
    print(f"\nNaN mask match          = {nan_match}")
    print(f"max|serial - paralelo|  = {max_diff:.3e}  (ignorando NaN)")
    print(f"relative                = {rel:.3e}")
    print(f"speedup                 = {t_s/t_p:.2f}x ({t_s:.2f}s -> {t_p:.2f}s)")

    tol = 1e-10
    if nan_match and max_diff < tol:
        print(f"PARIDAD OK (tol = {tol:.0e})")
    else:
        print(f"DIFERENCIA EXCEDE TOLERANCIA ({tol:.0e})")

    # Status consistency check
    st_s = img_s._stats["status"]
    st_p = img_p._stats["status"]
    c_s = {k: st_s.count(k) for k in set(st_s)}
    c_p = {k: st_p.count(k) for k in set(st_p)}
    print(f"\nSerial   statuses: {c_s}")
    print(f"Paralelo statuses: {c_p}")


if __name__ == "__main__":
    main()
