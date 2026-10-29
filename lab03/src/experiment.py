import csv
import sys
from pathlib import Path

import numpy as np

import transforms as T
from detectors import DETECTORS
from image_io import load_gray_float32
from repeatability import repeatability

BASE_DIR = Path(__file__).resolve().parent.parent


def build_grid():
    """Крок 16: сітка експериментів."""
    grid = []
    grid += [("rotation", float(a), lambda g, a=a: T.rotation(g, a)) for a in range(0, 91, 10)]
    grid += [("scale", float(s), lambda g, s=s: T.scale(g, s))
             for s in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)]
    grid += [("brightness", float(f), lambda g, f=f: T.brightness(g, f))
             for f in np.round(np.arange(0.5, 1.51, 0.1), 1)]
    grid += [("noise", float(sg), lambda g, sg=sg: T.gaussian_noise(g, sg))
             for sg in (0, 5, 10, 15, 20)]
    return grid


def main():
    data_dir = BASE_DIR / "data"
    if len(sys.argv) > 1:
        paths = [data_dir / n for n in sys.argv[1:]]
    else:
        paths = sorted(p for p in data_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print("Зображень:", len(paths), [p.name for p in paths])

    grid = build_grid()
    rows = []
    for path in paths:
        _, gray = load_gray_float32(str(path))
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        base = {name: det(gray_u8) for name, det in DETECTORS.items()}   # P1

        for i, (kind, param, fn) in enumerate(grid, 1):
            warped, H = fn(gray_u8)
            for name, det in DETECTORS.items():
                p2 = det(warped)                                          # P2
                rep, n1, n2 = repeatability(base[name], p2, H, gray_u8.shape, warped.shape)
                rows.append([path.name, kind, param, name, f"{rep:.4f}", n1, n2])
            print(f"{path.name}: {i}/{len(grid)} ({kind} {param})")

    (BASE_DIR / "results").mkdir(exist_ok=True)
    out = BASE_DIR / "results" / "repeatability.csv"
    try:
        f = open(out, "w", newline="", encoding="utf-8")
    except PermissionError:
        out = BASE_DIR / "results" / "repeatability_new.csv"
        print("repeatability.csv зайнятий іншою програмою, зберігаю в", out.name)
        f = open(out, "w", newline="", encoding="utf-8")
    with f:
        w = csv.writer(f)
        w.writerow(["image", "transform", "param", "detector", "repeatability", "n_p1", "n_p2"])
        w.writerows(rows)
    print("Збережено:", out)


if __name__ == "__main__":
    main()