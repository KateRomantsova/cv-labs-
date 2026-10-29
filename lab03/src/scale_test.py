import sys
from pathlib import Path

import cv2

from image_io import load_gray_float32
from harris_core import harris
from nms import threshold_points, non_max_suppression
from compare import match_fraction

BASE_DIR = Path(__file__).resolve().parent.parent
RATIO, R_MIN_DIST, N_POINTS = 0.01, 10, 1000   
SCALE = 2
TOL = 3.0


def detect(gray):
    response = harris(gray)
    xs, ys, scores = threshold_points(response, RATIO)
    return non_max_suppression(xs, ys, scores, R_MIN_DIST, N_POINTS)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    stem = Path(name).stem
    img_bgr, gray = load_gray_float32(str(BASE_DIR / "data" / name))

    small = cv2.resize(gray, None, fx=1 / SCALE, fy=1 / SCALE,
                       interpolation=cv2.INTER_AREA)
    print("Оригінал:", gray.shape, "| зменшене:", small.shape)

    pts_orig = detect(gray)
    pts_small = detect(small)
    print("Точок на оригіналі:", len(pts_orig))
    print("Точок на зменшеному:", len(pts_small))

    pts_small_up = pts_small * SCALE
    frac = match_fraction(pts_orig, pts_small_up, tol=TOL)
    frac_rev = match_fraction(pts_small_up, pts_orig, tol=TOL)
    print(f"Частка точок оригіналу з парою (<{TOL:.0f} px): {frac * 100:.1f} %")
    print(f"(зворотна) частка точок зменшеного з парою в оригіналі: {frac_rev * 100:.1f} %")

    vis = img_bgr.copy()
    for x, y in pts_orig:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    for x, y in pts_small_up:
        cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), 1)

    (BASE_DIR / "figures").mkdir(exist_ok=True)
    (BASE_DIR / "results").mkdir(exist_ok=True)
    cv2.imwrite(str(BASE_DIR / "figures" / f"scale_compare_{stem}.png"), vis)
    (BASE_DIR / "results" / f"metrics_scale_{stem}.txt").write_text(
        f"orig={len(pts_orig)}, small={len(pts_small)}, "
        f"match_{TOL:.0f}px={frac * 100:.1f}%, reverse={frac_rev * 100:.1f}%\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()