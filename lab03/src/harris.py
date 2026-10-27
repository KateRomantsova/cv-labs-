from pathlib import Path

import cv2

from image_io import load_gray_float32
from harris_core import harris
from nms import threshold_points, non_max_suppression
from compare import opencv_points, match_fraction

BASE_DIR = Path(__file__).resolve().parent.parent
RATIO, R_MIN_DIST, N_POINTS = 0.01, 10, 1000


def draw(img, pts, color):
    out = img.copy()
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), 5, color, 1)
    return out


def main():
    img_bgr, gray = load_gray_float32(str(BASE_DIR / "data" / "image.jpg"))
    print("Розмір:", gray.shape, "| тип:", gray.dtype)

    # кроки 2-5
    response = harris(gray)

    # кроки 6-7
    xs, ys, scores = threshold_points(response, RATIO)
    print("Кандидатів після порога:", len(xs))
    own = non_max_suppression(xs, ys, scores, R_MIN_DIST, N_POINTS)
    print("Власних точок після NMS:", len(own))

    # крок 8
    lib = opencv_points(gray, ratio=RATIO, r=R_MIN_DIST, n_max=N_POINTS)
    frac = match_fraction(own, lib, tol=2.0)
    print("Точок cv2.cornerHarris:", len(lib))
    print(f"Збіг із OpenCV (<2 px): {frac * 100:.1f} %")

    # збереження результатів
    (BASE_DIR / "figures").mkdir(exist_ok=True)
    (BASE_DIR / "results").mkdir(exist_ok=True)
    cv2.imwrite(str(BASE_DIR / "figures" / "harris_own.png"), draw(img_bgr, own, (0, 0, 255)))
    cv2.imwrite(str(BASE_DIR / "figures" / "harris_opencv.png"), draw(img_bgr, lib, (0, 255, 0)))
    (BASE_DIR / "results" / "metrics.txt").write_text(
        f"own={len(own)}, opencv={len(lib)}, match_2px={frac * 100:.1f}%\n", encoding="utf-8")


if __name__ == "__main__":
    main()