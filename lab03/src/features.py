import sys
import time
from pathlib import Path

import cv2
import numpy as np

from image_io import load_gray_float32

BASE_DIR = Path(__file__).resolve().parent.parent
REPEATS = 5


def run(detector, gray_u8, repeats=REPEATS):
    detector.detectAndCompute(gray_u8, None)           
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        kps, desc = detector.detectAndCompute(gray_u8, None)
        times.append(time.perf_counter() - t0)
    return kps, desc, float(np.mean(times))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    stem = Path(name).stem
    img_bgr, gray = load_gray_float32(str(BASE_DIR / "data" / name))

    # SIFT і ORB працюють із 8-бітним зображенням, тому float32 з кроку 1 повертаємо в uint8
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    print("Зображення:", gray_u8.shape, gray_u8.dtype)

    # крок 11
    detectors = {
        "SIFT": (cv2.SIFT_create(), (0, 255, 0)),
        "ORB": (cv2.ORB_create(nfeatures=1000), (0, 255, 255)),
    }

    (BASE_DIR / "figures").mkdir(exist_ok=True)
    (BASE_DIR / "results").mkdir(exist_ok=True)
    lines = []

    for label, (det, color) in detectors.items():
        # крок 12
        kps, desc, t = run(det, gray_u8)
        info = (f"{label}: точок = {len(kps)}, форма дескрипторів = {desc.shape}, "
                f"розмірність = {desc.shape[1]}, тип = {desc.dtype}, час = {t * 1000:.1f} мс")
        print(info)
        lines.append(info)

        # крок 13
        vis = cv2.drawKeypoints(img_bgr, kps, None, color=color,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(str(BASE_DIR / "figures" / f"{label.lower()}_{stem}.png"), vis)

    (BASE_DIR / "results" / f"metrics_features_{stem}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()