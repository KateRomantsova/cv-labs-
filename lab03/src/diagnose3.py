from pathlib import Path

import cv2
import numpy as np

from image_io import load_gray_float32
from harris_core import harris
from nms import threshold_points, non_max_suppression
from compare import opencv_points, match_fraction

BASE_DIR = Path(__file__).resolve().parent.parent
NAME = "image.jpg" 


def own_points(response, ratio, r=10, n=1000):
    xs, ys, sc = threshold_points(response, ratio)
    return non_max_suppression(xs, ys, sc, r, n)


img, gray = load_gray_float32(str(BASE_DIR / "data" / NAME))
resp = harris(gray)

print("ratio | own  lib | match 2px")
for ratio in (0.01, 0.05, 0.1, 0.2):
    own = own_points(resp, ratio)
    lib = opencv_points(gray, block_size=5, ratio=ratio)
    print(f"{ratio:<5} | {len(own):>4} {len(lib):>4} | {match_fraction(own, lib) * 100:>6.1f}%")

own = own_points(resp, 0.01)
lib = opencv_points(gray, block_size=5, ratio=0.01)
dist = np.linalg.norm(own[:, None] - lib[None], axis=2).min(axis=1)
out = img.copy()
for (x, y), d in zip(own, dist):
    color = (0, 255, 255) if d >= 2 else (0, 200, 0)   # жовті = без пари
    cv2.circle(out, (int(x), int(y)), 5, color, 1)
(BASE_DIR / "figures").mkdir(exist_ok=True)
cv2.imwrite(str(BASE_DIR / "figures" / "unmatched.png"), out)