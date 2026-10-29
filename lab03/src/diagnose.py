from pathlib import Path

from image_io import load_gray_float32
from harris_core import harris, harris_box
from nms import threshold_points, non_max_suppression
from compare import opencv_points, match_fraction

BASE_DIR = Path(__file__).resolve().parent.parent


def own_points(response, ratio, r=10, n=1000):
    xs, ys, sc = threshold_points(response, ratio)
    return non_max_suppression(xs, ys, sc, r, n)


_, gray = load_gray_float32(str(BASE_DIR / "data" / "image.jpg"))
r_gauss = harris(gray)

print("ratio  block | own(Gauss) own(box) lib | match Gauss  match box")
for bs in (5, 7, 9):
    r_box = harris_box(gray, bs)
    for ratio in (0.01, 0.05, 0.1):
        lib = opencv_points(gray, block_size=bs, ratio=ratio)
        g = own_points(r_gauss, ratio)
        b = own_points(r_box, ratio)
        print(f"{ratio:<6} {bs:<5} | {len(g):>9} {len(b):>8} {len(lib):>4} | "
              f"{match_fraction(g, lib) * 100:>10.1f}% {match_fraction(b, lib) * 100:>9.1f}%")