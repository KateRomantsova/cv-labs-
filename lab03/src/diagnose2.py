from pathlib import Path

from image_io import load_gray_float32
from harris_core import harris, harris_box
from nms import threshold_points, non_max_suppression
from compare import opencv_points, match_fraction

BASE_DIR = Path(__file__).resolve().parent.parent
NAME = "image.jpg"   


def own_points(response, ratio=0.01, r=10, n=1000):
    xs, ys, sc = threshold_points(response, ratio)
    return non_max_suppression(xs, ys, sc, r, n)


_, gray = load_gray_float32(str(BASE_DIR / "data" / NAME))
gauss = own_points(harris(gray))

print("block | match box (2px) | match Gauss: 2px   3px   5px")
for bs in (3, 5, 7):
    lib = opencv_points(gray, block_size=bs)
    box = own_points(harris_box(gray, bs))
    print(f"{bs:<5} | {match_fraction(box, lib) * 100:>13.1f}% | "
          + "   ".join(f"{match_fraction(gauss, lib, tol=t) * 100:>5.1f}%" for t in (2, 3, 5)))