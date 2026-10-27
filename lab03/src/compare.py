import cv2
import numpy as np

from nms import threshold_points, non_max_suppression


def opencv_points(gray, block_size=7, ksize=3, k=0.04,
                  ratio=0.01, r=10, n_max=1000):
    """Точки cv2.cornerHarris: той самий поріг і те саме придушення, що й у власній версії."""
    response = cv2.cornerHarris(gray, block_size, ksize, k)
    xs, ys, scores = threshold_points(response, ratio)
    return non_max_suppression(xs, ys, scores, r, n_max)


def match_fraction(own, ref, tol=2.0):
    """Крок 8. Частка власних точок, для яких є точка з бібліотеки ближче за tol пікселів."""
    if len(own) == 0 or len(ref) == 0:
        return 0.0
    dist = np.linalg.norm(own[:, None, :] - ref[None, :, :], axis=2)
    return float(np.mean(dist.min(axis=1) < tol))