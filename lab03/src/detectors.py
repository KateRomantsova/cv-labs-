import cv2
import numpy as np

from config import N_POINTS, NMS_RADIUS, HARRIS_RATIO
from harris_core import harris
from nms import threshold_points, non_max_suppression

_sift = cv2.SIFT_create()
_orb = cv2.ORB_create(nfeatures=20000)


def _select(xs, ys, scores):
    """Єдиний відбір для всіх: NMS з радіусом r, потім N найкращих за відгуком."""
    return non_max_suppression(xs, ys, scores, NMS_RADIUS, N_POINTS)


def detect_harris(gray_u8):
    response = harris(gray_u8.astype(np.float32))
    xs, ys, scores = threshold_points(response, HARRIS_RATIO)
    return _select(xs, ys, scores)


def _from_keypoints(kps):
    if not kps:
        return np.empty((0, 2), np.float32)
    pts = np.array([k.pt for k in kps], np.float32)
    resp = np.array([k.response for k in kps], np.float64)
    return _select(pts[:, 0], pts[:, 1], resp)


def detect_sift(gray_u8):
    return _from_keypoints(_sift.detect(gray_u8, None))


def detect_orb(gray_u8):
    return _from_keypoints(_orb.detect(gray_u8, None))


DETECTORS = {"Harris": detect_harris, "SIFT": detect_sift, "ORB": detect_orb}