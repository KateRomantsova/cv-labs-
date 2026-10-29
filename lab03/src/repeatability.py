import cv2
import numpy as np

from config import EPS, BORDER_MARGIN


def valid_mask(src_shape, H, dst_shape, margin=BORDER_MARGIN):
    """Область перетвореного кадру, яку покриває оригінал (мінус смужка біля межі)."""
    h, w = src_shape
    dh, dw = dst_shape
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (dw, dh),
                               flags=cv2.INTER_NEAREST)
    k = 2 * margin + 1
    mask = cv2.erode(mask, np.ones((k, k), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask > 0


def _inside(points, mask):
    inside = np.zeros(len(points), bool)
    if len(points) == 0:
        return inside
    x = np.round(points[:, 0]).astype(int)
    y = np.round(points[:, 1]).astype(int)
    ok = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    inside[ok] = mask[y[ok], x[ok]]
    return inside


def repeatability(p1, p2, H, src_shape, dst_shape, eps=EPS):
    """Крок 15. Повертає (повторюваність, |P1|, |P2|) у спільній області."""
    mask = valid_mask(src_shape, H, dst_shape)

    # 2) переносимо P1 через H
    if len(p1):
        p1 = cv2.perspectiveTransform(p1.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    p1 = p1[_inside(p1, mask)]
    p2 = p2[_inside(p2, mask)]

    denom = min(len(p1), len(p2))
    if denom == 0:
        return 0.0, len(p1), len(p2)

    # 3) для скількох точок P1 є точка P2 ближче за ε
    dist = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
    matched = int((dist.min(axis=1) < eps).sum())

    # 4) ділимо на min(|P1|, |P2|); обмеження 1.0 страхує від рідкісного випадку,
    #    коли дві сусідні точки P1 збігаються з однією точкою P2
    return min(1.0, matched / denom), len(p1), len(p2)