import cv2
import numpy as np


def load_gray_float32(path: str, max_width: int = 1000):
    """Крок 1. Повертає (кольорове BGR uint8, сіре float32)."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не вдалося прочитати зображення: {path}")

    h, w = img_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        img_bgr = cv2.resize(img_bgr, (max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img_bgr, gray