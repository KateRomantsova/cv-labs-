import cv2
import numpy as np


def rotation(gray_u8, theta):
    """Поворот на θ° навколо центру. H = getRotationMatrix2D + рядок (0, 0, 1)."""
    h, w = gray_u8.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)      # 2×3
    H = np.vstack([m, [0.0, 0.0, 1.0]])                          # 3×3
    return cv2.warpPerspective(gray_u8, H, (w, h)), H


def scale(gray_u8, s):
    """Масштаб s. H = diag(s, s, 1); полотно змінює розмір разом із зображенням."""
    h, w = gray_u8.shape
    H = np.diag([s, s, 1.0])
    size = (int(round(w * s)), int(round(h * s)))
    return cv2.warpPerspective(gray_u8, H, size), H


def brightness(gray_u8, factor):
    """Множення яскравості; геометрія не змінюється (H = I)."""
    out = np.clip(np.rint(gray_u8.astype(np.float32) * factor), 0, 255).astype(np.uint8)
    return out, np.eye(3)


def gaussian_noise(gray_u8, sigma, seed=0):
    """Гауссів шум із σ (фіксоване зерно для відтворюваності); H = I."""
    rng = np.random.default_rng(seed)
    noisy = gray_u8.astype(np.float32) + rng.normal(0.0, sigma, gray_u8.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8), np.eye(3)