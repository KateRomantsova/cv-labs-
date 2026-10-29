import cv2
import numpy as np

SIGMA = 1.5
K = 0.04


def compute_derivatives(gray):
    """Крок 2. Похідні Ix, Iy через Собеля з ядром 3."""
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return ix, iy


def compute_products(ix, iy):
    """Крок 3. Поелементні добутки Ix², Iy², Ix·Iy (ДО згладжування)."""
    return ix * ix, iy * iy, ix * iy


def smooth(arr, sigma=SIGMA):
    """Крок 4. Гаусове згладжування; розмір ядра обчислюється з σ."""
    return cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)


def harris_response(a, b, c, k=K):
    """Крок 5. R = (A·B − C²) − k·(A + B)². Рахуємо у float64 проти втрати точності."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)
    return (a * b - c ** 2) - k * (a + b) ** 2


def harris(gray, sigma=SIGMA, k=K):
    """Повний конвеєр кроків 2-5."""
    ix, iy = compute_derivatives(gray)
    ixx, iyy, ixy = compute_products(ix, iy)      # спочатку квадрати
    a, b, c = smooth(ixx, sigma), smooth(iyy, sigma), smooth(ixy, sigma)  # потім згладжування
    return harris_response(a, b, c, k)

def harris_box(gray, block_size=7, k=K):
    ix, iy = compute_derivatives(gray)
    ixx, iyy, ixy = compute_products(ix, iy)
    a, b, c = (cv2.blur(m, (block_size, block_size)) for m in (ixx, iyy, ixy))
    return harris_response(a, b, c, k)