
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if os.path.exists("texture.jpg"):
    tex = cv2.imread("texture.jpg")
    print("Використано texture.jpg")
else:
    # запасний варіант: дрібні діагональні смуги з періодом ~3 пікселі
    n = 1024
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    v = ((0.5 + 0.5 * np.sin(2 * np.pi * (x + 0.6 * y) / 3.0)) * 255).astype(np.uint8)
    tex = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    print("texture.jpg не знайдено: використано синтетичну смугасту текстуру.")

h, w = tex.shape[:2]
size = (w // 4, h // 4)                                   # зменшення в 4 рази

# Спосіб 1: пряме зменшення найближчим сусідом
nearest = cv2.resize(tex, size, interpolation=cv2.INTER_NEAREST)

# Спосіб 2: спершу розмиття (низькочастотний фільтр), потім зменшення
# sigma ~ коефіцієнт зменшення / 2, щоб прибрати частоти, які не вмістяться в меншу сітку
blurred = cv2.GaussianBlur(tex, (0, 0), sigmaX=2.0)
smooth = cv2.resize(blurred, size, interpolation=cv2.INTER_NEAREST)

cv2.imencode(".png", nearest)[1].tofile("step12_nearest.png")
cv2.imencode(".png", smooth)[1].tofile("step12_blur_then_resize.png")

# Рисунок для звіту: зверху повні зображення, знизу збільшені фрагменти
cy, cx = h // 2, w // 2
r = min(200, cy, cx)                                      # фрагмент 2r x 2r в оригіналі
sy, sx, rs = cy // 4, cx // 4, r // 4                     # той самий фрагмент у зменшеному

crop_orig = tex[cy - r:cy + r, cx - r:cx + r]
crop_near = nearest[sy - rs:sy + rs, sx - rs:sx + rs]
crop_smooth = smooth[sy - rs:sy + rs, sx - rs:sx + rs]

items = [(tex, "Оригінал"), (nearest, "INTER_NEAREST"), (smooth, "GaussianBlur + зменшення"),
         (crop_orig, "Фрагмент оригіналу"), (crop_near, "Фрагмент: NEAREST (муар)"),
         (crop_smooth, "Фрагмент: з розмиттям")]
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
for a, (im, t) in zip(ax.ravel(), items):
    a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation="nearest")
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
plt.savefig("step12_aliasing.png", dpi=150)
print(f"Розмір оригіналу {w}x{h}, зменшеного {size[0]}x{size[1]}")
print("Збережено: step12_aliasing.png, step12_nearest.png, step12_blur_then_resize.png")