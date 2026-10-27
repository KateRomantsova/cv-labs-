import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("photo.jpg")

# Некоректно: uint8 переповнюється (230 + 50 = 280 -> 280 - 256 = 24)
bright_bad = img + np.uint8(50)

# Коректно, спосіб 1: cv2.add з насиченням до 255
bright_ok1 = cv2.add(img, np.full_like(img, 50))

# Коректно, спосіб 2: int16 + np.clip
bright_ok2 = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
print("Обидва коректні способи однакові:", np.array_equal(bright_ok1, bright_ok2))

# Пікселі, які були яскравими й стали темними
g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g1 = cv2.cvtColor(bright_bad, cv2.COLOR_BGR2GRAY)
bad = (g0 > 200) & (g1 < 100)
print("Було яскравими, стали темними:", int(bad.sum()), "пікселів")

# Фрагмент 100x100, де таких пікселів найбільше
k = 100
box = cv2.boxFilter(bad.astype(np.float32), -1, (k, k))
y, x = np.unravel_index(np.argmax(box), box.shape)
y0 = int(np.clip(y - k // 2, 0, img.shape[0] - k))
x0 = int(np.clip(x - k // 2, 0, img.shape[1] - k))
sl = (slice(y0, y0 + k), slice(x0, x0 + k))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(ax, [img, bright_bad, bright_ok1],
                    ["Оригінал", "img + 50 (переповнення)", "cv2.add (коректно)"]):
    a.imshow(cv2.cvtColor(im[sl], cv2.COLOR_BGR2RGB), interpolation="nearest")
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
plt.savefig("step3_overflow.png", dpi=150)
plt.show()