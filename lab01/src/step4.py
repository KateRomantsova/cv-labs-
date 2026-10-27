import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("photo.jpg")

def psnr(a, b):
    # float64, щоб різниця не переповнила uint8
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

rows = []

# PNG (без втрат)
cv2.imwrite("test.png", img)
rows.append(("PNG", os.path.getsize("test.png"), psnr(img, cv2.imread("test.png"))))

# JPEG з різною якістю
for q in (95, 75, 50, 25, 10):
    name = f"test_q{q}.jpg"
    cv2.imwrite(name, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    rows.append((f"JPEG q={q}", os.path.getsize(name), psnr(img, cv2.imread(name))))

# Таблиця
print(f"{'Формат':<12}{'Розмір, КБ':>12}{'PSNR, дБ':>12}")
for n, s, p in rows:
    print(f"{n:<12}{s / 1024:>12.1f}{p:>12.2f}")

# Графік PSNR від розміру файлу (PNG має PSNR = ∞, тому його пропускаємо)
pts = [(s / 1024, p, n) for n, s, p in rows if np.isfinite(p)]
plt.figure(figsize=(6, 4))
plt.plot([a for a, _, _ in pts], [b for _, b, _ in pts], "o-")
for a, b, n in pts:
    plt.annotate(n, (a, b), textcoords="offset points", xytext=(4, 4), fontsize=8)
plt.xlabel("Розмір файлу, КБ")
plt.ylabel("PSNR, дБ")
plt.grid(True)
plt.title("Залежність PSNR від розміру файлу")
plt.tight_layout()
plt.savefig("step4_psnr.png", dpi=150)
plt.show()