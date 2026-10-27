
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILE = "photo.jpg"        # можна замінити на texture.jpg або інше фото
SIZE = 256                # розмір HR-фрагмента (кратний 8)
K = 8                     # коефіцієнт збільшення

img = cv2.imread(FILE)
if img is None:
    raise SystemExit(f"Не вдалося прочитати {FILE}")
H, W = img.shape[:2]
if min(H, W) < SIZE:
    raise SystemExit(f"Фото замале: потрібно щонайменше {SIZE}x{SIZE} пікселів")

# HR-фрагмент із центру фото
y0, x0 = (H - SIZE) // 2, (W - SIZE) // 2
hr = img[y0:y0 + SIZE, x0:x0 + SIZE]

# Малий фрагмент: зменшення в 8 разів (INTER_AREA усереднює пікселі, без аліасингу)
low = cv2.resize(hr, (SIZE // K, SIZE // K), interpolation=cv2.INTER_AREA)
print(f"HR-фрагмент: {hr.shape[1]}x{hr.shape[0]}, малий: {low.shape[1]}x{low.shape[0]}")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


methods = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}

results = {}
print(f"\n{'Метод':<16}{'PSNR, дБ':>10}")
with open("step13_psnr.csv", "w", encoding="utf-8") as f:
    f.write("method,psnr_db\n")
    for name, flag in methods.items():
        up = cv2.resize(low, (SIZE, SIZE), interpolation=flag)     # збільшення в 8 разів
        results[name] = up
        p = psnr(hr, up)
        print(f"{name:<16}{p:>10.2f}")
        f.write(f"{name},{p:.3f}\n")
        cv2.imencode(".png", up)[1].tofile(f"step13_{name}.png")

z = 48
zy = zx = (SIZE - z) // 2
panels = [("Оригінал HR", hr)] + [(f"{n}\nPSNR = {psnr(hr, im):.2f} дБ", im) for n, im in results.items()]
fig, ax = plt.subplots(2, 5, figsize=(18, 8))
for i, (title, im) in enumerate(panels):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax[0, i].imshow(rgb, interpolation="nearest")
    ax[0, i].set_title(title)
    ax[1, i].imshow(rgb[zy:zy + z, zx:zx + z], interpolation="nearest")
    ax[1, i].set_title("збільшена ділянка")
    ax[0, i].axis("off")
    ax[1, i].axis("off")
plt.tight_layout()
plt.savefig("step13_interp.png", dpi=150)
print("\nЗбережено: step13_interp.png, step13_psnr.csv, step13_<метод>.png")