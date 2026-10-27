
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = cv2.imread("photo.jpg")
if img is None:
    raise SystemExit("Не вдалося прочитати photo.jpg")

gammas = [0.5, 1.0, 2.2]
x = np.arange(256)


def lut(g):
    """Таблиця перетворення 0..255 -> 0..255 для заданої гамми."""
    return np.clip(255.0 * (x / 255.0) ** g, 0, 255).astype(np.uint8)


# --- 1) Застосування до зображення
fig, ax = plt.subplots(1, 3, figsize=(14, 6))
for a, g in zip(ax, gammas):
    out = cv2.LUT(img, lut(g))                                  # застосування через таблицю
    cv2.imencode(".png", out)[1].tofile(f"step14_gamma_{g}.png")
    a.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    a.set_title(f"γ = {g}")
    a.axis("off")
plt.tight_layout()
plt.savefig("step14_images.png", dpi=150)
plt.close()

# --- 2) Передавальні характеристики
plt.figure(figsize=(5.5, 5.5))
for g in gammas:
    plt.plot(x, lut(g), label=f"γ = {g}")
plt.xlabel("Вхідне значення")
plt.ylabel("Вихідне значення")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.grid(True)
plt.legend()
plt.title("Передавальні характеристики")
plt.tight_layout()
plt.savefig("step14_curves.png", dpi=150)
plt.close()

# --- 3) Чому середнє кодованих значень != середнє освітленостей
G = 2.2
a, b = 0, 255                                                    # чорний і білий піксель
lin = lambda v: (v / 255.0) ** G                                 # кодоване значення -> лінійна освітленість
enc = lambda L: 255.0 * L ** (1 / G)                             # лінійна освітленість -> кодоване значення

naive = (a + b) / 2                                              # середнє кодованих значень
L_naive = lin(naive)                                             # освітленість, яку воно реально дає
L_true = (lin(a) + lin(b)) / 2                                   # середнє освітленостей
v_true = enc(L_true)                                             # кодоване значення, що відповідає цьому середньому

print(f"Пікселі {a} і {b}, гамма {G}")
print(f"Середнє кодованих значень: {naive}  -> освітленість {L_naive:.3f}")
print(f"Середнє освітленостей:      {L_true:.3f}  -> кодоване значення {v_true:.1f}")

# Графік: де лежать два результати на кривій
v = np.linspace(0, 255, 256)
plt.figure(figsize=(6, 5))
plt.plot(v, lin(v), label=f"L = (v/255)^{G}")
plt.scatter([naive], [L_naive], color="red", zorder=3, label=f"середнє кодованих: v={naive:.1f}, L={L_naive:.3f}")
plt.scatter([v_true], [L_true], color="green", zorder=3, label=f"середнє освітленостей: L=0.5, v={v_true:.1f}")
plt.plot([0, 255], [0, 1], "k--", lw=0.8)
plt.axhline(L_true, color="green", lw=0.6, ls=":")
plt.xlabel("Кодоване значення v")
plt.ylabel("Освітленість L")
plt.grid(True)
plt.legend(fontsize=8)
plt.title("Усереднення кодованих значень і освітленостей")
plt.tight_layout()
plt.savefig("step14_averaging.png", dpi=150)
print("\nЗбережено: step14_images.png, step14_curves.png, step14_averaging.png")