
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("day.jpg")                                  # BGR
mask = cv2.imread("day_mask.png", cv2.IMREAD_GRAYSCALE) > 127  # еталон: True = предмет

# --- Підбір меж: початкові значення з кольорів пікселів усередині еталонної маски
px = img[mask]                                               # N x 3 (B, G, R)
lower = np.percentile(px, 10, axis=0).astype(np.uint8)
upper = np.percentile(px, 90, axis=0).astype(np.uint8)


# --- Порогування
seg = cv2.inRange(img, lower, upper)

# --- Оцінка (для підбору межі): частка правильних пікселів та IoU
p, g = seg > 0, mask
acc = (p == g).mean()
iou = (p & g).sum() / (p | g).sum()

print("Межі (порядок B, G, R):")
print("  нижні :", lower.tolist())
print("  верхні:", upper.tolist())
print("Ті самі межі в порядку R, G, B:")
print("  нижні :", lower[::-1].tolist())
print("  верхні:", upper[::-1].tolist())
print(f"Кадр 1: частка правильних = {acc:.3f}, IoU = {iou:.3f}")

# --- Зберігаємо межі для кроку 9 і малюнок для звіту
with open("step8_bounds.txt", "w") as f:
    f.write(" ".join(map(str, lower.tolist() + upper.tolist())))

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Кадр 1")
ax[1].imshow(seg, cmap="gray"); ax[1].set_title("Результат cv2.inRange")
ax[2].imshow(mask, cmap="gray"); ax[2].set_title("Еталонна маска")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("step8_result.png", dpi=150)
plt.show()