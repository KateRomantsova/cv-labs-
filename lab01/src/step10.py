
import cv2
import numpy as np

day, lamp = cv2.imread("day.jpg"), cv2.imread("lamp.jpg")
mask1 = cv2.imread("day_mask.png", cv2.IMREAD_GRAYSCALE) > 127
mask2 = cv2.imread("lamp_mask.png", cv2.IMREAD_GRAYSCALE) > 127

hsv1 = cv2.cvtColor(day, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(lamp, cv2.COLOR_BGR2HSV)

# --- Підбір меж на кадрі 1: перцентилі 10-90 серед пікселів усередині маски.
# H рахуємо циклічно, щоб коректно працювати і для червоного (біля 0 та 179).
px = hsv1[mask1].astype(np.float64)
ang = px[:, 0] * 2 * np.pi / 180
center = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 180 / (2 * np.pi)) % 180
h_shift = (px[:, 0] - center + 90) % 180
h_lo, h_hi = np.percentile(h_shift, [10, 90])
h_lo, h_hi = (h_lo - 90 + center) % 180, (h_hi - 90 + center) % 180
s_lo, s_hi = np.percentile(px[:, 1], [10, 90])
v_lo, v_hi = np.percentile(px[:, 2], [10, 90])
lower = (int(h_lo), int(s_lo), int(v_lo))
upper = (int(h_hi), int(s_hi), int(v_hi))


def in_range_hsv(hsv, lo, hi):
    if lo[0] <= hi[0]:                                        # звичайний випадок
        return cv2.inRange(hsv, lo, hi)
    m1 = cv2.inRange(hsv, (lo[0], lo[1], lo[2]), (179, hi[1], hi[2]))   # червоний: два інтервали
    m2 = cv2.inRange(hsv, (0, lo[1], lo[2]), (hi[0], hi[1], hi[2]))
    return cv2.bitwise_or(m1, m2)


def evaluate(hsv, gt):
    seg = in_range_hsv(hsv, lower, upper)
    pred = seg > 0
    acc = (pred == gt).mean()                                 # частка правильних пікселів
    iou = (pred & gt).sum() / (pred | gt).sum()
    return seg, acc, iou


print("Межі HSV (H, S, V): нижні", lower, "верхні", upper)
if lower[0] > upper[0]:
    print("  (нижня межа H > верхньої: використано два інтервали, червоний колір)")

rows = []
segs = []
for name, hsv, gt in [("Кадр 1 (день)", hsv1, mask1), ("Кадр 2 (лампа)", hsv2, mask2)]:
    seg, acc, iou = evaluate(hsv, gt)
    segs.append(seg)
    rows.append((name, acc, iou))
    print(f"{name}: частка правильних = {acc:.3f}, IoU = {iou:.3f}")

# --- Зберігаємо для звіту та кроку 11
with open("step10_results.csv", "w", encoding="utf-8") as f:
    f.write("frame,accuracy,iou\n")
    for name, acc, iou in rows:
        f.write(f"{name},{acc:.4f},{iou:.4f}\n")

h = 500
panels = []
for img, seg, gt in [(day, segs[0], mask1), (lamp, segs[1], mask2)]:
    w = int(img.shape[1] * h / img.shape[0])
    g3 = cv2.cvtColor(gt.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    s3 = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    panels.append(np.hstack([cv2.resize(x, (w, h)) for x in (img, s3, g3)]))
width = max(p.shape[1] for p in panels)
panels = [cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
cv2.imencode(".png", np.vstack(panels))[1].tofile("step10_compare.png")
print("Збережено: step10_compare.png, step10_results.csv")