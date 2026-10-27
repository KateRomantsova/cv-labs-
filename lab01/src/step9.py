
import cv2
import numpy as np


vals = list(map(int, open("step8_bounds.txt").read().split()))
lower = np.array(vals[:3], dtype=np.uint8)
upper = np.array(vals[3:], dtype=np.uint8)
print("Межі (B, G, R): нижні", lower.tolist(), "верхні", upper.tolist())


def evaluate(photo, mask_file):
    img = cv2.imread(photo)
    gt = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) > 127   # еталон
    pred = cv2.inRange(img, lower, upper) > 0                # результат порогування
    acc = (pred == gt).mean()                                # частка правильних пікселів
    iou = (pred & gt).sum() / (pred | gt).sum()              # IoU
    return acc, iou


with open("step9_results.csv", "w", encoding="utf-8") as f:   # для кроку 11
    f.write("frame,accuracy,iou\n")
    for name, photo, mask_file in [("Кадр 1 (день)", "day.jpg", "day_mask.png"),
                                   ("Кадр 2 (лампа)", "lamp.jpg", "lamp_mask.png")]:
        acc, iou = evaluate(photo, mask_file)
        print(f"{name}: частка правильних = {acc:.3f}, IoU = {iou:.3f}")
        f.write(f"{name},{acc:.4f},{iou:.4f}\n")
print("Збережено: step9_results.csv")