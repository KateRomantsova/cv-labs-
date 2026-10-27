
import os
import csv
import matplotlib.pyplot as plt


def read_iou(path, label):
    """Повертає (IoU кадру 1, IoU кадру 2) з csv або з введення."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return float(rows[0]["iou"]), float(rows[1]["iou"])
    print(f"Файл {path} не знайдено, введіть IoU для {label} вручну (через крапку).")
    a = float(input(f"  {label}, кадр 1 (IoU): ").replace(",", "."))
    b = float(input(f"  {label}, кадр 2 (IoU): ").replace(",", "."))
    return a, b


data = {
    "RGB": read_iou("step9_results.csv", "RGB"),
    "HSV": read_iou("step10_results.csv", "HSV"),
}

print(f"\n{'Простір':<8}{'Кадр 1 (IoU)':>14}{'Кадр 2 (IoU)':>14}{'Падіння':>10}{'Збережено':>11}")
with open("step11_table.csv", "w", encoding="utf-8") as f:
    f.write("space,iou_frame1,iou_frame2,drop,kept_percent\n")
    for space, (i1, i2) in data.items():
        drop = i1 - i2
        kept = 100 * i2 / i1 if i1 else 0
        print(f"{space:<8}{i1:>14.3f}{i2:>14.3f}{drop:>10.3f}{kept:>10.1f}%")
        f.write(f"{space},{i1:.4f},{i2:.4f},{drop:.4f},{kept:.1f}\n")

# Діаграма для звіту
spaces = list(data)
x = range(len(spaces))
plt.figure(figsize=(5, 4))
plt.bar([i - 0.2 for i in x], [data[s][0] for s in spaces], width=0.4, label="Кадр 1 (день)")
plt.bar([i + 0.2 for i in x], [data[s][1] for s in spaces], width=0.4, label="Кадр 2 (лампа)")
plt.xticks(list(x), spaces)
plt.ylabel("IoU")
plt.ylim(0, 1)
plt.legend()
plt.title("IoU: RGB проти HSV")
plt.tight_layout()
plt.savefig("step11_iou.png", dpi=150)
print("\nЗбережено: step11_table.csv, step11_iou.png")