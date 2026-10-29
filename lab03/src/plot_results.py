import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS = {
    "rotation": "Кут повороту, °",
    "scale": "Масштаб s",
    "brightness": "Множник яскравості",
    "noise": "σ гауссового шуму",
}

data = defaultdict(list)       # (перетворення, детектор, параметр) -> значення по зображеннях
with open(BASE_DIR / "results" / "repeatability.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        data[(r["transform"], r["detector"], float(r["param"]))].append(float(r["repeatability"]))

(BASE_DIR / "figures").mkdir(exist_ok=True)
for kind, xlabel in LABELS.items():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for det in ("Harris", "SIFT", "ORB"):
        params = sorted({p for (k, d, p) in data if k == kind and d == det})
        mean = np.array([np.mean(data[(kind, det, p)]) for p in params])
        std = np.array([np.std(data[(kind, det, p)]) for p in params])
        ax.plot(params, mean, marker="o", label=det)
        ax.fill_between(params, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Повторюваність")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE_DIR / "figures" / f"repeatability_{kind}.png", dpi=150)
    plt.close(fig)
print("Графіки збережено в figures/")