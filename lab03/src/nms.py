import numpy as np


def threshold_points(response, ratio=0.01):
    """Крок 6. Точки, де R > ratio·max(R). Повертає x, y та значення R."""
    thr = ratio * response.max()
    ys, xs = np.nonzero(response > thr)
    return xs, ys, response[ys, xs]


def non_max_suppression(xs, ys, scores, r=10, n_max=1000):
    """Крок 7. Жадібне придушення: за спаданням R беремо точку,
    якщо вона далі ніж r пікселів від усіх уже обраних."""
    order = np.argsort(-scores)
    pts = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)

    selected = np.empty((n_max, 2), dtype=np.float32)
    count = 0
    r2 = r * r
    for p in pts:
        if count == 0 or np.all(((selected[:count] - p) ** 2).sum(axis=1) > r2):
            selected[count] = p
            count += 1
            if count == n_max:
                break
    return selected[:count]          # масив (x, y)