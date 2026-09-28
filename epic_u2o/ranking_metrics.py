#!/usr/bin/env python3
"""Ranking metrics: Spearman rho and Kendall tau for the autotuning use case.

Relative order of candidate configs matters more than absolute accuracy
for autotuning. Numpy-only rank correlations with tie handling.
"""

import math
from itertools import combinations


def _ranks(x):
    """Average ranks (1-based) with tie handling."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b) -> float:
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def spearman(y_true, y_pred) -> float:
    """Spearman rank correlation."""
    assert len(y_true) == len(y_pred) and len(y_true) > 1
    return _pearson(_ranks(list(y_true)), _ranks(list(y_pred)))


def kendall_tau(y_true, y_pred) -> float:
    """Kendall tau-b with tie correction."""
    yt, yp = list(y_true), list(y_pred)
    assert len(yt) == len(yp) and len(yt) > 1
    conc = disc = t_x = t_y = 0
    for i, j in combinations(range(len(yt)), 2):
        dx = (yt[i] > yt[j]) - (yt[i] < yt[j])
        dy = (yp[i] > yp[j]) - (yp[i] < yp[j])
        if dx * dy > 0:
            conc += 1
        elif dx * dy < 0:
            disc += 1
        elif dx != 0:
            t_x += 1
        elif dy != 0:
            t_y += 1
    denom = math.sqrt((conc + disc + t_x) * (conc + disc + t_y))
    return (conc - disc) / denom if denom > 0 else 0.0


def topk_hit_rate(y_true, y_pred, k: int = 5) -> float:
    """Fraction of true top-k configs present in predicted top-k (lower is better)."""
    yt, yp = list(y_true), list(y_pred)
    k = min(k, len(yt))
    true_top = set(sorted(range(len(yt)), key=lambda i: yt[i])[:k])
    pred_top = set(sorted(range(len(yp)), key=lambda i: yp[i])[:k])
    return len(true_top & pred_top) / k


def ranking_metrics(y_true, y_pred, k: int = 5) -> dict:
    return {
        'spearman': spearman(y_true, y_pred),
        'kendall_tau': kendall_tau(y_true, y_pred),
        f'top{k}_hit_rate': topk_hit_rate(y_true, y_pred, k),
        'n': len(list(y_true)),
    }


if __name__ == '__main__':
    yt = [5.0, 2.0, 8.0, 1.0, 4.0, 7.0, 3.0, 6.0]
    yp = [5.2, 1.8, 8.5, 1.2, 4.1, 6.8, 3.3, 5.9]
    m = ranking_metrics(yt, yp, k=3)
    assert m['spearman'] > 0.95 and m['kendall_tau'] > 0.8
    assert m['top3_hit_rate'] == 1.0
    # Perfectly reversed should be -1
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9
    print(f"Ranking OK: spearman={m['spearman']:.4f} kendall={m['kendall_tau']:.4f} top3={m['top3_hit_rate']:.2f}")
