"""Metrics. Scores are oriented so that higher = more likely AI; labels: 1 = AI, 0 = human."""
from __future__ import annotations

import math

import numpy as np


def eer(scores, labels) -> float:
    """Equal error rate."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    order = np.argsort(s)
    y = y[order]
    fnr = np.cumsum(y) / y.sum()
    fpr = 1 - np.cumsum(1 - y) / (1 - y).sum()
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def auc(scores, labels) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, int)
    return float(roc_auc_score(y, scores)) if 0 < y.sum() < y.size else float("nan")


def tpr_at_fpr(scores, labels, fpr: float = 0.01) -> float:
    """Share of AI clips detected when the threshold is set so that `fpr` of human clips are flagged."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    neg = np.sort(s[y == 0])
    if neg.size == 0 or y.sum() == 0:
        return float("nan")
    thr = neg[min(neg.size - 1, int(math.ceil((1 - fpr) * neg.size)) - 1)]
    return float((s[y == 1] > thr).mean())


def per_group_eer(scores, labels, groups) -> dict[str, float]:
    """EER of each AI system against ALL human clips of the dataset."""
    s, y, g = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups)
    return {name: eer(s[(y == 0) | (g == name)], y[(y == 0) | (g == name)]) for name in sorted(set(g[y == 1]))}
