"""Split metrics and val-chosen thresholds (docs/PROTOCOL.md rule 3; D-018).

Binary tasks: AUROC needs no threshold; accuracy and macro-F1 use one threshold
chosen on val (FHM maximises accuracy; MAMI, HarMeme and CrisisMMD maximise
macro-F1) and reused unchanged on test. Multi-class tasks (MVSA-Single,
Food-101) predict by argmax with the smallest index winning ties; no threshold.

``find_threshold`` extends V3 ``src/train_utils.py:find_optimal_threshold``
(commit 75c6b9f): it searches every distinct score exactly instead of a
0.1-0.9 grid, and can maximise macro-F1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

THRESHOLD_METRIC = {"fhm": "accuracy"}  # every other binary dataset: macro_f1


def threshold_metric(dataset: str) -> str:
    return THRESHOLD_METRIC.get(dataset, "macro_f1")


def _score(labels: np.ndarray, preds: np.ndarray, metric: str, classes=(0, 1)) -> float:
    if metric == "accuracy":
        return float(accuracy_score(labels, preds))
    if metric == "macro_f1":  # every class counts; an empty class scores 0 (D-018)
        return float(f1_score(labels, preds, average="macro", zero_division=0,
                              labels=list(classes)))
    raise ValueError(f"unknown threshold metric {metric!r}")


def find_threshold(labels, probs, metric: str = "accuracy") -> tuple[float, float]:
    """Threshold t maximising ``metric`` for the rule ``prob >= t``.

    Candidates are the distinct scores (plus 1 + max, which predicts all
    negative); the smallest best threshold wins ties. Returns (threshold, value).
    """
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    if labels.shape != probs.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and probs must be non-empty 1-D arrays of the same length")
    candidates = np.append(np.unique(probs), probs.max() + 1.0)
    best_t, best = float(candidates[0]), -1.0
    for t in candidates:
        value = _score(labels, (probs >= t).astype(int), metric)
        if value > best:
            best_t, best = float(t), value
    return best_t, best


def binary_metrics(labels, probs, threshold: float) -> dict:
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    out = {"threshold": float(threshold), "accuracy": _score(labels, preds, "accuracy"),
           "macro_f1": _score(labels, preds, "macro_f1")}
    out["auroc"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else None
    return out


def multiclass_metrics(labels, probs) -> dict:
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0]:
        raise ValueError("probs must be [N, C] with one row per label")
    preds = probs.argmax(axis=1)  # first maximum = smallest class index on ties
    classes = range(probs.shape[1])
    return {"accuracy": _score(labels, preds, "accuracy"),
            "macro_f1": _score(labels, preds, "macro_f1", classes)}
