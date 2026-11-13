"""D-024 calibration: 15 equal-width bins, with no fitted parameters."""

import numpy as np

TOLERANCE = 1e-5  # float32 rounding; anything larger is a real error


def calibration(labels, probs) -> dict:
    """Binary bins compare mean P(y=1) with positive frequency (not accuracy).

    Categorical bins compare top-label confidence with argmax correctness.
    Bins are [lower, upper), except the last includes 1. Empty bins have null
    means and zero contribution, and remain in the reliability-diagram table.
    """
    y, p = np.asarray(labels), np.asarray(probs, dtype=np.float64)
    if y.ndim != 1 or not len(y) or p.ndim not in (1, 2) or len(p) != len(y):
        raise ValueError("expected nonempty labels [N] and probabilities [N] or [N,C]")
    # float32 mixing (sum of pi_k * p_k over 101 classes) can land a few ulps outside [0,1].
    if not np.isfinite(p).all() or (p < -TOLERANCE).any() or (p > 1 + TOLERANCE).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    p = np.clip(p, 0.0, 1.0)
    classes = 2 if p.ndim == 1 else p.shape[1]
    if classes < 2 or not np.isin(y, np.arange(classes)).all():
        raise ValueError("labels must be class indices")
    if p.ndim == 2 and not np.allclose(p.sum(axis=1), 1, atol=TOLERANCE):
        raise ValueError("class probabilities must sum to one")
    confidence = p if p.ndim == 1 else p.max(axis=1)
    observed = y if p.ndim == 1 else (p.argmax(axis=1) == y)
    edges = np.linspace(0, 1, 16)
    membership = np.searchsorted(edges[1:-1], confidence, side="right")
    bins, ece = [], 0.0
    for i in range(15):
        selected = membership == i
        count = int(selected.sum())
        mean = float(confidence[selected].mean()) if count else None
        frequency = float(observed[selected].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean - frequency)
        bins.append({"lower": float(edges[i]), "upper": float(edges[i + 1]),
                     "count": count, "mean_probability": mean, "observed_frequency": frequency})
    return {"ece": float(ece), "n_bins": 15,
            "kind": "binary_positive" if p.ndim == 1 else "top_label", "bins": bins}
