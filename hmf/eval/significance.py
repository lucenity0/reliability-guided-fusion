"""Paired significance tests for model comparisons (docs/MATH_SPEC.md §10).

``paired_bootstrap`` and ``mcnemar_test`` are adapted from V3 src/significance.py:
https://github.com/lucenity0/A-Unified-Adaptive-Framework-for-Multimodal-Data-Fusion-with-Dynamic-Modality-Reweighting
at commit 75c6b9f5424cfd65b04a3c29c89ded40a7ce38cd. The V3 checkpoint runner is not
ported. Changes: B defaults to 10,000; the p-value follows MATH_SPEC §10 exactly
(no add-one correction); metrics are computed with numpy rank/count formulas
instead of sklearn calls per resample, and checked against sklearn in the tests.

New: fast DeLong (DeLong et al. 1988; Sun & Xu 2014), paired t-test and Wilcoxon
across seeds, Holm-Bonferroni, the per-seed p-value summary, and the multi-class
paired accuracy bootstrap ``paired_bootstrap_accuracy``.

Model A is the proposed method and model B the baseline; every delta is A - B.
All functions are pure numpy/scipy and expect paired inputs in the same order.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _as_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a 1-D array of 0/1 values")
    if labels.min() == labels.max():
        raise ValueError("labels must contain both classes")
    return labels.astype(int)


def _as_paired_scores(labels: np.ndarray, *scores: np.ndarray) -> list[np.ndarray]:
    out = []
    for score in scores:
        score = np.asarray(score, dtype=float)
        if score.shape != labels.shape:
            raise ValueError("scores must have the same shape as labels")
        if not np.isfinite(score).all():
            raise ValueError("scores must be finite")
        out.append(score)
    return out


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank formula, with ties counted as 1/2."""
    labels = _as_binary_labels(labels)
    (scores,) = _as_paired_scores(labels, scores)
    ranks = stats.rankdata(scores)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ── DeLong ───────────────────────────────────────────────────────────────────


def delong_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Two-sided DeLong test for two correlated AUROCs (fast midrank algorithm).

    Returns both AUROCs, delta = auc_a - auc_b, its standard error, z, p and a
    normal-approximation 95% CI of delta. Identical predictors give p = 1.
    """
    labels = _as_binary_labels(labels)
    scores_a, scores_b = _as_paired_scores(labels, scores_a, scores_b)
    pos, neg = labels == 1, labels == 0
    m, n = int(pos.sum()), int(neg.sum())

    aucs, v01, v10 = [], [], []
    for score in (scores_a, scores_b):
        x, y = score[pos], score[neg]
        tx = stats.rankdata(x)
        ty = stats.rankdata(y)
        tz = stats.rankdata(np.concatenate([x, y]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)  # structural components per positive
        v10.append(1.0 - (tz[m:] - ty) / m)  # structural components per negative

    s01 = np.cov(np.vstack(v01)) if m > 1 else np.zeros((2, 2))
    s10 = np.cov(np.vstack(v10)) if n > 1 else np.zeros((2, 2))
    cov = s01 / m + s10 / n
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast)
    delta = float(aucs[0] - aucs[1])

    if var <= 1e-15:
        z = 0.0 if abs(delta) <= 1e-15 else np.copysign(np.inf, delta)
        se = 0.0
    else:
        se = var**0.5
        z = delta / se
    p = float(2 * stats.norm.sf(abs(z)))
    half = 1.959963984540054 * se
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta": delta,
        "se": se,
        "z": float(z),
        "p_value": min(1.0, p),
        "delta_lo": delta - half,
        "delta_hi": delta + half,
    }


# ── Paired bootstrap (ported from V3) ────────────────────────────────────────


def _f1(labels: np.ndarray, preds: np.ndarray, positive: int) -> float:
    tp = np.sum((preds == positive) & (labels == positive))
    denom = np.sum(preds == positive) + np.sum(labels == positive)
    return float(2 * tp / denom) if denom else 0.0


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Accuracy and macro-F1 at a fixed threshold (macro-F1 as sklearn, zero_division=0)."""
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "macro_f1": (_f1(labels, preds, 1) + _f1(labels, preds, 0)) / 2,
    }


def paired_bootstrap(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap of A - B over samples (MATH_SPEC §10.2).

    Each resample draws one index set and applies it to both models. Resamples
    with a single class are redrawn, since AUROC is undefined there. The CI is
    percentile; p = 2 min(Pr(delta* <= 0), Pr(delta* >= 0)), capped at 1.
    AUROC is always reported. Accuracy and macro-F1 are added when both
    val-chosen thresholds are given.
    """
    labels = _as_binary_labels(labels)
    scores_a, scores_b = _as_paired_scores(labels, scores_a, scores_b)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    with_thresholds = threshold_a is not None and threshold_b is not None
    if (threshold_a is None) != (threshold_b is None):
        raise ValueError("give both thresholds or neither")

    def metrics(y: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> dict:
        out = {"auroc": (auroc(y, sa), auroc(y, sb))}
        if with_thresholds:
            ma = _threshold_metrics(y, sa, threshold_a)
            mb = _threshold_metrics(y, sb, threshold_b)
            for name in ma:
                out[name] = (ma[name], mb[name])
        return out

    rng = np.random.default_rng(seed)
    size = labels.size
    point = metrics(labels, scores_a, scores_b)
    draws = {name: np.empty((n_boot, 2)) for name in point}
    done = 0
    while done < n_boot:
        idx = rng.integers(0, size, size=size)
        y = labels[idx]
        if y.min() == y.max():
            continue
        for name, pair in metrics(y, scores_a[idx], scores_b[idx]).items():
            draws[name][done] = pair
        done += 1

    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out: dict = {"n_boot": n_boot, "seed": seed, "ci": ci}
    for name, (a_point, b_point) in point.items():
        a, b = draws[name][:, 0], draws[name][:, 1]
        d = a - b
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        out[name] = {
            "a": a_point,
            "a_lo": float(np.percentile(a, lo_q)),
            "a_hi": float(np.percentile(a, hi_q)),
            "b": b_point,
            "b_lo": float(np.percentile(b, lo_q)),
            "b_hi": float(np.percentile(b, hi_q)),
            "delta": a_point - b_point,
            "delta_lo": float(np.percentile(d, lo_q)),
            "delta_hi": float(np.percentile(d, hi_q)),
            "p_value": float(min(1.0, p)),
        }
    return out


def _as_class_array(values: np.ndarray, name: str) -> np.ndarray:
    """1-D array of non-negative integer class indices (integer-valued floats allowed)."""
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D array")
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise ValueError(f"{name} must contain integer class indices")
    if np.issubdtype(arr.dtype, np.floating) and not (
        np.isfinite(arr).all() and (arr == np.round(arr)).all()
    ):
        raise ValueError(f"{name} must contain integer class indices")
    if (arr < 0).any():
        raise ValueError(f"{name} must be non-negative class indices")
    return arr.astype(np.int64)


def paired_bootstrap_accuracy(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap of accuracy A - B for any number of classes (MATH_SPEC §10).

    Inputs are aligned 1-D class labels and predicted classes (e.g. argmax of
    class probabilities). Each resample draws one index set with
    ``np.random.default_rng(seed)`` and applies it to both models; the
    replicate statistic is mean(correct_a - correct_b). The CI is percentile;
    p = 2 min(Pr(delta* <= 0), Pr(delta* >= 0)), capped at 1. No DeLong call
    and no threshold fitting.
    """
    labels = _as_class_array(labels, "labels")
    preds_a = _as_class_array(preds_a, "preds_a")
    preds_b = _as_class_array(preds_b, "preds_b")
    if not labels.shape == preds_a.shape == preds_b.shape:
        raise ValueError("labels and predictions must have the same length")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")

    correct_a = (preds_a == labels).astype(float)
    correct_b = (preds_b == labels).astype(float)
    diff = correct_a - correct_b

    rng = np.random.default_rng(seed)
    size = labels.size
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, size, size=size)
        d[i] = diff[idx].mean()

    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {
        "n_boot": n_boot,
        "seed": seed,
        "ci": ci,
        "acc_a": float(correct_a.mean()),
        "acc_b": float(correct_b.mean()),
        "delta": float(diff.mean()),
        "delta_lo": float(np.percentile(d, lo_q)),
        "delta_hi": float(np.percentile(d, hi_q)),
        "p_value": float(min(1.0, p)),
    }


# ── McNemar (ported from V3) ─────────────────────────────────────────────────


def mcnemar_test(labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """McNemar's test on paired thresholded predictions.

    n01 = A right and B wrong; n10 = A wrong and B right. Exact binomial test
    when n01 + n10 < 25, otherwise chi-square with continuity correction.
    """
    labels = np.asarray(labels)
    preds_a, preds_b = np.asarray(preds_a), np.asarray(preds_b)
    if not (labels.shape == preds_a.shape == preds_b.shape) or labels.ndim != 1:
        raise ValueError("labels and predictions must be 1-D arrays of the same length")
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    discordant = n01 + n10

    if discordant == 0:
        return {"n01": n01, "n10": n10, "method": "degenerate", "statistic": None, "p_value": 1.0}
    if discordant < 25:
        p = stats.binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue
        return {"n01": n01, "n10": n10, "method": "exact binomial", "statistic": None,
                "p_value": float(p)}
    statistic = (abs(n01 - n10) - 1) ** 2 / discordant
    return {
        "n01": n01,
        "n10": n10,
        "method": "chi-square (continuity-corrected)",
        "statistic": float(statistic),
        "p_value": float(stats.chi2.sf(statistic, df=1)),
    }


# ── Across seeds ─────────────────────────────────────────────────────────────


def _seed_deltas(metric_a: np.ndarray, metric_b: np.ndarray) -> np.ndarray:
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("need paired 1-D per-seed metrics with at least 2 seeds")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("per-seed metrics must be finite")
    return a - b


def paired_seed_ttest(metric_a: np.ndarray, metric_b: np.ndarray, *, ci: float = 0.95) -> dict:
    """Primary test (MATH_SPEC §10.1): paired t-test of A - B across seeds.

    Returns the mean delta, its t-based CI and the two-sided p. If every delta
    is identical the t statistic is undefined: p = 1 when they are all zero,
    otherwise p = 0 (a constant nonzero improvement on every seed).
    """
    d = _seed_deltas(metric_a, metric_b)
    s = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        p = 1.0 if mean == 0.0 else 0.0
        return {"n_seeds": s, "delta": mean, "delta_lo": mean, "delta_hi": mean,
                "t": None, "p_value": p}
    se = sd / s**0.5
    t = mean / se
    half = float(stats.t.ppf((1 + ci) / 2, df=s - 1)) * se
    return {
        "n_seeds": s,
        "delta": mean,
        "delta_lo": mean - half,
        "delta_hi": mean + half,
        "t": float(t),
        "p_value": float(2 * stats.t.sf(abs(t), df=s - 1)),
    }


def paired_seed_wilcoxon(metric_a: np.ndarray, metric_b: np.ndarray) -> dict:
    """Secondary test (MATH_SPEC §10.1): two-sided Wilcoxon signed-rank across seeds.

    With n = 5 its minimum two-sided p is 0.0625, so report it only at S = 10.
    """
    d = _seed_deltas(metric_a, metric_b)
    if np.all(d == 0):
        return {"n_seeds": d.size, "statistic": None, "p_value": 1.0}
    res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    return {"n_seeds": d.size, "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def summarize_seed_pvalues(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Per-seed tests are reported as the median p and the fraction with p < alpha."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0 or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be a nonempty 1-D array in [0, 1]")
    return {"n_seeds": int(p.size), "median_p": float(np.median(p)),
            "frac_significant": float(np.mean(p < alpha))}


def holm_bonferroni(p_values: dict[str, float] | list[float], alpha: float = 0.05) -> dict:
    """Holm step-down adjustment over a family (MATH_SPEC §10.3).

    Accepts a list or a {name: p} mapping and returns adjusted p-values and
    reject flags in the same structure.
    """
    names = list(p_values) if isinstance(p_values, dict) else None
    raw = np.asarray(list(p_values.values()) if names is not None else p_values, dtype=float)
    if raw.ndim != 1 or raw.size == 0 or ((raw < 0) | (raw > 1)).any():
        raise ValueError("p_values must be a nonempty collection of values in [0, 1]")
    m = raw.size
    order = np.argsort(raw, kind="stable")
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * raw[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    reject = adjusted <= alpha
    if names is not None:
        return {"adjusted": dict(zip(names, adjusted.tolist(), strict=True)),
                "reject": dict(zip(names, reject.tolist(), strict=True)), "alpha": alpha}
    return {"adjusted": adjusted.tolist(), "reject": reject.tolist(), "alpha": alpha}
