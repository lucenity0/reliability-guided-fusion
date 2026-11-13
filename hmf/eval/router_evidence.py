"""Post-hoc router evidence from prediction CSVs for #32; never used for selection."""

import csv

import numpy as np


def router_evidence(path, expert_names, *, num_classes: int, tau: float) -> dict:
    """Recover tempered CE targets, per-expert Pearson r and argmax agreement.

    Pass the run's frozen tau for the reliability-supervised model. Baselines
    without a trained tau require an explicit analysis temperature. Correlation
    is null when either vector is constant. Ties use the first expert in the
    supplied order. Exact zero likelihoods (saturated/rounded probabilities)
    are floored at float64's smallest positive normal before taking logs.
    The returned targets are in CSV row order, alongside IDs and expert names.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if num_classes < 2 or not expert_names:
        raise ValueError("need at least two classes and one named expert")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("prediction CSV is empty")
    labels = np.array([int(row["label"]) for row in rows])
    if not np.isin(labels, np.arange(num_classes)).all():
        raise ValueError("labels must be class indices")
    if num_classes == 2:
        p = np.array([[float(row[f"p_{name}"]) for name in expert_names] for row in rows])
        likelihood = np.where(labels[:, None] == 1, p, 1 - p)
    else:
        p = np.array([[[float(row[f"p_{name}_{c}"]) for c in range(num_classes)]
                       for name in expert_names] for row in rows])
        if not np.allclose(p.sum(axis=-1), 1, atol=1e-6):
            raise ValueError("expert class probabilities must sum to one")
        likelihood = np.take_along_axis(p, labels[:, None, None], axis=-1).squeeze(-1)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("expert probabilities must be finite and in [0,1]")
    log_likelihood = np.log(np.maximum(likelihood, np.finfo(np.float64).tiny))
    with np.errstate(over="ignore", under="ignore"):
        target = np.exp((log_likelihood - log_likelihood.max(axis=1, keepdims=True)) / tau)
    target /= target.sum(axis=1, keepdims=True)
    result = {"ids": [row["id"] for row in rows], "expert_names": list(expert_names),
              "tau": tau, "target": target}
    if all(f"pi_{name}" in rows[0] for name in expert_names):
        pi = np.array([[float(row[f"pi_{name}"]) for name in expert_names] for row in rows])
        if (not np.isfinite(pi).all() or (pi < 0).any()
                or not np.allclose(pi.sum(axis=1), 1, atol=5e-6)):
            raise ValueError("routing weights must be finite probability vectors")
        correlations = []
        for k in range(len(expert_names)):
            x, y = pi[:, k], target[:, k]
            correlations.append(None if np.ptp(x) == 0 or np.ptp(y) == 0
                                else float(np.corrcoef(x, y)[0, 1]))
        result.update(pi_mean=pi.mean(axis=0).tolist(), pearson=correlations,
                      argmax_agreement=float((pi.argmax(axis=1) == target.argmax(axis=1)).mean()))
    return result
