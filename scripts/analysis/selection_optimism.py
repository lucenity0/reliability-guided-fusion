"""Selection optimism of best-epoch val AUROC, Hate-CLIPper Align on FHM (#106; val only).

The official Hate-CLIPper code picks its checkpoint on dev_seen, the split it reports on
(``main.py`` L137, L143, L153). This measures how much that kind of selection inflates
AUROC, without touching the test split. Each seed's run must have been trained with
``SAVE_EPOCH_PREDS=1`` so that ``preds_val_epoch<NNN>.csv`` holds every epoch.

``--n-splits`` stratified half-splits (A, B) of val are drawn once with
``numpy.random.default_rng(--rng-seed)`` and shared by every seed. Within each class, A
gets the first ``n_c // 2`` indices of a permutation. For each seed and split:
``e* = argmax_e AUROC_A(e)`` (the earliest epoch on ties, as ``hmf.train`` selects),
``optimism = AUROC_A(e*) - AUROC_B(e*)``, plus ``AUROC_B(e*)`` and ``max_e AUROC_B(e)``.
Summaries give the mean, the std (ddof=1) and the 2.5/97.5 percentiles (numpy linear).

Usage::

    python scripts/analysis/selection_optimism.py --results-dir ~/runs/sel/results \\
        --seeds 0 1 2 3 4 --train-list ~/splits_predup/fhm/train.txt \\
        --out results/fhm/hateclipper_align/selection_optimism.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from hmf.eval.guard import git_sha

EPOCH_FILE = re.compile(r"preds_val_epoch(\d{3})\.csv")
QUANTITIES = ("optimism", "auroc_b_selected", "auroc_b_max")


def read_preds(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return ([row["id"] for row in rows], np.array([int(row["label"]) for row in rows]),
            np.array([float(row["prob"]) for row in rows]))


def load_run(run_dir: Path) -> dict:
    """One seed's per-epoch val probabilities ``[E, N]``, checked against its own outputs."""
    found = sorted((int(m.group(1)), p) for p in run_dir.iterdir()
                   if (m := EPOCH_FILE.fullmatch(p.name)))
    epochs = [epoch for epoch, _ in found]
    if not found or epochs != list(range(1, len(found) + 1)):
        raise ValueError(f"{run_dir}: expected preds_val_epoch001.csv onwards, got {epochs}")
    ids, labels, _ = read_preds(found[0][1])
    probs = []
    for _, path in found:
        epoch_ids, epoch_labels, prob = read_preds(path)
        if epoch_ids != ids or not np.array_equal(epoch_labels, labels):
            raise ValueError(f"{path}: ids or labels differ from epoch 1")
        probs.append(prob)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    by_epoch = [float(roc_auc_score(labels, prob)) for prob in probs]
    best = int(np.argmax(by_epoch)) + 1  # first maximum = earliest epoch
    if best != metrics["best_epoch"]:
        raise ValueError(f"{run_dir}: best epoch {best} from the files, "
                         f"{metrics['best_epoch']} in metrics.json")
    best_file = run_dir / f"preds_val_epoch{best:03d}.csv"
    return {"ids": ids, "labels": labels, "probs": np.stack(probs), "metrics": metrics,
            "auroc_by_epoch": by_epoch, "best_epoch": best,
            "best_file_matches_preds_val":
                best_file.read_bytes() == (run_dir / "preds_val.csv").read_bytes()}


def stratified_halves(labels: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean ``[n_splits, N]`` masks of half A; each class contributes ``n_c // 2``."""
    masks = np.zeros((n_splits, len(labels)), dtype=bool)
    for split in range(n_splits):
        for label in np.unique(labels):
            members = rng.permutation(np.flatnonzero(labels == label))
            masks[split, members[: len(members) // 2]] = True
    return masks


def selection_optimism(labels: np.ndarray, probs: np.ndarray, masks: np.ndarray) -> dict:
    """Per split: the epoch chosen on A, its optimism, AUROC_B there and the best AUROC_B."""
    auroc_a = np.array([[roc_auc_score(labels[m], p[m]) for p in probs] for m in masks])
    auroc_b = np.array([[roc_auc_score(labels[~m], p[~m]) for p in probs] for m in masks])
    chosen, rows = auroc_a.argmax(axis=1), np.arange(len(masks))  # earliest epoch on ties
    return {"epoch": chosen + 1, "optimism": auroc_a[rows, chosen] - auroc_b[rows, chosen],
            "auroc_b_selected": auroc_b[rows, chosen], "auroc_b_max": auroc_b.max(axis=1)}


def summary(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {"n": int(x.size), "mean": round(float(x.mean()), 6),
            "std": round(float(x.std(ddof=1)), 6),
            "p2_5": round(float(np.percentile(x, 2.5)), 6),
            "p97_5": round(float(np.percentile(x, 97.5)), 6)}


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode()).hexdigest()


def analyse(results_dir: Path, seeds: list[int], train_list: Path, *, n_splits: int = 100,
            rng_seed: int = 1234, dataset: str = "fhm",
            method: str = "hateclipper_align") -> dict:
    runs = {seed: load_run(results_dir / dataset / method / f"seed{seed}") for seed in seeds}
    first = runs[seeds[0]]
    for seed, run in runs.items():
        if run["ids"] != first["ids"] or not np.array_equal(run["labels"], first["labels"]):
            raise ValueError(f"seed {seed}: val ids or labels differ from seed {seeds[0]}")
    train_bytes = train_list.read_bytes()
    n_train = len(train_bytes.decode("utf-8").split())
    for seed, run in runs.items():
        if run["metrics"]["n_train"] != n_train:
            raise ValueError(f"seed {seed}: trained on {run['metrics']['n_train']} IDs, "
                             f"{train_list} has {n_train}")
    labels = first["labels"]
    masks = stratified_halves(labels, n_splits, np.random.default_rng(rng_seed))
    per_seed, pooled = {}, {name: [] for name in QUANTITIES}
    for seed, run in runs.items():
        result = selection_optimism(labels, run["probs"], masks)
        for name in QUANTITIES:
            pooled[name].extend(result[name])
        m = run["metrics"]
        per_seed[str(seed)] = {
            "config_hash": m["config_hash"], "git_sha": m["git_sha"], "n_train": m["n_train"],
            "epochs": len(run["auroc_by_epoch"]), "best_epoch": run["best_epoch"],
            "best_val_auroc": round(run["auroc_by_epoch"][run["best_epoch"] - 1], 6),
            "val_auroc_by_epoch": [round(a, 6) for a in run["auroc_by_epoch"]],
            "best_file_matches_preds_val": run["best_file_matches_preds_val"],
            "chosen_epoch_counts": {str(int(e)): int(c) for e, c in
                                    zip(*np.unique(result["epoch"], return_counts=True),
                                        strict=True)},
            **{name: summary(result[name]) for name in QUANTITIES},
        }
    return {
        "issue": 106, "dataset": dataset, "method": method, "split": "val",
        "git_sha": git_sha(),
        "train_list": {"sha256": hashlib.sha256(train_bytes).hexdigest(), "n": n_train},
        "val": {"n": len(labels), "n_positive": int(labels.sum()),
                "ids_sha256": sha256_lines(first["ids"])},
        "half_splits": {"n_splits": n_splits, "rng": f"numpy.random.default_rng({rng_seed})",
                        "stratified": True, "n_a": int(masks[0].sum()),
                        "n_b": int((~masks[0]).sum())},
        "definitions": {
            "e_star": "argmax_e AUROC_A(e), earliest epoch on ties",
            "optimism": "AUROC_A(e*) - AUROC_B(e*)",
            "auroc_b_selected": "AUROC_B(e*)", "auroc_b_max": "max_e AUROC_B(e)",
            "summary": "mean, std (ddof=1), 2.5/97.5 percentiles (numpy linear)",
            "auroc": "sklearn roc_auc_score on preds_val_epoch<NNN>.csv (6-decimal probs)"},
        "all": {name: summary(pooled[name]) for name in QUANTITIES},
        "seeds": per_seed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-list", type=Path, required=True,
                        help="the train ID list the runs used (its SHA-256 is recorded)")
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--dataset", default="fhm")
    parser.add_argument("--method", default="hateclipper_align")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyse(args.results_dir, args.seeds, args.train_list, n_splits=args.n_splits,
                     rng_seed=args.rng_seed, dataset=args.dataset, method=args.method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["all"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
