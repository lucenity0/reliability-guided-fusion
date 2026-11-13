import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "selection_optimism", ROOT / "scripts" / "analysis" / "selection_optimism.py")
so = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so)


def write_preds(path, ids, labels, probs):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "prob"])
        for row in zip(ids, labels, probs, strict=True):
            writer.writerow([row[0], int(row[1]), f"{row[2]:.6f}"])


def fake_run(run_dir, labels, probs, n_train=10):
    """A seed's outputs as hmf.train writes them with save_epoch_preds."""
    run_dir.mkdir(parents=True)
    ids = [f"v{i:03d}" for i in range(len(labels))]
    for epoch, prob in enumerate(probs, start=1):
        write_preds(run_dir / f"preds_val_epoch{epoch:03d}.csv", ids, labels, prob)
    rounded = np.round(probs, 6)
    best = int(np.argmax([roc_auc_score(labels, p) for p in rounded])) + 1
    write_preds(run_dir / "preds_val.csv", ids, labels, probs[best - 1])
    (run_dir / "metrics.json").write_text(json.dumps(
        {"best_epoch": best, "config_hash": "h", "git_sha": "g", "n_train": n_train}))


def test_stratified_halves_split_each_class_and_repeat_with_the_seed():
    labels = np.array([0] * 7 + [1] * 4)
    masks = so.stratified_halves(labels, 20, np.random.default_rng(1234))
    assert masks.shape == (20, 11)
    assert (masks[:, labels == 0].sum(axis=1) == 3).all()
    assert (masks[:, labels == 1].sum(axis=1) == 2).all()
    assert np.array_equal(masks, so.stratified_halves(labels, 20, np.random.default_rng(1234)))
    assert len({m.tobytes() for m in masks}) > 1


def test_selection_optimism_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 30)
    probs = rng.uniform(size=(4, 60)) + 0.2 * labels
    masks = so.stratified_halves(labels, 5, np.random.default_rng(1))
    result = so.selection_optimism(labels, probs, masks)
    for s, m in enumerate(masks):
        a = [roc_auc_score(labels[m], p[m]) for p in probs]
        b = [roc_auc_score(labels[~m], p[~m]) for p in probs]
        e = int(np.argmax(a))
        assert result["epoch"][s] == e + 1
        assert result["optimism"][s] == pytest.approx(a[e] - b[e])
        assert result["auroc_b_selected"][s] == pytest.approx(b[e])
        assert result["auroc_b_max"][s] == pytest.approx(max(b))


def test_a_perfect_epoch_has_no_optimism():
    labels = np.array([0, 1] * 20)
    probs = np.stack([labels * 0.5 + 0.25, np.full(40, 0.5)])
    masks = so.stratified_halves(labels, 10, np.random.default_rng(2))
    result = so.selection_optimism(labels, probs, masks)
    assert (result["epoch"] == 1).all() and (result["optimism"] == 0).all()


def test_analyse_end_to_end(tmp_path):
    rng = np.random.default_rng(3)
    labels = np.array([0] * 30 + [1] * 20)
    results = tmp_path / "results"
    for seed in (0, 1):
        probs = rng.uniform(size=(3, 50)) + 0.3 * labels
        fake_run(results / "fhm" / "hateclipper_align" / f"seed{seed}", labels, probs)
    train_list = tmp_path / "train.txt"
    train_list.write_text("".join(f"t{i}\n" for i in range(10)))
    out = tmp_path / "out.json"
    assert so.main(["--results-dir", str(results), "--seeds", "0", "1", "--n-splits", "8",
                    "--train-list", str(train_list), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["all"]["optimism"]["n"] == 16
    assert report["half_splits"] == {"n_splits": 8, "rng": "numpy.random.default_rng(1234)",
                                     "stratified": True, "n_a": 25, "n_b": 25}
    assert report["train_list"]["n"] == 10 and len(report["train_list"]["sha256"]) == 64
    for seed in ("0", "1"):
        run = report["seeds"][seed]
        assert run["best_file_matches_preds_val"] and run["optimism"]["n"] == 8
        assert run["best_val_auroc"] == max(run["val_auroc_by_epoch"])
        assert sum(run["chosen_epoch_counts"].values()) == 8


def test_analyse_rejects_a_mismatched_train_list(tmp_path):
    labels = np.array([0, 1] * 10)
    results = tmp_path / "results"
    fake_run(results / "fhm" / "hateclipper_align" / "seed0", labels,
             np.random.default_rng(4).uniform(size=(2, 20)), n_train=7650)
    train_list = tmp_path / "train.txt"
    train_list.write_text("a\nb\n")
    with pytest.raises(ValueError, match="7650"):
        so.analyse(results, [0], train_list)
