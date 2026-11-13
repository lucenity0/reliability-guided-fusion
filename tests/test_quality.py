"""D-027 mixed-quality inputs (#152; PROTOCOL "Mixed-quality inputs")."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmf.config import config_hash, load_config  # noqa: E402
from hmf.eval.quality import (  # noqa: E402
    MixedQualityFeatures,
    group_counts,
    quality_corruption,
    quality_group,
    quality_u,
)

ROOT = Path(__file__).resolve().parents[1]


def test_u_follows_the_preregistered_formula():
    digest = hashlib.sha256(b"1234:fhm:01235").digest()
    assert quality_u("fhm", "01235") == int.from_bytes(digest[:8], "big") / 2 ** 64


def test_assignment_is_deterministic_keyed_by_id_and_25_25_50():
    ids = [f"id{i}" for i in range(10_000)]
    counts = group_counts("mami", ids)
    assert counts["image_missing"] == pytest.approx(2500, abs=150)
    assert counts["text_missing"] == pytest.approx(2500, abs=150)
    assert counts["intact"] == pytest.approx(5000, abs=150)
    # The same ID gets the same group whatever its position or batch.
    assert [quality_group("mami", i) for i in reversed(ids)] == \
        [quality_group("mami", i) for i in ids][::-1]
    # The dataset name is part of the key.
    assert [quality_group("mami", i) for i in ids[:200]] != \
        [quality_group("fhm", i) for i in ids[:200]]


def test_unknown_corruption_is_rejected_and_absent_means_none():
    assert quality_corruption({}) == "none"
    with pytest.raises(ValueError, match="quality_corruption"):
        quality_corruption({"quality_corruption": "drop25"})


class FakeRole:
    def __init__(self, ids, d=4):
        self.ids = ids
        n = len(ids)
        rng = np.random.default_rng(0)
        self.data = {
            "image_tokens": torch.tensor(rng.normal(size=(n, 3, d)), dtype=torch.float32),
            "text_tokens": torch.tensor(rng.normal(size=(n, 2, d)), dtype=torch.float32),
            "text_mask": torch.tensor([[True, False]] * n),
            "text_pooled": torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32),
            "similarity": torch.zeros(n),
            "label": torch.zeros(n, dtype=torch.long),
        }
        self.data["image_pooled"] = self.data["image_tokens"][:, 0].clone()

    def __len__(self):
        return len(self.ids)

    def __getitems__(self, positions):
        return {k: v[positions].clone() for k, v in self.data.items()}


def test_missing_modality_becomes_the_mean_token_and_similarity_is_recomputed():
    ids = [f"x{i}" for i in range(64)]
    role = FakeRole(ids)
    stats = {"image": {"mean": np.full(4, 0.5)}, "text": {"mean": np.full(4, -0.25)}}
    view = MixedQualityFeatures(role, "toy", stats)
    positions = list(range(64))[::-1]  # any order
    batch, clean = view.__getitems__(positions), role.__getitems__(positions)
    for row, position in enumerate(positions):
        group = quality_group("toy", ids[position])
        image_dropped = torch.all(batch["image_tokens"][row] == 0.5)
        text_dropped = torch.all(batch["text_tokens"][row] == -0.25)
        assert image_dropped == (group == "image_missing")
        assert text_dropped == (group == "text_missing")
        if group == "image_missing":
            assert torch.all(batch["image_pooled"][row] == 0.5)
        if group == "text_missing":
            assert torch.all(batch["text_pooled"][row] == -0.25)
        if group == "intact":
            for key in ("image_tokens", "text_tokens", "image_pooled", "text_pooled"):
                assert torch.equal(batch[key][row], clean[key][row])
    assert torch.equal(batch["text_mask"], clean["text_mask"])
    expected = torch.nn.functional.cosine_similarity(batch["image_pooled"],
                                                     batch["text_pooled"], dim=-1)
    assert torch.allclose(batch["similarity"], expected)
    assert view.counts() == group_counts("toy", ids)
    # The wrapped role itself is never modified.
    assert torch.equal(role.__getitems__(positions)["image_tokens"], clean["image_tokens"])


def test_committed_clean_hashes_unchanged_and_drop50_changes_the_hash():
    listed = set(re.findall(r"^\| ([0-9a-f]{64}) \|",
                            (ROOT / "results/FINAL_CONFIGS.md").read_text(), re.M))
    for ds in ("fhm", "mami", "harm_c", "harm_p", "crisismmd", "mvsa"):
        for method in ("dynamic", "static", "global", "qmf", "pdf"):
            clean = load_config(ROOT / f"configs/{ds}/{method}.yaml", environ={})
            mq = load_config(ROOT / f"configs/{ds}/{method}_mq.yaml", environ={})
            assert "quality_corruption" not in clean
            assert config_hash(clean) in listed
            assert mq["quality_corruption"] == "drop50"
            assert mq["method"] == f"{clean['method']}_mq"
            assert {k for k in set(clean) | set(mq) if clean.get(k) != mq.get(k)} == \
                {"method", "quality_corruption"}
            assert config_hash(mq) != config_hash(clean)


def test_training_uses_clean_train_statistics_and_records_counts(tmp_path, monkeypatch):
    import hmf.train as training
    from hmf.train import train
    from tests.test_train import read_csv

    seen = []
    real = training.train_statistics

    def spy(dataset):
        seen.append(dataset)
        return real(dataset)

    monkeypatch.setattr(training, "train_statistics", spy)
    cfg = _toy_cfg(tmp_path, quality_corruption="drop50")
    metrics = train(cfg, device="cpu")
    assert len(seen) == 1
    assert type(seen[0]).__name__ == "CachedFeatureDataset"  # the clean train role
    train_ids = (tmp_path / "splits/toy2/train.txt").read_text().split()
    val_ids = (tmp_path / "splits/toy2/val.txt").read_text().split()
    assert list(seen[0].ids) == train_ids
    assert metrics["quality"] == {"corruption": "drop50",
                                  "train": group_counts("toy2", train_ids),
                                  "val": group_counts("toy2", val_ids)}
    rows = read_csv(tmp_path / "results/toy2/dyn_mq/seed0/preds_val.csv")
    assert [r["quality"] for r in rows] == [quality_group("toy2", r["id"]) for r in rows]


def test_final_pass_applies_the_rule_and_skips_d024_conditions(tmp_path):
    from hmf.evaluate import evaluate
    from hmf.train import train
    from tests.test_train import read_csv

    cfg = _toy_cfg(tmp_path, quality_corruption="drop50")
    train(cfg, device="cpu")
    run_dir = tmp_path / "results/toy2/dyn_mq/seed0"
    final_configs = tmp_path / "FINAL_CONFIGS.md"
    final_configs.write_text(f"| {config_hash(cfg)} | toy |\n")
    result = evaluate(run_dir, "test", final=True, final_configs=final_configs, device="cpu")
    assert "corruptions" not in result and result["robustness"] is None
    assert sorted(p.name for p in run_dir.glob("preds_test*.csv")) == ["preds_test.csv"]
    test_ids = (tmp_path / "splits/toy2/test.txt").read_text().split()
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["quality"]["test"] == group_counts("toy2", test_ids)
    rows = read_csv(run_dir / "preds_test.csv")
    assert [r["quality"] for r in rows] == [quality_group("toy2", i) for i in test_ids]


def test_drop50_changes_predictions_only_on_corrupted_samples(tmp_path):
    from hmf.evaluate import evaluate
    from hmf.train import train
    from tests.test_train import read_csv

    cfg = _toy_cfg(tmp_path)
    train(cfg, device="cpu")
    run_dir = tmp_path / "results/toy2/dyn/seed0"
    clean = {r["id"]: r["prob"] for r in read_csv(run_dir / "preds_val.csv")}
    # Score the same checkpoint with the rule switched on (val only).
    config = json.loads(json.dumps(cfg))
    config["quality_corruption"] = "drop50"
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    metrics["config_hash"] = config_hash(config)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    evaluate(run_dir, "val", device="cpu")
    for row in read_csv(run_dir / "preds_val.csv"):
        if row["quality"] == "intact":
            assert row["prob"] == clean[row["id"]]
    assert any(row["prob"] != clean[row["id"]]
               for row in read_csv(run_dir / "preds_val.csv") if row["quality"] != "intact")


def _toy_cfg(tmp_path, **overrides):
    from tests.test_train import make_split

    dataset = "toy2"
    source = [f"s{i:03d}" for i in range(100)]
    held = [f"h{i:03d}" for i in range(30)]
    root = tmp_path / "features" / dataset / "v1"
    make_split(root / "train", source, np.arange(100) % 2, 2, 0)
    make_split(root / "dev_seen", held, np.arange(30) % 2, 2, 1)
    splits = tmp_path / "splits" / dataset
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "train.txt").write_text("\n".join(source[:60]) + "\n")
    (splits / "val.txt").write_text("\n".join(source[60:]) + "\n")
    (splits / "test.txt").write_text("\n".join(held) + "\n")
    method = "dyn_mq" if overrides.get("quality_corruption") else "dyn"
    cfg = {
        "dataset": dataset, "method": method, "model": "toy_router", "cache_version": "v1",
        "num_classes": 2, "batch_size": 16, "num_epochs": 3, "lr": 0.05,
        "weight_decay": 0.0, "warmup_frac": 0.1, "patience": 10, "seed": 0,
        "deterministic": True, "max_samples": 0,
        "feature_dir": str(tmp_path / "features"), "results_dir": str(tmp_path / "results"),
        "checkpoint_dir": str(tmp_path / "ckpt"), "splits_dir": str(tmp_path / "splits"),
    }
    cfg.update(overrides)
    return cfg
