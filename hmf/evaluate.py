"""Prediction CSVs for a trained run: the only entry point that may read the test split.

Usage::

    python -m hmf.evaluate --run-dir results/fhm/dynamic/seed0 --split val
    python -m hmf.evaluate --run-dir results/fhm/dynamic/seed0 --split test --final

``--split test`` needs ``--final`` and the run's config hash in
``results/FINAL_CONFIGS.md`` (docs/PROTOCOL.md rules 1-2). Binary test metrics
reuse the threshold chosen on val during training; nothing is refitted on test.
``--expert-probs true_class`` writes each expert's argmax and probability of the label
instead of its full class distribution (multi-class only; ``train.write_predictions``),
so a many-class test pass (Food-101, nine files per run) stays small enough to download.
``--feature-dir`` and ``--checkpoint-dir`` relocate a run trained on another
machine; paths are not part of the config hash. The scored ID list is added to
``metrics.json``'s ``splits`` block, as training does for train and val (#110).

D-024: final evaluation also exports eight fixed corruptions, calibration and
expert probabilities. A temporary disk snapshot permits swaps while reading
the source test cache once. Set TMPDIR to a disk with room for the split in
float32 (roughly twice its fp16 cache size); the snapshot is removed on exit.

D-027: a config with ``quality_corruption: drop50`` scores val and test through
``MixedQualityFeatures`` (statistics from the clean train role), records the test
role's group counts under ``metrics.json``'s ``quality``, and skips the D-024
conditions (PROTOCOL "Mixed-quality inputs": they are not applied on top).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml

from hmf.config import config_hash, get_device
from hmf.data.common import id_list_record
from hmf.eval.corruptions import (
    CONDITIONS,
    MAX_BATCH_SIZE,
    SEED,
    CorruptedFeatures,
    StagedFeatures,
    StreamingFeatures,
    swap_permutation,
    train_statistics,
)
from hmf.eval.guard import FINAL_CONFIGS, check_final, git_sha
from hmf.eval.quality import MixedQualityFeatures, quality_corruption
from hmf.models.registry import build_model
from hmf.train import (
    EXPERT_PROBS,
    check_config,
    open_role,
    predict,
    split_list,
    split_metrics,
    write_predictions,
)
from hmf.utils.checkpointing import CheckpointManager
from hmf.utils.seed import restrict_sdpa, seed_everything


def evaluate(run_dir: str | Path, split: str, *, final: bool = False,
             final_configs: str | Path = FINAL_CONFIGS, feature_dir: str | None = None,
             checkpoint_dir: str | None = None, device=None,
             expert_probs: str = "full") -> dict:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    check_config(cfg)
    if split == "test":
        check_final(cfg, final, Path(final_configs))
    elif split != "val":
        raise ValueError("split must be 'val' or 'test'")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["config_hash"] != config_hash(cfg):
        raise ValueError(f"{run_dir}: config.yaml does not match metrics.json's config_hash")

    seed_everything(cfg["seed"], deterministic=cfg["deterministic"])
    device = torch.device(device) if device is not None else get_device()
    sdpa_backends = restrict_sdpa(device, deterministic=cfg["deterministic"])
    model = build_model(cfg).to(device)
    ckpt_dir = (Path(checkpoint_dir) / cfg["dataset"] / cfg["method"] / f"seed{cfg['seed']}"
                if checkpoint_dir else Path(metrics["checkpoint"]).parent)
    CheckpointManager(ckpt_dir).load_best(model, map_location=device)
    names = getattr(model, "expert_names", None)
    threshold = metrics["val"].get("threshold")
    if split == "test" and cfg["num_classes"] == 2 and threshold is None:
        raise ValueError("final evaluation requires the frozen clean-val threshold")

    def score(dataset, condition=None):
        pred = predict(model, dataset, eval_cfg, device)
        result = split_metrics(pred, cfg, threshold=threshold)
        if pred["pi"] is not None:
            result["pi_mean"] = pred["pi"].mean(axis=0).tolist()
        suffix = f"_{condition}" if condition else ""
        write_predictions(run_dir / f"preds_{split}{suffix}.csv", pred, cfg, names,
                          expert_probs)
        return {**result, "n": len(dataset)}

    eval_cfg = dict(cfg)
    mixed = quality_corruption(cfg) != "none"
    if split == "test":
        # A training batch can be large; final evaluation is sequential and bounded.
        eval_cfg.update(batch_size=min(cfg["batch_size"], MAX_BATCH_SIZE), num_workers=0)
        records = {role: id_list_record(split_list(cfg, role)) for role in ("train", "val")}
        for role, record in records.items():
            recorded = metrics.get("splits", {}).get(role)
            if recorded and record["sha256"] != recorded["sha256"]:
                raise ValueError(f"{role}-role IDs changed since training; cannot evaluate")
        train_ds = open_role(cfg, "train", feature_dir=feature_dir)
        statistics = train_statistics(train_ds)

        def quality_view(view):  # D-027: the same ID-keyed rule as training
            return MixedQualityFeatures(view, cfg["dataset"], statistics) if mixed else view

        # Upgrade older trained runs' val exports without selecting a new threshold.
        val_ds = open_role(cfg, "val", feature_dir=feature_dir)
        with StreamingFeatures(val_ds) as streamed_val:
            val_pred = predict(model, quality_view(streamed_val), eval_cfg, device)
        write_predictions(run_dir / "preds_val.csv", val_pred, cfg, names, expert_probs)
        metrics["val"].update(split_metrics(val_pred, cfg, threshold))
        dataset = open_role(cfg, "test", allow_test=True, feature_dir=feature_dir)
        permutation = swap_permutation(len(dataset))
        with TemporaryDirectory(prefix="hmf-final-") as temporary:
            staged = StagedFeatures(dataset, temporary, eval_cfg["batch_size"])
            if mixed:  # PROTOCOL: the D-024 conditions are not applied on top
                view = quality_view(staged)
                result = score(view)
                metrics.setdefault("quality", {})["test"] = view.counts()
            else:
                result = score(staged)
                result["corruptions"] = {
                    condition: score(CorruptedFeatures(staged, condition, statistics,
                                                       permutation), condition)
                    for condition in CONDITIONS}
        result["robustness"] = None if mixed else {
            "seed": SEED, "train_split": records["train"],
            "statistics": "population moments over all image and unmasked text tokens",
            "token_counts": {name: s["count"] for name, s in statistics.items()},
            "swap_donor_ids": [dataset.ids[i] for i in permutation],
        }
    else:
        dataset = open_role(cfg, "val", feature_dir=feature_dir)
        if mixed:
            statistics = train_statistics(open_role(cfg, "train", feature_dir=feature_dir))
            dataset = MixedQualityFeatures(dataset, cfg["dataset"], statistics)
        result = score(dataset)
    metrics[split] = {**result, "git_sha": git_sha(), "sdpa_backends": sdpa_backends}
    metrics.setdefault("splits", {})[split] = id_list_record(split_list(cfg, split))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics[split]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write predictions for a trained run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--final", action="store_true",
                        help="required for --split test (PROTOCOL rules 1-2)")
    parser.add_argument("--final-configs", default=str(FINAL_CONFIGS))
    parser.add_argument("--feature-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--expert-probs", choices=EXPERT_PROBS, default="full",
                        help="true_class: per-expert argmax and label probability only")
    args = parser.parse_args(argv)
    result = evaluate(args.run_dir, args.split, final=args.final, final_configs=args.final_configs,
                      feature_dir=args.feature_dir, checkpoint_dir=args.checkpoint_dir,
                      expert_probs=args.expert_probs)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
