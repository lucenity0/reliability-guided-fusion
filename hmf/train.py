"""Seeded, config-driven training on cached features (#19; the design notes §7).

Usage::

    python -m hmf.train --config configs/fhm/dynamic.yaml
    MAX_SAMPLES=64 SEED=1 python -m hmf.train --config configs/fhm/dynamic.yaml

Every training setting must be in the config (``REQUIRED_KEYS``): there are no
hidden defaults, so ``config_hash`` covers the whole experiment. Optional
runtime keys: ``splits_dir`` (default ``splits``), ``checkpoint_every`` (0 =
off), ``resume`` (false), ``num_workers`` (0) and ``save_epoch_preds`` (false; true
also writes ``preds_val_epoch<NNN>.csv`` after every epoch, #106). The command line
adds ``OPTIONAL_RUNTIME`` before env overrides, so ``HMF_SPLITS_DIR``,
``SAVE_EPOCH_PREDS``, ``RESUME`` and ``CHECKPOINT_EVERY`` work with configs that omit
those keys (#110). Optional experiment keys:
``grad_clip`` (max global gradient norm; absent or 0 = no clipping) and
``lr_schedule`` (``warmup_cosine`` when absent; ``constant`` keeps lr fixed and needs
``warmup_frac: 0``, as the official Hate-CLIPper optimiser does, #13).

- Data: IDs from ``<splits_dir>/<dataset>/<role>.txt``, features from the cache
  under ``<feature_dir>/<dataset>/<cache_version>/`` that holds them.
  ``max_samples`` > 0 keeps the first N IDs of each role (smoke runs).
- Optimiser: AdamW, linear warmup over ``warmup_frac`` of the steps, then cosine
  decay to 0 (MATH_SPEC §6), unless ``lr_schedule`` is ``constant``.
- Selection: early stopping with ``patience`` on val AUROC for binary tasks and
  val accuracy for multi-class tasks (D-018). The best epoch is restored.
- Outputs: ``<results_dir>/<dataset>/<method>/seed<k>/`` gets ``preds_val.csv``,
  ``metrics.json`` and ``config.yaml``; the best checkpoint goes to
  ``<checkpoint_dir>/<dataset>/<method>/seed<k>/best.pt``. ``metrics.json`` records
  each ID list read under ``splits`` (path, ID count, SHA-256 with LF endings), since
  ``splits_dir`` is outside ``config_hash`` (#110).

The test role is never read here; use ``hmf.evaluate --final`` (PROTOCOL rules 1-2).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from hmf.config import config_hash, get_device, load_config
from hmf.data.common import id_list_record, read_id_list
from hmf.eval.calibration import calibration
from hmf.eval.corruptions import train_statistics
from hmf.eval.guard import git_sha
from hmf.eval.metrics import binary_metrics, find_threshold, multiclass_metrics, threshold_metric
from hmf.eval.quality import MixedQualityFeatures, quality_corruption, quality_groups
from hmf.features.cache import load_cache
from hmf.features.dataset import CachedFeatureDataset, make_loader
from hmf.models.registry import build_model
from hmf.utils.checkpointing import CheckpointManager
from hmf.utils.seed import restrict_sdpa, seed_everything

REQUIRED_KEYS = (
    "dataset", "method", "model", "cache_version", "num_classes",
    "batch_size", "num_epochs", "lr", "weight_decay", "warmup_frac", "patience",
    "seed", "deterministic", "max_samples", "feature_dir", "results_dir", "checkpoint_dir",
)
ROUTER_STD_WARNING = 0.02  # note: warn when any std(pi_k) on val falls below this
OPTIONAL_RUNTIME = {"splits_dir": "splits", "save_epoch_preds": False,  # outside config_hash
                    "resume": False, "checkpoint_every": 0}


def check_config(cfg: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")
    if cfg["num_classes"] < 2:
        raise ValueError("num_classes must be at least 2")


def run_path(cfg: dict, root_key: str) -> Path:
    return Path(cfg[root_key]) / cfg["dataset"] / cfg["method"] / f"seed{cfg['seed']}"


def split_list(cfg: dict, role: str) -> Path:
    return Path(cfg.get("splits_dir", "splits")) / cfg["dataset"] / f"{role}.txt"


def open_role(cfg: dict, role: str, *, allow_test: bool = False,
              feature_dir: str | None = None) -> CachedFeatureDataset:
    """Dataset for a protocol role: its committed IDs, read from the cache holding them."""
    if role == "test" and not allow_test:
        raise PermissionError("the test role is for reporting only; use hmf.evaluate --final")
    ids = read_id_list(split_list(cfg, role))
    if cfg["max_samples"] > 0:
        ids = ids[: cfg["max_samples"]]
    root = Path(feature_dir or cfg["feature_dir"]) / cfg["dataset"] / str(cfg["cache_version"])
    wanted, matches = set(ids), []
    for path in sorted(p for p in root.iterdir() if (p / "manifest.json").exists()):
        # Inspect the ID index only: opening train/val must not map test arrays.
        cache_ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
        if wanted <= set(cache_ids):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"{root}: {len(matches)} caches hold every {role} ID; expected exactly 1")
    return CachedFeatureDataset(load_cache(matches[0]), ids)


def _to(batch: dict, device) -> dict:
    return {name: value.to(device) for name, value in batch.items()}


@torch.no_grad()
def predict(model, dataset: CachedFeatureDataset, cfg: dict, device) -> dict:
    model.eval()
    loader = make_loader(dataset, batch_size=cfg["batch_size"], shuffle=False, seed=0,
                         num_workers=cfg.get("num_workers", 0))
    probs, pis, labels, experts = [], [], [], []
    for batch in loader:
        out = model(_to(batch, device))
        probs.append(out["prob"].float().cpu())
        labels.append(batch["label"])
        if out.get("pi") is not None:
            pis.append(out["pi"].float().cpu())
        if out.get("z") is not None:
            z = out["z"].float()
            if cfg["num_classes"] == 2:
                p = (z.squeeze(-1) if z.ndim == 3 else z).sigmoid()
            else:
                p = z.softmax(dim=-1)
            experts.append(p.cpu())
    return {"ids": list(dataset.ids), "labels": torch.cat(labels).numpy(),
            "prob": torch.cat(probs).numpy(), "pi": torch.cat(pis).numpy() if pis else None,
            "expert_prob": torch.cat(experts).numpy() if experts else None}


def split_metrics(pred: dict, cfg: dict, threshold: float | None = None) -> dict:
    cal = {"calibration": calibration(pred["labels"], pred["prob"])}
    if cfg["num_classes"] > 2:
        return {**multiclass_metrics(pred["labels"], pred["prob"]), **cal}
    metric = threshold_metric(cfg["dataset"])
    if threshold is None:  # chosen on val only; test reuses the stored value
        threshold, _ = find_threshold(pred["labels"], pred["prob"], metric)
    return {**binary_metrics(pred["labels"], pred["prob"], threshold),
            "threshold_metric": metric, **cal}


def selection_value(metrics: dict, cfg: dict) -> float:
    value = metrics["accuracy"] if cfg["num_classes"] > 2 else metrics["auroc"]
    if value is None:
        raise ValueError("val AUROC is undefined (a single class); use more samples")
    return value


def router_health(pi: np.ndarray) -> dict:
    """Per-expert mean and std of pi over samples, and mean entropy normalised by log K."""
    entropy = -(pi * np.log(np.clip(pi, 1e-12, None))).sum(axis=1) / math.log(pi.shape[1])
    std = pi.std(axis=0)
    return {"pi_mean": pi.mean(axis=0).tolist(), "pi_std": std.tolist(),
            "pi_entropy": float(entropy.mean()), "collapsed": bool(std.min() < ROUTER_STD_WARNING)}


EXPERT_PROBS = ("full", "true_class")


def write_predictions(path: Path, pred: dict, cfg: dict, expert_names=None,
                      expert_probs: str = "full") -> None:
    """Binary: ``id,label,prob`` (P(y=1)). Multi-class: ``id,label,pred,prob`` (argmax and its
    probability). Append pi and per-expert probabilities where available.
    Categorical experts use ``p_<expert>_<class index>`` in class order. With
    ``expert_probs="true_class"`` (multi-class only), each expert instead gets
    ``pred_<expert>`` (its argmax) and ``p_<expert>_true`` (its probability of the label),
    which keeps every expert's accuracy and cross-entropy at a fraction of the size for
    many-class datasets (Food-101: 303 columns become 6).
    """
    if expert_probs not in EXPERT_PROBS:
        raise ValueError(f"expert_probs must be one of {EXPERT_PROBS}, not {expert_probs!r}")
    multiclass, pi = cfg["num_classes"] > 2, pred["pi"]
    compact = multiclass and expert_probs == "true_class"
    names = [] if pi is None else list(expert_names or range(pi.shape[1]))
    expert_prob = pred.get("expert_prob")
    p_names = ([] if expert_prob is None else
               list(expert_names or range(expert_prob.shape[1])))
    header = ["id", "label"] + (["pred"] if multiclass else []) + ["prob"]
    header += [f"pi_{name}" for name in names]
    groups = (quality_groups(cfg["dataset"], pred["ids"])
              if quality_corruption(cfg) != "none" else None)
    if compact:
        header += [f"pred_{name}" for name in p_names] + [f"p_{name}_true" for name in p_names]
    else:
        header += ([f"p_{name}_{c}" for name in p_names for c in range(cfg["num_classes"])]
                   if multiclass else [f"p_{name}" for name in p_names])
    if groups is not None:
        header.append("quality")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")  # LF on every OS (#110)
        writer.writerow(header)
        for i, sample_id in enumerate(pred["ids"]):
            label = int(pred["labels"][i])
            row = [sample_id, "" if label < 0 else label]
            if multiclass:
                k = int(pred["prob"][i].argmax())
                row += [k, f"{pred['prob'][i, k]:.6f}"]
            else:
                row.append(f"{pred['prob'][i]:.6f}")
            if pi is not None:
                row += [f"{p:.6f}" for p in pi[i]]
            if expert_prob is not None and compact:
                probs = expert_prob[i]  # (experts, classes)
                row += [int(k) for k in probs.argmax(axis=1)]
                row += ([""] * len(probs) if label < 0 else
                        [f"{p:.9g}" for p in probs[:, label]])
            elif expert_prob is not None:
                # Preserve float32 probabilities for downstream CE/reliability analysis.
                row += [f"{p:.9g}" for p in expert_prob[i].reshape(-1)]
            if groups is not None:
                row.append(groups[i])
            writer.writerow(row)


def warmup_cosine(total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def lr_factor(cfg: dict, total_steps: int):
    """LambdaLR factor for the config's ``lr_schedule`` (absent = ``warmup_cosine``)."""
    schedule = cfg.get("lr_schedule", "warmup_cosine")
    if schedule == "constant":
        if cfg["warmup_frac"] != 0:
            raise ValueError("lr_schedule constant needs warmup_frac: 0")
        return lambda step: 1.0
    if schedule != "warmup_cosine":
        raise ValueError(f"unknown lr_schedule {schedule!r}; use warmup_cosine or constant")
    return warmup_cosine(total_steps, round(cfg["warmup_frac"] * total_steps))


def train(cfg: dict, *, device=None) -> dict:
    check_config(cfg)
    seed_everything(cfg["seed"], deterministic=cfg["deterministic"])
    device = torch.device(device) if device is not None else get_device()
    sdpa_backends = restrict_sdpa(device, deterministic=cfg["deterministic"])
    train_ds, val_ds = open_role(cfg, "train"), open_role(cfg, "val")
    quality = None
    if quality_corruption(cfg) != "none":  # D-027: statistics from the clean train role
        statistics = train_statistics(train_ds)
        train_ds = MixedQualityFeatures(train_ds, cfg["dataset"], statistics)
        val_ds = MixedQualityFeatures(val_ds, cfg["dataset"], statistics)
        quality = {"corruption": quality_corruption(cfg),
                   "train": train_ds.counts(), "val": val_ds.counts()}
    model = build_model(cfg).to(device)
    if hasattr(model, "prepare_training"):
        model.prepare_training(train_ds.ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    total_steps = math.ceil(len(train_ds) / cfg["batch_size"]) * cfg["num_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor(cfg, total_steps))
    ckpt = CheckpointManager(run_path(cfg, "checkpoint_dir"),
                             periodic_every=cfg.get("checkpoint_every", 0))
    start, patience_ctr, state = 1, 0, {}
    if cfg.get("resume", False):
        start, patience_ctr, state = ckpt.load_resume(model, optimizer, scheduler,
                                                       map_location=device)
    history, best_epoch = state.get("history", []), state.get("best_epoch", 0)
    out_dir = run_path(cfg, "results_dir")
    if cfg.get("save_epoch_preds", False):
        out_dir.mkdir(parents=True, exist_ok=True)
    seconds_before, t0 = state.get("train_seconds", 0.0), time.perf_counter()

    for epoch in range(start, cfg["num_epochs"] + 1):
        if patience_ctr >= cfg["patience"]:
            break
        model.train()
        loader = make_loader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                             seed=cfg["seed"] * 100_003 + epoch,
                             num_workers=cfg.get("num_workers", 0))
        sums, seen = defaultdict(float), 0
        for batch in loader:
            batch = _to(batch, device)
            loss, parts = model.loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            size = len(batch["label"])
            seen += size
            sums["loss"] += loss.item() * size
            for name, value in parts.items():
                sums[name] += float(value) * size
        val_pred = predict(model, val_ds, cfg, device)
        if cfg.get("save_epoch_preds", False):  # #106: per-epoch val files
            write_predictions(out_dir / f"preds_val_epoch{epoch:03d}.csv", val_pred, cfg,
                              getattr(model, "expert_names", None))
        val = split_metrics(val_pred, cfg)
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                  **{f"train_{name}": total / seen for name, total in sums.items()}, "val": val}
        if val_pred["pi"] is not None:
            record["router"] = router_health(val_pred["pi"])
            if record["router"]["collapsed"]:
                warnings.warn(f"epoch {epoch}: router std(pi) below {ROUTER_STD_WARNING} on val "
                              f"{record['router']['pi_std']}", stacklevel=2)
        history.append(record)
        if ckpt.save_best_if_improved(selection_value(val, cfg), epoch, model):
            best_epoch, patience_ctr = epoch, 0
        else:
            patience_ctr += 1
        print(f"epoch {epoch}: loss {record['train_loss']:.4f} "
              f"val {selection_value(val, cfg):.4f} (best epoch {best_epoch})", flush=True)
        ckpt.save_periodic(epoch, model, optimizer, scheduler, patience_ctr=patience_ctr,
                           extra={"history": history, "best_epoch": best_epoch,
                                  "train_seconds": seconds_before + time.perf_counter() - t0})
    train_seconds = seconds_before + time.perf_counter() - t0

    ckpt.load_best(model, map_location=device)
    val_pred = predict(model, val_ds, cfg, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / "preds_val.csv", val_pred, cfg,
                      getattr(model, "expert_names", None))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    metrics = {
        "dataset": cfg["dataset"], "method": cfg["method"], "model": cfg["model"],
        "seed": cfg["seed"], "config_hash": config_hash(cfg), "git_sha": git_sha(),
        "device": str(device), "gpu_name": os.environ.get("HMF_GPU_NAME"),
        "sdpa_backends": sdpa_backends,
        "best_epoch": best_epoch, "epochs_run": len(history), "train_seconds": train_seconds,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_train": len(train_ds), "n_val": len(val_ds),
        "splits": {role: id_list_record(split_list(cfg, role)) for role in ("train", "val")},
        "selection_metric": "accuracy" if cfg["num_classes"] > 2 else "auroc",
        "val": split_metrics(val_pred, cfg),
        "router": router_health(val_pred["pi"]) if val_pred["pi"] is not None else None,
        "checkpoint": str(ckpt.best_path), "history": history,
    }
    if quality is not None:
        metrics["quality"] = quality
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one config and seed on cached features.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    metrics = train(load_config(args.config, defaults=OPTIONAL_RUNTIME))
    print(json.dumps({k: metrics[k] for k in ("config_hash", "best_epoch", "val")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
