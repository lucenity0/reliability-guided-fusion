"""Cross-dataset transfer (#28; PROTOCOL "Cross-dataset transfer").

Usage::

    python -m hmf.eval.transfer --runs results --checkpoints checkpoints \\
        --feature-dir features --out results/transfer

A model trained on dataset A (selected on A's val) scores dataset B's test set with no
adaptation. Only AUROC is reported: the label semantics differ and A's threshold doesn't
transfer. Datasets: FHM, MAMI, Harm-C, Harm-P. The source run's config hash must be in
``results/FINAL_CONFIGS.md`` (PROTOCOL rules 1-2), because B's test split is read.

Each (source run, target) writes ``<out>/<source>/<method>/seed<k>/<target>.json``. The
diagonal (A on A) is not recomputed here: it is A's own --final result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from hmf.config import config_hash, get_device
from hmf.eval.guard import FINAL_CONFIGS, check_final, git_sha
from hmf.eval.significance import auroc
from hmf.models.registry import build_model
from hmf.train import check_config, open_role, predict
from hmf.utils.checkpointing import CheckpointManager
from hmf.utils.seed import restrict_sdpa, seed_everything

DATASETS = ("fhm", "mami", "harm_c", "harm_p")
METHODS = ("dynamic", "static", "qmf", "pdf", "concat_mlp")


def transfer(run_dir: str | Path, target: str, *, checkpoint_dir: str | Path,
             feature_dir: str | None = None, final_configs: str | Path = FINAL_CONFIGS,
             device=None) -> dict:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    check_config(cfg)
    source = cfg["dataset"]
    if source not in DATASETS or target not in DATASETS:
        raise ValueError(f"transfer is defined only among {DATASETS}")
    if target == source:
        raise ValueError("the diagonal is the source's own --final result; don't rescore it")
    if cfg["num_classes"] != 2:
        raise ValueError("transfer needs a binary model")
    check_final(cfg, True, Path(final_configs))
    seed_everything(cfg["seed"], deterministic=cfg["deterministic"])
    device = torch.device(device) if device is not None else get_device()
    restrict_sdpa(device, deterministic=cfg["deterministic"])
    model = build_model(cfg).to(device)
    ckpt = Path(checkpoint_dir) / source / cfg["method"] / f"seed{cfg['seed']}"
    CheckpointManager(ckpt).load_best(model, map_location=device)
    target_cfg = {**cfg, "dataset": target, "max_samples": 0}
    dataset = open_role(target_cfg, "test", allow_test=True, feature_dir=feature_dir)
    pred = predict(model, dataset, target_cfg, device)
    labels = np.asarray(pred["labels"])
    return {"source": source, "target": target, "method": cfg["method"], "seed": cfg["seed"],
            "auroc": float(auroc(labels, np.asarray(pred["prob"]))), "n": int(len(labels)),
            "source_config_hash": config_hash(cfg), "git_sha": git_sha(), "device": str(device)}


def run_all(runs: Path, checkpoints: Path, out: Path, *, feature_dir: str | None = None,
            methods=METHODS, datasets=None, final_configs=FINAL_CONFIGS, device=None) -> int:
    datasets, count = datasets or DATASETS, 0
    for source in datasets:
        for method in methods:
            for run_dir in sorted((runs / source / method).glob("seed*")):
                if not (run_dir / "config.yaml").exists():
                    continue
                for target in datasets:
                    if target == source:
                        continue
                    path = out / source / method / run_dir.name / f"{target}.json"
                    if path.exists():
                        continue
                    result = transfer(run_dir, target, checkpoint_dir=checkpoints,
                                      feature_dir=feature_dir, final_configs=final_configs,
                                      device=device)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    print(f"{source}->{target} {method}/{run_dir.name}: "
                          f"auroc {result['auroc']:.4f}", flush=True)
                    count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score each source model on other test sets.")
    parser.add_argument("--runs", default="results", help="root holding <ds>/<method>/seed<k>")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--feature-dir")
    parser.add_argument("--out", default="results/transfer")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--final-configs", default=str(FINAL_CONFIGS))
    args = parser.parse_args(argv)
    n = run_all(Path(args.runs), Path(args.checkpoints), Path(args.out),
                feature_dir=args.feature_dir, methods=args.methods,
                final_configs=args.final_configs)
    print(f"wrote {n} transfer results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
