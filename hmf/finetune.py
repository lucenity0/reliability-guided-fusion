"""End-to-end training with LoRA on the last CLIP blocks (#30; dropped from the paper by
D-029 and kept ready in case a reviewer asks).

Usage (needs the raw images and a GPU)::

    python -m hmf.finetune --config configs/fhm/lora/dynamic.yaml --root ~/datasets/fhm

The fusion model, its config and its selection rule are those of ``hmf.train``; the only
difference is that features come from CLIP on every step instead of from the cache, with
``apply_lora`` adapters on the last ``lora_last_n`` blocks of both towers. Rows are built
exactly as the ``v1`` cache builds them (``encode_batch``), so a run with zero-initialised
adapters starts from the cached-feature model's inputs. Extra config keys: ``lora_rank``,
``lora_alpha``, ``lora_last_n``, ``lora_lr``. Train and val roles only: the test role is
never read here (PROTOCOL rules 1-2).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import yaml

from hmf.config import config_hash, get_device, load_config
from hmf.data.common import read_id_list
from hmf.eval.guard import git_sha
from hmf.features.extract import (
    SPECS,
    encode_batch,
    load_clip,
    load_source,
    make_batches,
    source_splits,
)
from hmf.models.lora import apply_lora, lora_state_dict
from hmf.models.registry import build_model
from hmf.train import (
    OPTIONAL_RUNTIME,
    check_config,
    lr_factor,
    run_path,
    selection_value,
    split_list,
    split_metrics,
    write_predictions,
)
from hmf.utils.seed import seed_everything

LORA_KEYS = ("lora_rank", "lora_alpha", "lora_last_n", "lora_lr")


class FineTuned(torch.nn.Module):
    """CLIP with adapters feeding a fusion model the rows a ``v1`` cache would hold."""

    def __init__(self, clip, fusion, spec_name: str = "v1"):
        super().__init__()
        self.clip, self.fusion, self.spec = clip, fusion, SPECS[spec_name]
        self.expert_names = getattr(fusion, "expert_names", None)

    def rows(self, pixels, input_ids, mask, labels) -> dict:
        out = encode_batch(self.clip, self.spec, pixels, input_ids, mask)
        image = out["image_tokens"].float()
        return {"image_tokens": image, "text_tokens": out["text_tokens"].float(),
                "text_mask": out["text_mask"].to(pixels.device),
                "image_pooled": image[:, 0], "text_pooled": out["text_pooled"].float(),
                "similarity": out["similarity"].float(), "label": labels}

    def forward(self, batch: dict) -> dict:
        return self.fusion(batch)

    def loss(self, out, batch):
        return self.fusion.loss(out, batch)


def role_examples(cfg: dict, root: str, role: str) -> list:
    if role == "test":
        raise PermissionError("the test role is for reporting only (PROTOCOL rules 1-2)")
    wanted = read_id_list(split_list(cfg, role))
    if cfg["max_samples"] > 0:
        wanted = wanted[: cfg["max_samples"]]
    pool = {e.id: e for split in source_splits(cfg["dataset"], [role])
            for e in load_source(cfg["dataset"], root, split)}
    return [pool[i] for i in wanted]


@torch.no_grad()
def _predict(model, loader, labels, device) -> dict:
    model.eval()
    probs, pis, experts, order = [], [], [], []
    for rows, pixels, input_ids, mask, _ in loader:
        y = torch.as_tensor(labels[rows.numpy()], device=device)
        out = model(model.rows(pixels.to(device), input_ids.to(device), mask.to(device), y))
        probs.append(out["prob"].float().cpu())
        order.append(rows)
        if out.get("pi") is not None:
            pis.append(out["pi"].float().cpu())
        if out.get("z") is not None:
            z = out["z"].float()
            experts.append((z.squeeze(-1).sigmoid() if z.shape[-1] == 1
                            else z.softmax(-1)).cpu())
    index = torch.cat(order).numpy()
    return {"index": index, "labels": labels[index], "prob": torch.cat(probs).numpy(),
            "pi": torch.cat(pis).numpy() if pis else None,
            "expert_prob": torch.cat(experts).numpy() if experts else None}


def finetune(cfg: dict, train_examples, val_examples, *, clip, preprocess, device=None,
             num_workers: int = 0) -> dict:
    check_config(cfg)
    missing = [k for k in LORA_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config is missing LoRA keys: {missing}")
    seed_everything(cfg["seed"], deterministic=cfg["deterministic"])
    device = torch.device(device) if device is not None else get_device()
    adapted = apply_lora(clip, last_n=cfg["lora_last_n"], rank=cfg["lora_rank"],
                         alpha=cfg["lora_alpha"])
    model = FineTuned(clip, build_model(cfg)).to(device)
    lora = [p for n, p in model.clip.named_parameters() if ".lora_" in n]
    optimizer = torch.optim.AdamW(
        [{"params": model.fusion.parameters(), "lr": cfg["lr"]},
         {"params": lora, "lr": cfg["lora_lr"]}], weight_decay=cfg["weight_decay"])
    labels = {name: torch.tensor([e.label for e in examples]).numpy()
              for name, examples in (("train", train_examples), ("val", val_examples))}
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = torch.utils.data.DataLoader(
        make_batches(train_examples, preprocess, cfg["batch_size"], num_workers).dataset,
        batch_size=cfg["batch_size"], shuffle=True, generator=generator,
        num_workers=num_workers, collate_fn=preprocess)
    val_loader = make_batches(val_examples, preprocess, cfg["batch_size"], num_workers)
    steps = math.ceil(len(train_examples) / cfg["batch_size"]) * cfg["num_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor(cfg, steps))
    out_dir, ckpt_dir = run_path(cfg, "results_dir"), run_path(cfg, "checkpoint_dir")
    best, best_epoch, patience, history, start = -math.inf, 0, 0, [], time.perf_counter()
    for epoch in range(1, cfg["num_epochs"] + 1):
        if patience >= cfg["patience"]:
            break
        model.train()
        for rows, pixels, input_ids, mask, _ in train_loader:
            y = torch.as_tensor(labels["train"][rows.numpy()], device=device)
            batch = model.rows(pixels.to(device), input_ids.to(device), mask.to(device), y)
            loss, _ = model.loss(model(batch), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        pred = _predict(model, val_loader, labels["val"], device)
        val = split_metrics(pred, cfg)
        history.append({"epoch": epoch, "val": val})
        if selection_value(val, cfg) > best:
            best, best_epoch, patience = selection_value(val, cfg), epoch, 0
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"fusion": model.fusion.state_dict(), "lora": lora_state_dict(clip),
                        "epoch": epoch}, ckpt_dir / "best.pt")
        else:
            patience += 1
        print(f"epoch {epoch}: val {selection_value(val, cfg):.4f} (best epoch {best_epoch})",
              flush=True)
    state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.fusion.load_state_dict(state["fusion"])
    clip.load_state_dict(state["lora"], strict=False)
    pred = _predict(model, val_loader, labels["val"], device)
    pred["ids"] = [val_examples[i].id for i in pred["index"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / "preds_val.csv", pred, cfg, model.expert_names)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    metrics = {"dataset": cfg["dataset"], "method": cfg["method"], "seed": cfg["seed"],
               "config_hash": config_hash(cfg), "git_sha": git_sha(), "device": str(device),
               "best_epoch": best_epoch, "train_seconds": time.perf_counter() - start,
               "lora_modules": adapted,
               "lora_params": sum(p.numel() for p in lora),
               "fusion_params": sum(p.numel() for p in model.fusion.parameters()),
               "n_train": len(train_examples), "n_val": len(val_examples),
               "val": split_metrics(pred, cfg), "history": history,
               "checkpoint": str(ckpt_dir / "best.pt")}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA plus fusion (#30).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True, help="the dataset's raw release")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, defaults=OPTIONAL_RUNTIME)
    device = get_device()
    clip, preprocess = load_clip(SPECS["v1"], device, torch.float32)
    metrics = finetune(cfg, role_examples(cfg, args.root, "train"),
                       role_examples(cfg, args.root, "val"), clip=clip, preprocess=preprocess,
                       device=device, num_workers=args.num_workers)
    print(json.dumps({k: metrics[k] for k in ("config_hash", "best_epoch", "val")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
