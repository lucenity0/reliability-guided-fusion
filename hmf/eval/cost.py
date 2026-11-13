"""Cost profile of the fusion heads and the frozen backbone (#31; MATH_SPEC §8, A-6).

Usage::

    python -m hmf.eval.cost --configs configs/fhm --out results/cost/cost.json --backbone

Every latency in one file comes from one device, so run it once on the reference GPU (T4).
For each config this records trainable parameters, FLOPs per sample, latency in ms per
sample at batch sizes 1 and 64, and peak GPU memory. FLOPs follow A-6's two-per-MAC
convention: ``torch``'s counter covers linear layers but not the fused attention kernel,
so each ``nn.MultiheadAttention`` adds ``4·Q·C·h`` from its actual token counts.
``closed_form`` is the A-6 ledger for the dynamic architecture at the same shapes;
MATH_SPEC §8 asks the profiler to agree with it within 10%.

Fusion costs use the cached shapes (P=145, L=77). The frozen backbone is reported
separately at its own 577 vision tokens, never at P=145.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from hmf.config import load_config
from hmf.models.registry import build_model

P, L = 145, 77
BACKBONE = "openai/clip-vit-large-patch14-336"
BATCH_SIZES = (1, 64)


def synthetic_batch(cfg: dict, batch_size: int, device="cpu") -> dict:
    d = cfg.get("feature_dim", 768)
    return {
        "image_tokens": torch.randn(batch_size, P, d, device=device),
        "text_tokens": torch.randn(batch_size, L, d, device=device),
        "text_mask": torch.ones(batch_size, L, dtype=torch.bool, device=device),
        "image_pooled": torch.randn(batch_size, d, device=device),
        "text_pooled": torch.randn(batch_size, d, device=device),
        "similarity": torch.zeros(batch_size, device=device),
        "label": torch.zeros(batch_size, dtype=torch.long, device=device),
        "index": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


@torch.no_grad()
def count_flops(model: nn.Module, forward, batch_size: int = 1) -> dict:
    """Per-sample FLOPs for ``forward()``: counted linear algebra plus attention contractions."""
    attention = [0]

    def on_attention(module, args, _output):
        query, key = args[0], args[1]
        attention[0] += 4 * query.shape[0] * query.shape[1] * key.shape[1] * module.embed_dim

    hooks = [m.register_forward_hook(on_attention) for m in model.modules()
             if isinstance(m, nn.MultiheadAttention)]
    try:
        with FlopCounterMode(display=False) as counter:
            forward()
        counted = int(counter.get_total_flops())
    finally:
        for hook in hooks:
            hook.remove()
    return {"counted": counted // batch_size, "attention": attention[0] // batch_size,
            "total": (counted + attention[0]) // batch_size}


def _matmul(rows: int, inner: int, columns: int) -> int:
    return 2 * rows * inner * columns


def _mlp(widths) -> int:
    return sum(_matmul(1, a, b) for a, b in zip(widths[:-1], widths[1:], strict=True))


def closed_form(cfg: dict, *, p: int = P, length: int = L, k: int = 3) -> int:
    """A-6 core ledger (MATH_SPEC §8, D-017 pools), for Variant P with FFN width 2h."""
    d, h, n = cfg["feature_dim"], cfg["hidden_dim"], cfg["cross_fusion_dim"]
    outputs = 1 if cfg["num_classes"] == 2 else cfg["num_classes"]
    layer = 8 * (p + length) * h**2 + 8 * p * length * h + 4 * (p + length) * h * (2 * h)
    return (2 * (p + length) * d * h + cfg["n_cross_layers"] * layer + 4 * d * n + n**2
            + 16 * h**2 + 8 * (p + length) * h
            + 2 * _mlp([2 * h, h, h]) + _mlp([2 * h + n**2, h, h])
            + 2 * _matmul(1, d, h) + _mlp([2 * h + k + 1, h, k])
            + k * _matmul(1, h, outputs) + _matmul(1, k, outputs))


@torch.no_grad()
def latency_ms(forward, batch_size: int, device, repeats: int = 50, warmup: int = 5) -> float:
    """Median milliseconds per sample over ``repeats`` calls after ``warmup`` discarded calls."""
    gpu = str(device).startswith("cuda")
    times = []
    for i in range(warmup + repeats):
        if gpu:
            torch.cuda.synchronize()
        start = time.perf_counter()
        forward()
        if gpu:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000 / batch_size)
    return statistics.median(times)


def _peak_memory(forward, device) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        forward()
    return int(torch.cuda.max_memory_allocated())


def profile_config(cfg: dict, device, repeats: int = 50) -> dict:
    cpu_model = build_model(cfg).eval()
    cpu_batch = synthetic_batch(cfg, 1)
    flops = count_flops(cpu_model, lambda: cpu_model(cpu_batch))
    model = build_model(cfg).to(device).eval()
    result = {"method": cfg["method"], "model": cfg["model"], "dataset": cfg["dataset"],
              "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "flops": flops, "latency_ms": {}, "peak_memory_bytes": {}}
    for size in BATCH_SIZES:
        batch = synthetic_batch(cfg, size, device)
        call = (lambda b: lambda: model(b))(batch)
        result["latency_ms"][str(size)] = latency_ms(call, size, device, repeats)
        result["peak_memory_bytes"][str(size)] = _peak_memory(call, device)
    return result


def profile_backbone(device, repeats: int = 20) -> dict:
    """Frozen CLIP forward: one image and one 77-token text, per sample."""
    from transformers import CLIPModel

    def load():
        return CLIPModel.from_pretrained(BACKBONE).eval().requires_grad_(False)

    def inputs(size, on):
        return (torch.randn(size, 3, 336, 336, device=on),
                torch.ones(size, 77, dtype=torch.long, device=on),
                torch.ones(size, 77, dtype=torch.long, device=on))

    cpu_model = load()
    pixels, ids, mask = inputs(1, "cpu")
    flops = {"image": count_flops(cpu_model.vision_model,
                                  lambda: cpu_model.vision_model(pixel_values=pixels)),
             "text": count_flops(cpu_model.text_model,
                                 lambda: cpu_model.text_model(input_ids=ids, attention_mask=mask))}
    model = load().to(device)
    result = {"backbone": BACKBONE, "flops": flops, "latency_ms": {}, "peak_memory_bytes": {},
              "params": sum(p.numel() for p in model.parameters())}
    for size in BATCH_SIZES:
        pixels, ids, mask = inputs(size, device)
        call = (lambda p, i, m: lambda: model(pixel_values=p, input_ids=i, attention_mask=m))(
            pixels, ids, mask)
        result["latency_ms"][str(size)] = latency_ms(call, size, device, repeats)
        result["peak_memory_bytes"][str(size)] = _peak_memory(call, device)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure fusion and backbone cost.")
    parser.add_argument("--configs", default="configs/fhm")
    parser.add_argument("--out", default="results/cost/cost.json")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--backbone", action="store_true")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    out = {"device": device, "gpu_name": gpu_name, "shapes": {"P": P, "L": L}, "models": {}}
    for path in sorted(Path(args.configs).glob("*.yaml")):
        cfg = load_config(path, environ={})
        if cfg.get("quality_corruption") or cfg["model"].startswith("hateclipper"):
            continue  # same cost as the clean config; Hate-CLIPper uses unprojected caches
        out["models"][cfg["method"]] = profile_config(cfg, device, args.repeats)
        if cfg["method"] == "dynamic":
            profiled, form = out["models"]["dynamic"]["flops"]["total"], closed_form(cfg)
            out["closed_form"] = {"dynamic": form, "profiled": profiled,
                                  "relative_error": abs(profiled - form) / form}
        print(cfg["method"], json.dumps(out["models"][cfg["method"]]["latency_ms"]), flush=True)
    if args.backbone:
        out["backbone"] = profile_backbone(device)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
