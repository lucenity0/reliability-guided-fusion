"""Low-rank adapters for the last CLIP blocks (#30: "LoRA on the last 4 CLIP blocks").

``apply_lora(model, last_n=4, rank=8, alpha=16)`` freezes every CLIP weight, then wraps
the query and value projections of the last ``last_n`` self-attention blocks of both
towers with ``W x + (alpha / rank) · B A x``. ``B`` starts at zero, so the adapted model
equals frozen CLIP until training moves it (Hu et al., 2022).
"""

from __future__ import annotations

import math

import torch
from torch import nn

TARGETS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale, self.dropout = alpha / rank, nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + self.scale * update


def _blocks(model) -> list:
    towers = [getattr(model, "vision_model", None), getattr(model, "text_model", None)]
    return [list(t.encoder.layers) for t in towers if t is not None]


def apply_lora(model: nn.Module, *, last_n: int = 4, rank: int = 8, alpha: float = 16.0,
               dropout: float = 0.0, targets=TARGETS) -> list[str]:
    """Freeze ``model`` and adapt ``targets`` in its last ``last_n`` blocks per tower.

    Returns the adapted module names. Only the ``lora_*`` parameters stay trainable.
    """
    model.requires_grad_(False)
    adapted = []
    for tower, layers in zip(("vision", "text"), _blocks(model), strict=False):
        if last_n > len(layers):
            raise ValueError(f"{tower} tower has {len(layers)} blocks, fewer than {last_n}")
        for index in range(len(layers) - last_n, len(layers)):
            attention = layers[index].self_attn
            for name in targets:
                base = getattr(attention, name)
                if isinstance(base, LoRALinear):
                    raise ValueError("LoRA is already applied")
                setattr(attention, name, LoRALinear(base, rank, alpha, dropout))
                adapted.append(f"{tower}.layers.{index}.self_attn.{name}")
    return adapted


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter weights: what a fine-tuned checkpoint needs beyond frozen CLIP."""
    return {k: v for k, v in model.state_dict().items() if ".lora_" in k}
