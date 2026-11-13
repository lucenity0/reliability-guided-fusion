"""Entropy-aware router from MATH_SPEC §3 (D-017 pooled maps and MLP)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from hmf.models.experts import InputProjection, _mlp


def normalized_entropy(z: torch.Tensor) -> torch.Tensor:
    """Return [B, K] entropies from [B, K, q] logits, without probability clamps."""
    if z.shape[-1] == 1:
        z = z.squeeze(-1)
        return -(z.sigmoid() * F.logsigmoid(z)
                 + (-z).sigmoid() * F.logsigmoid(-z)) / math.log(2)
    log_p = z.log_softmax(dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1) / math.log(z.shape[-1])


class Router(nn.Module):
    """Compute router logits g; the fusion model applies softmax over experts."""

    INPUTS = ("pooled", "entropy", "similarity")

    def __init__(self, d: int, h: int, k: int, dropout: float, entropy_stop_grad: bool,
                 inputs: tuple[str, ...] = INPUTS):
        super().__init__()
        unknown = set(inputs) - set(self.INPUTS)
        if unknown or not inputs:
            raise ValueError(f"router inputs must be a nonempty subset of {self.INPUTS}")
        # #29 ablation: a dropped input is zeroed, so the architecture and widths stay fixed.
        self.inputs = tuple(inputs)
        self.phi_v = InputProjection(d, h, dropout)
        self.phi_t = InputProjection(d, h, dropout)
        self.mlp = _mlp(2 * h + k + 1, h, k, dropout)
        self.entropy_stop_grad = entropy_stop_grad

    def forward(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        entropy = normalized_entropy(z.detach() if self.entropy_stop_grad else z)
        parts = [self.phi_v(batch["image_pooled"]), self.phi_t(batch["text_pooled"]), entropy,
                 batch["similarity"].reshape(-1, 1).to(entropy.dtype)]
        keep = ("pooled", "pooled", "entropy", "similarity")
        r = torch.cat([part if name in self.inputs else torch.zeros_like(part)
                       for part, name in zip(parts, keep, strict=True)], dim=-1)
        return self.mlp(r)
