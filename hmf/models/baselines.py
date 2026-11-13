"""design notes §5 rows 2 and 3: pooled-feature baselines (#22).

Both read ``batch["image_pooled"]`` [B, image_dim] and ``batch["text_pooled"]``
[B, text_dim] only. Every hyperparameter is a required config key; there are no
hidden defaults. The output width is q = 1 for binary tasks (sigmoid, BCE with
logits) and q = C for multi-class tasks (softmax, cross-entropy).

- ``concat_mlp``: [v; u] -> Linear(mlp_hidden) -> GELU -> Dropout -> Linear(q).
- ``linear_probe``: [v / ||v||; u / ||u||] -> Linear(q).
"""

import torch
from torch import nn
from torch.nn import functional as F

from hmf.models.baselines_v4 import _check_config, _Classifier
from hmf.models.registry import register_model

CONCAT_MLP_KEYS = ("num_classes", "image_dim", "text_dim", "mlp_hidden", "dropout")
LINEAR_PROBE_KEYS = ("num_classes", "image_dim", "text_dim")


class _PooledBaseline(_Classifier):
    """Shared forward/loss: ``features(batch)`` -> ``head`` -> logits."""

    head: nn.Module

    def __init__(self, cfg: dict, required: tuple[str, ...]):
        super().__init__()
        _check_config(cfg, required)
        self.binary = cfg["num_classes"] == 2
        self.q = 1 if self.binary else cfg["num_classes"]

    def features(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: dict) -> dict:
        logits = self.head(self.features(batch))
        return {"prob": self.probabilities(logits), "logits": logits}

    def loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        main = self.classification_loss(out["logits"], batch["label"])
        return main, {"main": main.detach().item()}


@register_model("concat_mlp")
class ConcatMLP(_PooledBaseline):
    """Concatenation MLP on raw pooled features; requires every CONCAT_MLP_KEYS entry."""

    def __init__(self, cfg: dict):
        super().__init__(cfg, CONCAT_MLP_KEYS)
        if not 0 <= cfg["dropout"] < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.head = nn.Sequential(
            nn.Linear(cfg["image_dim"] + cfg["text_dim"], cfg["mlp_hidden"]),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["mlp_hidden"], self.q),
        )

    def features(self, batch: dict) -> torch.Tensor:
        return torch.cat([batch["image_pooled"], batch["text_pooled"]], dim=-1)


@register_model("linear_probe")
class LinearProbe(_PooledBaseline):
    """CLIP linear probe: one Linear on the concatenated L2-normalised pooled vectors."""

    def __init__(self, cfg: dict):
        super().__init__(cfg, LINEAR_PROBE_KEYS)
        self.head = nn.Linear(cfg["image_dim"] + cfg["text_dim"], self.q)

    def features(self, batch: dict) -> torch.Tensor:
        return torch.cat([F.normalize(batch["image_pooled"], dim=-1),
                          F.normalize(batch["text_pooled"], dim=-1)], dim=-1)
