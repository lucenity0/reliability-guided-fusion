"""design notes §5 rows 1 and 7: unimodal expert and cached-feature V3 scalar gate.

V3 provenance: lucenity0/A-Unified-Adaptive-Framework-for-Multimodal-Data-Fusion-
with-Dynamic-Modality-Reweighting @ 75c6b9f5424cfd65b04a3c29c89ded40a7ce38cd:
``src/model.py`` L110–166 (DynamicGatingNetworkV3), L196–215 (auxiliary heads
and classifier), L227–261 (fuse_and_classify); ``src/regularization.py``
L22–28 (alpha_entropy_loss), L31–36 (alpha_diversity_loss).

Deviation: CLIP is frozen and cached (D-004), unlike V3's fine-tuned layers.
The gate, auxiliary heads and classifier act directly on L2-normalized cached
pooled vectors, omitting V3's pooled-token cross-attention. CE/BCE replaces
V3's focal objective; MixUp, R-Drop and other training tricks are not ported.
The V3 sample-variance regularizer is zero for singleton batches (where V3
returned NaN). Entropies remain differentiable, as in V3.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from hmf.models.experts import InputProjection, UnimodalExpert
from hmf.models.registry import register_model

UNIMODAL_KEYS = ("feature_dim", "hidden_dim", "dropout", "num_classes", "modality")
V3_KEYS = (
    "feature_dim", "num_classes", "gate_query_dim",
    "gate_hidden_dim_1", "gate_hidden_dim_2", "gate_dropout_1", "gate_dropout_2",
    "classifier_hidden_dim_1", "classifier_hidden_dim_2", "classifier_hidden_dim_3",
    "classifier_dropout_1", "classifier_dropout_2", "classifier_dropout_3",
    "lambda_aux", "lambda_entropy",
    "lambda_diversity", "normalize_eps", "entropy_eps", "alpha_entropy_eps",
)


def _check_config(cfg: dict, required: tuple[str, ...]) -> None:
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"config is missing required model keys: {missing}")
    if cfg["num_classes"] < 2:
        raise ValueError("num_classes must be at least 2")


class _Classifier(nn.Module):
    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.squeeze(-1).sigmoid() if self.binary else logits.softmax(dim=-1)

    def classification_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.binary:
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.to(logits.dtype))
        return F.cross_entropy(logits, y)


@register_model("unimodal")
class Unimodal(_Classifier):
    """One InputProjection and UnimodalExpert; requires every UNIMODAL_KEYS entry.

    Forward includes logits for stable CE/BCE and prob for the trainer, with no pi.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        _check_config(cfg, UNIMODAL_KEYS)
        if cfg["modality"] not in {"img", "txt"}:
            raise ValueError("modality must be img or txt")
        self.modality = cfg["modality"]
        self.expert_names = (self.modality,)
        self.binary = cfg["num_classes"] == 2
        q = 1 if self.binary else cfg["num_classes"]
        self.projection = InputProjection(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"])
        self.expert = UnimodalExpert(cfg["hidden_dim"], cfg["dropout"], q)

    def forward(self, batch: dict) -> dict:
        prefix = "image" if self.modality == "img" else "text"
        mask = batch["text_mask"] if self.modality == "txt" else None
        tokens = batch[f"{prefix}_tokens"]
        if mask is not None:
            tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        _, logits = self.expert(self.projection(tokens),
                                self.projection(batch[f"{prefix}_pooled"]), mask)
        return {"prob": self.probabilities(logits), "logits": logits, "z": logits[:, None]}

    def loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        main = self.classification_loss(out["logits"], batch["label"])
        return main, {"main": main.detach().item()}


def _v3_mlp(in_dim: int, widths: list[int], dropouts: list[float], out_dim: int):
    layers = []
    for index, (width, dropout) in enumerate(zip(widths, dropouts, strict=True)):
        layers.append(nn.Linear(in_dim, width))
        if index < 2:  # V3 normalizes the first two hidden layers in both MLPs.
            layers.append(nn.LayerNorm(width))
        layers.extend([nn.GELU(), nn.Dropout(dropout)])
        in_dim = width
    return nn.Sequential(*layers, nn.Linear(in_dim, out_dim))


@register_model("v3_alpha")
class V3Alpha(_Classifier):
    """V3 gate on pooled features; every V3_KEYS entry is required.

    gate_hidden_dim_1/2 and gate_dropout_1/2 configure the gate;
    classifier_hidden_dim_1/2/3 and classifier_dropout_1/2/3 configure the head.
    gate_query_dim also controls the width
    of the repeated attention score. V3's widths were 128, [512,128], [512,256,64]
    and dropout lists [0.4,0.3], [0.4,0.3,0.2]; none are implicit defaults here.
    normalize_eps, entropy_eps and alpha_entropy_eps control numerical stability.
    lambda_aux weights the mean of the two auxiliary CE/BCE losses;
    lambda_entropy and lambda_diversity weight negative entropy/sample variance.
    """

    expert_names = ("img", "txt")

    def __init__(self, cfg: dict):
        super().__init__()
        _check_config(cfg, V3_KEYS)
        for key in ("lambda_aux", "lambda_entropy", "lambda_diversity"):
            if not math.isfinite(cfg[key]) or cfg[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("normalize_eps", "entropy_eps", "alpha_entropy_eps"):
            if not math.isfinite(cfg[key]) or not 0 < cfg[key] < 0.5:
                raise ValueError(f"{key} must be finite and between 0 and 0.5")
        self.binary = cfg["num_classes"] == 2
        q = 1 if self.binary else cfg["num_classes"]
        d, a = cfg["feature_dim"], cfg["gate_query_dim"]
        self.normalize_eps, self.entropy_eps, self.alpha_entropy_eps = (
            cfg["normalize_eps"], cfg["entropy_eps"], cfg["alpha_entropy_eps"])
        self.lambda_aux, self.lambda_entropy, self.lambda_diversity = (
            cfg["lambda_aux"], cfg["lambda_entropy"], cfg["lambda_diversity"])
        self.query, self.key = nn.Linear(d, a), nn.Linear(d, a)
        self.gate = _v3_mlp(4 * d + a + 2,
                            [cfg[f"gate_hidden_dim_{i}"] for i in (1, 2)],
                            [cfg[f"gate_dropout_{i}"] for i in (1, 2)], 1)
        nn.init.zeros_(self.gate[-1].bias)
        self.img_aux_head, self.txt_aux_head = nn.Linear(d, q), nn.Linear(d, q)
        self.classifier = _v3_mlp(2 * d + 1,
                                  [cfg[f"classifier_hidden_dim_{i}"] for i in (1, 2, 3)],
                                  [cfg[f"classifier_dropout_{i}"] for i in (1, 2, 3)], q)

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        if self.binary:
            p = logits.sigmoid()
            return -(p * (p + self.entropy_eps).log()
                     + (1 - p) * (1 - p + self.entropy_eps).log())
        p = logits.softmax(dim=-1)
        return -(p * (p + self.entropy_eps).log()).sum(dim=-1, keepdim=True)

    def forward(self, batch: dict) -> dict:
        v = F.normalize(batch["image_pooled"], dim=-1, eps=self.normalize_eps)
        u = F.normalize(batch["text_pooled"], dim=-1, eps=self.normalize_eps)
        img_logits, txt_logits = self.img_aux_head(v), self.txt_aux_head(u)
        query, key = self.query(v), self.key(u)
        score = (query * key).sum(dim=-1, keepdim=True) / math.sqrt(query.shape[-1])
        descriptor = torch.cat([v, u, (v - u).abs(), v * u, score.expand_as(query),
                                self._entropy(img_logits), self._entropy(txt_logits)], dim=-1)
        alpha = self.gate(descriptor).sigmoid()
        fused = torch.cat([alpha * v, (1 - alpha) * u, alpha], dim=-1)
        logits = self.classifier(fused)
        return {"prob": self.probabilities(logits), "pi": torch.cat([alpha, 1 - alpha], dim=-1),
                "z": torch.stack([img_logits, txt_logits], dim=1),
                "logits": logits, "img_logits": img_logits, "txt_logits": txt_logits}

    def loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        y, alpha = batch["label"], out["pi"][:, 0]
        a = alpha.clamp(self.alpha_entropy_eps, 1 - self.alpha_entropy_eps)
        parts = {
            "main": self.classification_loss(out["logits"], y),
            "aux": (self.classification_loss(out["img_logits"], y)
                    + self.classification_loss(out["txt_logits"], y)) / 2,
            "alpha_entropy": (a * a.log() + (1 - a) * (1 - a).log()).mean(),
            "alpha_diversity": -alpha.var() if alpha.numel() > 1 else alpha.sum() * 0,
        }
        total = (parts["main"] + self.lambda_aux * parts["aux"]
                 + self.lambda_entropy * parts["alpha_entropy"]
                 + self.lambda_diversity * parts["alpha_diversity"])
        return total, {name: value.detach().item() for name, value in parts.items()}
