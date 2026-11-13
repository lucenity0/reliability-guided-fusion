"""design notes §5 row 4: Hate-CLIPper (align and cross fusion) on cached pooled CLIP features.

Ported from the official code, gokulkarthik/hateclipper @
86ad7f15daa145c36c13a9e4eb45730a094de7af, ``engine.py`` ``CLIPClassifier`` with
``use_pretrained_map`` false, as in ``run.sh``:

- L80–87: each modality's map is Linear(in, n) → Dropout(p_map), then for each extra
  mapping layer ReLU → Linear(n, n) → Dropout(p_map). The inputs are the frozen
  encoders' ``pooler_output`` (L172 image, 1024-d; L182 text, 768-d), which the ``hc224``
  cache stores (#11): ``image_tokens[:, 0]`` and ``text_pooled``.
- L191–192: both mapped vectors are L2-normalised.
- L196–197: ``align`` is their elementwise product (n-d). L200–206: ``cross`` is the outer
  product ``bmm(img[:, :, None], txt[:, None, :])`` flattened row-major (n²-d).
- L103–111: pre-output is Dropout(p_fusion), then ``num_pre_output_layers`` ×
  (Linear → ReLU → Dropout(p_pre_output)). L113: a final Linear to one logit.
- L156, L382–389 (``common_step``): logit = output(pre_output(features)).squeeze(1),
  P(y=1) = sigmoid(logit), loss = BCE with logits against the float label.
- ``main.py`` L47: the three dropouts are, in order, map, fusion and pre-output.

Deviations (no effect at the configured n = m = 1024):

- The official pre-output width is ``map_dim`` (L106, L109); here it is ``pre_output_dim``
  (the paper's m, Table 3), and the configs set m = n.
- Dropouts are three scalar keys because configs are flat (hmf/config.py).
- Only ``align`` and ``cross`` are ported, without the image/text/fine-grained auxiliary
  heads (L117–154; the official ``run.sh`` gives them weight 0).
- The encoders are frozen in the official run too (``run.sh`` ``--freeze_*_encoder t``), so
  reading their pooled outputs from the cache computes the same function.
"""

import torch
from torch import nn
from torch.nn import functional as F

from hmf.models.registry import register_model

HATECLIPPER_KEYS = (
    "cache_version", "image_dim", "text_dim", "map_dim", "num_mapping_layers", "fusion",
    "num_pre_output_layers", "pre_output_dim", "drop_map", "drop_fusion", "drop_pre_output",
    "num_classes",
)
FUSIONS = ("align", "cross")
CACHE_VERSION = "hc224"  # pre-projection pooler outputs at 224 px (#11, #13)


def _check_config(cfg: dict) -> None:
    missing = [key for key in HATECLIPPER_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"config is missing required model keys: {missing}")
    if cfg["cache_version"] != CACHE_VERSION:
        raise ValueError(f"hateclipper reads the {CACHE_VERSION!r} cache (pre-projection pooler "
                         f"outputs), got cache_version {cfg['cache_version']!r}")
    if cfg["num_classes"] != 2:
        raise ValueError("hateclipper is binary only (num_classes 2), as in the official FHM head")
    if cfg["fusion"] not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}, got {cfg['fusion']!r}")
    if cfg["num_mapping_layers"] < 1:
        raise ValueError("num_mapping_layers must be at least 1")


def _mapping(in_dim: int, n: int, layers: int, p: float) -> nn.Sequential:
    modules = [nn.Linear(in_dim, n), nn.Dropout(p)]  # engine.py L80-81
    for _ in range(1, layers):  # L82-84
        modules += [nn.ReLU(), nn.Linear(n, n), nn.Dropout(p)]
    return nn.Sequential(*modules)


def fused_dim(fusion: str, map_dim: int) -> int:
    return map_dim if fusion == "align" else map_dim ** 2  # engine.py L89-94


def num_parameters(cfg: dict) -> int:
    """Trainable parameters, computed analytically (the cross model has ~1.1B)."""
    n, m = cfg["map_dim"], cfg["pre_output_dim"]
    maps = sum(d * n + n + (cfg["num_mapping_layers"] - 1) * (n * n + n)
               for d in (cfg["image_dim"], cfg["text_dim"]))
    width, pre = fused_dim(cfg["fusion"], n), 0
    for _ in range(cfg["num_pre_output_layers"]):
        pre += width * m + m
        width = m
    return maps + pre + width + 1


@register_model("hateclipper")
class HateCLIPper(nn.Module):
    """Hate-CLIPper classifier; requires every HATECLIPPER_KEYS entry. Forward returns the
    logit and P(y=1), with no pi."""

    def __init__(self, cfg: dict):
        super().__init__()
        _check_config(cfg)
        self.fusion = cfg["fusion"]
        n, layers, p = cfg["map_dim"], cfg["num_mapping_layers"], cfg["drop_map"]
        self.image_map = _mapping(cfg["image_dim"], n, layers, p)
        self.text_map = _mapping(cfg["text_dim"], n, layers, p)
        pre_output, width = [nn.Dropout(cfg["drop_fusion"])], fused_dim(self.fusion, n)  # L103
        for _ in range(cfg["num_pre_output_layers"]):  # L105-109
            pre_output += [nn.Linear(width, cfg["pre_output_dim"]), nn.ReLU(),
                           nn.Dropout(cfg["drop_pre_output"])]
            width = cfg["pre_output_dim"]
        self.pre_output = nn.Sequential(*pre_output)
        self.output = nn.Linear(width, 1)  # L113

    def forward(self, batch: dict) -> dict:
        image = F.normalize(self.image_map(batch["image_pooled"]), p=2, dim=1)  # L172-173, L191
        text = F.normalize(self.text_map(batch["text_pooled"]), p=2, dim=1)  # L182-185, L192
        if self.fusion == "align":
            features = image * text  # L197
        else:
            features = torch.bmm(image.unsqueeze(2), text.unsqueeze(1)).flatten(1)  # L201, L206
        logits = self.output(self.pre_output(features)).squeeze(-1)  # L382-383
        return {"logits": logits, "prob": logits.sigmoid()}  # L386

    def loss(self, output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        loss = F.binary_cross_entropy_with_logits(  # engine.py L156, L389
            output["logits"], batch["label"].to(output["logits"].dtype))
        return loss, {"main": float(loss.detach())}
