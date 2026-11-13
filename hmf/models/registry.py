"""Model registry and the interface the training loop relies on (#19).

A model is a ``torch.nn.Module`` registered under the config key ``model``::

    @register_model("dynamic")
    def build(cfg: dict) -> nn.Module: ...

Its ``forward(batch)`` takes the batch dict from ``CachedFeatureDataset``
(``image_tokens``, ``text_tokens``, ``text_mask``, ``image_pooled``,
``text_pooled``, ``similarity``, ``label``) and returns a dict with:

- ``prob``: P(y=1) with shape [B] for binary tasks, or class probabilities
  [B, C] for multi-class tasks (``num_classes`` > 2). Used for every metric.
- ``pi`` (optional, may be None): routing weights [B, K]. If non-None, router health is
  logged each epoch and the weights are written to the prediction CSVs.
- ``z`` (optional): per-expert logits [B,K,1] for binary or [B,K,C] for
  categorical tasks. ``predict`` exports their probabilities for D-024.

It also implements ``loss(output, batch) -> (loss, parts)``: a scalar tensor to
minimise and a dict of float components to log (e.g. main, aux, rel). Optional
``expert_names`` (a sequence of K names) labels the ``pi_*`` CSV columns.
Optional ``prepare_training(ids)`` binds training-only model state to the
ordered training-role IDs before checkpoint restoration (used by QMF).

Model modules register themselves on import; add each one to ``MODEL_MODULES``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

MODELS: dict[str, Callable] = {}
MODEL_MODULES: tuple[str, ...] = (
    "hmf.models.fusion", "hmf.models.baselines_v4", "hmf.models.baselines",
    "hmf.models.hateclipper", "hmf.models.dynamic_baselines",
)


def register_model(name: str):
    def decorator(builder: Callable) -> Callable:
        if name in MODELS:
            raise ValueError(f"model {name!r} is already registered")
        MODELS[name] = builder
        return builder

    return decorator


def build_model(cfg: dict):
    for module in MODEL_MODULES:
        importlib.import_module(module)
    name = cfg["model"]
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](cfg)
