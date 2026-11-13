"""Pre-registered mixed-quality inputs (D-027; PROTOCOL "Mixed-quality inputs"; #152).

With ``quality_corruption: drop50`` every role (train, val and test) loses one modality
on a fixed half of its samples. For a sample ``id`` of dataset ``ds``::

    u = int.from_bytes(sha256(f"1234:{ds}:{id}").digest()[:8], "big") / 2**64
    u < 0.25         image missing: every image token and image_pooled become the
                     clean train-role mean image token (the D-024 drop)
    0.25 <= u < 0.5  text missing: every text token and text_pooled become the mean
                     text token; text_mask is unchanged
    otherwise        intact

The rule is keyed by sample ID, so every method, seed, epoch and batch order sees the
same samples corrupted. ``similarity`` is recomputed from the resulting pooled vectors,
as ``CorruptedFeatures`` does. Statistics come from ``train_statistics`` on the clean
train role only, computed once per run.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

QUALITY_CORRUPTIONS = ("none", "drop50")
GROUPS = ("intact", "image_missing", "text_missing")
SALT = "1234"


def quality_corruption(cfg: dict) -> str:
    """The config's corruption; absent means ``none``, so committed hashes are unchanged."""
    value = cfg.get("quality_corruption", "none")
    if value not in QUALITY_CORRUPTIONS:
        raise ValueError(f"quality_corruption must be one of {QUALITY_CORRUPTIONS}, "
                         f"not {value!r}")
    return value


def quality_u(dataset: str, sample_id: str) -> float:
    digest = hashlib.sha256(f"{SALT}:{dataset}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def quality_group(dataset: str, sample_id: str) -> str:
    u = quality_u(dataset, sample_id)
    if u < 0.25:
        return "image_missing"
    if u < 0.5:
        return "text_missing"
    return "intact"


def quality_groups(dataset: str, ids) -> list[str]:
    return [quality_group(dataset, str(sample_id)) for sample_id in ids]


def group_counts(dataset: str, ids) -> dict:
    counts = Counter(quality_groups(dataset, ids))
    return {group: counts.get(group, 0) for group in GROUPS}


class MixedQualityFeatures:
    """A view of a role whose samples are corrupted by the D-027 rule, keyed by ID.

    Wraps anything with ``ids``, ``__len__`` and ``__getitems__`` (a cached role, a
    streamed role or a staged test split).
    """

    def __init__(self, dataset, dataset_name: str, statistics: dict):
        self.dataset, self.ids, self.dataset_name = dataset, dataset.ids, dataset_name
        self.groups = np.array(quality_groups(dataset_name, self.ids))
        self.means = {modality: torch.as_tensor(statistics[modality]["mean"],
                                                dtype=torch.float32)
                      for modality in ("image", "text")}

    def __len__(self):
        return len(self.dataset)

    def counts(self) -> dict:
        return {group: int((self.groups == group).sum()) for group in GROUPS}

    def __getitems__(self, positions):
        batch = self.dataset.__getitems__(positions)
        groups = self.groups[np.asarray(positions, dtype=np.int64)]
        for modality in ("image", "text"):
            rows = torch.from_numpy(groups == f"{modality}_missing")
            if not rows.any():
                continue
            mean = self.means[modality]
            tokens, pooled = f"{modality}_tokens", f"{modality}_pooled"
            batch[tokens] = batch[tokens].clone()
            batch[pooled] = batch[pooled].clone()
            batch[tokens][rows] = mean.to(batch[tokens].dtype)
            batch[pooled][rows] = mean.to(batch[pooled].dtype)
        if batch["image_pooled"].shape[-1] == batch["text_pooled"].shape[-1]:
            batch["similarity"] = F.cosine_similarity(
                batch["image_pooled"], batch["text_pooled"], dim=-1)
        return batch
