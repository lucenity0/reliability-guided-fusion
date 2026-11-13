"""PyTorch Dataset over a feature cache (hmf/features/cache.py). CLIP never runs here.

Each item is a dict of tensors:

    image_tokens [T_img, d] float32, text_tokens [L, d] float32, text_mask [L] bool,
    image_pooled [d], text_pooled [d], similarity [], label [] (int64; -1 = unlabeled),
    index [] (row in the role's ID list)

Text is already padded to the split's longest sequence, so default collation
stacks items. ``__getitems__`` serves a whole batch with one sorted fancy-index
read per array, which keeps memory-mapped loading fast.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hmf.features.cache import FeatureCache, load_cache


class CachedFeatureDataset(Dataset):
    """Rows of one cached split, optionally restricted to (and ordered by) ``ids``.

    Pass ``ids`` from ``splits/<dataset>/<role>.txt`` when a role is a subset of a
    source split (e.g. the FHM Phase 1 train/val carve-out of ``train``).
    """

    def __init__(self, cache: FeatureCache | str | Path, ids: Sequence[str] | None = None,
                 *, require_labels: bool = True):
        self.cache = cache if isinstance(cache, FeatureCache) else load_cache(cache)
        self.ids = list(ids) if ids is not None else list(self.cache.ids)
        self.rows = self.cache.indices(self.ids)
        if require_labels and (self.cache.labels[np.sort(self.rows)] < 0).any():
            raise ValueError(f"{self.cache.path}: unlabeled rows; pass require_labels=False")

    def __len__(self) -> int:
        return len(self.rows)

    def _batch(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        rows = self.rows[positions]
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        undo = np.empty_like(order)
        undo[order] = np.arange(order.size)
        c = self.cache

        def take(array: np.ndarray) -> np.ndarray:
            return np.asarray(array[sorted_rows])[undo]

        image = take(c.image_tokens)
        return {
            "image_tokens": image.astype(np.float32),
            "text_tokens": take(c.text_tokens).astype(np.float32),
            "text_mask": take(c.text_mask),
            "image_pooled": image[:, 0].astype(np.float32),
            "text_pooled": take(c.text_pooled).astype(np.float32),
            "similarity": take(c.similarity).astype(np.float32),
            "label": take(c.labels),
            "index": positions.astype(np.int64),
        }

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        batch = self._batch(np.array([position]))
        return {name: torch.from_numpy(value[0]) if value.ndim > 1 else torch.tensor(value[0])
                for name, value in batch.items()}

    def __getitems__(self, positions: list[int]) -> dict[str, torch.Tensor]:
        """Batched fetch used by DataLoader (torch >= 2.2); pair with ``collate_fn=identity``."""
        batch = self._batch(np.asarray(positions, dtype=np.int64))
        return {name: torch.from_numpy(np.ascontiguousarray(value))
                for name, value in batch.items()}


def _identity(batch):
    return batch


def make_loader(dataset: CachedFeatureDataset, *, batch_size: int, shuffle: bool, seed: int,
                num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    """Seeded loader that fetches whole batches through ``__getitems__``."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=num_workers, drop_last=drop_last, collate_fn=_identity,
                      persistent_workers=num_workers > 0)
