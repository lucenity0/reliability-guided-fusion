"""Fixed cached-feature corruptions for D-024; no fitting on val or test.

Token moments weight each image token and each unmasked text token equally;
std is the population std (ddof=0). Pooled features follow the same corruption,
and cosine is recomputed from the resulting pooled features. Noise on the
pooled CLS/EOS uses that token's noise, so duplicated representations agree.
Swap moves tokens, pooled features and (for text) the donor's mask together.
"""

from contextlib import ExitStack
from copy import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hmf.features.cache import ARRAYS
from hmf.features.dataset import make_loader

SEED = 1234
CONDITIONS = ("drop-image", "drop-text", "noise-image-0.5", "noise-text-0.5",
              "noise-image-1.0", "noise-text-1.0", "swap-image", "swap-text")
# Bound feature/statistics buffers independently of a training batch size.
MAX_BATCH_SIZE = 16


class _BufferedRows:
    """Read selected rows without faulting the entire source mmap into RSS."""

    def __init__(self, array, stack):
        if not array.flags.c_contiguous:
            raise ValueError("streamed caches must have C-contiguous arrays")
        self.handle = stack.enter_context(open(array.filename, "rb"))
        self.shape, self.dtype, self.offset = array.shape, array.dtype, array.offset
        self.row_bytes = int(np.prod(array.shape[1:])) * array.dtype.itemsize

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        result = np.empty((len(rows), *self.shape[1:]), dtype=self.dtype)
        if not len(rows):
            return result
        # CachedFeatureDataset sorts the source rows. Read contiguous runs in
        # one operation, skipping val/dedup rows without loading their features.
        boundaries = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1, len(rows)]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            self.handle.seek(self.offset + int(rows[start]) * self.row_bytes)
            raw = self.handle.read(int(stop - start) * self.row_bytes)
            result[start:stop] = np.frombuffer(raw, dtype=self.dtype).reshape(
                stop - start, *self.shape[1:])
        return result


class StreamingFeatures:
    """Context-managed buffered view of an already-open memory-mapped cache.

    Each array file handle opens once and only selected rows are read. The
    original cache and its mappings are unchanged. This bounds resident source
    pages on platforms where mmap discard hints do not evict pages promptly.
    """

    def __init__(self, dataset):
        self.dataset, self.ids = copy(dataset), dataset.ids
        self.stack = ExitStack()

    def __enter__(self):
        try:
            self.dataset.cache = copy(self.dataset.cache)
            for name in ARRAYS:
                array = getattr(self.dataset.cache, name)
                if isinstance(array, np.memmap):
                    setattr(self.dataset.cache, name, _BufferedRows(array, self.stack))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, positions):
        return self.dataset.__getitems__(positions)


def train_statistics(dataset, *, batch_size: int = MAX_BATCH_SIZE) -> dict:
    """Stream the train-role Dataset, never the whole backing source cache.

    The caller opens the train role; this respects carve-outs and dedup IDs even
    when train and val share a source cache. Parallel-variance merging avoids
    cancellation for near-constant dimensions. Padding is excluded from text.
    """
    with StreamingFeatures(dataset) as streamed:
        return _token_statistics(streamed, batch_size)


def _token_statistics(dataset, batch_size):
    stats = {}
    loader = make_loader(dataset, batch_size=min(batch_size, MAX_BATCH_SIZE),
                         shuffle=False, seed=SEED, num_workers=0)
    for batch in loader:
        for modality in ("image", "text"):
            tokens = batch[f"{modality}_tokens"].numpy()
            values = (tokens.reshape(-1, tokens.shape[-1]) if modality == "image"
                      else tokens[batch["text_mask"].numpy()]).astype(np.float64)
            n = len(values)
            if not n:
                raise ValueError("train role has no valid tokens")
            mean = values.mean(axis=0)
            m2 = ((values - mean) ** 2).sum(axis=0)
            if modality in stats:
                old = stats[modality]
                total, delta = old["count"] + n, mean - old["mean"]
                m2 += old["m2"] + delta ** 2 * (old["count"] * n / total)
                mean = old["mean"] + delta * (n / total)
                n = total
            stats[modality] = {"mean": mean, "m2": m2, "count": n}
    if not stats:
        raise ValueError("train role is empty")
    return {name: {"mean": s["mean"], "std": np.sqrt(s["m2"] / s["count"]),
                   "count": s["count"]} for name, s in stats.items()}


def swap_permutation(n: int) -> np.ndarray:
    """Uniform random derangement by rejection, fixed seed for both modalities."""
    if n < 2:
        raise ValueError("swap corruption requires at least two test samples")
    rng = np.random.default_rng(SEED)
    while True:
        permutation = rng.permutation(n)
        if np.all(permutation != np.arange(n)):
            return permutation


class StagedFeatures:
    """Temporary disk snapshot: fetch each source row once, keep only a batch in RAM.

    Swap needs random donors. The snapshot allows all conditions to share one
    source-cache read without retaining the complete split in memory. Mappings
    are opened only for a batch and explicitly closed to bound resident pages.
    The caller owns a TemporaryDirectory and removes it on success or failure.
    """

    def __init__(self, dataset, directory: str | Path, batch_size: int):
        self.ids, self.directory = list(dataset.ids), Path(directory)
        self.arrays = {}
        with ExitStack() as stack:
            streamed = stack.enter_context(StreamingFeatures(dataset))
            loader = make_loader(streamed, batch_size=min(batch_size, MAX_BATCH_SIZE),
                                 shuffle=False, seed=SEED, num_workers=0)
            handles = {}
            for batch in loader:
                for name, tensor in batch.items():
                    array = tensor.numpy()
                    if name not in handles:
                        handles[name] = stack.enter_context((self.directory / name).open("wb"))
                        self.arrays[name] = (array.dtype, (len(self.ids), *array.shape[1:]))
                    handles[name].write(array.tobytes())

    def __len__(self):
        return len(self.ids)

    def __getitems__(self, positions):
        result = {}
        for name, (dtype, shape) in self.arrays.items():
            mapped = np.memmap(self.directory / name, mode="r", dtype=dtype, shape=shape)
            try:
                result[name] = torch.from_numpy(np.array(mapped[positions], copy=True))
            finally:
                mapped._mmap.close()
        return result


class CorruptedFeatures:
    """A lazy condition over the staged split, independent of batch/worker order."""

    def __init__(self, dataset, condition: str, statistics: dict, permutation: np.ndarray):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown corruption {condition!r}")
        self.dataset, self.ids = dataset, dataset.ids
        self.condition, self.statistics, self.permutation = condition, statistics, permutation

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, positions):
        batch = self.dataset.__getitems__(positions)
        parts = self.condition.split("-")
        kind, modality = parts[:2]
        token_key, pool_key = f"{modality}_tokens", f"{modality}_pooled"
        if kind == "swap":
            donor = self.dataset.__getitems__(self.permutation[positions])
            batch[token_key], batch[pool_key] = donor[token_key], donor[pool_key]
            if modality == "text":
                batch["text_mask"] = donor["text_mask"]
        elif kind == "drop":
            mean = torch.as_tensor(self.statistics[modality]["mean"], dtype=torch.float32)
            batch[token_key] = mean.expand_as(batch[token_key]).clone()
            batch[pool_key] = mean.expand_as(batch[pool_key]).clone()
        else:
            sigma = float(parts[2])
            std = self.statistics[modality]["std"]
            # Row-keyed streams keep noise identical across methods, batch sizes
            # and evaluation order. The two sigmas use the same standardized noise.
            noise = []
            for position in positions:
                rng = np.random.default_rng(np.random.SeedSequence(
                    [SEED, 0 if modality == "image" else 1, int(position)]))
                noise.append(rng.standard_normal(batch[token_key].shape[1:]) * (sigma * std))
            noise = torch.as_tensor(np.stack(noise), dtype=torch.float32)
            batch[token_key] = batch[token_key] + noise
            pooled_index = (torch.zeros(len(positions), dtype=torch.long) if modality == "image"
                            else batch["text_mask"].long().sum(dim=1) - 1)
            batch[pool_key] = batch[pool_key] + noise[torch.arange(len(positions)), pooled_index]
        if batch["image_pooled"].shape[-1] == batch["text_pooled"].shape[-1]:
            batch["similarity"] = F.cosine_similarity(
                batch["image_pooled"], batch["text_pooled"], dim=-1)
        # Pre-projection Hate-CLIPper caches have different dimensions. Those
        # models never consume similarity, which cannot be recovered here.
        return batch
