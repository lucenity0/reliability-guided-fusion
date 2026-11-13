"""Shared test helpers."""

import numpy as np
import pytest


def _synthetic_cache_arrays(n=12, t_img=5, length=7, d=8, seed=0, labels=None):
    rng = np.random.default_rng(seed)
    mask = np.zeros((n, length), dtype=bool)
    lengths = rng.integers(1, length - 1, size=n)  # longest is < length, so it gets trimmed
    for i, k in enumerate(lengths):
        mask[i, :k] = True
    return {
        "ids": [f"{i:05d}" for i in range(n)],
        "image_tokens": rng.normal(size=(n, t_img, d)).astype(np.float32),
        "text_tokens": rng.normal(size=(n, length, d)).astype(np.float32),
        "text_mask": mask,
        "text_pooled": rng.normal(size=(n, d)).astype(np.float32),
        "similarity": rng.uniform(-1, 1, size=n).astype(np.float32),
        "labels": np.array(labels if labels is not None else rng.integers(0, 2, size=n)),
        "meta": {"backbone": "clip-vit-l14-336", "cache_version": "v1", "git_sha": "abc"},
    }


@pytest.fixture
def cache_arrays():
    """Factory for small synthetic feature-cache arrays (hmf/features/cache.py)."""
    return _synthetic_cache_arrays
