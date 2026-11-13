"""On-disk format for frozen CLIP feature caches (docs/MATH_SPEC.md §1).

One directory per (dataset, cache version, source split)::

    <root>/<dataset>/<cache_version>/<split>/
        manifest.json        format, backbone, shapes, dtype, git_sha, n, ids_sha256
        ids.json             sample IDs in row order (the join key with splits/*.txt)
        image_tokens.npy     float16 [N, T_img, d]  row 0 = CLS (= CLIP image embedding)
        text_tokens.npy      float16 [N, L, d_t]    L = longest text in the split (≤ 77)
        text_mask.npy        bool    [N, L]
        text_pooled.npy      float16 [N, d_t]       EOS token (= CLIP text embedding)
        similarity.npy       float32 [N]            cos(image CLS, text EOS)
        labels.npy           int64   [N]            class index; -1 = unlabeled

The main cache stores projected features, so d_t = d. A pooled-only cache of
pre-projection pooler outputs (e.g. Hate-CLIPper's, #13) has T_img = L = 1 and
d_t != d (1024 vs 768 for CLIP ViT-L/14).

Arrays are plain ``.npy`` files so readers can memory-map them; nothing is
loaded into RAM until indexed. The cache holds derived features of licensed
data: keep it outside the repo and share it only privately (docs/COMPUTE.md).
Experiment identity lives in the manifest's ``backbone``, ``num_image_tokens``
and ``cache_version`` fields; ``config_hash`` ignores paths (#39).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "hmf-feature-cache"
FORMAT_VERSION = 1
ARRAYS = ("image_tokens", "text_tokens", "text_mask", "text_pooled", "similarity", "labels")
UNLABELED = -1


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def write_cache(
    out_dir: str | Path,
    *,
    ids: Sequence[str],
    image_tokens: np.ndarray,
    text_tokens: np.ndarray,
    text_mask: np.ndarray,
    text_pooled: np.ndarray,
    similarity: np.ndarray,
    labels: np.ndarray,
    meta: dict,
) -> Path:
    """Validate shapes and dtypes, then write one split's cache directory.

    ``meta`` must contain ``backbone``, ``cache_version`` and ``git_sha``; any
    other keys (image size, timing, device) are stored as given. Text arrays are
    trimmed to the longest unmasked sequence in the split.
    """
    missing = {"backbone", "cache_version", "git_sha"} - set(meta)
    if missing:
        raise ValueError(f"meta is missing {sorted(missing)}")
    n = len(ids)
    if len(set(ids)) != n:
        raise ValueError("ids must be unique")
    if image_tokens.ndim != 3 or text_tokens.ndim != 3 or image_tokens.shape[0] != n:
        raise ValueError("image_tokens and text_tokens must be [N, T, d] with N = len(ids)")
    d, d_text = image_tokens.shape[2], text_tokens.shape[2]
    expected = {
        "text_tokens": (n, text_tokens.shape[1], d_text),
        "text_mask": (n, text_tokens.shape[1]),
        "text_pooled": (n, d_text),
        "similarity": (n,),
        "labels": (n,),
    }
    arrays = {"text_tokens": text_tokens, "text_mask": text_mask, "text_pooled": text_pooled,
              "similarity": similarity, "labels": labels}
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} has shape {arrays[name].shape}, expected {shape}")
    # np.asarray instead of astype: arrays already in the stored dtype (e.g. fp16
    # memmaps from hmf/features/extract.py) stream to disk without an in-memory copy.
    mask = np.asarray(text_mask, dtype=bool)
    if n and not mask.any(axis=1).all():
        raise ValueError("every sample needs at least one unmasked text token")
    length = int(mask.sum(axis=1).max()) if n else 0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "image_tokens.npy", np.asarray(image_tokens, dtype=np.float16))
    np.save(out / "text_tokens.npy", np.asarray(text_tokens[:, :length], dtype=np.float16))
    np.save(out / "text_mask.npy", mask[:, :length])
    np.save(out / "text_pooled.npy", np.asarray(text_pooled, dtype=np.float16))
    np.save(out / "similarity.npy", np.asarray(similarity, dtype=np.float32))
    np.save(out / "labels.npy", np.asarray(labels, dtype=np.int64))
    (out / "ids.json").write_text(json.dumps(list(ids)), encoding="utf-8")
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "n": n,
        "dim": int(d),
        "text_dim": int(d_text),
        "num_image_tokens": int(image_tokens.shape[1]),
        "max_text_tokens": length,
        "dtype": "float16",
        "ids_sha256": ids_digest(ids),
        **meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    return out


@dataclass
class FeatureCache:
    """Memory-mapped view of one split's cache directory."""

    path: Path
    manifest: dict
    ids: list[str]
    image_tokens: np.ndarray
    text_tokens: np.ndarray
    text_mask: np.ndarray
    text_pooled: np.ndarray
    similarity: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def image_pooled(self) -> np.ndarray:
        """CLS row of the image tokens (CLIP's image embedding)."""
        return self.image_tokens[:, 0]

    def indices(self, ids: Sequence[str]) -> np.ndarray:
        """Row indices for the given IDs, in the given order; unknown IDs raise."""
        where = {sample_id: i for i, sample_id in enumerate(self.ids)}
        unknown = [i for i in ids if i not in where]
        if unknown:
            raise KeyError(f"{len(unknown)} ids not in cache {self.path}, e.g. {unknown[:3]}")
        return np.fromiter((where[i] for i in ids), dtype=np.int64, count=len(ids))


def load_cache(path: str | Path, *, mmap: bool = True) -> FeatureCache:
    """Open a cache directory, checking its manifest against the arrays."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: not a {FORMAT} v{FORMAT_VERSION} cache")
    ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
    if ids_digest(ids) != manifest["ids_sha256"] or len(ids) != manifest["n"]:
        raise ValueError(f"{path}: ids.json does not match the manifest")
    mode = "r" if mmap else None
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode=mode) for name in ARRAYS}
    for name, array in arrays.items():
        if array.shape[0] != manifest["n"]:
            raise ValueError(f"{path}: {name} has {array.shape[0]} rows, manifest says "
                             f"{manifest['n']}")
    return FeatureCache(path=path, manifest=manifest, ids=ids, **arrays)
