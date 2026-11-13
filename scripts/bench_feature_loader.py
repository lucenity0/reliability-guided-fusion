#!/usr/bin/env python3
"""Time one epoch of CachedFeatureDataset loading (issue #12 acceptance: < 60 s for FHM train).

    python scripts/bench_feature_loader.py \\
        --cache ~/features/fhm/v1/train --ids splits/fhm/train.txt
    python scripts/bench_feature_loader.py --synthetic 7650   # no real cache needed
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from hmf.data.common import read_id_list
from hmf.features.cache import write_cache
from hmf.features.dataset import CachedFeatureDataset, make_loader


def synthetic_cache(n: int, root: Path) -> Path:
    rng = np.random.default_rng(0)
    d, length = 768, 40
    mask = np.zeros((n, length), dtype=bool)
    mask[:, :20] = True
    return write_cache(
        root / "synthetic",
        ids=[f"{i:06d}" for i in range(n)],
        image_tokens=rng.standard_normal((n, 145, d), dtype=np.float32).astype(np.float16),
        text_tokens=rng.standard_normal((n, length, d), dtype=np.float32).astype(np.float16),
        text_mask=mask,
        text_pooled=rng.standard_normal((n, d), dtype=np.float32).astype(np.float16),
        similarity=rng.uniform(-1, 1, n).astype(np.float32),
        labels=rng.integers(0, 2, n),
        meta={"backbone": "synthetic", "cache_version": "bench", "git_sha": "n/a"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache")
    parser.add_argument("--ids")
    parser.add_argument("--synthetic", type=int)
    parser.add_argument("--workdir")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        cache = synthetic_cache(args.synthetic, Path(tmp)) if args.synthetic else Path(args.cache)
        ids = read_id_list(args.ids) if args.ids else None
        ds = CachedFeatureDataset(cache, ids=ids)
        for epoch in (1, 2):  # epoch 2 shows the warm page-cache case
            start = time.perf_counter()
            n = sum(len(b["label"]) for b in make_loader(ds, batch_size=args.batch_size,
                                                         shuffle=True, seed=epoch,
                                                         num_workers=args.workers))
            print(f"epoch {epoch}: {n} samples in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
