"""Split-integrity checks and cross-split duplicate reports (docs/PROTOCOL.md).

* Disjoint IDs: no ID may appear in more than one split.
* Duplicates across splits, found with a 64-bit difference hash (dHash) of
  each image and compared by Hamming distance:
    - ``image_text``: near-identical image **and** the same normalised text,
      i.e. the same meme. Removed from ``train`` only; val/test never change.
    - ``image_only``: near-identical image with different text. FHM builds
      "benign confounders" this way on purpose, so these are reported but not
      removed until the owner decides (raised on the PROTOCOL freeze gate, #10).
* Label distributions per split, for comparison with the source paper.

Reports are computed over the protocol roles (train, val, test), so the FHM and MAMI
val carve-outs are checked against their own train role (#9). Each pair records
whether its two labels agree. ``python -m hmf.data.integrity --apply splits/<ds>``
removes a report's ``train_ids_to_drop`` from ``train.txt`` (D-014); ``val.txt`` and
``test.txt`` are never touched.

Reports contain IDs and counts only, never text or pixels.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

from hmf.data.common import Example

HASH_BITS = 64
DEFAULT_MAX_DISTANCE = 4  # dHash bits; catches re-encoded copies, not distinct photos

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: str | Path, size: int = 8) -> int:
    """Difference hash: grayscale, resize to (size+1)×size, compare neighbours."""
    from PIL import Image  # optional dependency: the train/dev extras

    with Image.open(path) as image:
        gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def overlapping_ids(splits: Mapping[str, Sequence[Example]]) -> dict[str, list[str]]:
    """IDs shared by each pair of splits; an empty dict means all are disjoint."""
    ids = {name: {e.id for e in examples} for name, examples in splits.items()}
    out = {}
    for a, b in combinations(sorted(ids), 2):
        shared = sorted(ids[a] & ids[b])
        if shared:
            out[f"{a}|{b}"] = shared
    return out


def _as_bytes(hashes: Sequence[int]) -> np.ndarray:
    raw = b"".join(h.to_bytes(HASH_BITS // 8, "big") for h in hashes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hashes), HASH_BITS // 8)


def _near_pairs(ha: Sequence[int], hb: Sequence[int], max_distance: int,
                chunk: int = 256) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with Hamming(ha[i], hb[j]) <= max_distance."""
    if not ha or not hb:
        return []
    a, b = _as_bytes(ha), _as_bytes(hb)
    pairs = []
    for start in range(0, len(a), chunk):
        block = a[start:start + chunk]
        dist = _POPCOUNT[block[:, None, :] ^ b[None, :, :]].sum(axis=2, dtype=np.int32)
        for i, j in zip(*np.nonzero(dist <= max_distance), strict=True):
            pairs.append((start + int(i), int(j), int(dist[i, j])))
    return pairs


def find_cross_split_duplicates(
    splits: Mapping[str, Sequence[Example]],
    *,
    hash_fn: Callable[[str], int] = dhash,
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> list[dict]:
    """Near-duplicate image pairs between different splits, with their kind."""
    if not 0 <= max_distance <= HASH_BITS:
        raise ValueError(f"max_distance must be in [0, {HASH_BITS}]")
    hashes = {name: [hash_fn(e.image) for e in examples] for name, examples in splits.items()}
    found = []
    for a, b in combinations(sorted(splits), 2):
        for i, j, dist in _near_pairs(hashes[a], hashes[b], max_distance):
            ea, eb = splits[a][i], splits[b][j]
            same_text = normalise_text(ea.text) == normalise_text(eb.text)
            labeled = ea.label is not None and eb.label is not None
            found.append({
                "kind": "image_text" if same_text else "image_only",
                "splits": [a, b],
                "ids": [ea.id, eb.id],
                "distance": dist,
                "same_label": ea.label == eb.label if labeled else None,
            })
    return found


def label_distribution(examples: Sequence[Example], num_classes: int = 2) -> dict:
    """Binary: positive/negative counts. Multi-class (MVSA-Single, Food-101; D-018):
    ``class_counts`` indexed by class, so class indices are never summed as if 0/1."""
    labels = [e.label for e in examples]
    if any(label is None for label in labels):
        return {"n": len(labels), "labeled": False}
    if any(not 0 <= label < num_classes for label in labels):
        raise ValueError(f"labels outside [0, {num_classes}); pass the dataset's num_classes")
    if num_classes == 2:
        positives = int(sum(labels))
        return {"n": len(labels), "labeled": True, "positive": positives,
                "negative": len(labels) - positives}
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    return {"n": len(labels), "labeled": True, "class_counts": counts.tolist()}


def dedup_report(
    dataset: str,
    splits: Mapping[str, Sequence[Example]],
    *,
    train_split: str = "train",
    hash_fn: Callable[[str], int] = dhash,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    num_classes: int = 2,
) -> dict:
    """Build ``splits/<dataset>/dedup_report.json`` content (IDs and counts only).

    ``train_ids_to_drop`` lists train IDs in an ``image_text`` pair with any
    other split. Only train is ever altered; ``image_only`` pairs are reported.
    ``num_classes`` > 2 reports per-class label counts.
    """
    if train_split not in splits:
        raise ValueError(f"{train_split!r} is not among the splits")
    pairs = find_cross_split_duplicates(splits, hash_fn=hash_fn, max_distance=max_distance)
    drop = set()
    counts: dict[str, dict[str, int]] = {}
    agreement = {kind: {"same": 0, "different": 0} for kind in ("image_text", "image_only")}
    for pair in pairs:
        key = "|".join(pair["splits"])
        counts.setdefault(key, {"image_text": 0, "image_only": 0})[pair["kind"]] += 1
        if pair["same_label"] is not None:
            agreement[pair["kind"]]["same" if pair["same_label"] else "different"] += 1
        if pair["kind"] == "image_text" and train_split in pair["splits"]:
            other = pair["splits"].index(train_split)
            if pair["splits"][1 - other] != train_split:
                drop.add(pair["ids"][other])
    return {
        "dataset": dataset,
        "hash": f"dhash{HASH_BITS}",
        "max_distance": max_distance,
        "policy": "drop image_text duplicates from the train role only; image_only reported "
                  "(D-014)",
        "label_agreement": agreement,
        "split_sizes": {name: len(examples) for name, examples in splits.items()},
        "label_distribution": {name: label_distribution(ex, num_classes)
                               for name, ex in splits.items()},
        "overlapping_ids": overlapping_ids(splits),
        "pair_counts": counts,
        "pairs": pairs,
        "train_ids_to_drop": sorted(drop),
    }


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def drop_from_train(train: Sequence[Example], report: Mapping) -> list[Example]:
    """Apply a report's train-only removals; other splits are never touched."""
    drop = set(report["train_ids_to_drop"])
    return [e for e in train if e.id not in drop]


def apply_report(split_dir: str | Path) -> tuple[int, int]:
    """Rewrite ``<split_dir>/train.txt`` without the report's ``train_ids_to_drop`` (D-014).

    Idempotent. Refuses a report whose train size matches neither the current list
    (first application) nor the list after its drops (already applied). Returns the
    train sizes (before, after).
    """
    from hmf.data.common import read_id_list

    split_dir = Path(split_dir)
    report = json.loads((split_dir / "dedup_report.json").read_text(encoding="utf-8"))
    ids = read_id_list(split_dir / "train.txt")
    drop = set(report["train_ids_to_drop"])
    full = report["split_sizes"]["train"]
    if len(ids) not in (full, full - len(drop)):
        raise ValueError(f"{split_dir}: train.txt has {len(ids)} ids but the report's train "
                         f"role has {full} ({len(drop)} to drop); regenerate the report")
    kept = [i for i in ids if i not in drop]
    (split_dir / "train.txt").write_text("".join(f"{i}\n" for i in kept), encoding="utf-8")
    return len(ids), len(kept)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Apply dedup reports to train ID lists.")
    parser.add_argument("--apply", metavar="DIR", nargs="+", required=True,
                        help="split dirs with dedup_report.json and train.txt, e.g. splits/fhm")
    args = parser.parse_args(argv)
    for split_dir in args.apply:
        before, after = apply_report(split_dir)
        print(f"{split_dir}/train.txt: {before} -> {after} ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
