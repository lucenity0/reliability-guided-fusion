"""MVSA-Single loader (3-class sentiment), QMF/MMBT splits.

Splits: the ``{train,dev,test}.jsonl`` files in ``QingyangZhang/QMF`` @
``fe6c4c6`` (``text-image-classification/datasets/MVSA_Single/``), the MMBT
splits QMF and PDF report on (docs/PROTOCOL.md, D-012). Images come from the
Kaggle copy linked in QMF's README (issue #53). Expected layout (never committed):

    annotations/{train,dev,test}.jsonl
    images/data/<id>.jpg          # the jsonl ``img`` field is relative to images/

Each JSONL row has ``id``, ``img``, ``label`` and ``text``. ``label`` is a class
name; ``Example.label`` is its index in ``CLASSES`` (fixed order, #55), an int
in ``[0, 3)``. Multi-class training follows D-018 (#54).
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from hmf.data.common import (
    ROLES,
    Example,
    check_unique_ids,
    read_jsonl,
    resolve_role,
    write_id_list,
)

CLASSES = ("negative", "neutral", "positive")
SPLITS = ("train", "dev", "test")
DEFAULT_ROLES = {"train": "train", "val": "dev", "test": "test"}

# Rows per split in the QMF files (docs/PROTOCOL.md); train class counts from #55.
EXPECTED_COUNTS = {"train": 1555, "dev": 518, "test": 519}
EXPECTED_CLASS_COUNTS = {"train": {"negative": 422, "neutral": 282, "positive": 851}}


def load_mvsa_split(annotations: str | Path, images: str | Path, split: str, *,
                    max_samples: int = 0) -> list[Example]:
    """Load one source split in file order; ``max_samples > 0`` keeps the first N."""
    if split not in SPLITS:
        raise ValueError(f"unknown MVSA-Single split {split!r}; expected one of {SPLITS}")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    path = Path(annotations) / f"{split}.jsonl"
    examples = []
    for line, row in enumerate(read_jsonl(path), start=1):
        where = f"{path.name} row {line}"
        try:
            raw_id, img, text, label = row["id"], row["img"], row["text"], row["label"]
        except KeyError as exc:
            raise ValueError(f"{where}: missing field {exc.args[0]!r}") from exc
        if label not in CLASSES:
            raise ValueError(f"{where}: unknown class {label!r}")
        if not isinstance(text, str):
            raise ValueError(f"{where}: text must be a string")
        examples.append(Example(id=str(raw_id), image=str(Path(images) / img), text=text,
                                label=CLASSES.index(label), split=split))
    check_unique_ids(examples, f"MVSA-Single {split}")
    return examples[:max_samples] if max_samples else examples


def load_mvsa(annotations: str | Path, images: str | Path, role: str, *,
              roles: Mapping[str, str] = DEFAULT_ROLES, max_samples: int = 0,
              allow_test: bool = False) -> list[Example]:
    """Load the split behind a protocol role; ``test`` needs ``allow_test=True``."""
    split = resolve_role(role, roles, allow_test=allow_test)
    return load_mvsa_split(annotations, images, split, max_samples=max_samples)


def class_counts(examples: list[Example]) -> dict[str, int]:
    counts = Counter(e.label for e in examples)
    return {name: counts.get(i, 0) for i, name in enumerate(CLASSES)}


def count_mismatches(split: str, examples: list[Example]) -> list[str]:
    problems = []
    if len(examples) != EXPECTED_COUNTS[split]:
        problems.append(f"{split}: {len(examples)} examples, expected {EXPECTED_COUNTS[split]}")
    expected = EXPECTED_CLASS_COUNTS.get(split)
    if expected is not None and class_counts(examples) != expected:
        problems.append(f"{split}: class counts {class_counts(examples)}, expected {expected}")
    return problems


def missing_images(examples: list[Example]) -> list[str]:
    return [e.id for e in examples if not Path(e.image).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MVSA-Single counts and write split IDs.")
    parser.add_argument("--annotations", required=True, help="dir with QMF's split jsonl files")
    parser.add_argument("--images", help="dir the jsonl img paths are relative to (checks files)")
    parser.add_argument("--write-splits", metavar="DIR", help="e.g. splits/mvsa")
    parser.add_argument("--dedup-report", metavar="PATH",
                        help="e.g. splits/mvsa/dedup_report.json (needs --images and Pillow)")
    args = parser.parse_args(argv)
    if args.dedup_report and not args.images:
        parser.error("--dedup-report needs --images")

    loaded = {s: load_mvsa_split(args.annotations, args.images or ".", s) for s in SPLITS}
    problems = [p for s, ex in loaded.items() for p in count_mismatches(s, ex)]
    for split, examples in loaded.items():
        print(f"{split:5s} {len(examples):5d} examples, {class_counts(examples)}")
        if args.images:
            missing = missing_images(examples)
            print(f"{split:5s} {len(missing)} images missing")
            if missing:
                problems.append(f"{split}: {len(missing)} images missing")
    for problem in problems:
        print(f"MISMATCH {problem}")
    if args.write_splits:
        for role in ROLES:
            n = write_id_list(loaded[DEFAULT_ROLES[role]], Path(args.write_splits) / f"{role}.txt")
            print(f"wrote {args.write_splits}/{role}.txt ({n} ids)")
    if args.dedup_report:
        from hmf.data.integrity import dedup_report, write_report

        by_role = {role: loaded[DEFAULT_ROLES[role]] for role in ROLES}  # protocol roles (#9)
        report = dedup_report("mvsa", by_role, num_classes=len(CLASSES))
        write_report(report, args.dedup_report)
        print(f"wrote {args.dedup_report}: {len(report['pairs'])} cross-split pairs, "
              f"{len(report['train_ids_to_drop'])} train ids to drop, "
              f"overlapping ids: {sorted(report['overlapping_ids']) or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
