"""HarMeme V1 loader (Harm-C and Harm-P), binarised harmfulness.

Source: the V1 annotations in ``LCS2-IIITD/MOMENTA`` (``HarMeme_V1/Annotations``)
and the authors' image zip (issue #53). V1 only: V0's Harm-P has duplicates.
Expected layout (never committed):

    annotations/Harm-C/{train,val,test}.jsonl
    annotations/Harm-P/{train,val,test}_v1.jsonl
    HarMeme_Images/harmeme_images_covid_19/<image>   # Harm-C, e.g. covid_memes_18.png
    HarMeme_Images/harmeme_images_us_pol/<image>     # Harm-P, e.g. memes_100.png

``images`` is the ``HarMeme_Images`` directory from the zip (7,096 images); the
subset picks its sub-folder.

Each JSONL row has ``id``, ``image``, ``labels`` and ``text``. ``labels[0]`` is
the harmfulness level; a second entry, where present, is the target and is not
used. Binarisation (docs/PROTOCOL.md, following PromptHate and Pro-Cap):
``not harmful`` = 0, ``somewhat harmful`` and ``very harmful`` = 1.

One Harm-P ``train_v1.jsonl`` row has an empty label. Rows without a label are
dropped and their IDs reported (``read_split``, a warning, and the CLI); their
label is never guessed.
"""

from __future__ import annotations

import argparse
import warnings
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

SUBSETS = ("harm_c", "harm_p")
SPLITS = ("train", "val", "test")
DEFAULT_ROLES = {"train": "train", "val": "val", "test": "test"}
SUBSET_FILES = {"harm_c": ("Harm-C", "{split}.jsonl"), "harm_p": ("Harm-P", "{split}_v1.jsonl")}
IMAGE_DIRS = {"harm_c": "harmeme_images_covid_19", "harm_p": "harmeme_images_us_pol"}
LABELS = {"not harmful": 0, "somewhat harmful": 1, "very harmful": 1}

# (total, harmful) per split, counted over labeled rows (docs/PROTOCOL.md).
# Harm-C train/test equal PromptHate (Cao et al. 2022) Table 1; Harm-P matches
# the HarMeme V1 README table except for the unlabeled train row below.
EXPECTED_COUNTS = {
    "harm_c": {"train": (3013, 1064), "val": (177, 61), "test": (354, 124)},
    "harm_p": {"train": (2938, 1487), "val": (177, 89), "test": (355, 171)},
}
# Rows with an empty label, dropped on load.
EXPECTED_UNLABELED = {"harm_c": {}, "harm_p": {"train": 1}}


def split_file(annotations: str | Path, subset: str, split: str) -> Path:
    if subset not in SUBSETS:
        raise ValueError(f"unknown HarMeme subset {subset!r}; expected one of {SUBSETS}")
    if split not in SPLITS:
        raise ValueError(f"unknown HarMeme split {split!r}; expected one of {SPLITS}")
    folder, pattern = SUBSET_FILES[subset]
    return Path(annotations) / folder / pattern.format(split=split)


def binarise(labels) -> int | None:
    """0/1 from a ``labels`` list; ``None`` when the row has no label."""
    if not isinstance(labels, list):
        raise ValueError(f"labels must be a list, got {labels!r}")
    if not labels or labels[0] in ("", None):
        return None
    if labels[0] not in LABELS:
        raise ValueError(f"unknown harmfulness label {labels[0]!r}")
    return LABELS[labels[0]]


def read_split(annotations: str | Path, images: str | Path, subset: str,
               split: str) -> tuple[list[Example], list[str]]:
    """All labeled rows of one source split, plus the IDs of rows dropped as unlabeled."""
    path = split_file(annotations, subset, split)
    image_dir = Path(images) / IMAGE_DIRS[subset]
    examples, dropped = [], []
    for line, row in enumerate(read_jsonl(path), start=1):
        where = f"{path.name} row {line}"
        try:
            raw_id, image, text, labels = row["id"], row["image"], row["text"], row["labels"]
        except KeyError as exc:
            raise ValueError(f"{where}: missing field {exc.args[0]!r}") from exc
        if not isinstance(text, str):
            raise ValueError(f"{where}: text must be a string")
        try:
            label = binarise(labels)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if label is None:
            dropped.append(str(raw_id))
            continue
        examples.append(Example(id=str(raw_id), image=str(image_dir / image), text=text,
                                label=label, split=split))
    check_unique_ids(examples, f"HarMeme {subset} {split}")
    return examples, dropped


def load_harmeme_split(annotations: str | Path, images: str | Path, subset: str, split: str, *,
                       max_samples: int = 0) -> list[Example]:
    """Load one source split in file order; unlabeled rows are dropped with a warning."""
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    examples, dropped = read_split(annotations, images, subset, split)
    if dropped:
        warnings.warn(f"HarMeme {subset} {split}: dropped {len(dropped)} unlabeled row(s): "
                      f"{dropped}", stacklevel=2)
    return examples[:max_samples] if max_samples else examples


def load_harmeme(annotations: str | Path, images: str | Path, subset: str, role: str, *,
                 roles: Mapping[str, str] = DEFAULT_ROLES, max_samples: int = 0,
                 allow_test: bool = False) -> list[Example]:
    """Load the split behind a protocol role; ``test`` needs ``allow_test=True``."""
    split = resolve_role(role, roles, allow_test=allow_test)
    return load_harmeme_split(annotations, images, subset, split, max_samples=max_samples)


def count_mismatches(subset: str, split: str, examples: list[Example],
                     dropped: list[str]) -> list[str]:
    """Compare a full split with EXPECTED_COUNTS and EXPECTED_UNLABELED."""
    total, harmful = EXPECTED_COUNTS[subset][split]
    unlabeled = EXPECTED_UNLABELED[subset].get(split, 0)
    problems = []
    if len(examples) + len(dropped) != total:
        problems.append(f"{subset} {split}: {len(examples) + len(dropped)} rows, expected {total}")
    got = sum(e.label for e in examples)
    if got != harmful:
        problems.append(f"{subset} {split}: {got} harmful, expected {harmful}")
    if len(dropped) != unlabeled:
        problems.append(f"{subset} {split}: {len(dropped)} unlabeled rows, expected {unlabeled}")
    return problems


def missing_images(examples: list[Example]) -> list[str]:
    return [e.id for e in examples if not Path(e.image).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check HarMeme counts and write split ID lists.")
    parser.add_argument("--annotations", required=True,
                        help="HarMeme_V1/Annotations dir (holds Harm-C/ and Harm-P/)")
    parser.add_argument("--images", required=True, help="HarMeme_Images dir from the image zip")
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--write-splits", metavar="DIR", help="e.g. splits/harm_c")
    parser.add_argument("--dedup-report", metavar="PATH",
                        help="e.g. splits/harm_c/dedup_report.json (needs Pillow)")
    args = parser.parse_args(argv)

    loaded, problems = {}, []
    for split in SPLITS:
        examples, dropped = read_split(args.annotations, args.images, args.subset, split)
        loaded[split] = examples
        missing = missing_images(examples)
        print(f"{split:5s} {len(examples):5d} examples, {sum(e.label for e in examples)} harmful, "
              f"{len(missing)} images missing")
        if dropped:
            print(f"dropped unlabeled {split} rows: {', '.join(dropped)}")
        problems += count_mismatches(args.subset, split, examples, dropped)
        if missing:
            problems.append(f"{args.subset} {split}: {len(missing)} images missing")
    for problem in problems:
        print(f"MISMATCH {problem}")
    if args.write_splits:
        for role in ROLES:
            n = write_id_list(loaded[DEFAULT_ROLES[role]], Path(args.write_splits) / f"{role}.txt")
            print(f"wrote {args.write_splits}/{role}.txt ({n} ids)")
    if args.dedup_report:
        from hmf.data.integrity import dedup_report, write_report

        by_role = {role: loaded[DEFAULT_ROLES[role]] for role in ROLES}  # protocol roles (#9)
        report = dedup_report(args.subset, by_role)
        write_report(report, args.dedup_report)
        print(f"wrote {args.dedup_report}: {len(report['pairs'])} cross-split pairs, "
              f"{len(report['train_ids_to_drop'])} train ids to drop, "
              f"overlapping ids: {sorted(report['overlapping_ids']) or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
