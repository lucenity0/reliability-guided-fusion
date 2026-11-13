"""CrisisMMD v2.0 loader, informative task, agreed-label benchmark split.

Source: Ofli, Alam and Imran (2020), "Analysis of Social Media Data using
Multimodal Deep Learning for Disaster Response", ISCRAM 2020 (arXiv:2004.11838),
whose split files are distributed as ``crisismmd_datasplit_agreed_label.zip``
at crisisnlp.qcri.org. Expected layout (never committed):

    annotations/task_informative_text_img_agreed_lab_{train,dev,test}.tsv
    images/data_image/<event>/<date>/<tweet_id>_<k>.jpg

TSV columns: event_name, tweet_id, image_id, tweet_text, image, label,
label_text, label_image, label_text_image. In the agreed-label set the text
and image labels coincide, and this loader checks that they do.

**Several images per tweet.** Train contains tweets with more than one image
(9,601 image-text pairs from 8,293 tweets). Following Ofli et al. (Table 1,
"Image" columns), every (tweet, image) pair is one sample, keyed by its unique
``image_id`` (``<tweet_id>_<k>``); pairs from one tweet share the text. Tweets
must not cross splits, which ``overlapping_tweets`` checks.

Label: ``informative`` = 1, ``not_informative`` = 0.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

from hmf.data.common import ROLES, Example, check_unique_ids, resolve_role, write_id_list

SPLITS = ("train", "dev", "test")
DEFAULT_ROLES = {"train": "train", "val": "dev", "test": "test"}
FILE_PATTERN = "task_informative_text_img_agreed_lab_{split}.tsv"
LABELS = {"informative": 1, "not_informative": 0}
COLUMNS = ("tweet_id", "image_id", "tweet_text", "image", "label", "label_text", "label_image")

# Ofli et al. 2020, Table 1, p. 5 (arXiv:2004.11838v1): informative task,
# "Image" columns = image-text pairs. (total, informative) per split.
EXPECTED_COUNTS = {"train": (9601, 6345), "dev": (1573, 1056), "test": (1534, 1030)}


def tweet_of(example_id: str) -> str:
    """Tweet ID of a ``<tweet_id>_<k>`` sample ID."""
    tweet, sep, index = example_id.rpartition("_")
    if not sep or not tweet.isdigit() or not index.isdigit():
        raise ValueError(f"not a CrisisMMD image id: {example_id!r}")
    return tweet


def load_crisismmd_split(annotations: str | Path, images: str | Path, split: str, *,
                         max_samples: int = 0) -> list[Example]:
    if split not in SPLITS:
        raise ValueError(f"unknown CrisisMMD split {split!r}; expected one of {SPLITS}")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    path = Path(annotations) / FILE_PATTERN.format(split=split)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        examples = []
        for line, row in enumerate(reader, start=2):
            where = f"{path.name}:{line}"
            labels = {row["label"], row["label_text"], row["label_image"]}
            if len(labels) != 1:
                raise ValueError(f"{where}: text/image labels disagree in the agreed-label set")
            if row["label"] not in LABELS:
                raise ValueError(f"{where}: unknown label {row['label']!r}")
            if tweet_of(row["image_id"]) != row["tweet_id"]:
                raise ValueError(f"{where}: image_id does not belong to tweet_id")
            examples.append(Example(
                id=row["image_id"],
                image=str(Path(images) / row["image"]),
                text=row["tweet_text"],
                label=LABELS[row["label"]],
                split=split,
            ))
    check_unique_ids(examples, f"CrisisMMD {split}")
    return examples[:max_samples] if max_samples else examples


def load_crisismmd(annotations: str | Path, images: str | Path, role: str, *,
                   roles: Mapping[str, str] = DEFAULT_ROLES, max_samples: int = 0,
                   allow_test: bool = False) -> list[Example]:
    """Load the split behind a protocol role; ``test`` needs ``allow_test=True``."""
    split = resolve_role(role, roles, allow_test=allow_test)
    return load_crisismmd_split(annotations, images, split, max_samples=max_samples)


def overlapping_tweets(splits: Mapping[str, Sequence[Example]]) -> dict[str, list[str]]:
    """Tweets with samples in more than one split (must be empty)."""
    tweets = {name: {tweet_of(e.id) for e in ex} for name, ex in splits.items()}
    out = {}
    for a, b in combinations(sorted(tweets), 2):
        shared = sorted(tweets[a] & tweets[b])
        if shared:
            out[f"{a}|{b}"] = shared
    return out


def count_mismatches(split: str, examples: Sequence[Example]) -> list[str]:
    total, positive = EXPECTED_COUNTS[split]
    problems = []
    if len(examples) != total:
        problems.append(f"{split}: {len(examples)} examples, expected {total}")
    got = sum(e.label for e in examples)
    if got != positive:
        problems.append(f"{split}: {got} informative, expected {positive}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check CrisisMMD counts and write split IDs.")
    parser.add_argument("--annotations", required=True, help="dir with the agreed-label TSVs")
    parser.add_argument("--images", required=True, help="dir containing data_image/")
    parser.add_argument("--write-splits", metavar="DIR", help="e.g. splits/crisismmd")
    parser.add_argument("--dedup-report", metavar="PATH",
                        help="e.g. splits/crisismmd/dedup_report.json (needs Pillow)")
    args = parser.parse_args(argv)

    loaded = {s: load_crisismmd_split(args.annotations, args.images, s) for s in SPLITS}
    problems = [p for s, ex in loaded.items() for p in count_mismatches(s, ex)]
    problems += [f"tweets shared by {k}: {len(v)}" for k, v in overlapping_tweets(loaded).items()]
    for split, examples in loaded.items():
        print(f"{split:5s} {len(examples):5d} pairs, {len({tweet_of(e.id) for e in examples})} "
              f"tweets, {sum(e.label for e in examples)} informative")
    for problem in problems:
        print(f"MISMATCH {problem}")
    if args.write_splits:
        for role in ROLES:
            n = write_id_list(loaded[DEFAULT_ROLES[role]], Path(args.write_splits) / f"{role}.txt")
            print(f"wrote {args.write_splits}/{role}.txt ({n} ids)")
    if args.dedup_report:
        from hmf.data.integrity import dedup_report, write_report

        by_role = {role: loaded[DEFAULT_ROLES[role]] for role in ROLES}  # protocol roles (#9)
        report = dedup_report("crisismmd", by_role)
        write_report(report, args.dedup_report)
        print(f"wrote {args.dedup_report}: {len(report['pairs'])} cross-split pairs, "
              f"{len(report['train_ids_to_drop'])} train ids to drop, "
              f"overlapping ids: {sorted(report['overlapping_ids']) or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
