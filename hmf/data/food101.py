"""Food-101 (UPMC) loader (101 classes), QMF/MMBT splits.

Splits: the ``{train,dev,test}.jsonl`` files in ``QingyangZhang/QMF`` @
``fe6c4c6`` (``text-image-classification/datasets/food101/``), the MMBT splits
QMF and PDF report on (docs/PROTOCOL.md, D-012). ``test_hard_gt.jsonl`` from the
same folder is not used (#56). Images come from the Kaggle UPMC Food-101 copy
linked in QMF's README (issue #53). Expected layout (never committed):

    annotations/{train,dev,test}.jsonl
    images/images/{train,test}/<class>/<file>.jpg   # jsonl ``img`` is relative to images/

Each JSONL row has ``id``, ``img``, ``label`` and ``text``. ``label`` is a class
name; ``Example.label`` is its index in ``CLASSES`` (the 101 names sorted
alphabetically, #56), an int in ``[0, 101)``. Multi-class training follows
D-018 (#54).
"""

from __future__ import annotations

import argparse
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

CLASSES = (
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad",
    "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad",
    "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheese_plate", "cheesecake",
    "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse",
    "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
    "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup",
    "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi",
    "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
    "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles",
)
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
SPLITS = ("train", "dev", "test")
DEFAULT_ROLES = {"train": "train", "val": "dev", "test": "test"}

# Rows per split in the QMF files (docs/PROTOCOL.md, #56).
EXPECTED_COUNTS = {"train": 62971, "dev": 5000, "test": 22715}


def load_food101_split(annotations: str | Path, images: str | Path, split: str, *,
                       max_samples: int = 0) -> list[Example]:
    """Load one source split in file order; ``max_samples > 0`` keeps the first N."""
    if split not in SPLITS:
        raise ValueError(f"unknown Food-101 split {split!r}; expected one of {SPLITS}")
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
        if label not in CLASS_INDEX:
            raise ValueError(f"{where}: unknown class {label!r}")
        if not isinstance(text, str):
            raise ValueError(f"{where}: text must be a string")
        examples.append(Example(id=str(raw_id), image=str(Path(images) / img), text=text,
                                label=CLASS_INDEX[label], split=split))
    check_unique_ids(examples, f"Food-101 {split}")
    return examples[:max_samples] if max_samples else examples


def load_food101(annotations: str | Path, images: str | Path, role: str, *,
                 roles: Mapping[str, str] = DEFAULT_ROLES, max_samples: int = 0,
                 allow_test: bool = False) -> list[Example]:
    """Load the split behind a protocol role; ``test`` needs ``allow_test=True``."""
    split = resolve_role(role, roles, allow_test=allow_test)
    return load_food101_split(annotations, images, split, max_samples=max_samples)


def count_mismatches(split: str, examples: list[Example]) -> list[str]:
    if len(examples) != EXPECTED_COUNTS[split]:
        return [f"{split}: {len(examples)} examples, expected {EXPECTED_COUNTS[split]}"]
    return []


def missing_images(examples: list[Example]) -> list[str]:
    return [e.id for e in examples if not Path(e.image).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Food-101 counts and write split IDs.")
    parser.add_argument("--annotations", required=True, help="dir with QMF's split jsonl files")
    parser.add_argument("--images", help="dir the jsonl img paths are relative to (checks files)")
    parser.add_argument("--write-splits", metavar="DIR", help="e.g. splits/food101")
    parser.add_argument("--dedup-report", metavar="PATH",
                        help="e.g. splits/food101/dedup_report.json (needs --images and Pillow)")
    args = parser.parse_args(argv)
    if args.dedup_report and not args.images:
        parser.error("--dedup-report needs --images")

    loaded = {s: load_food101_split(args.annotations, args.images or ".", s) for s in SPLITS}
    problems = [p for s, ex in loaded.items() for p in count_mismatches(s, ex)]
    for split, examples in loaded.items():
        print(f"{split:5s} {len(examples):6d} examples, {len({e.label for e in examples})} classes")
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
        report = dedup_report("food101", by_role, num_classes=len(CLASSES))
        write_report(report, args.dedup_report)
        print(f"wrote {args.dedup_report}: {len(report['pairs'])} cross-split pairs, "
              f"{len(report['train_ids_to_drop'])} train ids to drop, "
              f"overlapping ids: {sorted(report['overlapping_ids']) or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
