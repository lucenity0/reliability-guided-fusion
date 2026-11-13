"""MAMI (SemEval-2022 Task 5) loader, Sub-task A: misogynous memes, binary.

Source: the organisers' release (per-requester access, #53). Cite Fersini et al. 2022,
"SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification", SemEval-2022, as
the release README asks. Expected layout under ``root`` (never committed): the
release's ``training.zip`` and ``test.zip`` unpacked, plus its ``test_labels.txt``:

    root/TRAINING/training.csv   file_name, misogynous, shaming, stereotype,
                                 objectification, violence, Text Transcription
    root/TRAINING/<file_name>    e.g. 1.jpg
    root/test/Test.csv           file_name, Text Transcription (test.zip is
                                 password-protected; the organisers send the password)
    root/test/<file_name>
    root/test_labels.txt         file_name and the five labels, no header

Files are tab-separated UTF-8 with a byte-order mark. They are not quoted CSV: some
transcriptions contain quote characters, so rows are read with ``QUOTE_NONE``. The label
is ``misogynous`` (Sub-task A); the Sub-task B columns are not used.

IDs: the image file stem for train (``"1"`` for ``1.jpg``), and ``"test_"`` plus the stem
for test (``"test_15001"``). The release reuses five file names (15001, 15002, 15004,
15005, 15006) in train and test for *different* memes (image bytes and texts differ), so
bare stems would put one ID in two splits (docs/PROTOCOL.md integrity rule).

There is no official val split. ``train:fit`` and ``train:val`` are a stratified 10%
carve-out of train with seed 1234 (#6), as for FHM Phase 1 (#57).
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path

from hmf.data.common import (
    ROLES,
    Example,
    check_unique_ids,
    resolve_role,
    stratified_carveout,
    write_id_list,
)

SPLITS = ("train", "test")
CARVEOUT = ("train:fit", "train:val")
CARVEOUT_FRACTION, CARVEOUT_SEED = 0.10, 1234
DEFAULT_ROLES = {"train": "train:fit", "val": "train:val", "test": "test"}
LABEL_COLUMNS = ("misogynous", "shaming", "stereotype", "objectification", "violence")
TEXT_COLUMN = "Text Transcription"
TEST_ID_PREFIX = "test_"  # train and test reuse five file names for different memes

# Fersini et al. 2022, SemEval-2022 Task 5, Table 2 ("Dataset characteristics"), p. 536,
# Sub-task A: (total, misogynous) per split.
EXPECTED_COUNTS = {"train": (10000, 5000), "test": (1000, 500)}


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
                if row]


def _with_header(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    header, *rows = _read_tsv(path)
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    out = []
    for line, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"{path.name}:{line}: {len(row)} fields, expected {len(header)}")
        out.append(dict(zip(header, row, strict=True)))
    return out


def _label(value: str, where: str) -> int:
    if value not in ("0", "1"):
        raise ValueError(f"{where}: misogynous must be 0 or 1, got {value!r}")
    return int(value)


def _train_examples(root: Path) -> list[Example]:
    path = root / "TRAINING" / "training.csv"
    rows = _with_header(path, ("file_name", "misogynous", TEXT_COLUMN))
    return [Example(id=Path(r["file_name"]).stem, image=str(root / "TRAINING" / r["file_name"]),
                    text=r[TEXT_COLUMN], label=_label(r["misogynous"], f"{path.name}:{i}"),
                    split="train")
            for i, r in enumerate(rows, start=2)]


def _test_examples(root: Path) -> list[Example]:
    labels_path, text_path = root / "test_labels.txt", root / "test" / "Test.csv"
    if not text_path.exists():
        raise FileNotFoundError(f"{text_path} not found: unpack the release's test.zip "
                                "(password from the organisers) into root/test/")
    texts = {r["file_name"]: r[TEXT_COLUMN]
             for r in _with_header(text_path, ("file_name", TEXT_COLUMN))}
    examples = []
    for line, row in enumerate(_read_tsv(labels_path), start=1):
        where = f"{labels_path.name}:{line}"
        if len(row) != 1 + len(LABEL_COLUMNS):
            raise ValueError(f"{where}: {len(row)} fields, expected {1 + len(LABEL_COLUMNS)}")
        file_name = row[0]
        if file_name not in texts:
            raise ValueError(f"{where}: {file_name!r} has no row in {text_path.name}")
        examples.append(Example(id=TEST_ID_PREFIX + Path(file_name).stem,
                                image=str(root / "test" / file_name),
                                text=texts[file_name], label=_label(row[1], where),
                                split="test"))
    if len(texts) != len(examples):
        raise ValueError(f"{text_path.name} has {len(texts)} rows, {labels_path.name} "
                         f"{len(examples)}")
    return examples


def load_mami_split(root: str | Path, split: str, *, max_samples: int = 0) -> list[Example]:
    """Load one source split in file order; ``max_samples > 0`` keeps the first N."""
    if split not in SPLITS:
        raise ValueError(f"unknown MAMI split {split!r}; expected one of {SPLITS}")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    root = Path(root)
    examples = _train_examples(root) if split == "train" else _test_examples(root)
    check_unique_ids(examples, f"MAMI {split}")
    return examples[:max_samples] if max_samples else examples


def load_mami(root: str | Path, role: str, *, roles: Mapping[str, str] = DEFAULT_ROLES,
              max_samples: int = 0, allow_test: bool = False) -> list[Example]:
    """Load the source split behind a protocol role; ``test`` needs ``allow_test=True``.

    ``max_samples`` truncates after the carve-out, so a smoke run never changes which
    IDs are held out.
    """
    split = resolve_role(role, roles, allow_test=allow_test)
    if split in CARVEOUT:
        fit, held = stratified_carveout(load_mami_split(root, "train"),
                                        fraction=CARVEOUT_FRACTION, seed=CARVEOUT_SEED)
        examples = fit if split == "train:fit" else held
        return examples[:max_samples] if max_samples else examples
    if split not in SPLITS:
        raise ValueError(f"role {role!r} maps to unknown MAMI split {split!r}")
    return load_mami_split(root, split, max_samples=max_samples)


def count_mismatches(split: str, examples: list[Example]) -> list[str]:
    total, misogynous = EXPECTED_COUNTS[split]
    problems = []
    if len(examples) != total:
        problems.append(f"{split}: {len(examples)} examples, expected {total}")
    got = sum(e.label for e in examples)
    if got != misogynous:
        problems.append(f"{split}: {got} misogynous, expected {misogynous}")
    return problems


def missing_images(examples: list[Example]) -> list[str]:
    return [e.id for e in examples if not Path(e.image).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MAMI counts and write split ID lists.")
    parser.add_argument("--root", required=True, help="unpacked MAMI release (not committed)")
    parser.add_argument("--write-splits", metavar="DIR", help="e.g. splits/mami")
    parser.add_argument("--dedup-report", metavar="PATH",
                        help="e.g. splits/mami/dedup_report.json (needs Pillow)")
    args = parser.parse_args(argv)

    loaded = {s: load_mami_split(args.root, s) for s in SPLITS}
    problems = [p for s, ex in loaded.items() for p in count_mismatches(s, ex)]
    for split, examples in loaded.items():
        missing = missing_images(examples)
        print(f"{split:5s} {len(examples):5d} examples, {sum(e.label for e in examples)} "
              f"misogynous, {len(missing)} images missing")
        if missing:
            problems.append(f"{split}: {len(missing)} images missing")
    for problem in problems:
        print(f"MISMATCH {problem}")
    if args.write_splits:
        for role in ROLES:
            examples = load_mami(args.root, role, allow_test=True)
            n = write_id_list(examples, Path(args.write_splits) / f"{role}.txt")
            print(f"wrote {args.write_splits}/{role}.txt ({n} ids, "
                  f"{sum(e.label for e in examples)} misogynous)")
    if args.dedup_report:
        from hmf.data.integrity import dedup_report, write_report

        # Over the protocol roles, so the val carve-out is checked against train (#9).
        by_role = {role: load_mami(args.root, role, allow_test=True) for role in ROLES}
        report = dedup_report("mami", by_role)
        write_report(report, args.dedup_report)
        print(f"wrote {args.dedup_report}: {len(report['pairs'])} cross-split pairs, "
              f"{len(report['train_ids_to_drop'])} train ids to drop, "
              f"overlapping ids: {sorted(report['overlapping_ids']) or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
