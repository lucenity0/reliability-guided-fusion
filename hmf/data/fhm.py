"""Facebook Hateful Memes (FHM) loader for the Phase 1 and Phase 2 releases.

Expected layout under ``root`` (never committed):

    root/img/<id>.png
    Phase 1 (D-012, the release we use): root/{train,dev,test}.jsonl
    Phase 2: root/{train,dev_seen,dev_unseen,test_seen,test_unseen}.jsonl

Phase 1's ``dev.jsonl`` is dev_seen and its ``test.jsonl`` is test_seen (same
IDs as Phase 2), but Phase 1 carries the pre-relabelling labels. The release is
detected from the files present (``detect_release``).

Each JSONL row has ``id``, ``img``, ``text`` and, where released, ``label``.
The canonical ID is the image file stem (e.g. ``"01235"``): the JSON ``id`` is
an integer and drops leading zeros, so it is only checked for consistency.

Protocol roles map to source splits through ``roles``. ``train:fit`` and
``train:val`` are the two parts of a stratified 10% carve-out of train (seed
1234). The Phase 1 default follows D-012: train = train:fit, val = train:val,
test = dev_seen.
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
    stratified_carveout,
    write_id_list,
)

SPLITS = ("train", "dev_seen", "dev_unseen", "test_seen", "test_unseen")
CARVEOUT = ("train:fit", "train:val")
CARVEOUT_FRACTION, CARVEOUT_SEED = 0.10, 1234

# Source file names per split; the first one present is used.
SPLIT_FILES = {
    "train": ("train.jsonl",),
    "dev_seen": ("dev_seen.jsonl", "dev.jsonl"),
    "dev_unseen": ("dev_unseen.jsonl",),
    "test_seen": ("test_seen.jsonl", "test.jsonl"),
    "test_unseen": ("test_unseen.jsonl",),
}

PHASE1_ROLES = {"train": "train:fit", "val": "train:val", "test": "dev_seen"}
PHASE2_ROLES = {"train": "train", "val": "dev_unseen", "test": "test_unseen"}
DEFAULT_ROLES = PHASE1_ROLES  # D-012

# (total, hateful) per split; None = unlabeled in that release.
EXPECTED_COUNTS = {
    # Phase 1 release as distributed (emily49/hateful-memes); these equal the FHM
    # counts in PromptHate (Cao et al. 2022) Table 1 and Pro-Cap Table 3.
    "phase1": {"train": (8500, 3050), "dev_seen": (500, 250), "test_seen": (1000, None)},
    # Kiela et al. 2021, "The Hateful Memes Challenge: Competition Report",
    # PMLR 133, Table 1, p. 349.
    "phase2": {
        "train": (8500, 3019),
        "dev_seen": (500, 247),
        "dev_unseen": (540, 200),
        "test_seen": (1000, 490),
        "test_unseen": (2000, 750),
    },
}


def split_file(root: str | Path, split: str) -> Path | None:
    for name in SPLIT_FILES[split]:
        if (Path(root) / name).exists():
            return Path(root) / name
    return None


def detect_release(root: str | Path) -> str:
    """``phase1`` if ``dev.jsonl`` is present without ``dev_seen.jsonl``, else ``phase2``."""
    root = Path(root)
    if (root / "dev.jsonl").exists() and not (root / "dev_seen.jsonl").exists():
        return "phase1"
    return "phase2"


def _parse_row(row: dict, split: str, root: Path, line: int) -> Example:
    where = f"{split}.jsonl row {line}"
    try:
        img, text, raw_id = row["img"], row["text"], row["id"]
    except KeyError as exc:
        raise ValueError(f"{where}: missing field {exc.args[0]!r}") from exc
    stem = Path(img).stem
    if not stem.isdigit() or int(stem) != int(raw_id):
        raise ValueError(f"{where}: id {raw_id!r} does not match image {img!r}")
    label = row.get("label")
    if label is not None and label not in (0, 1):
        raise ValueError(f"{where}: label must be 0 or 1, got {label!r}")
    if not isinstance(text, str):
        raise ValueError(f"{where}: text must be a string")
    return Example(id=stem, image=str(root / img), text=text, label=label, split=split)


def load_fhm_split(root: str | Path, split: str, *, max_samples: int = 0) -> list[Example]:
    """Load one source split in file order; ``max_samples > 0`` keeps the first N."""
    if split not in SPLITS:
        raise ValueError(f"unknown FHM split {split!r}; expected one of {SPLITS}")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    root = Path(root)
    path = split_file(root, split)
    if path is None:
        raise FileNotFoundError(f"no file for FHM {split} under {root} ({SPLIT_FILES[split]})")
    rows = read_jsonl(path)
    examples = [_parse_row(row, split, root, i) for i, row in enumerate(rows, start=1)]
    check_unique_ids(examples, f"FHM {split}")
    labels = {e.label is None for e in examples}
    if len(labels) > 1:
        raise ValueError(f"FHM {split}: some rows are labeled and some are not")
    return examples[:max_samples] if max_samples else examples


def load_fhm(
    root: str | Path,
    role: str,
    *,
    roles: Mapping[str, str] = DEFAULT_ROLES,
    max_samples: int = 0,
    allow_test: bool = False,
) -> list[Example]:
    """Load the source split behind a protocol role (train/val/test).

    ``test`` is refused unless ``allow_test=True`` (reporting and integrity
    checks only). Train and val must be labeled. ``max_samples`` truncates
    after the carve-out, so a smoke run never changes which IDs are held out.
    """
    split = resolve_role(role, roles, allow_test=allow_test)
    if split in CARVEOUT:
        fit, held = stratified_carveout(load_fhm_split(root, "train"),
                                        fraction=CARVEOUT_FRACTION, seed=CARVEOUT_SEED)
        examples = fit if split == "train:fit" else held
        return examples[:max_samples] if max_samples else examples
    if split not in SPLITS:
        raise ValueError(f"role {role!r} maps to unknown FHM split {split!r}")
    examples = load_fhm_split(root, split, max_samples=max_samples)
    if role != "test" and examples and examples[0].label is None:
        raise ValueError(f"FHM {split} is unlabeled and cannot serve as {role}")
    return examples


def count_mismatches(split: str, examples: list[Example], release: str = "phase1") -> list[str]:
    """Compare a full split against EXPECTED_COUNTS; empty list means it matches.

    Hateful counts are only compared for labeled splits.
    """
    if split not in EXPECTED_COUNTS[release]:
        return [f"{split}: not part of the {release} release"]
    total, hateful = EXPECTED_COUNTS[release][split]
    problems = []
    if len(examples) != total:
        problems.append(f"{split}: {len(examples)} examples, expected {total}")
    if examples and examples[0].label is not None and hateful is not None:
        got = sum(e.label for e in examples)
        if got != hateful:
            problems.append(f"{split}: {got} hateful, expected {hateful}")
    return problems


def write_splits(root: str | Path, out_dir: str | Path,
                 roles: Mapping[str, str] = DEFAULT_ROLES) -> dict[str, int]:
    """Write ``out_dir/{train,val,test}.txt`` (IDs only) for the given role map."""
    counts = {}
    for role in ROLES:
        examples = load_fhm(root, role, roles=roles, allow_test=True)
        counts[role] = write_id_list(examples, Path(out_dir) / f"{role}.txt")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FHM counts and write split ID lists.")
    parser.add_argument("--root", required=True, help="FHM release directory (not committed)")
    parser.add_argument("--write-splits", metavar="DIR", help="e.g. splits/fhm")
    parser.add_argument("--dedup-report", metavar="PATH",
                        help="e.g. splits/fhm/dedup_report.json (needs Pillow)")
    for role in ROLES:
        parser.add_argument(f"--{role}-split", choices=SPLITS + CARVEOUT,
                            help="default: the detected release's role map")
    args = parser.parse_args(argv)

    release = detect_release(args.root)
    defaults = PHASE1_ROLES if release == "phase1" else PHASE2_ROLES
    print(f"release: {release}")
    problems = []
    for split in SPLITS:
        if split_file(args.root, split) is not None:
            examples = load_fhm_split(args.root, split)
            labeled = "labeled" if examples and examples[0].label is not None else "unlabeled"
            print(f"{split:12s} {len(examples):5d} examples, {labeled}")
            problems += count_mismatches(split, examples, release)
        else:
            print(f"{split:12s} missing")
    for problem in problems:
        print(f"MISMATCH {problem}")
    roles = {role: getattr(args, f"{role}_split") or defaults[role] for role in ROLES}
    if args.write_splits:
        for role, n in write_splits(args.root, args.write_splits, roles).items():
            print(f"wrote {args.write_splits}/{role}.txt ({n} ids, {roles[role]})")
    if args.dedup_report:
        from hmf.data.integrity import dedup_report, write_report

        # Over the protocol roles, so the val carve-out is checked against train (#9).
        by_role = {role: load_fhm(args.root, role, roles=roles, allow_test=True)
                   for role in ROLES}
        report = dedup_report("fhm", by_role)
        write_report(report, args.dedup_report)
        print(f"wrote {args.dedup_report}: {len(report['pairs'])} cross-split pairs, "
              f"{len(report['train_ids_to_drop'])} train ids to drop, "
              f"overlapping ids: {sorted(report['overlapping_ids']) or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
