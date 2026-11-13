import json
import os
from pathlib import Path

import pytest

from hmf.data.common import read_id_list, resolve_role, stratified_carveout
from hmf.data.fhm import (
    DEFAULT_ROLES,
    EXPECTED_COUNTS,
    PHASE1_ROLES,
    PHASE2_ROLES,
    SPLITS,
    count_mismatches,
    detect_release,
    load_fhm,
    load_fhm_split,
    main,
    write_splits,
)

P2 = PHASE2_ROLES


def write_split(root: Path, split: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def row(i: int, label: int | None = 0, text: str = "a meme") -> dict:
    out = {"id": i, "img": f"img/{i:05d}.png", "text": text}
    if label is not None:
        out["label"] = label
    return out


@pytest.fixture
def fhm_root(tmp_path):
    root = tmp_path / "fhm"
    write_split(root, "train", [row(i, i % 2) for i in range(1, 101)])
    write_split(root, "dev_seen", [row(i, i % 2) for i in range(201, 211)])
    write_split(root, "dev_unseen", [row(i, i % 2) for i in range(301, 321)])
    write_split(root, "test_seen", [row(i, None) for i in range(401, 411)])
    write_split(root, "test_unseen", [row(i, None) for i in range(501, 531)])
    return root


def test_schema_and_canonical_ids(fhm_root):
    examples = load_fhm(fhm_root, "train", roles=P2)
    first = examples[0]
    assert first.id == "00001"  # leading zeros from the image name are kept
    assert first.image == str(fhm_root / "img/00001.png")
    assert (first.text, first.label, first.split) == ("a meme", 1, "train")
    assert len(examples) == 100


def test_phase2_role_map(fhm_root):
    assert detect_release(fhm_root) == "phase2"
    assert {e.split for e in load_fhm(fhm_root, "val", roles=P2)} == {"dev_unseen"}
    custom = {"train": "train", "val": "dev_seen", "test": "dev_unseen"}
    test = load_fhm(fhm_root, "test", roles=custom, allow_test=True)
    assert {e.split for e in test} == {"dev_unseen"}


def test_test_role_requires_explicit_permission(fhm_root):
    with pytest.raises(PermissionError):
        load_fhm(fhm_root, "test", roles=P2)
    test = load_fhm(fhm_root, "test", roles=P2, allow_test=True)
    assert len(test) == 30 and all(e.label is None for e in test)


def test_unlabeled_split_cannot_be_train_or_val(fhm_root):
    roles = {"train": "train", "val": "test_seen", "test": "test_unseen"}
    with pytest.raises(ValueError, match="unlabeled"):
        load_fhm(fhm_root, "val", roles=roles)


def test_max_samples_smoke(fhm_root):
    assert [e.id for e in load_fhm(fhm_root, "train", roles=P2, max_samples=64)] == [
        f"{i:05d}" for i in range(1, 65)
    ]
    with pytest.raises(ValueError):
        load_fhm_split(fhm_root, "train", max_samples=-1)


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"id": 7, "img": "img/00008.png", "text": "x", "label": 0}, "does not match"),
        ({"id": 7, "img": "img/00007.png", "text": "x", "label": 2}, "label must be"),
        ({"id": 7, "img": "img/00007.png", "label": 0}, "missing field"),
        ({"id": 7, "img": "img/00007.png", "text": 3, "label": 0}, "text must be"),
    ],
)
def test_malformed_rows_are_rejected(tmp_path, bad, message):
    write_split(tmp_path, "train", [bad])
    with pytest.raises(ValueError, match=message):
        load_fhm_split(tmp_path, "train")


def test_duplicate_ids_and_mixed_labels_are_rejected(tmp_path):
    write_split(tmp_path, "train", [row(1), row(1)])
    with pytest.raises(ValueError, match="duplicate id"):
        load_fhm_split(tmp_path, "train")
    write_split(tmp_path, "dev_seen", [row(1, 0), row(2, None)])
    with pytest.raises(ValueError, match="some rows are labeled"):
        load_fhm_split(tmp_path, "dev_seen")


def test_role_validation():
    with pytest.raises(ValueError):
        resolve_role("dev", DEFAULT_ROLES, allow_test=False)
    with pytest.raises(ValueError):
        resolve_role("train", {"train": "train"}, allow_test=False)


def test_count_mismatches(fhm_root):
    problems = count_mismatches("dev_unseen", load_fhm_split(fhm_root, "dev_unseen"), "phase2")
    assert problems == ["dev_unseen: 20 examples, expected 540",
                        "dev_unseen: 10 hateful, expected 200"]


def test_write_splits_writes_ids_only(fhm_root, tmp_path):
    out = tmp_path / "splits" / "fhm"
    assert write_splits(fhm_root, out, P2) == {"train": 100, "val": 20, "test": 30}
    assert read_id_list(out / "val.txt")[:2] == ["00301", "00302"]
    assert "meme" not in (out / "train.txt").read_text()


def test_cli_reports_mismatches(fhm_root, tmp_path, capsys):
    assert main(["--root", str(fhm_root), "--write-splits", str(tmp_path / "s")]) == 1
    out = capsys.readouterr().out
    assert "release: phase2" in out
    assert "MISMATCH train: 100 examples, expected 8500" in out
    assert (tmp_path / "s" / "test.txt").exists()


# ── Real data (runs only where the owner's FHM copy is available) ───────────

REAL_ROOT = os.environ.get("HMF_FHM_ROOT")


@pytest.mark.skipif(not REAL_ROOT, reason="set HMF_FHM_ROOT to the FHM release directory")
@pytest.mark.parametrize("split", SPLITS)
def test_real_fhm_counts_match_competition_report(split):
    from hmf.data.fhm import split_file

    if split_file(REAL_ROOT, split) is None:
        pytest.skip(f"{split} not present in this release")
    examples = load_fhm_split(REAL_ROOT, split)
    assert count_mismatches(split, examples, detect_release(REAL_ROOT)) == []
    assert len(load_fhm_split(REAL_ROOT, split, max_samples=64)) == 64


def test_expected_counts_cover_releases():
    assert set(EXPECTED_COUNTS["phase2"]) == set(SPLITS)
    assert set(EXPECTED_COUNTS["phase1"]) == {"train", "dev_seen", "test_seen"}
    assert DEFAULT_ROLES == PHASE1_ROLES == {"train": "train:fit", "val": "train:val",
                                             "test": "dev_seen"}


# ── Phase 1 release (D-012) ──────────────────────────────────────────────────


@pytest.fixture
def phase1_root(tmp_path):
    root = tmp_path / "fhm1"
    # 200 train rows, 60 hateful (stratification check), dev = dev_seen, test unlabeled.
    write_split(root, "train", [row(i, int(i % 10 < 3)) for i in range(1, 201)])
    (root / "dev.jsonl").write_text("".join(json.dumps(row(i, i % 2)) + "\n"
                                            for i in range(301, 321)))
    (root / "test.jsonl").write_text("".join(json.dumps(row(i, None)) + "\n"
                                             for i in range(401, 411)))
    return root


def test_phase1_layout_and_default_roles(phase1_root):
    assert detect_release(phase1_root) == "phase1"
    fit = load_fhm(phase1_root, "train")
    val = load_fhm(phase1_root, "val")
    test = load_fhm(phase1_root, "test", allow_test=True)
    assert (len(fit), len(val), len(test)) == (180, 20, 20)
    assert {e.id for e in fit}.isdisjoint({e.id for e in val})
    assert sum(e.label for e in val) == 6  # 10% of 60 hateful, stratified
    assert {e.split for e in test} == {"dev_seen"}  # dev.jsonl is dev_seen
    with pytest.raises(PermissionError):
        load_fhm(phase1_root, "test")


def test_carveout_is_deterministic_and_order_preserving(phase1_root):
    train = load_fhm_split(phase1_root, "train")
    a = stratified_carveout(train, fraction=0.1, seed=1234)
    b = stratified_carveout(list(train), fraction=0.1, seed=1234)
    assert [e.id for e in a[1]] == [e.id for e in b[1]]
    assert [e.id for e in a[1]] == sorted(e.id for e in a[1])  # file order kept
    other = stratified_carveout(train, fraction=0.1, seed=7)
    assert {e.id for e in other[1]} != {e.id for e in a[1]}
    with pytest.raises(ValueError):
        stratified_carveout(load_fhm_split(phase1_root, "test_seen"))


def test_max_samples_does_not_change_the_carveout(phase1_root):
    full = load_fhm(phase1_root, "val")
    assert load_fhm(phase1_root, "val", max_samples=5) == full[:5]


def test_phase1_counts_and_cli(phase1_root, tmp_path, capsys):
    assert count_mismatches("dev_seen", load_fhm_split(phase1_root, "dev_seen"), "phase1") == [
        "dev_seen: 20 examples, expected 500", "dev_seen: 10 hateful, expected 250"]
    assert count_mismatches("dev_unseen", [], "phase1") == [
        "dev_unseen: not part of the phase1 release"]
    out_dir = tmp_path / "s"
    assert main(["--root", str(phase1_root), "--write-splits", str(out_dir)]) == 1
    out = capsys.readouterr().out
    assert "release: phase1" in out and "dev_unseen   missing" in out
    assert len(read_id_list(out_dir / "val.txt")) == 20
    assert read_id_list(out_dir / "test.txt")[0] == "00301"
