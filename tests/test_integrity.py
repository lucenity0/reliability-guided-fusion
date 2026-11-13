import json
import os
from pathlib import Path

import numpy as np
import pytest

from hmf.data.common import Example, read_id_list
from hmf.data.integrity import (
    _near_pairs,
    apply_report,
    dedup_report,
    dhash,
    drop_from_train,
    find_cross_split_duplicates,
    hamming,
    label_distribution,
    main,
    normalise_text,
    overlapping_ids,
    write_report,
)

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"

Image = pytest.importorskip("PIL.Image")


def ex(i, split, image="img.png", text="t", label=0):
    return Example(id=str(i), image=image, text=text, label=label, split=split)


def make_image(path: Path, seed: int, noise: float = 0.0) -> str:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(64, 64), dtype=np.uint8).astype(float)
    if noise:
        base = np.clip(base + np.random.default_rng(seed + 1000).normal(0, noise, base.shape), 0,
                       255)
    Image.fromarray(base.astype(np.uint8)).save(path)
    return str(path)


def test_overlapping_ids():
    splits = {"train": [ex(1, "train"), ex(2, "train")], "val": [ex(3, "val")],
              "test": [ex(2, "test")]}
    assert overlapping_ids(splits) == {"test|train": ["2"]}
    assert overlapping_ids({"train": [ex(1, "train")], "val": [ex(3, "val")]}) == {}


def test_dhash_is_stable_under_reencoding_and_separates_images(tmp_path):
    a = make_image(tmp_path / "a.png", seed=0)
    b = make_image(tmp_path / "b.png", seed=1)
    Image.open(a).save(tmp_path / "a.jpg", quality=85)
    assert dhash(a) == dhash(a)
    assert hamming(dhash(a), dhash(str(tmp_path / "a.jpg"))) <= 4
    assert hamming(dhash(a), dhash(b)) > 16


def test_near_pairs_matches_brute_force():
    rng = np.random.default_rng(0)
    ha = [int(x) for x in rng.integers(0, 2**63, size=40, dtype=np.int64)]
    hb = [int(x) for x in rng.integers(0, 2**63, size=30, dtype=np.int64)]
    hb[5] = ha[7] ^ 0b101  # distance 2
    hb[9] = ha[3]  # distance 0
    got = {(i, j) for i, j, _ in _near_pairs(ha, hb, 4, chunk=7)}
    want = {(i, j) for i in range(40) for j in range(30) if hamming(ha[i], hb[j]) <= 4}
    assert got == want and {(7, 5), (3, 9)} <= got


def test_dedup_kinds_and_train_only_removal(tmp_path):
    shared = make_image(tmp_path / "shared.png", seed=10)
    shared_jpg = tmp_path / "shared.jpg"
    Image.open(shared).save(shared_jpg, quality=90)
    other = [make_image(tmp_path / f"o{i}.png", seed=20 + i) for i in range(4)]
    splits = {
        "train": [
            ex("t1", "train", shared, "Look   how many people love you"),  # same meme as val
            ex("t2", "train", shared, "a benign confounder caption"),  # same image only
            ex("t3", "train", other[0], "unrelated"),
        ],
        "val": [ex("v1", "val", str(shared_jpg), "look how many people love you")],
        "test": [ex("x1", "test", other[1], "unrelated", None),
                 ex("x2", "test", other[2], "other", None)],
    }
    report = dedup_report("toy", splits)
    kinds = {(tuple(p["ids"]), p["kind"]) for p in report["pairs"]}
    assert kinds == {(("t1", "v1"), "image_text"), (("t2", "v1"), "image_only")}
    assert report["train_ids_to_drop"] == ["t1"]
    assert report["pair_counts"] == {"train|val": {"image_text": 1, "image_only": 1}}
    assert report["label_distribution"]["test"] == {"n": 2, "labeled": False}
    assert report["label_agreement"] == {"image_text": {"same": 1, "different": 0},
                                         "image_only": {"same": 1, "different": 0}}
    assert [e.id for e in drop_from_train(splits["train"], report)] == ["t2", "t3"]

    path = tmp_path / "splits" / "toy" / "dedup_report.json"
    write_report(report, path)
    saved = json.loads(path.read_text())
    assert saved["train_ids_to_drop"] == ["t1"]
    assert "love" not in path.read_text()  # IDs and counts only, never content


def test_duplicates_between_non_train_splits_never_drop_anything():
    splits = {"train": [ex("t", "train", "a", "x")], "val": [ex("v", "val", "b", "same")],
              "test": [ex("s", "test", "b", "same")]}
    report = dedup_report("toy", splits, hash_fn=lambda path: 0 if path == "b" else 2**63 - 1)
    assert [p["ids"] for p in report["pairs"]] == [["s", "v"]]
    assert report["train_ids_to_drop"] == []


def test_max_distance_validation_and_text_normalisation():
    with pytest.raises(ValueError):
        find_cross_split_duplicates({"a": [], "b": []}, max_distance=65)
    assert normalise_text("  It's\tTIME\n") == "it's time"
    assert label_distribution([ex(1, "s", label=1), ex(2, "s", label=0)]) == {
        "n": 2, "labeled": True, "positive": 1, "negative": 1}


def test_label_distribution_is_class_aware():
    # MVSA-Single / Food-101 (D-018): class indices are counted per class, never summed.
    examples = [ex(i, "s", label=label) for i, label in enumerate([2, 0, 2, 1, 2])]
    assert label_distribution(examples, num_classes=3) == {
        "n": 5, "labeled": True, "class_counts": [1, 1, 3]}
    assert label_distribution(examples[:2], num_classes=4)["class_counts"] == [1, 0, 1, 0]
    with pytest.raises(ValueError, match="num_classes"):
        label_distribution(examples)  # binary default with labels up to 2
    splits = {"train": examples[:3], "val": examples[3:]}
    report = dedup_report("toy", splits, hash_fn=lambda path: 0, num_classes=3)
    assert report["label_distribution"]["val"]["class_counts"] == [0, 1, 1]


def test_pairs_record_label_agreement():
    splits = {"train": [ex("a", "train", "p", "x", 1), ex("b", "train", "q", "y", 0)],
              "val": [ex("c", "val", "p", "x", 0)],
              "test": [ex("d", "test", "q", "y", None)]}
    report = dedup_report("toy", splits, hash_fn=lambda path: {"p": 0, "q": 2**63 - 1}[path])
    assert {tuple(p["ids"]): p["same_label"] for p in report["pairs"]} == {
        ("d", "b"): None, ("a", "c"): False}  # unlabeled test pairs are not counted
    assert report["label_agreement"]["image_text"] == {"same": 0, "different": 1}
    assert report["train_ids_to_drop"] == ["a", "b"]  # D-014: dropped whatever the labels


def test_apply_report_rewrites_train_only_and_is_idempotent(tmp_path, capsys):
    d = tmp_path / "splits" / "toy"
    d.mkdir(parents=True)
    (d / "train.txt").write_text("a\nb\nc\n")
    (d / "val.txt").write_text("v\n")
    (d / "test.txt").write_text("t\n")
    write_report({"split_sizes": {"train": 3, "val": 1, "test": 1},
                  "train_ids_to_drop": ["b"]}, d / "dedup_report.json")
    assert apply_report(d) == (3, 2)
    assert read_id_list(d / "train.txt") == ["a", "c"]
    assert apply_report(d) == (2, 2)  # already applied: no change
    assert (d / "val.txt").read_text() == "v\n" and (d / "test.txt").read_text() == "t\n"
    (d / "train.txt").write_text("a\n")
    with pytest.raises(ValueError, match="regenerate the report"):
        apply_report(d)
    (d / "train.txt").write_text("a\nb\nc\n")
    assert main(["--apply", str(d)]) == 0 and "3 -> 2 ids" in capsys.readouterr().out


@pytest.mark.parametrize("report_path", sorted(SPLITS_DIR.glob("*/dedup_report.json")),
                         ids=lambda p: p.parent.name)
def test_committed_dedup_reports_are_applied(report_path):
    # #9 / D-014: committed train lists exclude every train id the role-level report drops.
    report = json.loads(report_path.read_text())
    sizes, drop = report["split_sizes"], set(report["train_ids_to_drop"])
    assert set(sizes) == {"train", "val", "test"} and report["overlapping_ids"] == {}
    train = read_id_list(report_path.parent / "train.txt")
    assert not drop & set(train) and len(train) == sizes["train"] - len(drop)
    assert len(read_id_list(report_path.parent / "val.txt")) == sizes["val"]
    assert len(read_id_list(report_path.parent / "test.txt")) == sizes["test"]


# ── Real data (owner's copy only) ────────────────────────────────────────────


@pytest.mark.skipif(not os.environ.get("HMF_FHM_ROOT"), reason="set HMF_FHM_ROOT")
def test_real_fhm_role_splits_are_disjoint():
    from hmf.data.fhm import load_fhm

    root = os.environ["HMF_FHM_ROOT"]
    roles = {role: load_fhm(root, role, allow_test=True) for role in ("train", "val", "test")}
    assert overlapping_ids(roles) == {}
