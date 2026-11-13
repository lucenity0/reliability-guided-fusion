import json
import os

import pytest

from hmf.data.common import read_id_list
from hmf.data.food101 import (
    CLASSES,
    EXPECTED_COUNTS,
    SPLITS,
    count_mismatches,
    load_food101,
    load_food101_split,
    main,
)


def row(i, label="apple_pie", part="train", text="a recipe"):
    name = f"{label}_{i}.jpg"
    return {"id": f"{label}_{name}", "label": label, "text": text,
            "img": f"images/{part}/{label}/{name}"}


def write(ann, split, rows):
    ann.mkdir(parents=True, exist_ok=True)
    (ann / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def ann(tmp_path):
    root = tmp_path / "ann"
    write(root, "train", [row(1), row(2, "waffles"), row(3, "pho")])
    write(root, "dev", [row(4, "sushi"), row(5)])
    write(root, "test", [row(6, "waffles", part="test")])
    return root


def test_classes_are_101_sorted_names():
    assert len(CLASSES) == 101 == len(set(CLASSES))
    assert list(CLASSES) == sorted(CLASSES)
    assert CLASSES[0] == "apple_pie" and CLASSES[-1] == "waffles"


def test_labels_are_class_indices_and_paths_resolve(ann, tmp_path):
    train = load_food101(ann, tmp_path / "img", "train")
    assert [e.id for e in train] == ["apple_pie_apple_pie_1.jpg", "waffles_waffles_2.jpg",
                                     "pho_pho_3.jpg"]
    assert [e.label for e in train] == [0, 100, CLASSES.index("pho")]
    assert train[0].image == str(tmp_path / "img" / "images/train/apple_pie/apple_pie_1.jpg")
    assert len(load_food101(ann, tmp_path, "train", max_samples=1)) == 1


def test_roles_and_test_guard(ann, tmp_path):
    assert {e.split for e in load_food101(ann, tmp_path, "val")} == {"dev"}
    with pytest.raises(PermissionError):
        load_food101(ann, tmp_path, "test")
    assert [e.label for e in load_food101(ann, tmp_path, "test", allow_test=True)] == [100]


def test_malformed_rows_are_rejected(ann, tmp_path):
    write(ann, "dev", [row(1, "pizza_margherita")])
    with pytest.raises(ValueError, match="dev.jsonl row 1: unknown class 'pizza_margherita'"):
        load_food101_split(ann, tmp_path, "dev")
    write(ann, "dev", [{"id": "x", "label": "pho", "img": "images/train/pho/x.jpg"}])
    with pytest.raises(ValueError, match="missing field 'text'"):
        load_food101_split(ann, tmp_path, "dev")
    write(ann, "dev", [row(1), row(1)])
    with pytest.raises(ValueError, match="duplicate id"):
        load_food101_split(ann, tmp_path, "dev")
    with pytest.raises(ValueError, match="unknown Food-101 split"):
        load_food101_split(ann, tmp_path, "test_hard_gt")


def test_cli_writes_id_lists_and_checks_images(ann, tmp_path, capsys):
    out_dir = tmp_path / "splits" / "food101"
    rc = main(["--annotations", str(ann), "--write-splits", str(out_dir)])
    assert rc == 1 and "MISMATCH dev: 2 examples, expected 5000" in capsys.readouterr().out
    assert read_id_list(out_dir / "val.txt") == ["sushi_sushi_4.jpg", "apple_pie_apple_pie_5.jpg"]
    assert read_id_list(out_dir / "test.txt") == ["waffles_waffles_6.jpg"]

    for e in load_food101_split(ann, tmp_path / "img", "train"):
        os.makedirs(os.path.dirname(e.image), exist_ok=True)
        open(e.image, "wb").close()
    main(["--annotations", str(ann), "--images", str(tmp_path / "img")])
    out = capsys.readouterr().out
    missing = [ln for ln in out.splitlines() if ln.startswith("MISMATCH") and "missing" in ln]
    assert missing == ["MISMATCH dev: 2 images missing", "MISMATCH test: 1 images missing"]


def test_cli_dedup_report_counts_classes(ann, tmp_path):
    import numpy as np
    from PIL import Image

    examples = [e for s in SPLITS for e in load_food101_split(ann, tmp_path / "img", s)]
    for seed, e in enumerate(examples):
        os.makedirs(os.path.dirname(e.image), exist_ok=True)
        noise = np.random.default_rng(seed).integers(0, 256, (16, 16), dtype=np.uint8)
        Image.fromarray(noise).save(e.image)
    out = tmp_path / "dedup_report.json"
    main(["--annotations", str(ann), "--images", str(tmp_path / "img"), "--dedup-report", str(out)])
    counts = json.loads(out.read_text())["label_distribution"]["train"]["class_counts"]
    assert len(counts) == 101 and sum(counts) == 3
    assert counts[0] == counts[CLASSES.index("pho")] == counts[100] == 1
    with pytest.raises(SystemExit):
        main(["--annotations", str(ann), "--dedup-report", str(out)])


def test_expected_counts_match_protocol():
    assert EXPECTED_COUNTS == {"train": 62971, "dev": 5000, "test": 22715}


@pytest.mark.skipif(not os.environ.get("HMF_FOOD101_ANN"),
                    reason="set HMF_FOOD101_ANN to the QMF food101 split directory")
def test_real_food101_counts_and_disjoint_ids():
    ann = os.environ["HMF_FOOD101_ANN"]
    splits = {s: load_food101_split(ann, ".", s) for s in SPLITS}
    assert [p for s, ex in splits.items() for p in count_mismatches(s, ex)] == []
    assert {e.label for e in splits["train"]} == set(range(len(CLASSES)))
    ids = {s: {e.id for e in ex} for s, ex in splits.items()}
    assert not (ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"])
