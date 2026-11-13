import json
import os

import pytest

from hmf.data.common import read_id_list
from hmf.data.mvsa import (
    CLASSES,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_COUNTS,
    SPLITS,
    class_counts,
    count_mismatches,
    load_mvsa,
    load_mvsa_split,
    main,
)


def row(i, label="positive", text="a tweet"):
    return {"id": str(i), "label": label, "text": text, "img": f"data/{i}.jpg"}


def write(ann, split, rows):
    ann.mkdir(parents=True, exist_ok=True)
    (ann / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def ann(tmp_path):
    root = tmp_path / "ann"
    write(root, "train", [row(1), row(2, "negative"), row(3, "neutral")])
    write(root, "dev", [row(4, "neutral"), row(5)])
    write(root, "test", [row(6, "negative")])
    return root


def test_class_order_is_fixed():
    assert CLASSES == ("negative", "neutral", "positive")


def test_labels_are_class_indices_and_paths_resolve(ann, tmp_path):
    train = load_mvsa(ann, tmp_path / "img", "train")
    assert [e.id for e in train] == ["1", "2", "3"]
    assert [e.label for e in train] == [2, 0, 1]
    assert train[0].image == str(tmp_path / "img" / "data" / "1.jpg")
    assert class_counts(train) == {"negative": 1, "neutral": 1, "positive": 1}
    assert len(load_mvsa(ann, tmp_path, "train", max_samples=2)) == 2


def test_roles_and_test_guard(ann, tmp_path):
    assert {e.split for e in load_mvsa(ann, tmp_path, "val")} == {"dev"}
    with pytest.raises(PermissionError):
        load_mvsa(ann, tmp_path, "test")
    assert [e.label for e in load_mvsa(ann, tmp_path, "test", allow_test=True)] == [0]


def test_malformed_rows_are_rejected(ann, tmp_path):
    write(ann, "dev", [row(1, "angry")])
    with pytest.raises(ValueError, match="dev.jsonl row 1: unknown class 'angry'"):
        load_mvsa_split(ann, tmp_path, "dev")
    write(ann, "dev", [{"id": "1", "label": "neutral", "text": "x"}])
    with pytest.raises(ValueError, match="missing field 'img'"):
        load_mvsa_split(ann, tmp_path, "dev")
    write(ann, "dev", [row(1), row(1)])
    with pytest.raises(ValueError, match="duplicate id"):
        load_mvsa_split(ann, tmp_path, "dev")
    with pytest.raises(ValueError, match="unknown MVSA-Single split"):
        load_mvsa_split(ann, tmp_path, "val")


def test_count_mismatches(ann, tmp_path):
    assert count_mismatches("train", load_mvsa_split(ann, tmp_path, "train")) == [
        "train: 3 examples, expected 1555",
        "train: class counts {'negative': 1, 'neutral': 1, 'positive': 1}, expected "
        "{'negative': 422, 'neutral': 282, 'positive': 851}",
    ]
    assert count_mismatches("dev", load_mvsa_split(ann, tmp_path, "dev")) == [
        "dev: 2 examples, expected 518"]


def test_cli_writes_id_lists_and_checks_images(ann, tmp_path, capsys):
    out_dir = tmp_path / "splits" / "mvsa"
    rc = main(["--annotations", str(ann), "--write-splits", str(out_dir)])
    assert rc == 1 and "MISMATCH train: 3 examples, expected 1555" in capsys.readouterr().out
    assert read_id_list(out_dir / "val.txt") == ["4", "5"]
    assert read_id_list(out_dir / "test.txt") == ["6"]

    (tmp_path / "img" / "data").mkdir(parents=True)
    for i in range(1, 6):
        (tmp_path / "img" / "data" / f"{i}.jpg").write_bytes(b"")
    main(["--annotations", str(ann), "--images", str(tmp_path / "img")])
    out = capsys.readouterr().out
    missing = [ln for ln in out.splitlines() if ln.startswith("MISMATCH") and "missing" in ln]
    assert missing == ["MISMATCH test: 1 images missing"]


def test_cli_dedup_report_counts_classes(ann, tmp_path):
    import numpy as np
    from PIL import Image

    images = tmp_path / "img" / "data"
    images.mkdir(parents=True)
    for i in range(1, 7):
        noise = np.random.default_rng(i).integers(0, 256, (16, 16), dtype=np.uint8)
        Image.fromarray(noise).save(images / f"{i}.jpg")
    out = tmp_path / "dedup_report.json"
    main(["--annotations", str(ann), "--images", str(tmp_path / "img"), "--dedup-report", str(out)])
    report = json.loads(out.read_text())
    assert report["split_sizes"] == {"train": 3, "val": 2, "test": 1}  # protocol roles (#9)
    assert report["label_distribution"]["train"]["class_counts"] == [1, 1, 1]
    assert report["label_distribution"]["val"]["class_counts"] == [0, 1, 1]
    with pytest.raises(SystemExit):
        main(["--annotations", str(ann), "--dedup-report", str(out)])  # needs --images


def test_expected_counts_match_protocol():
    assert EXPECTED_COUNTS == {"train": 1555, "dev": 518, "test": 519}
    assert EXPECTED_CLASS_COUNTS == {
        "train": {"negative": 422, "neutral": 282, "positive": 851}}


@pytest.mark.skipif(not os.environ.get("HMF_MVSA_ANN"),
                    reason="set HMF_MVSA_ANN to the QMF MVSA_Single split directory")
def test_real_mvsa_counts_and_disjoint_ids():
    ann = os.environ["HMF_MVSA_ANN"]
    splits = {s: load_mvsa_split(ann, ".", s) for s in SPLITS}
    assert [p for s, ex in splits.items() for p in count_mismatches(s, ex)] == []
    ids = {s: {e.id for e in ex} for s, ex in splits.items()}
    assert not (ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"])
