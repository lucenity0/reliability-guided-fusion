import os

import pytest

from hmf.data.common import read_id_list
from hmf.data.mami import (
    EXPECTED_COUNTS,
    count_mismatches,
    load_mami,
    load_mami_split,
    main,
)

HEADER = ["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence",
          "Text Transcription"]


def tsv(path, rows):
    # The release's format: tab-separated, UTF-8 with a byte-order mark, CRLF line ends.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("﻿" + "".join("\t".join(r) + "\r\n" for r in rows)).encode("utf-8"))


def train_row(i, label, text="a meme"):
    return [f"{i}.jpg", str(label), "0", "0", "0", "0", text]


@pytest.fixture
def root(tmp_path):
    rows = [train_row(i, i % 2) for i in range(1, 21)]
    rows[2][-1] = '"I\'M NOT FAT, I\'M JUST BIG-BONED." THERE IS NO STOMACH BONE'
    tsv(tmp_path / "TRAINING" / "training.csv", [HEADER, *rows])
    tsv(tmp_path / "test" / "Test.csv",
        [["file_name", "Text Transcription"], ["15236.jpg", "test text"], ["15805.jpg", "x"]])
    tsv(tmp_path / "test_labels.txt", [["15805.jpg", "1", "0", "0", "1", "0"],
                                       ["15236.jpg", "0", "0", "0", "0", "0"]])
    return tmp_path


def test_train_schema_quotes_and_bom(root):
    train = load_mami_split(root, "train")
    assert [e.id for e in train[:3]] == ["1", "2", "3"]
    assert [e.label for e in train[:4]] == [1, 0, 1, 0]
    assert train[0].image == str(root / "TRAINING" / "1.jpg")
    assert train[2].text == '"I\'M NOT FAT, I\'M JUST BIG-BONED." THERE IS NO STOMACH BONE'
    assert {e.split for e in train} == {"train"}


def test_test_split_joins_labels_and_texts_in_label_file_order(root):
    test = load_mami(root, "test", allow_test=True)
    assert [(e.id, e.label, e.text) for e in test] == [("test_15805", 1, "x"),
                                                       ("test_15236", 0, "test text")]
    assert test[0].image == str(root / "test" / "15805.jpg")
    with pytest.raises(PermissionError):
        load_mami(root, "test")


def test_reused_file_names_get_distinct_ids(root):
    # The release has 15001.jpg etc. in both train and test, as different memes.
    rows = [train_row(i, i % 2) for i in range(1, 21)] + [train_row(15805, 1, "a train meme")]
    tsv(root / "TRAINING" / "training.csv", [HEADER, *rows])
    train_ids = {e.id for e in load_mami_split(root, "train")}
    test_ids = {e.id for e in load_mami_split(root, "test")}
    assert "15805" in train_ids and "test_15805" in test_ids and not train_ids & test_ids
    test = {e.id: e for e in load_mami_split(root, "test")}
    assert test["test_15805"].image == str(root / "test" / "15805.jpg")


def test_stratified_carveout_is_seeded_and_disjoint(root):
    fit, val = load_mami(root, "train"), load_mami(root, "val")
    assert len(val) == 2 and sorted(e.label for e in val) == [0, 1]  # round(0.1 * 10) per class
    assert len(fit) == 18 and not {e.id for e in fit} & {e.id for e in val}
    assert [e.id for e in load_mami(root, "val")] == [e.id for e in val]
    assert len(load_mami(root, "train", max_samples=5)) == 5


def test_missing_test_text_file_explains_the_password(root):
    (root / "test" / "Test.csv").unlink()
    with pytest.raises(FileNotFoundError, match="test.zip"):
        load_mami_split(root, "test")


def test_malformed_files_are_rejected(root):
    tsv(root / "test_labels.txt", [["99.jpg", "1", "0", "0", "0", "0"]])
    with pytest.raises(ValueError, match="no row in Test.csv"):
        load_mami_split(root, "test")
    tsv(root / "test_labels.txt", [["15236.jpg", "0", "0", "0", "0", "0"]])
    with pytest.raises(ValueError, match="Test.csv has 2 rows"):
        load_mami_split(root, "test")
    tsv(root / "TRAINING" / "training.csv", [HEADER, train_row(1, 2)])
    with pytest.raises(ValueError, match="misogynous must be 0 or 1"):
        load_mami_split(root, "train")
    tsv(root / "TRAINING" / "training.csv", [HEADER, train_row(1, 0), train_row(1, 1)])
    with pytest.raises(ValueError, match="duplicate id"):
        load_mami_split(root, "train")
    tsv(root / "TRAINING" / "training.csv", [HEADER[:2], ["1.jpg", "0"]])
    with pytest.raises(ValueError, match="missing columns"):
        load_mami_split(root, "train")
    with pytest.raises(ValueError, match="unknown MAMI split"):
        load_mami_split(root, "dev")


def test_cli_writes_role_id_lists(root, capsys):
    out = root / "splits"
    rc = main(["--root", str(root), "--write-splits", str(out)])
    text = capsys.readouterr().out
    assert rc == 1 and "MISMATCH train: 20 examples, expected 10000" in text
    assert "images missing" in text
    assert len(read_id_list(out / "train.txt")) == 18 and len(read_id_list(out / "val.txt")) == 2
    assert read_id_list(out / "test.txt") == ["test_15805", "test_15236"]


def test_expected_counts_match_fersini_2022_table_2():
    assert EXPECTED_COUNTS == {"train": (10000, 5000), "test": (1000, 500)}
    assert count_mismatches("test", []) == ["test: 0 examples, expected 1000",
                                            "test: 0 misogynous, expected 500"]


@pytest.mark.skipif(not os.environ.get("HMF_MAMI_ROOT"),
                    reason="set HMF_MAMI_ROOT to the unpacked MAMI release")
def test_real_mami_counts_and_carveout():
    root = os.environ["HMF_MAMI_ROOT"]
    train = load_mami_split(root, "train")
    assert count_mismatches("train", train) == []
    fit, val = load_mami(root, "train"), load_mami(root, "val")
    assert (len(fit), sum(e.label for e in fit)) == (9000, 4500)
    assert (len(val), sum(e.label for e in val)) == (1000, 500)
    if not os.path.exists(os.path.join(root, "test", "Test.csv")):
        pytest.skip("test.zip not unpacked (password from the organisers)")
    test = load_mami_split(root, "test")
    assert count_mismatches("test", test) == []
    assert not {e.id for e in test} & {e.id for e in train}
