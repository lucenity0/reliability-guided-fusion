import json
import os
from pathlib import Path

import pytest

from hmf.data.common import read_id_list
from hmf.data.harmeme import (
    EXPECTED_COUNTS,
    EXPECTED_UNLABELED,
    SPLITS,
    SUBSETS,
    binarise,
    count_mismatches,
    load_harmeme,
    load_harmeme_split,
    main,
    read_split,
    split_file,
)


def row(i, level="not harmful", target=None, text="a meme"):
    labels = [level] if target is None else [level, target]
    return {"id": f"covid_memes_{i}", "image": f"covid_memes_{i}.png", "labels": labels,
            "text": text}


def write(ann, subset, split, rows):
    path = split_file(ann, subset, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def ann(tmp_path):
    root = tmp_path / "ann"
    for subset in SUBSETS:
        write(root, subset, "train", [row(1), row(2, "somewhat harmful", "individual"),
                                      row(3, "very harmful", "society")])
        write(root, subset, "val", [row(4), row(5, "very harmful", "community")])
        write(root, subset, "test", [row(6, "somewhat harmful", "organization")])
    return root


def test_binarisation_follows_labels0():
    assert binarise(["not harmful"]) == 0
    assert binarise(["somewhat harmful", "individual"]) == 1
    assert binarise(["very harmful", "society"]) == 1
    assert binarise([]) is None and binarise([""]) is None
    with pytest.raises(ValueError, match="unknown harmfulness"):
        binarise(["harmful"])
    with pytest.raises(ValueError, match="must be a list"):
        binarise("not harmful")


def test_file_names_per_subset(tmp_path):
    assert split_file(tmp_path, "harm_c", "val") == tmp_path / "Harm-C" / "val.jsonl"
    assert split_file(tmp_path, "harm_p", "train") == tmp_path / "Harm-P" / "train_v1.jsonl"
    with pytest.raises(ValueError, match="unknown HarMeme subset"):
        split_file(tmp_path, "harm-c", "train")
    with pytest.raises(ValueError, match="unknown HarMeme split"):
        split_file(tmp_path, "harm_c", "dev")


def test_schema_ids_and_image_paths(ann, tmp_path):
    train = load_harmeme(ann, tmp_path / "img", "harm_c", "train")
    assert [e.id for e in train] == ["covid_memes_1", "covid_memes_2", "covid_memes_3"]
    assert [e.label for e in train] == [0, 1, 1]
    assert {e.split for e in train} == {"train"}
    assert train[0].image == str(tmp_path / "img" / "harmeme_images_covid_19" / "covid_memes_1.png")
    harm_p = load_harmeme(ann, tmp_path / "img", "harm_p", "train", max_samples=2)
    assert len(harm_p) == 2
    assert Path(harm_p[0].image).parent == tmp_path / "img" / "harmeme_images_us_pol"


def test_roles_and_test_guard(ann, tmp_path):
    assert {e.split for e in load_harmeme(ann, tmp_path, "harm_p", "val")} == {"val"}
    with pytest.raises(PermissionError):
        load_harmeme(ann, tmp_path, "harm_c", "test")
    test = load_harmeme(ann, tmp_path, "harm_c", "test", allow_test=True)
    assert [e.label for e in test] == [1]


def test_unlabeled_row_is_dropped_and_reported(ann, tmp_path):
    rows = [row(1), {"id": "memes_9", "image": "memes_9.png", "labels": [], "text": "x"},
            row(2, "very harmful", "society")]
    write(ann, "harm_p", "train", rows)
    examples, dropped = read_split(ann, tmp_path, "harm_p", "train")
    assert [e.id for e in examples] == ["covid_memes_1", "covid_memes_2"]
    assert dropped == ["memes_9"]
    with pytest.warns(UserWarning, match="dropped 1 unlabeled row.*memes_9"):
        load_harmeme_split(ann, tmp_path, "harm_p", "train")


def test_malformed_rows_are_rejected(ann, tmp_path):
    write(ann, "harm_c", "val", [{"id": "a", "image": "a.png", "text": "x"}])
    with pytest.raises(ValueError, match="missing field 'labels'"):
        load_harmeme_split(ann, tmp_path, "harm_c", "val")
    write(ann, "harm_c", "val", [row(1, "maybe harmful")])
    with pytest.raises(ValueError, match="val.jsonl row 1: unknown harmfulness"):
        load_harmeme_split(ann, tmp_path, "harm_c", "val")
    write(ann, "harm_c", "val", [row(1), row(1)])
    with pytest.raises(ValueError, match="duplicate id"):
        load_harmeme_split(ann, tmp_path, "harm_c", "val")
    with pytest.raises(ValueError, match="max_samples"):
        load_harmeme_split(ann, tmp_path, "harm_c", "train", max_samples=-1)


def test_count_mismatches_include_unlabeled_rows(ann, tmp_path):
    examples, dropped = read_split(ann, tmp_path, "harm_p", "train")
    assert count_mismatches("harm_p", "train", examples, dropped) == [
        "harm_p train: 3 rows, expected 2938",
        "harm_p train: 2 harmful, expected 1487",
        "harm_p train: 0 unlabeled rows, expected 1",
    ]


def test_cli_writes_id_lists_and_flags_missing_images(ann, tmp_path, capsys):
    images = tmp_path / "img"
    (images / "harmeme_images_covid_19").mkdir(parents=True)
    for i in range(1, 7):
        (images / "harmeme_images_covid_19" / f"covid_memes_{i}.png").write_bytes(b"")
    out_dir = tmp_path / "splits" / "harm_c"
    rc = main(["--annotations", str(ann), "--images", str(images), "--subset", "harm_c",
               "--write-splits", str(out_dir)])
    out = capsys.readouterr().out
    assert rc == 1 and "MISMATCH harm_c train: 3 rows, expected 3013" in out
    assert not [ln for ln in out.splitlines() if ln.startswith("MISMATCH") and "missing" in ln]
    assert read_id_list(out_dir / "val.txt") == ["covid_memes_4", "covid_memes_5"]
    assert read_id_list(out_dir / "test.txt") == ["covid_memes_6"]

    (images / "harmeme_images_covid_19" / "covid_memes_5.png").unlink()
    main(["--annotations", str(ann), "--images", str(images), "--subset", "harm_c"])
    assert "MISMATCH harm_c val: 1 images missing" in capsys.readouterr().out


def test_expected_counts_match_protocol():
    assert EXPECTED_COUNTS == {
        "harm_c": {"train": (3013, 1064), "val": (177, 61), "test": (354, 124)},
        "harm_p": {"train": (2938, 1487), "val": (177, 89), "test": (355, 171)},
    }
    assert EXPECTED_UNLABELED == {"harm_c": {}, "harm_p": {"train": 1}}


@pytest.mark.skipif(not os.environ.get("HMF_HARMEME_ROOT"),
                    reason="set HMF_HARMEME_ROOT to the HarMeme V1 Annotations directory")
@pytest.mark.parametrize("subset", SUBSETS)
def test_real_harmeme_counts(subset):
    ann = Path(os.environ["HMF_HARMEME_ROOT"])
    problems, ids = [], {}
    for split in SPLITS:
        examples, dropped = read_split(ann, ".", subset, split)
        problems += count_mismatches(subset, split, examples, dropped)
        ids[split] = {e.id for e in examples}
    assert problems == []
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
