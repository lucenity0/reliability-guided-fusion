import os

import pytest

from hmf.data.common import read_id_list
from hmf.data.crisismmd import (
    EXPECTED_COUNTS,
    FILE_PATTERN,
    SPLITS,
    count_mismatches,
    load_crisismmd,
    load_crisismmd_split,
    main,
    overlapping_tweets,
    tweet_of,
)

HEADER = ("event_name\ttweet_id\timage_id\ttweet_text\timage\tlabel\tlabel_text\t"
          "label_image\tlabel_text_image")


def line(tweet, k, label="informative", text="flood in the city", text_label=None):
    text_label = text_label or label
    return (f"harvey\t{tweet}\t{tweet}_{k}\t{text}\tdata_image/harvey/1_9_2017/{tweet}_{k}.jpg\t"
            f"{label}\t{text_label}\t{label}\tPositive")


def write(root, split, lines):
    root.mkdir(parents=True, exist_ok=True)
    (root / FILE_PATTERN.format(split=split)).write_text("\n".join([HEADER, *lines]) + "\n")


@pytest.fixture
def ann(tmp_path):
    root = tmp_path / "ann"
    write(root, "train", [line(100, 0), line(100, 1), line(101, 0, "not_informative")])
    write(root, "dev", [line(200, 0), line(201, 0, "not_informative")])
    write(root, "test", [line(300, 0, text='quote " inside, no quoting')])
    return root


def test_pairs_are_samples_and_share_tweet_text(ann, tmp_path):
    train = load_crisismmd(ann, tmp_path / "img", "train")
    assert [e.id for e in train] == ["100_0", "100_1", "101_0"]
    assert [e.label for e in train] == [1, 1, 0]
    assert train[0].text == train[1].text
    assert train[0].image == str(tmp_path / "img" / "data_image/harvey/1_9_2017/100_0.jpg")
    assert {tweet_of(e.id) for e in train} == {"100", "101"}


def test_roles_and_test_guard(ann, tmp_path):
    assert {e.split for e in load_crisismmd(ann, tmp_path, "val")} == {"dev"}
    with pytest.raises(PermissionError):
        load_crisismmd(ann, tmp_path, "test")
    test = load_crisismmd(ann, tmp_path, "test", allow_test=True)
    assert test[0].text == 'quote " inside, no quoting'


def test_disagreeing_or_unknown_labels_are_rejected(tmp_path):
    write(tmp_path, "train", [line(1, 0, "informative", text_label="not_informative")])
    with pytest.raises(ValueError, match="disagree"):
        load_crisismmd_split(tmp_path, tmp_path, "train")
    write(tmp_path, "train", [line(1, 0, "maybe")])
    with pytest.raises(ValueError, match="unknown label"):
        load_crisismmd_split(tmp_path, tmp_path, "train")


def test_image_must_belong_to_tweet(tmp_path):
    bad = line(1, 0).replace("\t1_0\t", "\t2_0\t")
    write(tmp_path, "dev", [bad])
    with pytest.raises(ValueError, match="does not belong"):
        load_crisismmd_split(tmp_path, tmp_path, "dev")


def test_missing_column_and_duplicates(tmp_path):
    (tmp_path / FILE_PATTERN.format(split="dev")).write_text("tweet_id\timage_id\n1\t1_0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_crisismmd_split(tmp_path, tmp_path, "dev")
    write(tmp_path, "dev", [line(1, 0), line(1, 0)])
    with pytest.raises(ValueError, match="duplicate id"):
        load_crisismmd_split(tmp_path, tmp_path, "dev")


def test_tweet_overlap_detection(ann, tmp_path):
    splits = {s: load_crisismmd_split(ann, tmp_path, s) for s in SPLITS}
    assert overlapping_tweets(splits) == {}
    write(ann, "dev", [line(100, 2)])  # another image of a train tweet
    splits["dev"] = load_crisismmd_split(ann, tmp_path, "dev")
    assert overlapping_tweets(splits) == {"dev|train": ["100"]}
    with pytest.raises(ValueError):
        tweet_of("not-an-id")


def test_counts_cli_and_split_files(ann, tmp_path, capsys):
    assert count_mismatches("dev", load_crisismmd_split(ann, tmp_path, "dev")) == [
        "dev: 2 examples, expected 1573", "dev: 1 informative, expected 1056"]
    out_dir = tmp_path / "splits"
    rc = main(["--annotations", str(ann), "--images", str(tmp_path),
               "--write-splits", str(out_dir)])
    assert rc == 1 and "MISMATCH train" in capsys.readouterr().out
    assert read_id_list(out_dir / "val.txt") == ["200_0", "201_0"]


def test_cli_dedup_report(ann, tmp_path):
    import json

    import numpy as np
    from PIL import Image

    for seed, e in enumerate(e for s in SPLITS for e in load_crisismmd_split(ann, tmp_path, s)):
        os.makedirs(os.path.dirname(e.image), exist_ok=True)
        noise = np.random.default_rng(seed).integers(0, 256, (16, 16), dtype=np.uint8)
        Image.fromarray(noise).save(e.image)
    out = tmp_path / "dedup_report.json"
    main(["--annotations", str(ann), "--images", str(tmp_path), "--dedup-report", str(out)])
    report = json.loads(out.read_text())
    assert report["split_sizes"] == {"train": 3, "val": 2, "test": 1}  # protocol roles (#9)
    assert report["label_distribution"]["train"] == {"n": 3, "labeled": True, "positive": 2,
                                                     "negative": 1}
    assert report["overlapping_ids"] == {}


def test_expected_counts_match_paper_table():
    assert EXPECTED_COUNTS == {"train": (9601, 6345), "dev": (1573, 1056),
                               "test": (1534, 1030)}


@pytest.mark.skipif(not os.environ.get("HMF_CRISISMMD_ANN"),
                    reason="set HMF_CRISISMMD_ANN to the agreed-label TSV directory")
def test_real_crisismmd_counts_and_tweet_disjointness():
    ann = os.environ["HMF_CRISISMMD_ANN"]
    splits = {s: load_crisismmd_split(ann, ".", s) for s in SPLITS}
    assert [p for s, ex in splits.items() for p in count_mismatches(s, ex)] == []
    assert overlapping_tweets(splits) == {}
