import csv
import json
import math

import pytest

from hmf.config import config_hash
from hmf.data.common import Example
from hmf.lmm_zeroshot import (
    REFUSAL_SCORE,
    check_final,
    load_templates,
    render_prompt,
    run_config,
    score_examples,
    write_outputs,
    yes_probability,
)

YES, YES_SPACE, NO, OTHER = 0, 1, 2, 3


def logits(yes=0.0, yes_space=-50.0, no=0.0, other=-50.0):
    return [yes, yes_space, no, other]


def test_yes_probability_two_way_softmax():
    p, refused = yes_probability(logits(yes=2.0, no=0.0), [YES], [NO])
    assert p == pytest.approx(1 / (1 + math.exp(-2.0))) and not refused


def test_spellings_pool_their_probability_mass():
    # Yes split over two spellings with equal logits = twice the mass of No.
    p, _ = yes_probability(logits(yes=0.0, yes_space=0.0, no=0.0), [YES, YES_SPACE], [NO])
    assert p == pytest.approx(2 / 3)


def test_refusal_when_greedy_token_is_neither():
    p, refused = yes_probability(logits(yes=1.0, no=0.5, other=5.0), [YES], [NO])
    assert refused and 0.5 < p < 1


def test_invalid_id_sets():
    with pytest.raises(ValueError):
        yes_probability(logits(), [YES], [YES])
    with pytest.raises(ValueError):
        yes_probability(logits(), [], [NO])


def example(i, label=1, text="  look how many people love you "):
    return Example(id=f"{i:05d}", image=f"img/{i:05d}.png", text=text, label=label, split="dev")


def test_render_prompt_fills_text_and_task():
    out = render_prompt('Is this {task}? "{text}" Yes or No.', example(1), "harmful")
    assert out == 'Is this harmful? "look how many people love you" Yes or No.'
    with pytest.raises(ValueError):
        render_prompt("no slot", example(1))


def test_scores_refusals_as_half_and_records_latency():
    responses = {"img/00001.png": logits(yes=3.0), "img/00002.png": logits(other=9.0)}
    seen = []

    def fake(image, prompt):
        seen.append(prompt)
        return responses[image]

    scores = score_examples([example(1), example(2, 0)], "{task}: {text}", fake, [YES], [NO],
                            task="misogynistic")
    assert scores[0].prob_yes > 0.9 and not scores[0].refused
    assert scores[1].prob_yes == REFUSAL_SCORE and scores[1].refused
    assert all(s.latency_ms >= 0 for s in scores)
    assert seen[0].startswith("misogynistic: look how")


def test_outputs_follow_the_stable_interface(tmp_path):
    exs = [example(1, 1), example(2, 0), example(3, 0)]
    scores = score_examples(exs, "{text}", lambda i, p: logits(yes=3.0) if i.endswith("1.png")
                            else logits(yes=-3.0), [YES], [NO])
    metrics = write_outputs(tmp_path, "val", exs, scores, {"git_sha": "abc"})
    rows = list(csv.reader((tmp_path / "preds_val.csv").open()))
    assert rows[0] == ["id", "label", "prob"] and rows[1][:2] == ["00001", "1"]
    assert metrics["auroc"] == 1.0 and metrics["n_refused"] == 0
    assert json.loads((tmp_path / "metrics.json").read_text())["git_sha"] == "abc"


def test_test_role_needs_final_and_a_listed_hash(tmp_path):
    cfg = run_config("qwen2.5-vl-7b", "{text}", "fhm", "nf4-4bit-fp16")
    final_configs = tmp_path / "FINAL_CONFIGS.md"
    with pytest.raises(PermissionError, match="--final"):
        check_final(cfg, False, final_configs)
    with pytest.raises(PermissionError, match="not in"):
        check_final(cfg, True, final_configs)
    final_configs.write_text(f"{config_hash(cfg)} 2026-10-08 #24\n")
    assert check_final(cfg, True, final_configs) == config_hash(cfg)


def test_config_identity_changes_with_prompt_and_dataset():
    base = config_hash(run_config("qwen2.5-vl-7b", "{text}", "fhm", "nf4-4bit-fp16"))
    assert base != config_hash(run_config("qwen2.5-vl-7b", "{text}!", "fhm", "nf4-4bit-fp16"))
    assert base != config_hash(run_config("qwen2.5-vl-7b", "{text}", "mami", "nf4-4bit-fp16"))
    assert base != config_hash(run_config("llava-1.6-7b", "{text}", "fhm", "nf4-4bit-fp16"))


def test_candidate_file_parsing(tmp_path):
    from pathlib import Path

    templates = load_templates(Path("configs/lmm_prompt_candidates.txt"))
    assert len(templates) == 3 and all("{text}" in x and "{task}" in x for x in templates)
    single = tmp_path / "p.txt"
    single.write_text("# frozen\nIs it {task}? {text}\n")
    assert load_templates(single) == ["Is it {task}? {text}"]
    single.write_text("a {text}\n---\nb {text}\n---\nc {text}\n---\nd {text}\n")
    with pytest.raises(ValueError, match="at most 3"):
        load_templates(single)


def test_val_then_test_share_metrics_and_the_val_threshold(tmp_path):
    from hmf.eval.metrics import binary_metrics, find_threshold

    exs = [example(i, i % 2) for i in range(1, 9)]
    fake = lambda image, prompt: logits(yes=float(int(image[-5]) % 3) - 1.0)  # noqa: E731
    scores = score_examples(exs, "{text}", fake, [YES], [NO])
    meta = {"config_hash": "h1", "dataset": "mami", "git_sha": "abc"}
    write_outputs(tmp_path, "val", exs, scores, meta)
    result = write_outputs(tmp_path, "test", exs, scores, {**meta, "git_sha": "def"})
    saved = json.loads((tmp_path / "metrics.json").read_text())
    probs = [float(r[2]) for r in list(csv.reader((tmp_path / "preds_val.csv").open()))[1:]]
    labels = [e.label for e in exs]
    threshold, _ = find_threshold(labels, probs, "macro_f1")
    assert saved["val"]["threshold"] == threshold and saved["val"]["threshold_metric"] == "macro_f1"
    assert saved["test"]["threshold"] == threshold  # reused, never re-chosen on test
    expected = binary_metrics(labels, probs, threshold)
    assert saved["test"]["macro_f1"] == expected["macro_f1"] == result["macro_f1"]
    assert (saved["val"]["git_sha"], saved["test"]["git_sha"]) == ("abc", "def")
    assert (tmp_path / "preds_test.csv").read_bytes().count(b"\r") == 0  # LF endings


def test_test_without_val_threshold_reports_auroc_only(tmp_path):
    exs = [example(1, 1), example(2, 0)]
    scores = score_examples(exs, "{text}", lambda i, p: logits(yes=1.0 if i.endswith("1.png")
                                                                else -1.0), [YES], [NO])
    result = write_outputs(tmp_path, "test", exs, scores, {"config_hash": "h", "dataset": "fhm"})
    assert result["auroc"] == 1.0 and "threshold" not in result


def test_metrics_of_another_config_are_not_overwritten(tmp_path):
    exs = [example(1, 1), example(2, 0)]
    scores = score_examples(exs, "{text}", lambda i, p: logits(), [YES], [NO])
    write_outputs(tmp_path, "val", exs, scores, {"config_hash": "h1", "dataset": "fhm"})
    with pytest.raises(ValueError, match="separate --out-dir"):
        write_outputs(tmp_path, "test", exs, scores, {"config_hash": "h2", "dataset": "fhm"})


def test_scored_ids_must_equal_the_committed_list(tmp_path):
    from hmf.lmm_zeroshot import check_split_ids

    (tmp_path / "mami").mkdir()
    (tmp_path / "mami" / "val.txt").write_text("00001\n00002\n")
    check_split_ids([example(2), example(1)], "mami", "val", tmp_path)  # order-free
    with pytest.raises(ValueError, match="1 extra, 0 missing"):
        check_split_ids([example(1), example(2), example(3)], "mami", "val", tmp_path)
    with pytest.raises(ValueError, match="0 extra, 1 missing"):
        check_split_ids([example(1)], "mami", "val", tmp_path)
    with pytest.raises(FileNotFoundError):
        check_split_ids([example(1)], "mami", "test", tmp_path)


@pytest.mark.parametrize("dataset", ["harm_c", "harm_p"])
def test_load_role_dispatches_harmeme(monkeypatch, dataset):
    import hmf.data.harmeme as harmeme
    from hmf.lmm_zeroshot import load_role

    calls = []
    monkeypatch.setattr(harmeme, "load_harmeme", lambda *a, **k: calls.append((a, k)) or [])
    load_role(dataset, "/ann:/img", "test", allow_test=True, max_samples=3)
    assert calls == [(("/ann", "/img", dataset, "test"), {"allow_test": True, "max_samples": 3})]


def test_load_role_dispatches_mami(monkeypatch):
    import hmf.data.mami as mami
    from hmf.lmm_zeroshot import load_role

    calls = []
    monkeypatch.setattr(mami, "load_mami", lambda *a, **k: calls.append((a, k)) or [])
    load_role("mami", "/root", "val", allow_test=False)
    assert calls == [(("/root", "val"), {"allow_test": False, "max_samples": 0})]


def test_every_lmm_dataset_has_a_loader():
    from hmf.lmm_zeroshot import TASK_PHRASES, load_role

    for dataset in TASK_PHRASES:
        with pytest.raises(Exception) as info:
            load_role(dataset, "/nonexistent:/nonexistent", "val", allow_test=False)
        assert "no loader" not in str(info.value)


def test_final_rows_cover_every_model_and_dataset():
    from hmf.lmm_zeroshot import MODELS, QUANTIZATION, TASK_PHRASES, final_rows

    rows = final_rows()
    assert len(rows) == len(MODELS) * len(TASK_PHRASES) == 10
    (template,) = load_templates("configs/lmm_prompt.txt")
    digest = config_hash(run_config("qwen2.5-vl-7b", template, "mami", QUANTIZATION))
    assert any(row.startswith(f"| {digest} | lmm_zeroshot qwen2.5-vl-7b mami") for row in rows)
