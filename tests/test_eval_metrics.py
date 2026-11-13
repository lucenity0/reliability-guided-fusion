import numpy as np
import pytest

from hmf.eval.metrics import binary_metrics, find_threshold, multiclass_metrics, threshold_metric


def test_threshold_metric_follows_protocol_rule_3():
    assert threshold_metric("fhm") == "accuracy"
    assert {threshold_metric(d) for d in ("mami", "harm_c", "harm_p", "crisismmd")} == {"macro_f1"}


def test_find_threshold_is_exact_on_separable_scores():
    labels = np.array([0, 0, 0, 1, 1])
    probs = np.array([0.05, 0.2, 0.31, 0.32, 0.9])
    t, value = find_threshold(labels, probs, "accuracy")
    assert (t, value) == (0.32, 1.0)  # the 0.1-0.9 grid in V3 could miss this gap
    assert binary_metrics(labels, probs, t)["accuracy"] == 1.0


def test_find_threshold_macro_f1_can_differ_from_accuracy():
    labels = np.array([0] * 8 + [1] * 2)
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.95, 0.65, 0.99])
    t_acc, _ = find_threshold(labels, probs, "accuracy")
    t_f1, f1 = find_threshold(labels, probs, "macro_f1")
    assert 0 < f1 <= 1 and t_acc >= t_f1


def test_find_threshold_validates_input():
    with pytest.raises(ValueError):
        find_threshold([0, 1], [0.2], "accuracy")
    with pytest.raises(ValueError):
        find_threshold([0, 1], [0.2, 0.8], "auroc")


def test_binary_metrics_reports_auroc_without_threshold():
    out = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.6, 0.4], 0.5)
    assert out["auroc"] == 0.75 and out["accuracy"] == 0.5
    assert binary_metrics([1, 1], [0.2, 0.8], 0.5)["auroc"] is None


def test_multiclass_argmax_ties_and_absent_classes():
    probs = np.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3]])
    out = multiclass_metrics([0, 2, 1], probs)  # row 0 ties -> class 0
    assert out["accuracy"] == 1.0 and out["macro_f1"] == 1.0
    out = multiclass_metrics([0, 0, 0], probs[:, [0, 1, 2]])
    assert out["macro_f1"] == pytest.approx((0.5 + 0 + 0) / 3)  # all C classes count


def test_git_sha_prefers_the_runner_env_on_a_vm(monkeypatch, tmp_path):
    from hmf.eval.guard import git_sha

    monkeypatch.setenv("HMF_GIT_SHA", "abc1234")
    assert git_sha() == "abc1234"
    monkeypatch.delenv("HMF_GIT_SHA")
    monkeypatch.chdir(tmp_path)  # not a git checkout, no env: unknown
    assert git_sha() == "unknown"
