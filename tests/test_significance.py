import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from hmf.eval.significance import (
    _threshold_metrics,
    auroc,
    delong_test,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap,
    paired_bootstrap_accuracy,
    paired_seed_ttest,
    paired_seed_wilcoxon,
    summarize_seed_pvalues,
)


def synthetic(n=400, gap=1.0, seed=0, ties=False):
    """Labels plus a strong model A and a weak model B sharing per-sample noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    y[:2] = [0, 1]
    shared = rng.normal(size=n)
    a = (1.0 + gap) * y + shared + 0.5 * rng.normal(size=n)
    b = 1.0 * y + shared + 0.5 * rng.normal(size=n)
    if ties:
        a, b = np.round(a, 1), np.round(b, 1)
    return y, a, b


def slow_delong(y, a, b):
    """Direct O(m n) DeLong from the placement-value definition, as a reference."""
    pos, neg = y == 1, y == 0

    def components(s):
        psi = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
        return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)

    auc_a, v01a, v10a = components(a)
    auc_b, v01b, v10b = components(b)
    var = np.var(v01a - v01b, ddof=1) / pos.sum() + np.var(v10a - v10b, ddof=1) / neg.sum()
    return auc_a - auc_b, var


@pytest.mark.parametrize("ties", [False, True])
def test_auroc_matches_sklearn(ties):
    y, a, _ = synthetic(ties=ties)
    assert auroc(y, a) == pytest.approx(roc_auc_score(y, a), abs=1e-12)


@pytest.mark.parametrize("ties", [False, True])
def test_fast_delong_matches_direct_definition(ties):
    y, a, b = synthetic(n=300, gap=0.3, seed=3, ties=ties)
    res = delong_test(y, a, b)
    delta, var = slow_delong(y, a, b)
    assert res["delta"] == pytest.approx(delta, abs=1e-12)
    assert res["se"] ** 2 == pytest.approx(var, rel=1e-9)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, a), abs=1e-12)
    assert res["auc_b"] == pytest.approx(roc_auc_score(y, b), abs=1e-12)


def test_identical_predictors_give_p_one():
    y, a, _ = synthetic()
    assert delong_test(y, a, a)["p_value"] == 1.0
    assert paired_bootstrap(y, a, a, n_boot=200)["auroc"]["p_value"] == 1.0
    preds = (a > 0.5).astype(int)
    assert mcnemar_test(y, preds, preds)["p_value"] == 1.0
    per_seed = np.array([0.70, 0.72, 0.71, 0.69, 0.73])
    assert paired_seed_ttest(per_seed, per_seed)["p_value"] == 1.0
    assert paired_seed_wilcoxon(per_seed, per_seed)["p_value"] == 1.0


def test_clearly_better_predictor_is_significant():
    y, a, b = synthetic(gap=1.5)
    assert delong_test(y, a, b)["p_value"] < 0.01
    assert paired_bootstrap(y, a, b, n_boot=2000)["auroc"]["p_value"] < 0.01
    assert mcnemar_test(y, (a > 1.25).astype(int), (b > 0.5).astype(int))["p_value"] < 0.01
    better = np.array([0.80, 0.82, 0.81, 0.83, 0.80, 0.82, 0.81, 0.84, 0.80, 0.82])
    worse = better - np.array([0.05, 0.06, 0.04, 0.05, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06])
    assert paired_seed_ttest(better, worse)["p_value"] < 0.01


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_delong_and_bootstrap_agree_in_sign_and_magnitude(seed):
    y, a, b = synthetic(n=500, gap=0.25, seed=seed)
    dl = delong_test(y, a, b)
    bs = paired_bootstrap(y, a, b, n_boot=3000, seed=seed)["auroc"]
    assert np.sign(dl["delta"]) == np.sign(bs["delta"])
    assert dl["delta"] == pytest.approx(bs["delta"], abs=1e-12)
    # Both CIs estimate the same sampling spread: widths within 25 %.
    assert (dl["delta_hi"] - dl["delta_lo"]) == pytest.approx(
        bs["delta_hi"] - bs["delta_lo"], rel=0.25
    )


def test_bootstrap_is_reproducible_and_reports_threshold_metrics():
    y, a, b = synthetic(n=200)
    first = paired_bootstrap(y, a, b, threshold_a=1.0, threshold_b=0.5, n_boot=300, seed=7)
    second = paired_bootstrap(y, a, b, threshold_a=1.0, threshold_b=0.5, n_boot=300, seed=7)
    assert first == second
    assert set(first) >= {"auroc", "accuracy", "macro_f1"}
    for name in ("auroc", "accuracy", "macro_f1"):
        r = first[name]
        assert r["delta_lo"] <= r["delta_hi"]
        assert r["a_lo"] <= r["a_hi"] and r["b_lo"] <= r["b_hi"]
    with pytest.raises(ValueError, match="both thresholds"):
        paired_bootstrap(y, a, b, threshold_a=0.5, n_boot=10)


@pytest.mark.parametrize("threshold", [-1.0, 0.5, 1.0, 9.0])
def test_threshold_metrics_match_sklearn(threshold):
    y, a, _ = synthetic(n=150, seed=4)
    preds = (a >= threshold).astype(int)
    got = _threshold_metrics(y, a, threshold)
    assert got["accuracy"] == pytest.approx(np.mean(preds == y))
    assert got["macro_f1"] == pytest.approx(f1_score(y, preds, average="macro", zero_division=0))


def test_mcnemar_branches():
    y = np.ones(100, dtype=int)
    a = y.copy()
    b = y.copy()
    b[:10] = 0  # 10 discordant pairs, all favouring A
    exact = mcnemar_test(y, a, b)
    assert exact["method"] == "exact binomial"
    assert exact["p_value"] == pytest.approx(stats.binomtest(0, 10, 0.5).pvalue)
    b[:40] = 0
    a[:5] = 0  # 35 A-right/B-wrong (indices 5..39), 0 the other way
    chi = mcnemar_test(y, a, b)
    assert chi["method"].startswith("chi-square")
    assert (chi["n01"], chi["n10"]) == (35, 0)
    assert chi["statistic"] == pytest.approx((35 - 1) ** 2 / 35)


def test_seed_ttest_matches_scipy_and_handles_constant_gain():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.01, size=10)
    b = a - rng.normal(0.01, 0.01, size=10)
    res = paired_seed_ttest(a, b)
    ref = stats.ttest_rel(a, b)
    assert res["p_value"] == pytest.approx(ref.pvalue)
    assert res["t"] == pytest.approx(ref.statistic)
    assert res["delta_lo"] < res["delta"] < res["delta_hi"]
    assert paired_seed_ttest(b + 0.02, b)["p_value"] == 0.0
    with pytest.raises(ValueError):
        paired_seed_ttest([0.7], [0.6])


def test_wilcoxon_minimum_p_at_five_seeds():
    a = np.array([0.80, 0.81, 0.82, 0.83, 0.84])
    res = paired_seed_wilcoxon(a, a - 0.01 * np.arange(1, 6))
    assert res["p_value"] == pytest.approx(0.0625)


def test_holm_bonferroni_reference_example():
    raw = {"fhm": 0.01, "mami": 0.04, "harm_c": 0.03, "harm_p": 0.005, "crisis": 0.20}
    res = holm_bonferroni(raw)
    # sorted: 0.005*5, 0.01*4, 0.03*3, 0.04*2 -> monotone max, then 0.20*1
    expected = {"harm_p": 0.025, "fhm": 0.04, "harm_c": 0.09, "mami": 0.09, "crisis": 0.20}
    for name, value in expected.items():
        assert res["adjusted"][name] == pytest.approx(value)
    assert res["reject"] == {"fhm": True, "mami": False, "harm_c": False, "harm_p": True,
                             "crisis": False}
    assert holm_bonferroni([0.9, 0.8])["adjusted"] == pytest.approx([1.0, 1.0])


def test_summarize_seed_pvalues():
    res = summarize_seed_pvalues([0.01, 0.2, 0.03, 0.04, 0.5])
    assert res == {"n_seeds": 5, "median_p": 0.04, "frac_significant": 0.6}


@pytest.mark.parametrize(
    "labels, scores",
    [([0, 1, 2], [0.1, 0.2, 0.3]), ([1, 1, 1], [0.1, 0.2, 0.3]), ([0, 1], [0.1, np.nan])],
)
def test_invalid_inputs_raise(labels, scores):
    with pytest.raises(ValueError):
        delong_test(np.array(labels), np.array(scores), np.array(scores))


# ── Multi-class: paired accuracy bootstrap and McNemar correctness ───────────


def multiclass(n_classes, n=600, acc_a=0.8, acc_b=0.6, seed=0):
    """Class labels plus predictions from A and B with correlated per-sample correctness.

    Wrong predictions are a uniformly random wrong class, so A and B can both be
    wrong with different labels.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    shared = rng.uniform(size=n)

    def predict(acc):
        right = shared + 0.1 * rng.normal(size=n) < acc
        wrong = (y + rng.integers(1, n_classes, size=n)) % n_classes
        return np.where(right, y, wrong)

    return y, predict(acc_a), predict(acc_b)


@pytest.mark.parametrize("n_classes", [3, 101])
def test_accuracy_bootstrap_identical_predictors(n_classes):
    y, a, _ = multiclass(n_classes)
    res = paired_bootstrap_accuracy(y, a, a, n_boot=500)
    assert res["delta"] == 0.0
    assert res["delta_lo"] == res["delta_hi"] == 0.0
    assert res["p_value"] == 1.0
    assert res["acc_a"] == res["acc_b"] == pytest.approx(np.mean(a == y))


@pytest.mark.parametrize("n_classes", [3, 101])
def test_accuracy_bootstrap_clear_improvement(n_classes):
    y, a, b = multiclass(n_classes)
    res = paired_bootstrap_accuracy(y, a, b, n_boot=2000, seed=1)
    assert res["acc_a"] == pytest.approx(np.mean(a == y))
    assert res["acc_b"] == pytest.approx(np.mean(b == y))
    assert res["delta"] == pytest.approx(res["acc_a"] - res["acc_b"])
    assert res["delta"] > 0.1
    assert 0 < res["delta_lo"] <= res["delta"] <= res["delta_hi"]
    assert res["p_value"] < 0.01
    # Swapping A and B flips the sign and keeps the p-value.
    swapped = paired_bootstrap_accuracy(y, b, a, n_boot=2000, seed=1)
    assert swapped["delta"] == pytest.approx(-res["delta"])
    assert swapped["p_value"] == res["p_value"]


@pytest.mark.parametrize("n_classes", [3, 101])
def test_accuracy_bootstrap_resamples_pairs(n_classes):
    rng = np.random.default_rng(5)
    n = 500
    y = rng.integers(0, n_classes, size=n)
    wrong = (y + 1) % n_classes
    # About 50 % accuracy for both; A differs from B only on 10 samples, where A is right.
    b = np.where(rng.uniform(size=n) < 0.5, y, wrong)
    a = b.copy()
    flip = np.flatnonzero(b != y)[:10]
    a[flip] = y[flip]
    res = paired_bootstrap_accuracy(y, a, b, n_boot=2000, seed=3)
    # Paired resampling: every replicate delta is >= 0. Resampling the models
    # independently would give a CI spanning roughly +/- 0.06 around 0.02.
    assert res["delta"] == pytest.approx(10 / n)
    assert res["delta_lo"] >= 0.0
    # Same result as a direct reference loop over the same seeded index draws.
    ref_rng = np.random.default_rng(3)
    ref = np.array([
        np.mean((a[idx] == y[idx]).astype(float) - (b[idx] == y[idx]))
        for idx in (ref_rng.integers(0, n, size=n) for _ in range(2000))
    ])
    assert res["delta_lo"] == pytest.approx(np.percentile(ref, 2.5))
    assert res["delta_hi"] == pytest.approx(np.percentile(ref, 97.5))
    p_ref = min(1.0, 2 * min(np.mean(ref <= 0), np.mean(ref >= 0)))
    assert res["p_value"] == pytest.approx(p_ref)


@pytest.mark.parametrize("n_classes", [3, 101])
def test_accuracy_bootstrap_is_reproducible(n_classes):
    y, a, b = multiclass(n_classes, acc_a=0.62, acc_b=0.6)
    first = paired_bootstrap_accuracy(y, a, b, n_boot=500, seed=11)
    second = paired_bootstrap_accuracy(y, a, b, n_boot=500, seed=11)
    other = paired_bootstrap_accuracy(y, a, b, n_boot=500, seed=12)
    assert first == second
    assert (first["delta_lo"], first["delta_hi"]) != (other["delta_lo"], other["delta_hi"])
    assert first["delta"] == other["delta"]


@pytest.mark.parametrize("n_classes", [3, 101])
def test_accuracy_bootstrap_all_correct_and_all_wrong(n_classes):
    rng = np.random.default_rng(0)
    y = rng.integers(0, n_classes, size=200)
    right = y.copy()
    wrong_1 = (y + 1) % n_classes
    wrong_2 = (y + 2) % n_classes  # a different wrong class (equals y when C = 2 only)

    both_right = paired_bootstrap_accuracy(y, right, right, n_boot=200)
    assert (both_right["acc_a"], both_right["delta"], both_right["p_value"]) == (1.0, 0.0, 1.0)

    both_wrong = paired_bootstrap_accuracy(y, wrong_1, wrong_2, n_boot=200)
    assert (both_wrong["acc_a"], both_wrong["acc_b"]) == (0.0, 0.0)
    assert (both_wrong["delta"], both_wrong["p_value"]) == (0.0, 1.0)

    right_vs_wrong = paired_bootstrap_accuracy(y, right, wrong_1, n_boot=200)
    assert right_vs_wrong["delta"] == 1.0
    assert right_vs_wrong["delta_lo"] == right_vs_wrong["delta_hi"] == 1.0
    assert right_vs_wrong["p_value"] == 0.0


@pytest.mark.parametrize(
    "labels, preds_a, preds_b, kwargs",
    [
        ([0, 1, 2], [0, 1], [0, 1, 2], {}),  # length mismatch
        ([[0, 1], [2, 0]], [[0, 1], [2, 0]], [[0, 1], [2, 0]], {}),  # 2-D
        ([], [], [], {}),  # empty
        ([0, -1, 2], [0, 1, 2], [0, 1, 2], {}),  # negative label
        ([0, 1, 2], [0, 1.5, 2], [0, 1, 2], {}),  # non-integer prediction
        ([0, 1, 2], [0, 1, 2], [0, np.nan, 2], {}),  # NaN prediction
        (["a", "b", "c"], [0, 1, 2], [0, 1, 2], {}),  # non-numeric labels
        ([0, 1, 2], [0, 1, 2], [0, 1, 2], {"n_boot": 0}),
        ([0, 1, 2], [0, 1, 2], [0, 1, 2], {"ci": 1.0}),
    ],
)
def test_accuracy_bootstrap_invalid_inputs_raise(labels, preds_a, preds_b, kwargs):
    with pytest.raises(ValueError):
        paired_bootstrap_accuracy(np.array(labels), np.array(preds_a), np.array(preds_b),
                                  **kwargs)


@pytest.mark.parametrize("n_classes", [3, 101])
def test_mcnemar_multiclass_counts_correctness(n_classes):
    y, a, b = multiclass(n_classes, n=400)
    res = mcnemar_test(y, a, b)
    correct_a, correct_b = a == y, b == y
    assert res["n01"] == int(np.sum(correct_a & ~correct_b))
    assert res["n10"] == int(np.sum(~correct_a & correct_b))
    assert res["p_value"] < 0.01
    # Both wrong with different wrong classes is concordant, not discordant.
    wrong_1, wrong_2 = (y + 1) % n_classes, (y + 2) % n_classes
    assert mcnemar_test(y, wrong_1, wrong_2) == {
        "n01": 0, "n10": 0, "method": "degenerate", "statistic": None, "p_value": 1.0
    }
