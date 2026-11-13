import numpy as np
import pytest

from hmf.eval.calibration import calibration


def test_binary_known_cases_and_empty_bins():
    perfect = calibration([0, 0, 1, 1], [0, 0, 1, 1])
    assert perfect["ece"] == 0
    assert len(perfect["bins"]) == 15
    assert perfect["bins"][0]["count"] == perfect["bins"][-1]["count"] == 2
    assert perfect["bins"][1]["mean_probability"] is None
    assert perfect["bins"][1]["observed_frequency"] is None
    assert calibration([0, 1], [1, 0])["ece"] == 1
    assert calibration([0, 1], [0.5, 0.5])["ece"] == 0
    # Low probabilities compare positive frequency, not classification accuracy.
    assert calibration([0, 0, 0, 1], [0.25] * 4)["ece"] == 0
    assert calibration([0, 0, 1, 1], [0.2, 0.2, 0.8, 0.8])["ece"] == pytest.approx(0.2)


def test_bin_boundaries_go_right_and_one_stays_in_last_bin():
    edges = np.linspace(0, 1, 16)
    result = calibration(np.zeros(16), edges)
    assert [b["count"] for b in result["bins"]] == [1] * 14 + [2]
    assert result["ece"] == pytest.approx(edges.mean())


def test_multiclass_top_label_with_argmax_tie():
    p = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.5, 0.5, 0]]
    result = calibration([0, 2, 0], p)
    assert result["kind"] == "top_label"
    assert result["ece"] == pytest.approx((2 * abs(0.8 - 0.5) + abs(0.5 - 1)) / 3)


@pytest.mark.parametrize("y,p", [([], []), ([0], [float("nan")]), ([0], [-0.1]),
                                  ([0], [1.1]), ([2], [0.5]), ([0], [[0.4, 0.4, 0.4]])])
def test_invalid_calibration_inputs(y, p):
    with pytest.raises(ValueError):
        calibration(y, p)


def test_float32_rounding_just_outside_unit_interval_is_accepted():
    # Food-101 dynamic seed 8 (#27): a 101-class mixture rounded a few ulps above 1.
    p = np.full((4, 101), 1 / 101, dtype=np.float64)
    p[0] = 0.0
    p[0, 3] = 1 + 3e-7
    result = calibration(np.array([3, 0, 1, 2]), p)
    assert 0 <= result["ece"] <= 1


def test_real_out_of_range_probabilities_still_raise():
    with pytest.raises(ValueError):
        calibration(np.array([1, 0]), np.array([1.01, 0.2]))
