import csv

import numpy as np
import pytest

from hmf.eval.router_evidence import router_evidence


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return path


def test_binary_targets_include_negative_labels_and_temperature(tmp_path):
    path = write(tmp_path / "preds.csv", [
        ["id", "label", "p_img", "p_txt", "pi_img", "pi_txt"],
        ["a", 1, 0.8, 0.2, 0.8, 0.2], ["b", 0, 0.6, 0.4, 0.4, 0.6],
        ["c", 0, 0.5, 0.5, 0.5, 0.5]])
    result = router_evidence(path, ["img", "txt"], num_classes=2, tau=1)
    np.testing.assert_allclose(result["target"], [[0.8, 0.2], [0.4, 0.6], [0.5, 0.5]])
    np.testing.assert_allclose(result["pearson"], [1, 1])
    assert result["argmax_agreement"] == 1
    sharp = router_evidence(path, ["img", "txt"], num_classes=2, tau=0.5)
    np.testing.assert_allclose(sharp["target"][0], np.array([0.64, 0.04]) / 0.68)


def test_categorical_targets_and_constant_router(tmp_path):
    path = write(tmp_path / "preds.csv", [
        ["id", "label", "p_img_0", "p_img_1", "p_img_2", "p_txt_0", "p_txt_1", "p_txt_2",
         "pi_img", "pi_txt"],
        ["a", 2, 0.1, 0.1, 0.8, 0.4, 0.4, 0.2, 0.5, 0.5],
        ["b", 1, 0.5, 0.2, 0.3, 0.1, 0.6, 0.3, 0.5, 0.5]])
    result = router_evidence(path, ["img", "txt"], num_classes=3, tau=1)
    np.testing.assert_allclose(result["target"], [[0.8, 0.2], [0.25, 0.75]])
    assert result["pearson"] == [None, None]
    assert result["argmax_agreement"] == 0.5


def test_saturated_probabilities_are_finite_and_no_router_is_supported(tmp_path):
    path = write(tmp_path / "preds.csv", [["id", "label", "p_img", "p_txt"],
                                         ["a", 0, 1, 1], ["b", 1, 1, 0]])
    result = router_evidence(path, ["img", "txt"], num_classes=2, tau=1e-300)
    np.testing.assert_allclose(result["target"], [[0.5, 0.5], [1, 0]])
    assert "pearson" not in result
    with pytest.raises(ValueError, match="tau"):
        router_evidence(path, ["img", "txt"], num_classes=2, tau=0)
