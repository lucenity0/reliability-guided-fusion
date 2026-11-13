"""hmf.report regenerates tables from a results tree (#32)."""

import csv
import json

import numpy as np
import pytest

from hmf.report import build, compare, load_runs, mixed_quality_router


def write_run(root, dataset, method, seed, test, *, corruptions=None, preds=None, ece=0.1):
    run = root / dataset / method / f"seed{seed}"
    run.mkdir(parents=True)
    block = {**test, "calibration": {"ece": ece}}
    if corruptions:
        block["corruptions"] = corruptions
    history = [{"epoch": e, "val": {"auroc": 0.5 + 0.1 * e, "accuracy": 0.5}} for e in (1, 2)]
    (run / "metrics.json").write_text(json.dumps(
        {"dataset": dataset, "method": method, "seed": seed, "best_epoch": 2,
         "history": history, "test": block}))
    if preds:
        with (run / "preds_test.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(preds[0]))
            writer.writeheader()
            writer.writerows(preds)


@pytest.fixture
def results(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "results"
    for seed in range(5):
        for dataset in ("fhm", "harm_c"):
            base = 0.7 + 0.01 * rng.standard_normal()
            write_run(root, dataset, "static", seed, {"auroc": base},
                      corruptions={"drop-image": {"auroc": base - 0.1}})
            write_run(root, dataset, "dynamic", seed, {"auroc": base + 0.02},
                      corruptions={"drop-image": {"auroc": base - 0.05}},
                      preds=[{"id": "a", "label": 1, "prob": 0.7, "pi_img": 0.5,
                              "pi_txt": 0.2, "pi_int": 0.3}])
        write_run(root, "mvsa", "static", seed, {"accuracy": 0.8})
        write_run(root, "mvsa", "dynamic", seed, {"accuracy": 0.8 + 0.001 * seed})
        write_run(root, "mami", "dynamic_mq", seed, {"auroc": 0.8}, preds=[
            {"id": "a", "label": 1, "prob": 0.7, "pi_img": 0.1, "pi_txt": 0.6,
             "pi_int": 0.3, "quality": "image_missing"},
            {"id": "b", "label": 0, "prob": 0.2, "pi_img": 0.4, "pi_txt": 0.3,
             "pi_int": 0.3, "quality": "intact"}])
    # A trial directory that is not a seed run is ignored.
    (root / "fhm" / "sweeps").mkdir()
    return root


def test_compare_is_paired_by_seed_and_holm_over_the_family(results):
    runs = load_runs(results)
    result = compare(runs, "dynamic", "static")
    assert set(result["rows"]) == {"fhm", "harm_c", "mvsa"}
    fhm = result["rows"]["fhm"]
    assert fhm["delta"] == pytest.approx(0.02) and fhm["wins"] == 5
    assert result["rows"]["mvsa"]["metric"] == "accuracy"
    assert not result["family_complete"]  # three of the six datasets
    raw = sorted(r["p_value"] for r in result["rows"].values())
    assert max(r["p_holm"] for r in result["rows"].values()) >= raw[-1]


def test_mixed_quality_router_pools_seeds_by_group(results):
    router = mixed_quality_router(load_runs(results))
    assert router["mami"]["image_missing"] == {"n": 5, "img": pytest.approx(0.1),
                                              "txt": pytest.approx(0.6),
                                              "int": pytest.approx(0.3)}
    assert router["mami"]["intact"]["n"] == 5


def test_build_writes_every_table(results, tmp_path):
    out = tmp_path / "docs"
    summary = build(results, out, with_figures=False)
    for name in ("main", "significance", "robustness", "calibration", "mixed_quality_router"):
        assert (out / "tables" / f"{name}.tex").exists()
        assert (out / "tables" / f"{name}.md").exists()
    main = (out / "tables" / "main.md").read_text()
    assert "Dynamic (ours)" in main and "0.802 ± 0.002" in main  # MVSA accuracy column
    rob = summary["robustness"]["fhm"]["drop-image"]
    assert rob["delta"] == pytest.approx(0.05)
    tex = (out / "tables" / "significance.tex").read_text()
    assert r"\begin{tabular}" in tex and "Harm-C" in tex
    assert json.loads((out / "tables" / "summary.json").read_text())["main"]


def test_figures_are_written_when_matplotlib_is_available(results, tmp_path):
    pytest.importorskip("matplotlib")
    summary = build(results, tmp_path / "docs")
    assert set(summary["figures"]) == {"pi_distributions", "val_curves", "mq_router_evidence"}
    assert (tmp_path / "docs" / "figures" / "pi_distributions.pdf").exists()


def test_logged_tests_fill_only_missing_blocks(tmp_path):
    from hmf.report import apply_logged_tests

    root = tmp_path / "results"
    write_run(root, "food101", "static", 0, {"accuracy": 0.9})
    run = root / "food101" / "static" / "seed1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"seed": 1, "val": {"accuracy": 0.95}}))
    (root / "food101" / "first_pass_log.json").write_text(json.dumps({"runs": {
        "food101/static/seed0": {"accuracy": 0.1, "ece": 0.1},
        "food101/static/seed1": {"accuracy": 0.95, "ece": 0.08}}}))
    runs = load_runs(root)
    assert apply_logged_tests(runs, root) == 1
    assert runs[("food101", "static")][0]["test"]["accuracy"] == 0.9  # never overwritten
    assert runs[("food101", "static")][1]["test"]["source"] == "first_pass_log"


def test_ablation_and_transfer_tables(tmp_path):
    from hmf.report import ablation_table, render_transfer, transfer_table

    root = tmp_path / "results"
    for seed in range(3):
        for name, value in (("abl_full", 0.80), ("abl_no_int", 0.79)):
            run = root / "fhm" / name / f"seed{seed}"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({"val": {"auroc": value + seed / 100}}))
        target = root / "transfer" / "fhm" / "dynamic" / f"seed{seed}"
        target.mkdir(parents=True)
        (target / "mami.json").write_text(json.dumps(
            {"source": "fhm", "target": "mami", "method": "dynamic", "auroc": 0.6}))
    runs = load_runs(root)
    table = ablation_table(runs)
    assert table["abl_no_int"]["delta"] == pytest.approx(-0.01)
    transfer = transfer_table(root, runs)
    assert transfer["fhm->mami"]["dynamic"] == {"mean": 0.6, "n": 3}
    assert "FHM → MAMI" in render_transfer(transfer)[0]
