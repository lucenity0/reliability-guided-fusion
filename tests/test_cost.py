"""#31 cost profiler: FLOP accounting against A-6, and the report table."""

import json

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from hmf.config import load_config  # noqa: E402
from hmf.eval.cost import (  # noqa: E402
    closed_form,
    count_flops,
    latency_ms,
    profile_config,
    synthetic_batch,
)
from hmf.report import build  # noqa: E402


def test_attention_is_added_from_token_counts():
    mha = nn.MultiheadAttention(16, 4, batch_first=True).eval()
    q, c = torch.randn(1, 5, 16), torch.randn(1, 7, 16)
    flops = count_flops(mha, lambda: mha(q, c, c, need_weights=False))
    assert flops["attention"] == 4 * 5 * 7 * 16
    assert flops["total"] == flops["counted"] + flops["attention"]


def test_profiled_dynamic_flops_match_the_a6_ledger_within_ten_percent():
    cfg = load_config("configs/fhm/dynamic.yaml", environ={})
    from hmf.models.registry import build_model

    model = build_model(cfg).eval()
    batch = synthetic_batch(cfg, 1)
    total = count_flops(model, lambda: model(batch))["total"]
    assert abs(total - closed_form(cfg)) / closed_form(cfg) < 0.10


def test_counting_does_not_depend_on_batch_size():
    linear = nn.Linear(8, 4).eval()
    one = count_flops(linear, lambda: linear(torch.randn(1, 8)), 1)
    many = count_flops(linear, lambda: linear(torch.randn(16, 8)), 16)
    assert one["total"] == many["total"]


def test_latency_is_per_sample():
    assert latency_ms(lambda: None, 64, "cpu", repeats=3, warmup=0) >= 0


def test_profile_config_records_both_batch_sizes():
    cfg = load_config("configs/fhm/linear_probe.yaml", environ={})
    result = profile_config(cfg, "cpu", repeats=1)
    assert set(result["latency_ms"]) == {"1", "64"} and result["params"] > 0


def test_report_renders_the_cost_table(tmp_path):
    results = tmp_path / "results"
    (results / "cost").mkdir(parents=True)
    model = {"params": 1_000_000, "flops": {"total": 5_000_000},
             "latency_ms": {"1": 1.5, "64": 0.25}, "peak_memory_bytes": {"64": 2**20}}
    (results / "cost" / "cost.json").write_text(json.dumps(
        {"device": "cuda", "gpu_name": "T4", "models": {"dynamic": model}}))
    summary = build(results, tmp_path / "docs", with_figures=False)
    assert summary["cost"]["rows"]["dynamic"]["ms_64"] == 0.25
    text = (tmp_path / "docs/tables/cost.md").read_text()
    assert "Dynamic (ours)" in text and "T4" in text and "5.0" in text
