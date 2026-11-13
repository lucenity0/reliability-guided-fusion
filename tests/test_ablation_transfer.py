"""#29 ablation switches and #28 cross-dataset transfer."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmf.config import config_hash, load_config  # noqa: E402
from hmf.models.registry import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def batch(b=4, d=768, p=5, length=3):
    return {"image_tokens": torch.randn(b, p, d), "text_tokens": torch.randn(b, length, d),
            "text_mask": torch.ones(b, length, dtype=torch.bool),
            "image_pooled": torch.randn(b, d), "text_pooled": torch.randn(b, d),
            "similarity": torch.rand(b), "label": torch.randint(0, 2, (b,))}


def cfg(**overrides):
    base = load_config(ROOT / "configs/fhm/dynamic.yaml", environ={})
    return {**base, **overrides}


def test_expert_subset_routes_over_the_kept_experts_only():
    model = build_model(cfg(experts="img,int")).eval()
    out = model(batch())
    assert model.expert_names == ("img", "int")
    assert out["pi"].shape == (4, 2) and out["z"].shape == (4, 2, 1)
    loss, parts = model.loss(model.train()(batch()), batch())
    assert torch.isfinite(loss)


@pytest.mark.parametrize("bad", ["", "img,img", "img,audio"])
def test_bad_expert_lists_are_rejected(bad):
    with pytest.raises(ValueError, match="experts"):
        build_model(cfg(experts=bad))


def test_dropped_router_inputs_have_no_effect_on_the_routing():
    torch.manual_seed(0)
    model = build_model(cfg(router_inputs="pooled,entropy")).eval()
    x = batch()
    y = {**x, "similarity": torch.rand(4) * 5}
    with torch.no_grad():
        assert torch.allclose(model(x)["pi"], model(y)["pi"])
    full = build_model(cfg()).eval()
    full.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert not torch.allclose(full(x)["pi"], full(y)["pi"])


def test_unknown_router_input_is_rejected():
    with pytest.raises(ValueError, match="router inputs"):
        build_model(cfg(router_inputs="pooled,audio"))


def test_committed_hashes_are_unchanged_and_ablations_differ():
    listed = (ROOT / "results/FINAL_CONFIGS.md").read_text()
    assert config_hash(load_config(ROOT / "configs/fhm/dynamic.yaml", environ={})) in listed
    for path in sorted((ROOT / "configs/fhm/ablations").glob("*.yaml")):
        ablation = load_config(path, environ={})
        assert ablation["method"].startswith("abl_")
        assert config_hash(ablation) not in listed  # val only: never cleared for test
        build_model(ablation)


def test_transfer_refuses_the_diagonal_and_unapproved_sources(tmp_path, monkeypatch):
    import hmf.eval.transfer as transfer
    from hmf.train import train
    from tests.test_train import make_split

    monkeypatch.setattr(transfer, "DATASETS", ("toy2", "toyb"))
    for name, seed in (("toy2", 0), ("toyb", 5)):
        ids = [f"{name}_{i:03d}" for i in range(80)]
        make_split(tmp_path / "features" / name / "v1" / "all", ids, np.arange(80) % 2, 2, seed)
        splits = tmp_path / "splits" / name
        splits.mkdir(parents=True)
        (splits / "train.txt").write_text("\n".join(ids[:50]) + "\n")
        (splits / "val.txt").write_text("\n".join(ids[50:65]) + "\n")
        (splits / "test.txt").write_text("\n".join(ids[65:]) + "\n")
    config = {"dataset": "toy2", "method": "dyn", "model": "toy_router", "cache_version": "v1",
              "num_classes": 2, "batch_size": 16, "num_epochs": 2, "lr": 0.05,
              "weight_decay": 0.0, "warmup_frac": 0.1, "patience": 5, "seed": 0,
              "deterministic": True, "max_samples": 0,
              "feature_dir": str(tmp_path / "features"),
              "results_dir": str(tmp_path / "results"), "checkpoint_dir": str(tmp_path / "ckpt"),
              "splits_dir": str(tmp_path / "splits")}
    train(config, device="cpu")
    run = tmp_path / "results/toy2/dyn/seed0"
    approvals = tmp_path / "FINAL_CONFIGS.md"
    approvals.write_text("")
    with pytest.raises(PermissionError):
        transfer.transfer(run, "toyb", checkpoint_dir=tmp_path / "ckpt",
                          final_configs=approvals, device="cpu")
    approvals.write_text(f"| {config_hash(config)} | toy |\n")
    with pytest.raises(ValueError, match="diagonal"):
        transfer.transfer(run, "toy2", checkpoint_dir=tmp_path / "ckpt",
                          final_configs=approvals, device="cpu")
    out = tmp_path / "transfer"
    n = transfer.run_all(tmp_path / "results", tmp_path / "ckpt", out, methods=("dyn",),
                         final_configs=approvals, device="cpu")
    assert n == 1
    result = json.loads((out / "toy2/dyn/seed0/toyb.json").read_text())
    assert result["n"] == 15 and 0 <= result["auroc"] <= 1
    assert transfer.run_all(tmp_path / "results", tmp_path / "ckpt", out, methods=("dyn",),
                            final_configs=approvals, device="cpu") == 0  # never rescored
