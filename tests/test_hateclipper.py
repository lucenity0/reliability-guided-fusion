from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from hmf.config import load_config  # noqa: E402
from hmf.features.cache import write_cache  # noqa: E402
from hmf.models.hateclipper import HateCLIPper, num_parameters  # noqa: E402
from hmf.models.registry import build_model  # noqa: E402
from hmf.train import train  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "fhm"


def small(**overrides):
    cfg = {"model": "hateclipper", "cache_version": "hc224", "image_dim": 12, "text_dim": 8,
           "map_dim": 4, "num_mapping_layers": 1, "fusion": "align",
           "num_pre_output_layers": 3, "pre_output_dim": 4, "drop_map": 0.2,
           "drop_fusion": 0.4, "drop_pre_output": 0.1, "num_classes": 2}
    cfg.update(overrides)
    return cfg


def batch(n=5, image_dim=12, text_dim=8, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    return {"image_pooled": torch.randn(n, image_dim, generator=g, dtype=dtype),
            "text_pooled": torch.randn(n, text_dim, generator=g, dtype=dtype),
            "label": torch.tensor([0, 1, 1, 0, 1][:n])}


@pytest.mark.parametrize("fusion, layers", [("align", 3), ("cross", 1)])
def test_output_shapes_and_registry(fusion, layers):
    model = build_model(small(fusion=fusion, num_pre_output_layers=layers)).eval()
    assert isinstance(model, HateCLIPper)
    out = model(batch())
    assert out["logits"].shape == (5,) and out["prob"].shape == (5,) and "pi" not in out
    torch.testing.assert_close(out["prob"], out["logits"].sigmoid())
    loss, parts = model.loss(out, batch())
    assert loss.ndim == 0 and set(parts) == {"main"}


def test_layer_order_follows_the_official_engine():
    # engine.py@86ad7f1 L80-84 (maps) and L103-109 (pre-output)
    model = HateCLIPper(small(num_mapping_layers=2, num_pre_output_layers=2))
    kinds = [type(m) for m in model.image_map]
    assert kinds == [nn.Linear, nn.Dropout, nn.ReLU, nn.Linear, nn.Dropout]
    assert [m.p for m in model.image_map if isinstance(m, nn.Dropout)] == [0.2, 0.2]
    assert model.image_map[0].in_features == 12 and model.text_map[0].in_features == 8
    kinds = [type(m) for m in model.pre_output]
    assert kinds == [nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout]
    assert [m.p for m in model.pre_output if isinstance(m, nn.Dropout)] == [0.4, 0.1, 0.1]
    assert model.output.out_features == 1


def test_fusions_match_their_definitions():
    for fusion in ("align", "cross"):
        model = HateCLIPper(small(fusion=fusion, num_pre_output_layers=0)).eval()
        b = batch()
        img = nn.functional.normalize(model.image_map(b["image_pooled"]), dim=1)
        txt = nn.functional.normalize(model.text_map(b["text_pooled"]), dim=1)
        fused = img * txt if fusion == "align" else torch.einsum("bi,bj->bij", img, txt).flatten(1)
        assert model.output.in_features == fused.shape[1] == (4 if fusion == "align" else 16)
        torch.testing.assert_close(model(b)["logits"], model.output(fused).squeeze(-1))


def test_parameter_counts():
    align = load_config(CONFIGS / "hateclipper_align.yaml", environ={})
    model = build_model(align)
    built = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # maps 1024·1024+1024 and 768·1024+1024, 3 × (1024·1024+1024), output 1024+1
    assert built == num_parameters(align) == 4_986_881  # paper Table 1: about 5M
    cross = load_config(CONFIGS / "hateclipper_cross.yaml", environ={})
    # the 1024²-d outer product into a 1024-wide layer; analytic only (about 4.3 GB of fp32)
    assert num_parameters(cross) == 1_837_056 + (1024**2 * 1024 + 1024) + 1025 == 1_075_580_929
    for cfg in (small(), small(fusion="cross", num_pre_output_layers=1, num_mapping_layers=2)):
        model = HateCLIPper(cfg)
        assert num_parameters(cfg) == sum(p.numel() for p in model.parameters())


def test_align_gradients():
    # 32 wide: at width 4 a whole ReLU layer can be dead at init (align features are small),
    # which stops gradient below it without any bug.
    torch.manual_seed(0)
    model = HateCLIPper(small(map_dim=32, pre_output_dim=32, drop_map=0.0, drop_fusion=0.0,
                              drop_pre_output=0.0)).double()
    b = batch(dtype=torch.float64)
    inputs = (b["image_pooled"].requires_grad_(), b["text_pooled"].requires_grad_())
    assert torch.autograd.gradcheck(
        lambda v, u: model({"image_pooled": v, "text_pooled": u})["logits"], inputs)
    loss, _ = model.loss(model(b), b)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("change, error, message", [
    ({"cache_version": "v1"}, ValueError, "hc224"),
    ({"num_classes": 3}, ValueError, "binary only"),
    ({"fusion": "concat"}, ValueError, "fusion must be"),
    ({"num_mapping_layers": 0}, ValueError, "at least 1"),
])
def test_config_errors(change, error, message):
    with pytest.raises(error, match=message):
        HateCLIPper(small(**change))


def test_missing_model_key_is_named():
    cfg = small()
    del cfg["drop_fusion"]
    with pytest.raises(ValueError, match="drop_fusion"):
        HateCLIPper(cfg)


def test_configs_hold_the_published_settings():
    for name, fusion, layers in (("align", "align", 3), ("cross", "cross", 1)):
        cfg = load_config(CONFIGS / f"hateclipper_{name}.yaml", environ={})
        assert (cfg["fusion"], cfg["num_mapping_layers"], cfg["num_pre_output_layers"]) == (
            fusion, 1, layers)
        assert (cfg["map_dim"], cfg["pre_output_dim"], cfg["batch_size"]) == (1024, 1024, 64)
        assert (cfg["lr"], cfg["weight_decay"], cfg["grad_clip"]) == (1e-4, 1e-4, 0.1)
        assert (cfg["num_epochs"], cfg["lr_schedule"], cfg["warmup_frac"]) == (20, "constant", 0)
        assert cfg["patience"] >= cfg["num_epochs"] and cfg["cache_version"] == "hc224"
        assert (cfg["drop_map"], cfg["drop_fusion"], cfg["drop_pre_output"]) == (0.2, 0.4, 0.1)


def test_trains_end_to_end_on_a_pooled_cache(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "features" / "toyhc" / "hc224" / "train"
    n, ids = 40, [f"s{i:02d}" for i in range(40)]
    labels = np.arange(n) % 2
    image = rng.normal(size=(n, 1, 12))
    image[:, 0, 0] += 3.0 * labels
    text = rng.normal(size=(n, 1, 8))
    write_cache(root, ids=ids, image_tokens=image, text_tokens=text,
                text_mask=np.ones((n, 1), bool), text_pooled=text[:, 0],
                similarity=np.zeros(n), labels=labels,
                meta={"backbone": "toy", "cache_version": "hc224", "git_sha": "test"})
    splits = tmp_path / "splits" / "toyhc"
    splits.mkdir(parents=True)
    (splits / "train.txt").write_text("\n".join(ids[:30]) + "\n")
    (splits / "val.txt").write_text("\n".join(ids[30:]) + "\n")
    cfg = {**small(), "dataset": "toyhc", "method": "hateclipper_align", "batch_size": 8,
           "num_epochs": 3, "lr": 1e-3, "weight_decay": 1e-4, "lr_schedule": "constant",
           "warmup_frac": 0.0, "grad_clip": 0.1, "patience": 20, "seed": 0,
           "deterministic": True, "max_samples": 0, "feature_dir": str(tmp_path / "features"),
           "results_dir": str(tmp_path / "results"), "checkpoint_dir": str(tmp_path / "ckpt"),
           "splits_dir": str(tmp_path / "splits")}
    metrics = train(cfg, device="cpu")
    assert metrics["epochs_run"] == 3 and metrics["router"] is None
    assert {h["lr"] for h in metrics["history"]} == {1e-3}
