import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from hmf.features.cache import write_cache  # noqa: E402
from hmf.models.baselines_v4 import UNIMODAL_KEYS, V3_KEYS  # noqa: E402
from hmf.models.registry import build_model  # noqa: E402
from hmf.train import train  # noqa: E402

D = 8


def config(model, num_classes=2, modality="img"):
    common = {"model": model, "feature_dim": D, "num_classes": num_classes}
    if model == "unimodal":
        return {**common, "modality": modality, "hidden_dim": 8, "dropout": 0.0}
    return {**common, "gate_query_dim": 4, "gate_hidden_dim_1": 12, "gate_hidden_dim_2": 6,
            "gate_dropout_1": 0.0, "gate_dropout_2": 0.0,
            "classifier_hidden_dim_1": 12, "classifier_hidden_dim_2": 8,
            "classifier_hidden_dim_3": 4, "classifier_dropout_1": 0.0,
            "classifier_dropout_2": 0.0, "classifier_dropout_3": 0.0, "lambda_aux": 0.3,
            "lambda_entropy": 0.2, "lambda_diversity": 0.4,
            "normalize_eps": 1e-12, "entropy_eps": 1e-8, "alpha_entropy_eps": 1e-7}


def batch(num_classes=2, size=6):
    rng = torch.Generator().manual_seed(7)
    return {"image_tokens": torch.randn(size, 5, D, generator=rng),
            "text_tokens": torch.randn(size, 3, D, generator=rng),
            "text_mask": torch.tensor([[True, True, False]]).expand(size, -1),
            "image_pooled": torch.randn(size, D, generator=rng),
            "text_pooled": torch.randn(size, D, generator=rng),
            "similarity": torch.randn(size, generator=rng),
            "label": torch.arange(size) % num_classes}


@pytest.mark.parametrize("name", ["unimodal", "v3_alpha"])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_shapes_probabilities_and_backward(name, num_classes):
    model = build_model(config(name, num_classes))
    inputs = batch(num_classes)
    out = model(inputs)
    q = 1 if num_classes == 2 else num_classes
    assert out["logits"].shape == (6, q)
    assert out["prob"].shape == ((6,) if num_classes == 2 else (6, num_classes))
    assert ((out["prob"] >= 0) & (out["prob"] <= 1)).all()
    if num_classes > 2:
        torch.testing.assert_close(out["prob"].sum(-1), torch.ones(6))
    loss, parts = model.loss(out, inputs)
    expected = (F.binary_cross_entropy_with_logits(out["logits"].squeeze(-1),
                                                  inputs["label"].float())
                if num_classes == 2 else F.cross_entropy(out["logits"], inputs["label"]))
    assert parts["main"] == pytest.approx(expected.item())
    if name == "unimodal":
        assert "pi" not in out and set(parts) == {"main"}
        torch.testing.assert_close(loss, expected)
    else:
        assert out["pi"].shape == (6, 2)
        assert ((out["pi"] > 0) & (out["pi"] < 1)).all()
        torch.testing.assert_close(out["pi"].sum(-1), torch.ones(6))
        assert model.expert_names == ("img", "txt")
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("modality", ["img", "txt"])
def test_unimodal_ignores_other_modality(modality):
    model = build_model(config("unimodal", modality=modality)).eval()
    inputs = batch()
    expected = model(inputs)["prob"]
    other = "text" if modality == "img" else "image"
    altered = {**inputs, f"{other}_tokens": torch.full_like(inputs[f"{other}_tokens"], torch.nan),
               f"{other}_pooled": torch.full_like(inputs[f"{other}_pooled"], torch.nan),
               "similarity": torch.full_like(inputs["similarity"], torch.nan)}
    if modality == "img":
        altered["text_mask"] = torch.zeros_like(inputs["text_mask"])
    torch.testing.assert_close(model(altered)["prob"], expected, rtol=0, atol=0)
    # It must also work when the unused modality is absent altogether.
    keep = "image" if modality == "img" else "text"
    minimal = {k: v for k, v in inputs.items() if k.startswith(keep)}
    torch.testing.assert_close(model(minimal)["prob"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_classes", [2, 3])
def test_v3_fusion_and_regularizers(num_classes):
    cfg = config("v3_alpha", num_classes)
    model = build_model(cfg).eval()
    captured = []
    hook = model.classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    inputs = batch(num_classes)
    out = model(inputs)
    hook.remove()
    a = out["pi"][:, :1]
    expected = torch.cat([a * F.normalize(inputs["image_pooled"], dim=-1),
                          (1 - a) * F.normalize(inputs["text_pooled"], dim=-1), a], dim=-1)
    assert captured[0].shape == (6, 2 * D + 1)
    torch.testing.assert_close(captured[0], expected)
    loss, parts = model.loss(out, inputs)
    assert set(parts) == {"main", "aux", "alpha_entropy", "alpha_diversity"}
    assert parts["alpha_entropy"] == pytest.approx(
        (a * a.log() + (1 - a) * (1 - a).log()).mean().item())
    assert parts["alpha_diversity"] == pytest.approx(-a.var().item())
    assert loss.item() == pytest.approx(parts["main"] + cfg["lambda_aux"] * parts["aux"]
                                       + cfg["lambda_entropy"] * parts["alpha_entropy"]
                                       + cfg["lambda_diversity"] * parts["alpha_diversity"])


def test_v3_singleton_batch_has_finite_loss():
    model = build_model(config("v3_alpha"))
    inputs = batch(size=1)
    loss, parts = model.loss(model(inputs), inputs)
    assert torch.isfinite(loss) and parts["alpha_diversity"] == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("name,keys", [("unimodal", UNIMODAL_KEYS), ("v3_alpha", V3_KEYS)])
def test_all_model_hyperparameters_are_required(name, keys):
    for key in keys:
        cfg = config(name)
        del cfg[key]
        with pytest.raises(ValueError, match=key):
            build_model(cfg)


def test_invalid_modality():
    with pytest.raises(ValueError, match="modality"):
        build_model(config("unimodal", modality="both"))


@pytest.fixture
def synthetic_cache(tmp_path):
    # Small train/val-only version of tests/test_train.py's synthetic-cache fixture.
    rng = np.random.default_rng(0)
    ids = [f"s{i:03d}" for i in range(24)]
    labels = np.arange(len(ids)) % 2
    pooled = rng.normal(size=(len(ids), D))
    pooled[np.arange(len(ids)), labels] += 3.0
    image = np.concatenate([pooled[:, None], rng.normal(size=(len(ids), 4, D))], axis=1)
    root = tmp_path / "features/toy2/v1/train"
    write_cache(root, ids=ids, image_tokens=image,
                text_tokens=rng.normal(size=(len(ids), 3, D)),
                text_mask=np.ones((len(ids), 3), bool),
                text_pooled=pooled + rng.normal(size=(len(ids), D)),
                similarity=rng.uniform(size=len(ids)), labels=labels,
                meta={"backbone": "toy", "cache_version": "v1", "git_sha": "test"})
    splits = tmp_path / "splits/toy2"
    splits.mkdir(parents=True)
    (splits / "train.txt").write_text("\n".join(ids[:16]) + "\n")
    (splits / "val.txt").write_text("\n".join(ids[16:]) + "\n")
    return {"dataset": "toy2", "cache_version": "v1", "batch_size": 8, "num_epochs": 1,
            "lr": 0.01, "weight_decay": 0.0, "warmup_frac": 0.1, "patience": 2,
            "seed": 0, "deterministic": True, "max_samples": 0,
            "feature_dir": str(tmp_path / "features"), "results_dir": str(tmp_path / "results"),
            "checkpoint_dir": str(tmp_path / "ckpt"), "splits_dir": str(tmp_path / "splits")}


@pytest.mark.parametrize("name", ["unimodal", "v3_alpha"])
def test_train_smoke(synthetic_cache, tmp_path, name):
    cfg = {**synthetic_cache, **config(name), "method": name}
    metrics = train(cfg, device="cpu")
    assert (metrics["n_train"], metrics["n_val"], metrics["epochs_run"]) == (16, 8, 1)
    assert np.isfinite(metrics["history"][0]["train_loss"])
    assert (metrics["router"] is None) == (name == "unimodal")
    run_dir = tmp_path / "results/toy2" / name / "seed0"
    with (run_dir / "preds_val.csv").open() as handle:
        reader = csv.DictReader(handle)
        expected = ["id", "label", "prob"]
        if name == "v3_alpha":
            expected += ["pi_img", "pi_txt"]
            assert "train_alpha_entropy" in metrics["history"][0]
            assert "train_alpha_diversity" in metrics["history"][0]
        expected += ["p_img", "p_txt"] if name == "v3_alpha" else ["p_img"]
        assert reader.fieldnames == expected
        assert len(list(reader)) == 8
    assert not (run_dir / "preds_test.csv").exists()
