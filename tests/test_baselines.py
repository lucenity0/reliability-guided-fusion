import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from hmf.features.cache import write_cache  # noqa: E402
from hmf.models.baselines import CONCAT_MLP_KEYS, LINEAR_PROBE_KEYS  # noqa: E402
from hmf.models.registry import build_model  # noqa: E402
from hmf.train import train  # noqa: E402

D = 8  # equal image/text widths, as in tests/test_train.py
MODELS = ("concat_mlp", "linear_probe")


def config(model, num_classes=2, image_dim=D, text_dim=D):
    cfg = {"model": model, "num_classes": num_classes, "image_dim": image_dim,
           "text_dim": text_dim}
    if model == "concat_mlp":
        cfg.update(mlp_hidden=16, dropout=0.0)
    return cfg


def batch(num_classes=2, size=6, image_dim=D, text_dim=D):
    rng = torch.Generator().manual_seed(7)
    return {"image_pooled": torch.randn(size, image_dim, generator=rng),
            "text_pooled": torch.randn(size, text_dim, generator=rng),
            "label": torch.arange(size) % num_classes}


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("num_classes", [2, 3])
def test_shapes_probabilities_and_loss(name, num_classes):
    model = build_model(config(name, num_classes, image_dim=6, text_dim=10))
    data = batch(num_classes, image_dim=6, text_dim=10)
    out = model(data)
    if num_classes == 2:
        assert out["logits"].shape == (6, 1) and out["prob"].shape == (6,)
        assert torch.allclose(out["prob"], out["logits"].squeeze(-1).sigmoid())
        expected = F.binary_cross_entropy_with_logits(out["logits"].squeeze(-1),
                                                      data["label"].float())
    else:
        assert out["logits"].shape == (6, 3) and out["prob"].shape == (6, 3)
        assert torch.allclose(out["prob"].sum(-1), torch.ones(6))
        expected = F.cross_entropy(out["logits"], data["label"])
    assert ((out["prob"] >= 0) & (out["prob"] <= 1)).all()
    assert "pi" not in out
    loss, parts = model.loss(out, data)
    assert torch.allclose(loss, expected) and set(parts) == {"main"}
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_concat_mlp_architecture():
    model = build_model({**config("concat_mlp", 3, image_dim=6, text_dim=10), "dropout": 0.25})
    linear_1, gelu, dropout, linear_2 = model.head
    assert (linear_1.in_features, linear_1.out_features) == (16, 16)
    assert isinstance(gelu, torch.nn.GELU) and dropout.p == 0.25
    assert (linear_2.in_features, linear_2.out_features) == (16, 3)


def test_linear_probe_is_one_linear_on_normalised_features():
    model = build_model(config("linear_probe", 2, image_dim=6, text_dim=10))
    assert isinstance(model.head, torch.nn.Linear)
    assert sum(p.numel() for p in model.parameters()) == (6 + 10) * 1 + 1
    data = batch(image_dim=6, text_dim=10)
    scaled = {**data, "image_pooled": 5 * data["image_pooled"],
              "text_pooled": 0.1 * data["text_pooled"]}
    assert torch.allclose(model(data)["prob"], model(scaled)["prob"], atol=1e-6)


@pytest.mark.parametrize("name, keys", [("concat_mlp", CONCAT_MLP_KEYS),
                                        ("linear_probe", LINEAR_PROBE_KEYS)])
def test_all_model_hyperparameters_are_required(name, keys):
    cfg = config(name)
    assert set(keys) == set(cfg) - {"model"}
    for key in keys:
        with pytest.raises(ValueError, match=key):
            build_model({k: v for k, v in cfg.items() if k != key})


def test_invalid_values_raise():
    with pytest.raises(ValueError, match="num_classes"):
        build_model(config("linear_probe", num_classes=1))
    with pytest.raises(ValueError, match="dropout"):
        build_model({**config("concat_mlp"), "dropout": 1.0})


# ── End-to-end smoke runs (fixture pattern from tests/test_train.py) ─────────


def make_split(root, ids, labels, seed):
    rng = np.random.default_rng(seed)
    n = len(ids)
    pooled = rng.normal(size=(n, D))
    pooled[np.arange(n), labels % D] += 3.0  # a learnable signal per class
    image = np.concatenate([pooled[:, None], rng.normal(size=(n, 4, D))], axis=1)
    text = rng.normal(size=(n, 3, D))
    write_cache(root, ids=ids, image_tokens=image, text_tokens=text,
                text_mask=np.ones((n, 3), bool), text_pooled=pooled + rng.normal(size=(n, D)),
                similarity=rng.uniform(size=n), labels=labels,
                meta={"backbone": "toy", "cache_version": "v1", "git_sha": "test"})


@pytest.fixture
def setup(tmp_path):
    def build(model, num_classes=2):
        dataset = f"toy{num_classes}"
        source = [f"s{i:03d}" for i in range(100)]
        root = tmp_path / "features" / dataset / "v1"
        make_split(root / "train", source, np.arange(100) % num_classes, 0)
        splits = tmp_path / "splits" / dataset
        splits.mkdir(parents=True, exist_ok=True)
        (splits / "train.txt").write_text("\n".join(source[:60]) + "\n")
        (splits / "val.txt").write_text("\n".join(source[60:]) + "\n")
        return {
            **config(model, num_classes),
            "dataset": dataset, "method": model, "cache_version": "v1",
            "batch_size": 16, "num_epochs": 3, "lr": 0.05,
            "weight_decay": 0.0, "warmup_frac": 0.1, "patience": 10, "seed": 0,
            "deterministic": True, "max_samples": 0,
            "feature_dir": str(tmp_path / "features"), "results_dir": str(tmp_path / "results"),
            "checkpoint_dir": str(tmp_path / "ckpt"), "splits_dir": str(tmp_path / "splits"),
        }

    return build


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("num_classes", [2, 3])
def test_train_smoke(setup, name, num_classes):
    cfg = setup(name, num_classes)
    metrics = train(cfg, device="cpu")
    assert metrics["model"] == name and metrics["epochs_run"] == 3
    assert metrics["router"] is None and metrics["num_params"] > 0
    if num_classes == 2:
        assert metrics["val"]["auroc"] > 0.7
    else:
        assert metrics["val"]["accuracy"] > 0.5
