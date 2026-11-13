"""Real-module gradient checks and synthetic-cache integration for issue #18."""

import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from hmf.features.cache import write_cache  # noqa: E402
from hmf.models.fusion import REQUIRED_KEYS, Fusion, reliability_target  # noqa: E402
from hmf.models.registry import build_model  # noqa: E402
from hmf.models.router import normalized_entropy  # noqa: E402
from hmf.train import train  # noqa: E402


def config(**overrides):
    cfg = dict(model="dynamic", feature_dim=8, hidden_dim=4, num_heads=2,
               n_cross_layers=1, cross_fusion_dim=2, dropout=0.0, fusion_variant="L",
               router_mode="dynamic", tau=0.7, lambda_aux=0.5, lambda_rel=0.2,
               entropy_stop_grad=True, num_classes=2)
    cfg.update(overrides)
    return cfg


def batch(num_classes=2):
    torch.manual_seed(123)
    return dict(image_tokens=torch.randn(3, 5, 8, dtype=torch.float64),
                text_tokens=torch.randn(3, 4, 8, dtype=torch.float64),
                text_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]).bool(),
                image_pooled=torch.randn(3, 8, dtype=torch.float64),
                text_pooled=torch.randn(3, 8, dtype=torch.float64),
                similarity=torch.tensor([0.2, -0.1, 0.8], dtype=torch.float64),
                label=torch.arange(3) % num_classes)


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("variant", ["L", "P"])
def test_router_gradients_match_derivations_and_finite_differences(num_classes, variant):
    data = batch(num_classes)
    model = build_model(config(num_classes=num_classes, fusion_variant=variant)).double()
    # Obtain the real modules' outputs, then hold the experts fixed as in A-5/A-7.
    experts = model.experts(data)
    h = torch.stack([experts[k][0] for k in model.expert_names], 1).detach()
    z = torch.stack([experts[k][1] for k in model.expert_names], 1).detach()
    g = model.router(data, z).detach().requires_grad_()

    def components(logits):
        return model.loss_components(model.fuse(h, z, logits), data)

    assert torch.autograd.gradcheck(
        lambda logits: tuple(components(logits).values()), (g,), eps=1e-6, atol=1e-7)
    out = model.fuse(h, z, g)
    parts = model.loss_components(out, data)
    pi, y, size = out["pi"], data["label"], len(data["label"])
    if num_classes == 2:
        p_k = z.squeeze(-1).sigmoid()
        likelihood = torch.where(y[:, None].bool(), p_k, 1 - p_k)
        expert_loss = F.binary_cross_entropy_with_logits(
            z.squeeze(-1), y[:, None].double().expand_as(p_k), reduction="none")
        mixed = (pi * z.squeeze(-1)).sum(-1)
        expected_l = pi * (mixed.sigmoid() - y)[:, None] * (z.squeeze(-1) - mixed[:, None])
    else:
        p_k = z.softmax(-1)
        likelihood = p_k.gather(-1, y[:, None, None].expand(-1, 3, 1)).squeeze(-1)
        expert_loss = -likelihood.log()
        mixed = (pi.unsqueeze(-1) * z).sum(1)
        error = mixed.softmax(-1) - F.one_hot(y, num_classes)
        expected_l = pi * (error[:, None] * (z - mixed[:, None])).sum(-1)
    responsibility = pi * likelihood / (pi * likelihood).sum(-1, keepdim=True)
    expected_main = expected_l if variant == "L" else pi - responsibility
    expected_rel = pi - (-expert_loss / model.tau).softmax(-1)
    for name, expected in (("main", expected_main), ("rel", expected_rel)):
        actual, = torch.autograd.grad(parts[name], g, retain_graph=True)
        torch.testing.assert_close(actual, expected / size, atol=1e-12, rtol=1e-10)
    if num_classes == 2 and variant == "P":
        p = out["prob"]
        scalar_form = ((p - y) / (p * (1 - p)))[:, None] * pi * (p_k - p[:, None])
        torch.testing.assert_close(scalar_form, expected_main)
    total, logged = model.loss(out, data)
    actual, = torch.autograd.grad(total, g)
    torch.testing.assert_close(actual, (expected_main + model.lambda_rel * expected_rel) / size)
    assert logged.keys() == {"main", "aux", "rel"}
    assert all(isinstance(value, float) for value in logged.values())


def test_temperature_limits_ties_and_detachment():
    losses = torch.tensor([[1., 2., 4.], [3., 1., 1.]], dtype=torch.float64, requires_grad=True)
    high = reliability_target(losses, 1e12)
    torch.testing.assert_close(high, torch.full_like(high, 1 / 3), atol=1e-12, rtol=0)
    low = reliability_target(losses, 1e-12)
    torch.testing.assert_close(low, torch.tensor([[1., 0., 0.], [0., .5, .5]]).double())
    assert not high.requires_grad and not low.requires_grad


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("variant", ["L", "P", "F"])
@pytest.mark.parametrize("mode", ["dynamic", "static", "global"])
def test_forward_loss_and_backward(num_classes, variant, mode):
    data = batch(num_classes)
    model = build_model(config(num_classes=num_classes, fusion_variant=variant,
                               router_mode=mode)).double()
    out = model(data)
    assert model.expert_names == ("img", "txt", "int")
    assert out["prob"].shape == ((3,) if num_classes == 2 else (3, num_classes))
    assert out["z"].shape == (3, 3, 1 if num_classes == 2 else num_classes)
    torch.testing.assert_close(out["pi"].sum(-1), torch.ones(3).double())
    assert ((out["prob"] >= 0) & (out["prob"] <= 1)).all()
    if num_classes > 2:
        torch.testing.assert_close(out["prob"].sum(-1), torch.ones(3).double())
    total, parts = model.loss(out, data)
    assert parts["rel"] >= -1e-14
    assert total.item() == pytest.approx(parts["main"] + .5 * parts["aux"] + .2 * parts["rel"])
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if variant == "F":
        assert model.fusion_head[0].weight.shape == (4, 4)
        assert model.fusion_head[-1].weight.shape == (1 if num_classes == 2 else 3, 4)
    if mode == "static":
        assert torch.equal(out["pi"], torch.full_like(out["pi"], 1 / 3))
        assert model.router is None
        assert set(dict(model.named_parameters())) == {
            f"experts.{k}" for k, _ in model.experts.named_parameters()
        } | ({f"fusion_head.{k}" for k, _ in model.fusion_head.named_parameters()}
             if variant == "F" else set())
    if mode == "global":
        assert model.router is None and model.g.numel() == 3
        with torch.no_grad():
            model.g.copy_(torch.tensor([-1., 0., 2.]))
        pi = model(data)["pi"]
        assert torch.equal(pi, model.g.softmax(-1).expand(3, -1))
        assert not torch.equal(pi, torch.full_like(pi, 1 / 3))


def test_controls_share_identical_experts():
    data = batch()
    models = []
    for mode in ("dynamic", "static", "global"):
        torch.manual_seed(12)
        models.append(Fusion(config(router_mode=mode)).double())
    for model in models[1:]:
        for name, value in models[0].experts.state_dict().items():
            assert torch.equal(value, model.experts.state_dict()[name])
        for name, outputs in models[0].experts(data).items():
            for expected, actual in zip(outputs, model.experts(data)[name], strict=True):
                torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("stop", [True, False])
def test_entropy_switch_controls_gradient_to_experts(num_classes, stop):
    data = batch(num_classes)
    model = Fusion(config(num_classes=num_classes, entropy_stop_grad=stop)).double()
    # Only pi participates, isolating the entropy path from main and auxiliary losses.
    model(data)["pi"][:, 0].sum().backward()
    for expert in (model.experts.img, model.experts.txt, model.experts.interaction):
        grad = expert.head.weight.grad
        assert grad is None if stop else grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("q", [1, 3])
def test_entropy_normalization_and_saturation(q):
    z = torch.zeros(2, 3, q, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(normalized_entropy(z), torch.ones(2, 3).double())
    saturated = torch.full_like(z, -1000.)
    saturated[..., 0] = 1000.
    saturated.requires_grad_()
    entropy = normalized_entropy(saturated)
    torch.testing.assert_close(entropy, torch.zeros_like(entropy))
    entropy.sum().backward()
    assert torch.isfinite(saturated.grad).all()
    assert torch.autograd.gradcheck(normalized_entropy, (torch.randn_like(z).requires_grad_(),))


def test_categorical_probability_loss_is_stable_without_clamping():
    model = Fusion(config(num_classes=3, fusion_variant="P")).double()
    z = torch.tensor([[[-1000., 0., 1.], [-1100., 2., 0.], [-1200., 1., 2.]]],
                     dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[-1000., 0., 2.]], dtype=torch.float64, requires_grad=True)
    out = model.fuse(torch.zeros(1, 3, 4).double(), z, g)
    loss, _ = model.loss(out, {"label": torch.tensor([0])})
    assert torch.isfinite(loss) and loss > 1000
    loss.backward()
    assert torch.isfinite(g.grad).all() and g.grad.abs().sum() > 0
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


@pytest.mark.parametrize("sign,label", [(1, 0), (-1, 1)])
def test_binary_probability_clamp_has_flat_main_gradient(sign, label):
    model = Fusion(config(fusion_variant="P")).double()
    z = sign * torch.tensor([[[30.], [40.], [50.]]], dtype=torch.float64)
    g = torch.tensor([[.2, -.5, .7]], dtype=torch.float64, requires_grad=True)
    out = model.fuse(torch.zeros(1, 3, 4).double(), z, g)
    main = model.loss_components(out, {"label": torch.tensor([label])})["main"]
    assert main.item() == pytest.approx(-np.log(1e-6))
    gradient, = torch.autograd.grad(main, g)
    assert torch.equal(gradient, torch.zeros_like(g))


def test_reliability_target_has_no_expert_gradient():
    model = Fusion(config()).double()
    data = batch()
    parts = model.loss_components(model(data), data)
    params = tuple(model.experts.parameters())
    gradients = torch.autograd.grad(parts["rel"], params, allow_unused=True)
    assert all(gradient is None for gradient in gradients)


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_all_hyperparameters_are_required(key):
    cfg = config()
    del cfg[key]
    with pytest.raises(ValueError, match=key):
        build_model(cfg)


@pytest.mark.parametrize("key,value", [("tau", 0), ("tau", float("inf")), ("lambda_aux", 0),
                                       ("lambda_rel", -1), ("num_classes", 1),
                                       ("fusion_variant", "bad"), ("router_mode", "bad"),
                                       ("entropy_stop_grad", "false")])
def test_invalid_config(key, value):
    with pytest.raises(ValueError, match=key):
        build_model(config(**{key: value}))


@pytest.fixture
def synthetic_cache(tmp_path):
    def setup(num_classes, variant):
        rng = np.random.default_rng(10)
        ids = [f"s{i:03d}" for i in range(24)]
        root = tmp_path / "features" / "toy" / "v1" / "train"
        write_cache(root, ids=ids, image_tokens=rng.normal(size=(24, 5, 8)),
                    text_tokens=rng.normal(size=(24, 3, 8)), text_mask=np.ones((24, 3), bool),
                    text_pooled=rng.normal(size=(24, 8)), similarity=rng.uniform(size=24),
                    labels=np.arange(24) % num_classes,
                    meta={"backbone": "toy", "cache_version": "v1", "git_sha": "test"})
        splits = tmp_path / "splits" / "toy"
        splits.mkdir(parents=True)
        (splits / "train.txt").write_text("\n".join(ids[:18]) + "\n")
        (splits / "val.txt").write_text("\n".join(ids[18:]) + "\n")
        return config(num_classes=num_classes, fusion_variant=variant, dataset="toy",
                      method="dyn", cache_version="v1", batch_size=6, num_epochs=1,
                      lr=.001, weight_decay=.01, warmup_frac=.1, patience=2, seed=0,
                      deterministic=True, max_samples=0,
                      **{key: str(tmp_path / directory) for key, directory in
                         (("feature_dir", "features"), ("results_dir", "results"),
                          ("checkpoint_dir", "ckpt"), ("splits_dir", "splits"))})
    return setup


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("variant", ["L", "P", "F"])
@pytest.mark.filterwarnings("ignore:epoch .*router std:UserWarning")
def test_training_smoke(synthetic_cache, tmp_path, num_classes, variant):
    cfg = synthetic_cache(num_classes, variant)
    metrics = train(cfg, device="cpu")
    assert metrics["epochs_run"] == 1 and metrics["n_train"] == 18 and metrics["n_val"] == 6
    run = tmp_path / "results" / "toy" / "dyn" / "seed0"
    saved = json.loads((run / "metrics.json").read_text())
    assert all(np.isfinite(saved["history"][0][f"train_{key}"]) for key in ("main", "aux", "rel"))
    with (run / "preds_val.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {"pi_img", "pi_txt", "pi_int"} <= rows[0].keys()
    assert not (run / "preds_test.csv").exists()
