import csv
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmf.config import load_config  # noqa: E402
from hmf.evaluate import evaluate  # noqa: E402
from hmf.features.cache import write_cache  # noqa: E402
from hmf.models.dynamic_baselines import (  # noqa: E402
    COMMON_KEYS,
    PDF_DEFAULTS,
    QMF_DEFAULTS,
    LossHistory,
    _TwoExperts,
    collaborative_belief,
    energy,
    relative_calibration,
)
from hmf.models.registry import build_model  # noqa: E402
from hmf.train import train  # noqa: E402
from hmf.utils.checkpointing import CheckpointManager  # noqa: E402


def config(num_classes=2):
    return {"feature_dim": 8, "hidden_dim": 8, "dropout": 0.0, "num_classes": num_classes}


def model_config(name, num_classes=2):
    return {**config(num_classes), "model": name,
            **(QMF_DEFAULTS if name == "qmf" else PDF_DEFAULTS)}


def batch(num_classes=2, size=6):
    generator = torch.Generator().manual_seed(23)
    return {
        "image_tokens": torch.randn(size, 5, 8, generator=generator),
        "text_tokens": torch.randn(size, 3, 8, generator=generator),
        "image_pooled": torch.randn(size, 8, generator=generator),
        "text_pooled": torch.randn(size, 8, generator=generator),
        "text_mask": torch.tensor([[True, True, False]]).expand(size, -1),
        "label": torch.arange(size) % num_classes,
        "index": torch.arange(size),
    }


def test_qmf_energy_by_hand():
    logits = torch.tensor([[0.0, 0.0], [math.log(2), math.log(3)]], dtype=torch.float64)
    torch.testing.assert_close(energy(logits, 1), torch.tensor(
        [-math.log(2), -math.log(5)], dtype=torch.float64))
    torch.testing.assert_close(energy(logits * 2, 2), 2 * energy(logits, 1))
    # A large offset must not overflow exp, unlike the official direct formula.
    torch.testing.assert_close(energy(logits + 1000, 1), energy(logits, 1) - 1000)


def test_pdf_confidence_by_hand():
    # log(1/4) = 2 log(1/2); other-modality losses give holo=(2/3,1/3).
    mono = torch.tensor([[0.5, 0.25]], dtype=torch.float64)
    holo, belief = collaborative_belief(mono)
    torch.testing.assert_close(holo, torch.tensor([[2 / 3, 1 / 3]], dtype=torch.float64))
    torch.testing.assert_close(belief, torch.tensor([[7 / 6, 7 / 12]], dtype=torch.float64))
    prob = torch.tensor([[[0.9, 0.1], [0.7, 0.3]]], dtype=torch.float64)
    du, calibration = relative_calibration(prob)
    torch.testing.assert_close(du, torch.tensor([[0.4, 0.2]], dtype=torch.float64))
    torch.testing.assert_close(calibration, torch.tensor([[1.0, 0.5]], dtype=torch.float64))
    torch.testing.assert_close(belief * calibration,
                               torch.tensor([[7 / 6, 7 / 24]], dtype=torch.float64))


@pytest.mark.parametrize("classes", [2, 101])
def test_pdf_uniform_boundary(classes):
    uniform = torch.full((3, 2, classes), 1 / classes)
    _, calibration = relative_calibration(uniform)
    torch.testing.assert_close(calibration, torch.ones(3, 2))
    uniform[0, 1] = 0
    uniform[0, 1, 0] = 1
    _, calibration = relative_calibration(uniform)
    torch.testing.assert_close(calibration[0], torch.tensor([0.0, 1.0]))
    holo, belief = collaborative_belief(torch.tensor([[0., 0.], [1., 1.], [0., 1.]]))
    assert torch.isfinite(holo).all() and torch.isfinite(belief).all()
    # The official +1e-8 offset perturbs the sum near mono=(1,1).
    # Equal confidences must still give equal finite beliefs and simplex pi.
    torch.testing.assert_close(holo[:2, 0], holo[:2, 1])
    torch.testing.assert_close(belief.softmax(-1).sum(-1), torch.ones(3))


def test_history_is_cumulative_and_indexed_by_training_order():
    history = LossHistory()
    history.prepare(["third", "first", "second"])
    history.update(torch.tensor([2, 0]), torch.tensor([[2., 3.], [1., 4.]], requires_grad=True))
    history.update(torch.tensor([0, 2]), torch.tensor([[5., 6.], [7., 8.]]))
    torch.testing.assert_close(history.values, torch.tensor(
        [[6., 10.], [0., 0.], [9., 11.]], dtype=torch.float64))
    assert not history.values.requires_grad


def test_history_ranking_matches_pinned_margin_ranking_loss():
    history = LossHistory()
    history.prepare(["a", "b", "c", "unvisited"])
    index = torch.tensor([2, 0, 1])
    history.update(index, torch.tensor([[1., 2.], [2., 4.], [4., 1.]]))
    confidence = torch.tensor([[0.2, 0.5], [0.9, 0.2], [0.1, 0.8]], requires_grad=True)
    actual = history.ranking_loss(confidence, index)
    values = history.values
    normalized = (values - values.amin(0)) / (values.amax(0) - values.amin(0))
    delta = (normalized[index] - normalized[index.roll(-1)]).float()
    target, margin = delta.sign(), delta.abs()
    nonzero = torch.where(target == 0, 1, target)
    expected = sum(torch.nn.functional.margin_ranking_loss(
        confidence[:, m], confidence[:, m].roll(-1) + margin[:, m] / nonzero[:, m],
        -target[:, m]) for m in range(2))
    torch.testing.assert_close(actual, expected)
    # The two modality means are added, not averaged together.
    assert actual.item() > 0
    actual_grad, = torch.autograd.grad(actual, confidence, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, confidence)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad.abs().sum() > 0


@pytest.mark.parametrize("size", [1, 3])
def test_history_ties_are_zero_with_finite_gradients(size):
    history = LossHistory()
    history.prepare([str(i) for i in range(size)])
    index = torch.arange(size)
    history.update(index, torch.ones(size, 2))
    confidence = torch.randn(size, 2, requires_grad=True)
    ranking = history.ranking_loss(confidence, index)
    assert ranking.item() == 0
    ranking.backward()
    torch.testing.assert_close(confidence.grad, torch.zeros_like(confidence))


def test_history_checkpoint_roundtrip_and_split_mismatch(tmp_path):
    history = LossHistory()
    history.prepare(["a", "b"])
    history.update(torch.tensor([1]), torch.tensor([[1.2, 2.4]]))
    path = tmp_path / "history.pt"
    torch.save(history.state_dict(), path)
    saved = torch.load(path, weights_only=True)
    # torch.load(map_location="mps") must not move float64 history to MPS.
    assert isinstance(saved["_extra_state"]["values"], list)
    restored = LossHistory()
    restored.load_state_dict(saved)
    assert restored.ids == ("a", "b")
    torch.testing.assert_close(restored.values, history.values)
    reordered = LossHistory()
    reordered.prepare(["b", "a"])
    with pytest.raises(ValueError, match="training IDs differ"):
        reordered.load_state_dict(saved)
    with pytest.raises(ValueError, match="training IDs differ"):
        restored.prepare(["b", "a"])


@pytest.mark.parametrize("index", [torch.tensor([-1]), torch.tensor([2]),
                                   torch.tensor([0, 0]), torch.tensor([0.0])])
def test_history_rejects_invalid_indices(index):
    history = LossHistory()
    history.prepare(["a", "b"])
    with pytest.raises(ValueError, match="training indices"):
        history.update(index, torch.ones(len(index), 2))


@pytest.mark.parametrize("classes", [2, 101])
def test_shared_experts_shapes_and_gradients(classes):
    model = _TwoExperts(config(classes))
    inputs = batch(classes)
    hidden, z = model.encode(inputs)
    assert hidden.shape == (6, 2, 8)
    assert z.shape == (6, 2, 1 if classes == 2 else classes)
    out = model.fuse(z, torch.full((6, 2), 0.5))
    main, losses = model.classification_losses(out, inputs)
    (main + losses.mean(0).sum()).backward()
    for module in (model.phi_v, model.phi_t, model.img, model.txt):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0


@pytest.mark.parametrize("name", ["qmf", "pdf"])
@pytest.mark.parametrize("classes", [2, 101])
def test_registered_models_shapes_losses_and_gradients(name, classes):
    model = build_model(model_config(name, classes))
    inputs = batch(classes)
    if name == "qmf":
        model.prepare_training([str(i) for i in range(6)])
    out = model(inputs)
    assert out["logits"].shape == (6, 1 if classes == 2 else classes)
    assert out["prob"].shape == ((6,) if classes == 2 else (6, classes))
    assert torch.isfinite(out["prob"]).all()
    assert ((0 <= out["prob"]) & (out["prob"] <= 1)).all()
    if classes > 2:
        torch.testing.assert_close(out["prob"].sum(-1), torch.ones(6))
    if name == "qmf":
        assert out["pi"] is None and out["qmf_conf"].shape == (6, 2)
    else:
        assert out["pi"].shape == (6, 2) and (out["pi"] >= 0).all()
        torch.testing.assert_close(out["pi"].sum(-1), torch.ones(6))
    loss, parts = model.loss(out, inputs)
    main, per_expert = model.classification_losses(out, inputs)
    assert parts["main"] == pytest.approx(main.item())
    assert parts["aux"] == pytest.approx(per_expert.mean(0).sum().item())
    assert loss.item() == pytest.approx(sum(parts.values()))
    loss.backward()
    for module in (model.phi_v, model.phi_t, model.img, model.txt):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    if name == "pdf":
        for head in model.confidence_heads:
            assert sum(p.grad.abs().sum() for p in head.parameters()) > 0


@pytest.mark.parametrize("classes", [2, 3, 101])
def test_qmf_official_fusion_and_detach(monkeypatch, classes):
    model = build_model(model_config("qmf", classes))
    q = 1 if classes == 2 else classes
    # Includes negative categorical log partitions: never clamp or normalize them.
    z = torch.linspace(-10., 3., 4 * q).reshape(2, 2, q).requires_grad_()
    monkeypatch.setattr(model, "encode", lambda _: (None, z))
    out = model({})
    categorical = torch.cat((-z / 2, z / 2), -1) if classes == 2 else z
    reference = torch.log(torch.exp(categorical).sum(-1)) / 10
    torch.testing.assert_close(out["qmf_conf"], reference)
    torch.testing.assert_close(out["logits"], (reference.detach().unsqueeze(-1) * z).sum(1))
    gradient, = torch.autograd.grad(out["logits"].sum(), z)
    torch.testing.assert_close(gradient, reference.detach().unsqueeze(-1).expand_as(z))
    assert out["pi"] is None
    if classes > 2:
        assert reference[0, 0] < 0
    else:
        original = out["prob"].detach()
        monkeypatch.setattr(model, "encode", lambda _: (None, -z))
        swapped = model({})
        torch.testing.assert_close(swapped["qmf_conf"], out["qmf_conf"])
        torch.testing.assert_close(swapped["prob"], 1 - original)


@pytest.mark.parametrize("classes", [2, 3, 101])
def test_pdf_pinned_forward_and_l1_parity(monkeypatch, classes):
    model = build_model(model_config("pdf", classes)).double()
    q = 1 if classes == 2 else classes
    z = torch.linspace(-2., 3., 4 * q, dtype=torch.float64).reshape(2, 2, q)
    hidden = torch.randn(2, 2, 8, dtype=torch.float64)
    monkeypatch.setattr(model, "encode", lambda _: (hidden, z))
    for training in (True, False):
        model.train(training)
        out = model({})
        mono = torch.cat([head(hidden[:, k]) for k, head in enumerate(model.confidence_heads)], -1)
        # Literal pinned code, independently evaluated per modality.
        holo = torch.stack((mono[:, 1].log(), mono[:, 0].log()), -1)
        holo = holo / ((mono[:, 0] * mono[:, 1]).log()[:, None] + 1e-8)
        belief = mono + holo
        prob = (torch.cat((1 - z.sigmoid(), z.sigmoid()), -1) if classes == 2
                else z.softmax(-1))
        du = (prob - 1 / classes).abs().mean(-1)
        calibration = torch.minimum(du / du.flip(-1), torch.ones_like(du))
        expected = (belief if training else belief * calibration).softmax(-1)
        torch.testing.assert_close(out["pi"], expected)
        torch.testing.assert_close(out["logits"], (expected.unsqueeze(-1) * z).sum(1))
        labels = torch.tensor([0, 1])
        target = prob.gather(-1, labels[:, None, None].expand(-1, 2, 1)).squeeze(-1)
        parts = model.loss_components(out, {"label": labels})
        torch.testing.assert_close(parts["confidence"], (mono - target).abs().mean(0).sum())


def test_pdf_calibration_is_eval_only_and_detach_boundaries(monkeypatch):
    model = build_model(model_config("pdf"))
    z = torch.tensor([[[4.], [0.2]], [[-3.], [0.1]]], requires_grad=True)
    hidden = torch.randn(2, 2, 8, requires_grad=True)
    monkeypatch.setattr(model, "encode", lambda _: (hidden, z))
    inputs = {"label": torch.tensor([0, 1])}
    trained = model(inputs)
    torch.testing.assert_close(trained["calibration"], torch.ones(2, 2))
    parts = model.loss_components(trained, inputs)
    heads = list(model.confidence_heads.parameters())
    # Confidence regression must not train encoder features or the true-class target.
    gradients = torch.autograd.grad(parts["confidence"], [hidden, z, *heads],
                                    allow_unused=True, retain_graph=True)
    assert gradients[0] is None and gradients[1] is None
    assert all(g is not None and torch.isfinite(g).all() for g in gradients[2:])
    # Fused CE must not train confidence heads, even indirectly through pi.
    gradients = torch.autograd.grad(parts["main"], [z, *heads], allow_unused=True)
    assert gradients[0].abs().sum() > 0 and all(g is None for g in gradients[1:])
    model.eval()
    evaluated = model(inputs)
    assert (evaluated["calibration"][:, 1] < 1).all()
    assert not torch.allclose(trained["pi"], evaluated["pi"])
    gradients = torch.autograd.grad(evaluated["logits"].sum(), [z, *heads], allow_unused=True)
    torch.testing.assert_close(gradients[0], evaluated["pi"].detach().unsqueeze(-1))
    assert all(g is None for g in gradients[1:])


def test_qmf_history_updates_once_in_loss_and_never_at_prediction():
    model = build_model(model_config("qmf"))
    model.prepare_training([str(i) for i in range(6)])
    inputs = batch()
    out = model(inputs)
    _, losses = model.classification_losses(out, inputs)
    assert model.history.values.sum() == 0
    model.loss(out, inputs)
    torch.testing.assert_close(model.history.values, losses.detach().double())
    saved = model.history.values.clone()
    model.loss_components(out, inputs)
    torch.testing.assert_close(model.history.values, saved)
    model.eval()
    # Evaluation may have no index at all; its indices never refer to train rows.
    inputs.pop("index")
    model.loss(model(inputs), inputs)
    torch.testing.assert_close(model.history.values, saved)


@pytest.mark.parametrize("name", ["qmf", "pdf"])
def test_model_ignores_labels_and_masked_padding_in_forward(name):
    model = build_model(model_config(name)).eval()
    inputs = batch()
    original = model(inputs)["prob"]
    inputs["text_tokens"][:, 2] = torch.nan
    inputs.pop("label")
    inputs.pop("index")
    torch.testing.assert_close(model(inputs)["prob"], original)


@pytest.mark.parametrize("name,defaults", [("qmf", QMF_DEFAULTS), ("pdf", PDF_DEFAULTS)])
def test_configs_keep_shared_settings_and_fix_official_constants(name, defaults):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / f"configs/fhm/{name}.yaml", environ={})
    dynamic = load_config(root / "configs/fhm/dynamic.yaml", environ={})
    assert cfg["method"] == cfg["model"] == name
    swept = set(pytest.importorskip("hmf.sweep").SEARCH_SPACES["dynamic"])  # #20 tunes per method
    for key in cfg.keys() & dynamic.keys() - {"method", "model"} - swept:
        assert cfg[key] == dynamic[key]
    for key, value in defaults.items():
        assert cfg[key] == value
        with pytest.raises(ValueError, match=key):
            build_model({**model_config(name), key: value * 2})
    for key in (*COMMON_KEYS, *defaults):
        incomplete = model_config(name)
        del incomplete[key]
        with pytest.raises(ValueError, match=key):
            build_model(incomplete)


@pytest.fixture
def synthetic_cache(tmp_path, cache_arrays):
    def make(classes=2):
        arrays = cache_arrays(n=24, labels=np.arange(24) % min(classes, 3))
        write_cache(tmp_path / "features/toy2/v1/train", **arrays)
        split_dir = tmp_path / "splits/toy2"
        split_dir.mkdir(parents=True, exist_ok=True)
        for role, ids in (("train", arrays["ids"][:16]), ("val", arrays["ids"][16:])):
            (split_dir / f"{role}.txt").write_text("\n".join(ids) + "\n")
        return {"dataset": "toy2", "cache_version": "v1", "batch_size": 5, "num_epochs": 2,
                "lr": 0.01, "weight_decay": 0.0, "warmup_frac": 0.1, "patience": 5,
                "seed": 0, "deterministic": True, "max_samples": 0,
                "feature_dir": str(tmp_path / "features"), "results_dir": str(tmp_path / "results"),
                "checkpoint_dir": str(tmp_path / "ckpt"), "splits_dir": str(tmp_path / "splits")}
    return make


@pytest.mark.parametrize("name", ["qmf", "pdf"])
@pytest.mark.parametrize("classes", [2, 101])
@pytest.mark.parametrize("seed", [0, 4])
def test_two_epoch_cpu_smoke_and_val_reload(synthetic_cache, name, classes, seed):
    cfg = {**synthetic_cache(classes), **model_config(name, classes), "method": name, "seed": seed}
    metrics = train(cfg, device="cpu")
    assert (metrics["n_train"], metrics["n_val"], metrics["epochs_run"]) == (16, 8, 2)
    assert all(math.isfinite(epoch["train_loss"]) for epoch in metrics["history"])
    assert (metrics["router"] is None) == (name == "qmf")
    run_dir = Path(cfg["results_dir"]) / "toy2" / name / f"seed{seed}"
    predictions = (run_dir / "preds_val.csv").read_bytes()
    with (run_dir / "preds_val.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == (["id", "label"] + (["pred"] if classes > 2 else [])
                                    + ["prob"] + (["pi_img", "pi_txt"] if name == "pdf" else [])
                                    + ([f"p_{expert}_{c}" for expert in ("img", "txt")
                                        for c in range(classes)] if classes > 2 else
                                       ["p_img", "p_txt"]))
        assert len(list(reader)) == 8
    evaluate(run_dir, "val", device="cpu")
    assert (run_dir / "preds_val.csv").read_bytes() == predictions
    assert not (run_dir / "preds_test.csv").exists()


def test_qmf_resume_preserves_history_and_training(synthetic_cache, tmp_path, monkeypatch):
    cfg = {**synthetic_cache(), **model_config("qmf"), "method": "qmf", "checkpoint_every": 1,
           "dropout": 0.2}
    uninterrupted = train(cfg, device="cpu")
    path = Path(cfg["checkpoint_dir"]) / "toy2/qmf/seed0/best_last.pt"
    expected = torch.load(path, weights_only=False)["model_state"]
    interrupted_cfg = {**cfg, "checkpoint_dir": str(tmp_path / "interrupted_ckpt"),
                       "results_dir": str(tmp_path / "interrupted_results")}
    save = CheckpointManager.save_periodic

    def interrupt_after_save(self, epoch, *args, **kwargs):
        save(self, epoch, *args, **kwargs)
        raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(CheckpointManager, "save_periodic", interrupt_after_save)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(interrupted_cfg, device="cpu")
    resumed = train({**interrupted_cfg, "resume": True}, device="cpu")
    assert resumed["history"] == uninterrupted["history"]
    actual_path = Path(interrupted_cfg["checkpoint_dir"]) / "toy2/qmf/seed0/best_last.pt"
    actual = torch.load(actual_path, weights_only=False)["model_state"]
    for key in expected:
        if key == "history._extra_state":
            assert actual[key]["ids"] == expected[key]["ids"]
            torch.testing.assert_close(actual[key]["values"], expected[key]["values"],
                                       rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
