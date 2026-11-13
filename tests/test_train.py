import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from hmf.config import config_hash, load_config  # noqa: E402
from hmf.data.common import id_list_record  # noqa: E402
from hmf.evaluate import evaluate  # noqa: E402
from hmf.features.cache import write_cache  # noqa: E402
from hmf.models.registry import MODELS, build_model, register_model  # noqa: E402
from hmf.train import OPTIONAL_RUNTIME, open_role, router_health, run_path, train  # noqa: E402

D = 8
ROOT = Path(__file__).resolve().parents[1]


class ToyRouter(nn.Module):
    """Three linear experts on pooled features mixed by a softmax router (Variant P)."""

    expert_names = ("img", "txt", "int")

    def __init__(self, cfg, collapse=False):
        super().__init__()
        self.img, self.txt, self.int = nn.Linear(D, 1), nn.Linear(D, 1), nn.Linear(2 * D, 1)
        self.router = nn.Linear(2 * D, 3)
        self.collapse = collapse

    def forward(self, batch):
        v, u = batch["image_pooled"], batch["text_pooled"]
        vu = torch.cat([v, u], dim=-1)
        z = torch.cat([self.img(v), self.txt(u), self.int(vu)], dim=-1)
        logits = self.router(vu)
        pi = torch.softmax(logits * 0 if self.collapse else logits, dim=-1)
        return {"prob": (pi * torch.sigmoid(z)).sum(-1), "pi": pi, "z": z}

    def loss(self, out, batch):
        y = batch["label"].float()
        main = F.binary_cross_entropy(out["prob"].clamp(1e-6, 1 - 1e-6), y)
        aux = F.binary_cross_entropy_with_logits(out["z"], y[:, None].expand_as(out["z"]))
        return main + 0.5 * aux, {"main": main.item(), "aux": aux.item()}


class ToyMulticlass(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head = nn.Linear(2 * D, cfg["num_classes"])

    def forward(self, batch):
        logits = self.head(torch.cat([batch["image_pooled"], batch["text_pooled"]], dim=-1))
        return {"prob": torch.softmax(logits, -1), "logits": logits}

    def loss(self, out, batch):
        loss = F.cross_entropy(out["logits"], batch["label"])
        return loss, {}


for name, builder in {"toy_router": ToyRouter,
                      "toy_collapsed": lambda cfg: ToyRouter(cfg, collapse=True),
                      "toy_multiclass": ToyMulticlass}.items():
    if name not in MODELS:
        register_model(name)(builder)


def make_split(root, ids, labels, num_classes, seed):
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
    def build(num_classes=2, model="toy_router", **overrides):
        dataset = f"toy{num_classes}"
        source = [f"s{i:03d}" for i in range(100)]
        held = [f"h{i:03d}" for i in range(30)]
        root = tmp_path / "features" / dataset / "v1"
        make_split(root / "train", source, np.arange(100) % num_classes, num_classes, 0)
        make_split(root / "dev_seen", held, np.arange(30) % num_classes, num_classes, 1)
        splits = tmp_path / "splits" / dataset
        splits.mkdir(parents=True, exist_ok=True)
        (splits / "train.txt").write_text("\n".join(source[:60]) + "\n")
        (splits / "val.txt").write_text("\n".join(source[60:]) + "\n")
        (splits / "test.txt").write_text("\n".join(held) + "\n")
        cfg = {
            "dataset": dataset, "method": "dyn", "model": model, "cache_version": "v1",
            "num_classes": num_classes, "batch_size": 16, "num_epochs": 4, "lr": 0.05,
            "weight_decay": 0.0, "warmup_frac": 0.1, "patience": 10, "seed": 0,
            "deterministic": True, "max_samples": 0,
            "feature_dir": str(tmp_path / "features"), "results_dir": str(tmp_path / "results"),
            "checkpoint_dir": str(tmp_path / "ckpt"), "splits_dir": str(tmp_path / "splits"),
        }
        cfg.update(overrides)
        return cfg

    return build


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_end_to_end_binary_run_writes_the_stable_interface(setup):
    cfg = setup()
    metrics = train(cfg, device="cpu")
    assert metrics["checkpoint"].endswith("best.pt")
    run_dir = f"{cfg['results_dir']}/toy2/dyn/seed0"
    saved = json.loads(open(f"{run_dir}/metrics.json").read())
    for key in ("git_sha", "config_hash", "best_epoch", "train_seconds", "device", "sdpa_backends"):
        assert key in saved
    assert saved["sdpa_backends"] == {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    assert saved["config_hash"] == config_hash(cfg)
    assert 1 <= saved["best_epoch"] <= 4 and len(saved["history"]) == 4
    assert saved["val"]["auroc"] > 0.8 and saved["val"]["threshold_metric"] == "macro_f1"
    assert set(saved["router"]) == {"pi_mean", "pi_std", "pi_entropy", "collapsed"}
    rows = read_csv(f"{run_dir}/preds_val.csv")
    assert list(rows[0]) == ["id", "label", "prob", "pi_img", "pi_txt", "pi_int",
                             "p_img", "p_txt", "p_int"]
    assert len(rows) == 40 and rows[0]["id"] == "s060"
    assert abs(sum(float(rows[0][f"pi_{k}"]) for k in ("img", "txt", "int")) - 1) < 1e-5
    assert b"\r" not in Path(f"{run_dir}/preds_val.csv").read_bytes()  # LF on every OS (#110)


def test_metrics_record_the_split_lists_read(setup, tmp_path):
    # #110: splits_dir is outside config_hash, so metrics.json names the lists by content.
    saved = train(setup(num_epochs=1), device="cpu")["splits"]
    assert set(saved) == {"train", "val"}
    for role, n in (("train", 60), ("val", 40)):
        path = tmp_path / "splits/toy2" / f"{role}.txt"
        assert saved[role] == {"path": path.as_posix(), "n": n,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_split_list_hash_ignores_crlf(tmp_path):
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes(b"a\nb\nc\n")
    crlf.write_bytes(b"a\r\nb\r\nc\r\n")
    assert id_list_record(lf)["sha256"] == id_list_record(crlf)["sha256"]
    assert id_list_record(crlf)["sha256"] == hashlib.sha256(b"a\nb\nc\n").hexdigest()
    assert id_list_record(crlf)["n"] == 3


def test_max_samples_env_override_gives_a_smoke_run(setup, tmp_path):
    cfg = setup()
    path = tmp_path / "smoke.yaml"
    path.write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()) + "\n")
    smoke = load_config(path, environ={"MAX_SAMPLES": "16", "NUM_EPOCHS": "2"})
    metrics = train(smoke, device="cpu")
    assert (metrics["n_train"], metrics["n_val"], metrics["epochs_run"]) == (16, 16, 2)
    assert metrics["config_hash"] != config_hash(cfg)  # max_samples is part of the experiment


def test_save_epoch_preds_smoke_run_writes_every_epoch(setup, tmp_path):
    # #106: SAVE_EPOCH_PREDS reaches a config without the key and leaves the hash alone.
    path = tmp_path / "smoke.yaml"
    path.write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in setup().items()) + "\n")
    env = {"MAX_SAMPLES": "64", "NUM_EPOCHS": "2"}
    smoke = load_config(path, environ={**env, "SAVE_EPOCH_PREDS": "1"}, defaults=OPTIONAL_RUNTIME)
    metrics = train(smoke, device="cpu")
    run_dir = tmp_path / "results/toy2/dyn/seed0"
    epoch_files = sorted(p.name for p in run_dir.glob("preds_val_epoch*.csv"))
    assert epoch_files == ["preds_val_epoch001.csv", "preds_val_epoch002.csv"]
    best = run_dir / f"preds_val_epoch{metrics['best_epoch']:03d}.csv"
    assert best.read_bytes() == (run_dir / "preds_val.csv").read_bytes()
    plain = load_config(path, environ=env, defaults=OPTIONAL_RUNTIME)
    assert plain["save_epoch_preds"] is False
    assert metrics["config_hash"] == config_hash(plain)


def test_epoch_preds_are_off_by_default(setup, tmp_path):
    train(setup(num_epochs=2), device="cpu")
    assert not list((tmp_path / "results/toy2/dyn/seed0").glob("preds_val_epoch*"))


def test_106_overrides_keep_the_cleared_align_hash():
    # #106 runs the committed Align config with these overrides; the hash listed in
    # results/FINAL_CONFIGS.md for #13 must not move. #110 adds RESUME and CHECKPOINT_EVERY.
    path = ROOT / "configs/fhm/hateclipper_align.yaml"
    env = {"SAVE_EPOCH_PREDS": "1", "HMF_SPLITS_DIR": "/predup", "SEED": "3",
           "RESUME": "1", "CHECKPOINT_EVERY": "2"}
    cfg = load_config(path, environ=env, defaults=OPTIONAL_RUNTIME)
    assert (cfg["save_epoch_preds"], cfg["splits_dir"], cfg["seed"]) == (True, "/predup", 3)
    assert (cfg["resume"], cfg["checkpoint_every"]) == (True, 2)
    assert config_hash(cfg) == config_hash(load_config(path, environ={}))
    assert f"| {config_hash(cfg)} | configs/fhm/hateclipper_align.yaml |" in (
        ROOT / "results/FINAL_CONFIGS.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", sorted((ROOT / "configs").rglob("*.yaml")),
                         ids=lambda p: p.relative_to(ROOT).as_posix())
def test_optional_runtime_keeps_every_committed_config_hash(path):
    # #110: adding OPTIONAL_RUNTIME (and overriding it) never moves a config hash.
    env = {"RESUME": "1", "CHECKPOINT_EVERY": "3", "SAVE_EPOCH_PREDS": "1",
           "HMF_SPLITS_DIR": "/elsewhere"}
    assert config_hash(load_config(path, environ=env, defaults=OPTIONAL_RUNTIME)) == (
        config_hash(load_config(path, environ={})))


def test_resume_and_checkpoint_every_reach_a_config_without_them(setup, tmp_path):
    # #110: RESUME and CHECKPOINT_EVERY were silently ignored when the YAML omits them.
    path = tmp_path / "run.yaml"
    path.write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in setup().items()) + "\n")
    first = load_config(path, environ={"NUM_EPOCHS": "2", "CHECKPOINT_EVERY": "1"},
                        defaults=OPTIONAL_RUNTIME)
    assert "checkpoint_every" not in load_config(path, environ={"CHECKPOINT_EVERY": "1"})
    assert (first["resume"], first["checkpoint_every"]) == (False, 1)
    before = train(first, device="cpu")
    assert (run_path(first, "checkpoint_dir") / "best_last.pt").exists()
    again = load_config(path, environ={"NUM_EPOCHS": "4", "CHECKPOINT_EVERY": "1",
                                       "RESUME": "1"}, defaults=OPTIONAL_RUNTIME)
    metrics = train(again, device="cpu")
    assert [h["epoch"] for h in metrics["history"]] == [1, 2, 3, 4]
    # Epochs 1-2 come from the checkpoint: a fresh 4-epoch run has a different lr schedule.
    assert metrics["history"][:2] == before["history"]
    assert metrics["config_hash"] == config_hash(load_config(path, environ={"NUM_EPOCHS": "4"}))


def test_training_is_reproducible_for_a_seed(setup, tmp_path):
    cfg = setup(model="dynamic", feature_dim=D, hidden_dim=8, num_heads=2,
                n_cross_layers=1, cross_fusion_dim=2, dropout=0.2,
                fusion_variant="P", router_mode="dynamic", tau=1.0,
                lambda_aux=0.5, lambda_rel=0.1, entropy_stop_grad=True,
                num_epochs=2, max_samples=16)
    first = train(cfg, device="cpu")["val"]
    first_csv = (run_path(cfg, "results_dir") / "preds_val.csv").read_bytes()
    cfg.update(results_dir=str(tmp_path / "repeat_results"),
               checkpoint_dir=str(tmp_path / "repeat_ckpt"))
    second = train(cfg, device="cpu")["val"]
    second_csv = (run_path(cfg, "results_dir") / "preds_val.csv").read_bytes()
    assert first == second
    assert first_csv == second_csv


def test_training_never_reads_the_test_role(setup):
    with pytest.raises(PermissionError):
        open_role(setup(), "test")


def test_missing_keys_are_rejected(setup):
    cfg = setup()
    del cfg["lr"]
    with pytest.raises(ValueError, match="lr"):
        train(cfg, device="cpu")


def test_test_split_guard(setup, tmp_path):
    cfg = setup()
    train(cfg, device="cpu")
    run_dir = tmp_path / "results/toy2/dyn/seed0"
    final_configs = tmp_path / "FINAL_CONFIGS.md"
    with pytest.raises(PermissionError, match="--final"):
        evaluate(run_dir, "test", final_configs=final_configs, device="cpu")
    with pytest.raises(PermissionError, match="not in"):
        evaluate(run_dir, "test", final=True, final_configs=final_configs, device="cpu")
    assert not (run_dir / "preds_test.csv").exists()

    final_configs.write_text(f"| {config_hash(cfg)} | 2026-10-08 | #19 |\n")
    result = evaluate(run_dir, "test", final=True, final_configs=final_configs, device="cpu")
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert result["threshold"] == saved["val"]["threshold"]  # chosen on val, reused on test
    assert result["n"] == 30 and len(read_csv(run_dir / "preds_test.csv")) == 30
    test_list = tmp_path / "splits/toy2/test.txt"  # #110: the scored list joins train and val
    assert set(saved["splits"]) == {"train", "val", "test"}
    assert saved["splits"]["test"] == {"path": test_list.as_posix(), "n": 30,
                                       "sha256": hashlib.sha256(test_list.read_bytes()).hexdigest()}


@pytest.mark.parametrize("classes,mode", [(2, "dynamic"), (3, "dynamic"), (101, "dynamic"),
                                         (2, "static"), (2, "global"), (2, "concat_mlp"),
                                         (2, "qmf"), (3, "pdf")])
def test_final_reads_test_cache_once_and_writes_all_conditions(setup, tmp_path, monkeypatch,
                                                             classes, mode):
    import hmf.train as training
    from hmf.eval.corruptions import CONDITIONS
    from hmf.eval.metrics import binary_metrics, multiclass_metrics
    from hmf.eval.router_evidence import router_evidence
    from hmf.evaluate import main
    from hmf.features.dataset import CachedFeatureDataset
    from hmf.models.dynamic_baselines import PDF_DEFAULTS, QMF_DEFAULTS

    cfg = setup(num_classes=classes,
                model=mode if mode in {"concat_mlp", "qmf", "pdf"} else "dynamic",
                feature_dim=D, hidden_dim=8, num_heads=2, n_cross_layers=1,
                cross_fusion_dim=2, dropout=0, fusion_variant="P", router_mode=mode,
                tau=0.5, lambda_aux=0.5, lambda_rel=0.1, entropy_stop_grad=True,
                image_dim=D, text_dim=D, mlp_hidden=8, num_epochs=1)
    if mode in {"qmf", "pdf"}:
        cfg.update(QMF_DEFAULTS if mode == "qmf" else PDF_DEFAULTS)
    train(cfg, device="cpu")
    run_dir = run_path(cfg, "results_dir")
    frozen = json.loads((run_dir / "metrics.json").read_text())
    # Emulate a run trained before #100: final must upgrade the val evidence.
    del frozen["val"]["calibration"]
    (run_dir / "metrics.json").write_text(json.dumps(frozen))
    final_configs = tmp_path / "FINAL_CONFIGS.md"
    final_configs.write_text(config_hash(cfg))
    opens, read_positions, thresholds = [], [], []
    original_load, original_batch = training.load_cache, CachedFeatureDataset._batch
    original_threshold = training.find_threshold

    def load(path):
        opens.append(Path(path).name)
        return original_load(path)

    def batch(self, positions):
        if self.cache.path.name == "dev_seen":
            read_positions.extend(positions.tolist())
        return original_batch(self, positions)

    def find(*args, **kwargs):
        thresholds.append(True)
        return original_threshold(*args, **kwargs)

    monkeypatch.setattr(training, "load_cache", load)
    monkeypatch.setattr(CachedFeatureDataset, "_batch", batch)
    monkeypatch.setattr(training, "find_threshold", find)
    monkeypatch.setattr("hmf.evaluate.get_device", lambda: torch.device("cpu"))
    assert main(["--run-dir", str(run_dir), "--split", "test", "--final",
                 "--final-configs", str(final_configs)]) == 0
    assert opens.count("dev_seen") == 1
    assert read_positions == list(range(30))  # each source row read exactly once, in order
    assert thresholds == []  # no threshold fitting on clean or corrupted test (or val)
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["val"]["calibration"]["n_bins"] == 15
    assert set(saved["test"]["corruptions"]) == set(CONDITIONS)
    assert saved["config_hash"] == frozen["config_hash"]
    assert len(list(run_dir.glob("preds_test*.csv"))) == 9
    clean = read_csv(run_dir / "preds_test.csv")
    assert len(set(saved["test"]["robustness"]["swap_donor_ids"])) == len(clean)
    assert all(row["id"] != donor for row, donor in
               zip(clean, saved["test"]["robustness"]["swap_donor_ids"], strict=True))
    for condition in (None, *CONDITIONS):
        path = run_dir / (f"preds_test_{condition}.csv" if condition else "preds_test.csv")
        rows = read_csv(path)
        metrics = saved["test"]["corruptions"][condition] if condition else saved["test"]
        assert [row["id"] for row in rows] == [row["id"] for row in clean]
        labels = np.array([int(row["label"]) for row in rows])
        if classes == 2:
            assert metrics["threshold"] == frozen["val"]["threshold"]
            expected = binary_metrics(labels, [float(row["prob"]) for row in rows],
                                      frozen["val"]["threshold"])
            assert metrics["accuracy"] == expected["accuracy"]
            assert metrics["macro_f1"] == expected["macro_f1"]
        else:
            assert "auroc" not in metrics and "threshold" not in metrics
            one_hot = np.eye(classes)[[int(row["pred"]) for row in rows]]
            expected = multiclass_metrics(labels, one_hot)
            assert metrics["accuracy"] == expected["accuracy"]
            assert metrics["macro_f1"] == expected["macro_f1"]
        if mode != "concat_mlp":
            names = ("img", "txt") if mode in {"qmf", "pdf"} else ("img", "txt", "int")
            evidence = router_evidence(path, names,
                                       num_classes=classes, tau=cfg["tau"])
            assert evidence["target"].shape == (30, len(names))
            np.testing.assert_allclose(evidence["target"].sum(axis=1), 1)
            if mode != "qmf":
                np.testing.assert_allclose(metrics["pi_mean"], evidence["pi_mean"], atol=1e-6)
            else:
                assert "pi_mean" not in metrics
        else:
            assert not any(key.startswith(("pi_", "p_")) for key in rows[0])


def test_final_rejects_missing_val_threshold_before_opening_test(setup, tmp_path, monkeypatch):
    import hmf.evaluate as evaluation

    cfg = setup(num_epochs=1)
    train(cfg, device="cpu")
    run_dir = run_path(cfg, "results_dir")
    path = run_dir / "metrics.json"
    metrics = json.loads(path.read_text())
    del metrics["val"]["threshold"]
    path.write_text(json.dumps(metrics))
    registered = tmp_path / "FINAL_CONFIGS.md"
    registered.write_text(config_hash(cfg))
    monkeypatch.setattr(evaluation, "open_role", lambda *a, **kw: pytest.fail("opened a role"))
    with pytest.raises(ValueError, match="frozen clean-val threshold"):
        evaluate(run_dir, "test", final=True, final_configs=registered, device="cpu")


def test_evaluate_val_reproduces_training_predictions(setup, tmp_path):
    train(setup(), device="cpu")
    run_dir = tmp_path / "results/toy2/dyn/seed0"
    before = (run_dir / "preds_val.csv").read_bytes()
    training = json.loads((run_dir / "metrics.json").read_text())
    training_backends = training["sdpa_backends"]
    result = evaluate(run_dir, "val", device="cpu")
    assert (run_dir / "preds_val.csv").read_bytes() == before
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["splits"] == training["splits"]
    assert saved["sdpa_backends"] == training_backends
    assert saved["val"]["sdpa_backends"] == result["sdpa_backends"] == training_backends


def test_evaluate_rejects_an_edited_config(setup, tmp_path):
    train(setup(), device="cpu")
    run_dir = tmp_path / "results/toy2/dyn/seed0"
    path = run_dir / "config.yaml"
    path.write_text(path.read_text().replace("lr: 0.05", "lr: 0.01"))
    with pytest.raises(ValueError, match="config_hash"):
        evaluate(run_dir, "val", device="cpu")


def test_resume_continues_from_the_rolling_checkpoint(setup):
    train(setup(num_epochs=2, checkpoint_every=1), device="cpu")
    metrics = train(setup(num_epochs=4, checkpoint_every=1, resume=True), device="cpu")
    assert [h["epoch"] for h in metrics["history"]] == [1, 2, 3, 4]


def test_early_stopping_respects_patience(setup):
    metrics = train(setup(num_epochs=30, patience=2, lr=0.2), device="cpu")
    assert metrics["epochs_run"] <= metrics["best_epoch"] + 2


def test_multiclass_run_selects_on_accuracy(setup):
    cfg = setup(num_classes=3, model="toy_multiclass")
    metrics = train(cfg, device="cpu")
    assert metrics["selection_metric"] == "accuracy" and metrics["router"] is None
    assert set(metrics["val"]) == {"accuracy", "macro_f1", "calibration"}
    assert metrics["val"]["accuracy"] > 0.6
    rows = read_csv(f"{cfg['results_dir']}/toy3/dyn/seed0/preds_val.csv")
    assert list(rows[0]) == ["id", "label", "pred", "prob"]


def test_collapsed_router_warns(setup):
    with pytest.warns(UserWarning, match="router std"):
        metrics = train(setup(model="toy_collapsed", num_epochs=1), device="cpu")
    assert metrics["router"]["collapsed"]


def test_router_health_values():
    pi = np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    health = router_health(pi)
    np.testing.assert_allclose(health["pi_std"], pi.std(axis=0))
    assert 0 < health["pi_entropy"] < 1 and not health["collapsed"]
    assert router_health(np.full((4, 3), 1 / 3))["pi_entropy"] == pytest.approx(1.0)


def test_registry_errors():
    with pytest.raises(KeyError, match="unknown model"):
        build_model({"model": "nope"})
    with pytest.raises(ValueError, match="already registered"):
        register_model("toy_router")(ToyRouter)


def test_grad_clip_is_applied_when_configured(setup, monkeypatch):
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm):
        calls.append(max_norm)
        return original(params, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    train(setup(num_epochs=1), device="cpu")
    assert calls == []
    train(setup(num_epochs=1, grad_clip=0.1), device="cpu")
    assert calls and set(calls) == {0.1}


def test_lr_schedule_constant_and_default(setup):
    # Hate-CLIPper's AdamW has no scheduler (#13); absent keeps warmup + cosine.
    constant = train(setup(num_epochs=3, lr_schedule="constant", warmup_frac=0.0), device="cpu")
    assert [h["lr"] for h in constant["history"]] == [0.05, 0.05, 0.05]
    default = train(setup(num_epochs=3), device="cpu")
    assert default["history"][-1]["lr"] < 0.05
    explicit = train(setup(num_epochs=3, lr_schedule="warmup_cosine"), device="cpu")
    assert [h["lr"] for h in explicit["history"]] == [h["lr"] for h in default["history"]]
    with pytest.raises(ValueError, match="warmup_frac: 0"):
        train(setup(lr_schedule="constant"), device="cpu")
    with pytest.raises(ValueError, match="unknown lr_schedule"):
        train(setup(lr_schedule="linear"), device="cpu")


def test_write_predictions_true_class_keeps_expert_accuracy_and_ce(tmp_path):
    from hmf.train import write_predictions

    rng = np.random.default_rng(0)
    expert_prob = rng.dirichlet(np.ones(5), size=(4, 3)).astype(np.float32)  # (n, experts, C)
    pi = rng.dirichlet(np.ones(3), size=4)
    prob = (pi[:, :, None] * expert_prob).sum(1)
    pred = {"ids": ["a", "b", "c", "d"], "labels": np.array([0, 4, 2, -1]), "prob": prob,
            "pi": pi, "expert_prob": expert_prob}
    cfg = {"num_classes": 5}
    full, compact = tmp_path / "full.csv", tmp_path / "compact.csv"
    write_predictions(full, pred, cfg, ("img", "txt", "int"))
    write_predictions(compact, pred, cfg, ("img", "txt", "int"), expert_probs="true_class")
    rows_full, rows = read_csv(full), read_csv(compact)
    assert list(rows[0]) == ["id", "label", "pred", "prob", "pi_img", "pi_txt", "pi_int",
                             "pred_img", "pred_txt", "pred_int",
                             "p_img_true", "p_txt_true", "p_int_true"]
    for i, (row, row_full) in enumerate(zip(rows, rows_full, strict=True)):
        assert {k: row[k] for k in ("id", "label", "pred", "prob", "pi_img")} == \
            {k: row_full[k] for k in ("id", "label", "pred", "prob", "pi_img")}
        for name in ("img", "txt", "int"):
            dist = [float(row_full[f"p_{name}_{c}"]) for c in range(5)]
            assert int(row[f"pred_{name}"]) == int(np.argmax(dist))
            label = pred["labels"][i]
            if label < 0:
                assert row[f"p_{name}_true"] == ""
            else:
                assert float(row[f"p_{name}_true"]) == pytest.approx(dist[label], rel=1e-6)
    assert compact.stat().st_size < full.stat().st_size


def test_write_predictions_true_class_leaves_binary_files_unchanged(tmp_path):
    from hmf.train import write_predictions

    pred = {"ids": ["a", "b"], "labels": np.array([0, 1]), "prob": np.array([0.2, 0.9]),
            "pi": np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]]),
            "expert_prob": np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.9]])}
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_predictions(a, pred, {"num_classes": 2}, ("img", "txt", "int"))
    write_predictions(b, pred, {"num_classes": 2}, ("img", "txt", "int"), expert_probs="true_class")
    assert a.read_bytes() == b.read_bytes()
    with pytest.raises(ValueError, match="expert_probs"):
        write_predictions(a, pred, {"num_classes": 2}, expert_probs="none")
