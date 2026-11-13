import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hmf.config import apply_env_overrides, config_hash, get_device, load_config


def test_yaml_env_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 0\nnum_workers: 2\ndeterministic: true\nfeature_dir: cache\n")
    monkeypatch.setenv("SEED", "4")
    monkeypatch.setenv("NUM_WORKERS", "")
    monkeypatch.setenv("DETERMINISTIC", "false")
    monkeypatch.setenv("HMF_FEATURE_DIR", "/new/cache")
    monkeypatch.setenv("BATCH_SIZE", "99")
    assert load_config(path) == {
        "seed": 4,
        "num_workers": 2,
        "deterministic": False,
        "feature_dir": "/new/cache",
    }
    assert load_config(path, environ={})["seed"] == 0


def test_v3_overrides_and_generic_scalar_types():
    cfg = {"batch_size": 8, "fusion_lr": 0.01, "resume": False, "dropout": 0.1, "name": "a"}
    assert (
        apply_env_overrides(
            cfg,
            {
                "BATCH_SIZE": "16",
                "FUSION_LR": "1e-4",
                "RESUME": "YES",
                "HMF_DROPOUT": "0.3",
                "HMF_NAME": "b",
            },
        )
        is cfg
    )
    assert cfg == {"batch_size": 16, "fusion_lr": 1e-4, "resume": True, "dropout": 0.3, "name": "b"}


def test_generic_overrides_require_prefix(monkeypatch):
    cfg = {"path": "project/bin", "home": "project/home", "feature_dir": "cache"}
    monkeypatch.setenv("PATH", "/ambient/bin")
    monkeypatch.setenv("HOME", "/ambient/home")
    monkeypatch.setenv("FEATURE_DIR", "/ambient/cache")
    for key in ("HMF_PATH", "HMF_HOME", "HMF_FEATURE_DIR"):
        monkeypatch.delenv(key, raising=False)
    assert apply_env_overrides(cfg.copy()) == cfg
    monkeypatch.setenv("HMF_PATH", "/requested/bin")
    monkeypatch.setenv("HMF_HOME", "/requested/home")
    monkeypatch.setenv("HMF_FEATURE_DIR", "/requested/cache")
    assert apply_env_overrides(cfg) == {
        "path": "/requested/bin",
        "home": "/requested/home",
        "feature_dir": "/requested/cache",
    }


def test_explicit_env_map_keeps_unprefixed_names():
    cfg = {"seed": 0, "batch_size": 8}
    assert apply_env_overrides(cfg.copy(), {"HMF_SEED": "1", "HMF_BATCH_SIZE": "16"}) == cfg
    assert apply_env_overrides(cfg, {"SEED": "2", "BATCH_SIZE": "32"}) == {
        "seed": 2,
        "batch_size": 32,
    }


@pytest.mark.parametrize("value", ["true", "TRUE", "yes", "on", "1", " true "])
def test_boolean_true(value):
    assert apply_env_overrides({"enabled": False}, {"HMF_ENABLED": value})["enabled"] is True


@pytest.mark.parametrize("value", ["false", "FALSE", "no", "off", "0", " false "])
def test_boolean_false(value):
    assert apply_env_overrides({"enabled": True}, {"HMF_ENABLED": value})["enabled"] is False


@pytest.mark.parametrize("env", [{"SEED": "abc"}, {"DETERMINISTIC": "maybe"}])
def test_invalid_overrides_leave_input_unchanged(env):
    cfg = {"name": "old", "seed": 0, "deterministic": True}
    with pytest.raises(ValueError, match="override"):
        apply_env_overrides(cfg, {"HMF_NAME": "new", **env})
    assert cfg == {"name": "old", "seed": 0, "deterministic": True}


@pytest.mark.parametrize(
    "document",
    [
        "",
        "[]",
        "null",
        "x: null",
        "x: [1]",
        "x: {y: 1}",
        "1: value",
        "Seed: 1",
        "x: .nan",
        "x: .inf",
    ],
)
def test_invalid_yaml_config(tmp_path, document):
    path = tmp_path / "config.yaml"
    path.write_text(document)
    with pytest.raises(ValueError):
        load_config(path, environ={})


def test_defaults_let_env_reach_keys_the_yaml_omits(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 0\nsplits_dir: mine\n")
    env = {"SAVE_EPOCH_PREDS": "yes", "HMF_SPLITS_DIR": "/s"}
    defaults = {"splits_dir": "splits", "save_epoch_preds": False}
    assert load_config(path, environ=env) == {"seed": 0, "splits_dir": "/s"}
    assert load_config(path, environ={}, defaults=defaults) == {
        "seed": 0, "splits_dir": "mine", "save_epoch_preds": False}  # YAML wins
    assert load_config(path, environ=env, defaults=defaults) == {
        "seed": 0, "splits_dir": "/s", "save_epoch_preds": True}
    path.write_text("[]")
    with pytest.raises(ValueError):
        load_config(path, environ={}, defaults=defaults)


def test_nonfinite_env_rejected():
    with pytest.raises(ValueError, match="finite"):
        apply_env_overrides({"rate": 0.1}, {"HMF_RATE": "nan"})


@pytest.mark.parametrize("literal", ["1e-4", "'1e-4'", "'1.0e-4'", "'-2E+3'", "'.5'", "'0.25'"])
def test_float_strings_rejected(tmp_path, literal):
    path = tmp_path / "config.yaml"
    path.write_text(f"lr: {literal}\n")
    with pytest.raises(ValueError, match=r"lr:.*1\.0e-4 without quotes"):
        load_config(path, environ={})


def test_yaml_scientific_float_and_non_numeric_strings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lr: 1.0e-4\nname: run-1e-4\nclip_model: openai/clip-vit-large-patch14-336\n")
    cfg = load_config(path, environ={})
    assert cfg["lr"] == 0.0001
    assert isinstance(cfg["lr"], float)
    assert cfg["name"] == "run-1e-4"
    assert apply_env_overrides(cfg, {"HMF_LR": "2e-4"})["lr"] == 0.0002


def test_hash_stability_and_sensitivity(tmp_path):
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("seed: 0\nname: run\nrate: 0.25\n")
    second.write_text(
        '# formatting and source path are irrelevant\n{rate: 0.25, name: "run", seed: 0}'
    )
    cfg = load_config(first, environ={})
    digest = config_hash(cfg)
    assert len(digest) == 64
    assert digest == config_hash(load_config(second, environ={}))
    assert digest == config_hash(yaml.safe_load(yaml.safe_dump(cfg)))
    assert digest == config_hash(load_config(first, environ={"SEED": "1"}))
    assert digest != config_hash({**cfg, "rate": 0.5})
    assert config_hash({"x": 1}) != config_hash({"x": 1.0})
    assert cfg == {"seed": 0, "name": "run", "rate": 0.25}


@pytest.mark.parametrize(
    "key,original,changed",
    [
        ("seed", 0, 9),
        ("num_workers", 0, 8),
        ("resume", False, True),
        ("save_epoch_preds", False, True),
        ("splits_dir", "splits", "/home/me/splits_predup"),
        ("data_dir", "data", "/content/drive/data"),
        ("feature_dir", "features", "/content/drive/features"),
        ("results_dir", "results", "/content/drive/results"),
        ("train_parquet", "train.parquet", "/content/train.parquet"),
        ("val_parquet", "val.parquet", "/content/val.parquet"),
        ("checkpoint_dir", "checkpoints", "/content/checkpoints"),
        ("checkpoint_name", "best.pt", "chosen.pt"),
        ("checkpoint_every", 2, 5),
        ("checkpoint_keep", 1, 3),
        ("path", "local", "/content"),
        ("home", "local", "/content/home"),
        ("cache_path", "cache", "/content/cache"),
        ("log_dir", "logs", "/content/logs"),
        ("cache_directory", "cache", "/content/cache"),
    ],
)
def test_hash_excludes_runtime_settings(key, original, changed):
    experiment = {"clip_model": "openai/clip-vit-large-patch14-336", "lr": 0.0001}
    cfg = {**experiment, key: original}
    before = cfg.copy()
    assert config_hash(cfg) == config_hash({**cfg, key: changed}) == config_hash(experiment)
    assert cfg == before


@pytest.mark.parametrize(
    "key,original,changed",
    [
        ("clip_model", "openai/clip-vit-large-patch14-336", "openai/clip-vit-base-patch32"),
        ("lr", 0.0001, 0.0002),
        ("weight_decay", 0.01, 0.05),
        ("batch_size", 8, 16),
        ("num_epochs", 20, 30),
        ("deterministic", True, False),
        ("max_samples", 0, 100),
    ],
)
def test_hash_keeps_experiment_settings(key, original, changed):
    assert config_hash({key: original}) != config_hash({key: changed})


def test_hash_stable_across_processes():
    cfg = {"seed": 0, "name": "répétable", "deterministic": True}
    code = (
        "import json, sys; from hmf.config import config_hash; "
        "print(config_hash(json.loads(sys.argv[1])))"
    )
    for hash_seed in ("1", "99"):
        output = subprocess.check_output(
            [sys.executable, "-c", code, json.dumps(cfg)],
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert output.strip() == config_hash(cfg)


def test_runtime_defaults():
    path = Path(__file__).resolve().parents[1] / "configs/default.yaml"
    cfg = load_config(path, environ={})
    assert cfg["seed"] == 0
    assert cfg["deterministic"] is True
    assert cfg["clip_model"] == "openai/clip-vit-large-patch14-336"


@pytest.mark.parametrize(
    "mps,cuda,expected",
    [
        (True, True, "mps"),
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
        (None, True, "cuda"),
        (None, False, "cpu"),
    ],
)
def test_device_precedence(monkeypatch, mps, cuda, expected):
    backends = SimpleNamespace()
    if mps is not None:
        backends.mps = SimpleNamespace(is_available=lambda: mps)
    torch = SimpleNamespace(
        backends=backends,
        cuda=SimpleNamespace(is_available=lambda: cuda),
        device=str,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert get_device() == expected


def test_device_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="train"):
        get_device()


def test_fhm_config_path_overrides_named_in_its_header():
    # configs/fhm/dynamic.yaml tells runs to use these variables (#19, #88).
    path = Path(__file__).resolve().parents[1] / "configs" / "fhm" / "dynamic.yaml"
    base = load_config(path, environ={})
    env = {"HMF_FEATURE_DIR": "/f", "HMF_RESULTS_DIR": "/r", "CHECKPOINT_DIR": "/c"}
    cfg = load_config(path, environ=env)
    assert (cfg["feature_dir"], cfg["results_dir"], cfg["checkpoint_dir"]) == ("/f", "/r", "/c")
    assert config_hash(cfg) == config_hash(base)  # runtime keys only
    ignored = load_config(path, environ={"HMF_CHECKPOINT_DIR": "/c"})
    assert ignored["checkpoint_dir"] == base["checkpoint_dir"]  # ENV_MAP keeps the V3 name
