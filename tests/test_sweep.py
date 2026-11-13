import json
from pathlib import Path

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")
pytest.importorskip("torch")

import yaml  # noqa: E402

from hmf import sweep  # noqa: E402
from hmf.config import config_hash, load_config  # noqa: E402
from hmf.features.cache import write_cache  # noqa: E402

D = 8
ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_TOY = {"method": "dynamic", "model": "dynamic", "feature_dim": D, "hidden_dim": 8,
               "num_heads": 2, "n_cross_layers": 1, "cross_fusion_dim": 2, "dropout": 0.2,
               "fusion_variant": "P", "router_mode": "dynamic", "tau": 1.0, "lambda_aux": 0.5,
               "lambda_rel": 0.1, "entropy_stop_grad": True}


@pytest.fixture
def toy(tmp_path):
    """A toy binary dataset; build(**overrides) writes a complete config file for it."""
    ids = [f"s{i:03d}" for i in range(100)]
    labels = np.arange(100) % 2
    rng = np.random.default_rng(0)
    pooled = rng.normal(size=(100, D))
    pooled[np.arange(100), labels] += 3.0  # a learnable signal per class
    write_cache(tmp_path / "features/toy/v1/train", ids=ids,
                image_tokens=np.concatenate([pooled[:, None], rng.normal(size=(100, 4, D))], 1),
                text_tokens=rng.normal(size=(100, 3, D)), text_mask=np.ones((100, 3), bool),
                text_pooled=pooled + rng.normal(size=(100, D)), similarity=rng.uniform(size=100),
                labels=labels, meta={"backbone": "toy", "cache_version": "v1", "git_sha": "test"})
    splits = tmp_path / "splits/toy"
    splits.mkdir(parents=True)
    (splits / "train.txt").write_text("\n".join(ids[:60]) + "\n")
    (splits / "val.txt").write_text("\n".join(ids[60:]) + "\n")

    def build(**overrides):
        cfg = {"seed": 0, "deterministic": True, "feature_dir": str(tmp_path / "features"),
               "results_dir": "results", "checkpoint_dir": "checkpoints",
               "splits_dir": str(tmp_path / "splits"), "num_workers": 0,
               "dataset": "toy", "method": "linear_probe", "model": "linear_probe",
               "cache_version": "v1", "num_classes": 2, "max_samples": 0,
               "image_dim": D, "text_dim": D, "lr": 1.0e-3, "weight_decay": 0.01,
               "warmup_frac": 0.05, "batch_size": 16, "num_epochs": 2, "patience": 5}
        cfg.update(overrides)
        path = tmp_path / f"{cfg['method']}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return path

    return build


def counted_params(record):
    return [(t["params"], t["value"]) for t in record["trials"] if not t["interrupted"]]


def test_search_spaces_match_math_spec_and_the_issue():
    # MATH_SPEC §6 ranges; the #20 spec's families (D-026).
    def unit(low, high, log=False):
        return {"type": "float", "low": low, "high": high, "log": log}

    shared = {"lr": unit(1e-5, 5e-4, log=True), "dropout": unit(0.1, 0.5),
              "weight_decay": unit(0.01, 0.05)}
    warmup = {"warmup_frac": {"type": "categorical", "choices": [0.05, 0.1]}}
    assert sweep.SEARCH_SPACES["dynamic"] == {
        **shared, "lambda_aux": unit(0.1, 1.0), "lambda_rel": unit(0.0, 1.0),
        "tau": {"type": "categorical", "choices": [0.25, 0.5, 1.0, 2.0, 4.0]},
        "hidden_dim": {"type": "categorical", "choices": [256, 512]}, **warmup}
    static = sweep.SEARCH_SPACES["static"]
    assert set(static) == set(sweep.SEARCH_SPACES["dynamic"]) - {"tau", "lambda_rel"}
    # Every baseline family also sweeps warmup_frac, as dynamic and static do (#130 review).
    assert sweep.SEARCH_SPACES["qmf"] == sweep.SEARCH_SPACES["pdf"] == {**shared, **warmup}
    assert sweep.SEARCH_SPACES["unimodal"] == {
        **shared, "hidden_dim": static["hidden_dim"], **warmup}
    assert sweep.SEARCH_SPACES["concat_mlp"] == {
        **shared, "mlp_hidden": static["hidden_dim"], **warmup}
    assert sweep.SEARCH_SPACES["linear_probe"] == {
        **{k: shared[k] for k in ("lr", "weight_decay")}, **warmup}
    assert sweep.SEARCH_SPACES["v3_alpha"] == sweep.SEARCH_SPACES["linear_probe"]
    assert all(sweep.SEARCH_SPACES[name]["warmup_frac"] == warmup["warmup_frac"]
               for name in sweep.SEARCH_SPACES)
    assert "global" not in sweep.SEARCH_SPACES


@pytest.mark.parametrize("name", sorted(sweep.SEARCH_SPACES))
def test_suggestions_stay_inside_the_bounds(name):
    space = sweep.SEARCH_SPACES[name]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    for _ in range(200):
        params = sweep.suggest(study.ask(), space)
        assert set(params) == set(space)
        for key, value in params.items():
            spec = space[key]
            if spec["type"] == "float":
                assert spec["low"] <= value <= spec["high"]
            else:
                assert value in spec["choices"] and type(value) is type(spec["choices"][0])


def test_family_follows_router_mode_and_global_is_not_swept():
    assert sweep.family({"model": "dynamic", "router_mode": "dynamic"}) == "dynamic"
    assert sweep.family({"model": "dynamic", "router_mode": "static"}) == "static"
    assert sweep.family({"model": "qmf"}) == "qmf"
    with pytest.raises(ValueError, match="D-026"):
        sweep.family({"model": "dynamic", "router_mode": "global"})
    with pytest.raises(ValueError, match="no search space"):
        sweep.family({"model": "hateclipper"})


def test_dataset_override_sets_the_class_count():
    cfg = {"dataset": "fhm", "num_classes": 2, "lr": 1e-4}
    assert sweep.for_dataset(cfg, "mvsa") == {"dataset": "mvsa", "num_classes": 3, "lr": 1e-4}
    assert sweep.for_dataset(cfg, "harm_p")["num_classes"] == 2
    with pytest.raises(ValueError, match="unknown dataset"):
        sweep.for_dataset(cfg, "fhm2")


@pytest.mark.parametrize("n_trials", [0, sweep.MAX_TRIALS + 1])
def test_trial_cap(toy, tmp_path, n_trials):
    assert sweep.MAX_TRIALS == 50  # PROTOCOL rule 5
    with pytest.raises(ValueError, match="PROTOCOL rule 5"):
        sweep.run_sweep(toy(), n_trials=n_trials, work_dir=tmp_path / "w", device="cpu")


def test_sweep_state_must_stay_outside_the_repo(toy, tmp_path):
    with pytest.raises(ValueError, match="inside the repo"):
        sweep.run_sweep(toy(), n_trials=1, work_dir=ROOT / "sweeps", device="cpu")
    with pytest.raises(ValueError, match="inside the repo"):
        sweep.run_sweep(toy(), n_trials=1, work_dir=tmp_path / "w", device="cpu",
                        storage=ROOT / "study.db")
    assert not (ROOT / "sweeps").exists()


def test_seeded_sampler_gives_the_same_first_suggestions(tmp_path):
    space = sweep.SEARCH_SPACES["dynamic"]
    draws = []
    for _ in range(2):
        sampler = sweep._load_sampler(tmp_path / "absent.pkl", resuming=False)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        draws.append([sweep.suggest(study.ask(), space) for _ in range(5)])
    assert draws[0] == draws[1]


def test_cpu_smoke_two_trials_write_the_frozen_config_and_record(toy, tmp_path):
    base = toy(**DYNAMIC_TOY)
    out = tmp_path / "out"
    record = sweep.run_sweep(base, n_trials=2, work_dir=tmp_path / "w", out_root=out,
                             device="cpu")
    trials = record["trials"]
    assert [t["state"] for t in trials] == ["COMPLETE", "COMPLETE"]
    assert record["best"]["value"] == max(t["value"] for t in trials)
    assert (record["family"], record["selection_metric"], record["seed"]) == (
        "dynamic", "auroc", 0)
    assert record["overrides"] == {}
    winner = trials[record["best"]["trial"]]
    for t in trials:  # each trial ran seed 0 in its own directory; checkpoints are gone
        run = tmp_path / f"w/toy/dynamic/trial{t['number']:03d}"
        metrics = json.loads((run / "results/toy/dynamic/seed0/metrics.json").read_text())
        assert metrics["seed"] == 0 and metrics["val"]["auroc"] == t["value"]
        assert (metrics["config_hash"], metrics["best_epoch"]) == (t["config_hash"],
                                                                  t["best_epoch"])
        assert not (run / "checkpoints").exists()
    frozen_path = out / "configs/toy/dynamic.yaml"
    assert json.loads((out / "results/toy/sweeps/dynamic.json").read_text()) == record
    frozen, raw = load_config(frozen_path, environ={}), yaml.safe_load(base.read_text())
    assert list(frozen) == list(raw)  # the base config's keys, in its order
    assert config_hash(frozen) == record["best"]["config_hash"] == winner["config_hash"]
    assert {k: frozen[k] for k in winner["params"]} == winner["params"]
    assert (frozen["results_dir"], frozen["seed"]) == ("results", 0)  # base runtime values
    unswept = set(raw) - set(sweep.SEARCH_SPACES["dynamic"])
    assert {k: frozen[k] for k in unswept} == {k: raw[k] for k in unswept}


def test_resume_matches_an_uninterrupted_sweep(toy, tmp_path, monkeypatch):
    base = toy()
    straight = sweep.run_sweep(base, n_trials=3, work_dir=tmp_path / "a",
                               out_root=tmp_path / "out_a", device="cpu")
    calls, real_train = [], sweep.train

    def interrupted_second_trial(cfg, device=None):
        calls.append(cfg)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_train(cfg, device=device)

    monkeypatch.setattr(sweep, "train", interrupted_second_trial)
    with pytest.raises(KeyboardInterrupt):
        sweep.run_sweep(base, n_trials=3, work_dir=tmp_path / "b", device="cpu")
    monkeypatch.setattr(sweep, "train", real_train)
    resumed = sweep.run_sweep(base, n_trials=3, work_dir=tmp_path / "b",
                              out_root=tmp_path / "out_b", device="cpu")
    states = [(t["state"], t["interrupted"], t["retry_of"]) for t in resumed["trials"]]
    assert states == [("COMPLETE", None, None), ("FAIL", True, None),
                      ("COMPLETE", None, 1), ("COMPLETE", None, None)]
    assert resumed["trials"][2]["params"] == resumed["trials"][1]["params"]
    assert counted_params(resumed) == counted_params(straight)
    assert (tmp_path / "out_b/configs/toy/linear_probe.yaml").read_text() == (
        tmp_path / "out_a/configs/toy/linear_probe.yaml").read_text()


def test_a_trial_left_running_by_a_crash_is_retried(toy, tmp_path):
    base, work = toy(), tmp_path / "w"
    sweep.run_sweep(base, n_trials=1, work_dir=work, out_root=tmp_path / "out", device="cpu")
    db = f"sqlite:///{work / 'toy/linear_probe/study.db'}"
    study = optuna.load_study(study_name="toy/linear_probe", storage=db)
    killed = sweep.suggest(study.ask(), sweep.SEARCH_SPACES["linear_probe"])  # never told
    record = sweep.run_sweep(base, n_trials=2, work_dir=work, out_root=tmp_path / "out",
                             device="cpu")
    states = [(t["state"], t["interrupted"], t["retry_of"]) for t in record["trials"]]
    assert states == [("COMPLETE", None, None), ("FAIL", True, None), ("COMPLETE", None, 1)]
    assert record["trials"][1]["params"] == record["trials"][2]["params"] == killed


def test_env_overrides_of_experiment_keys_are_recorded(toy, tmp_path):
    env = {"HMF_FUSION_VARIANT": "L", "HMF_FEATURE_DIR": str(tmp_path / "features")}
    record = sweep.run_sweep(toy(**DYNAMIC_TOY), n_trials=1, work_dir=tmp_path / "w",
                             out_root=tmp_path / "out", device="cpu", environ=env)
    assert record["overrides"] == {"fusion_variant": "L"}  # runtime paths are not listed
    frozen = load_config(tmp_path / "out/configs/toy/dynamic.yaml", environ={})
    assert frozen["fusion_variant"] == "L" and config_hash(frozen) == record["best"]["config_hash"]


def test_a_diverged_trial_counts_as_a_failed_trial(toy, tmp_path, monkeypatch):
    real_train, calls = sweep.train, []

    def diverge_first(cfg, device=None):
        calls.append(cfg)
        if len(calls) == 1:
            raise FloatingPointError("non-finite loss at epoch 1")
        return real_train(cfg, device=device)

    monkeypatch.setattr(sweep, "train", diverge_first)
    record = sweep.run_sweep(toy(), n_trials=2, work_dir=tmp_path / "w",
                             out_root=tmp_path / "out", device="cpu")
    assert [(t["state"], t["error"]) for t in record["trials"]] == [
        ("FAIL", "non-finite loss at epoch 1"), ("COMPLETE", None)]
    assert record["best"]["trial"] == 1 and len(calls) == 2
