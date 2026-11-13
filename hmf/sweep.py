"""Optuna hyperparameter sweep on val only (#20; PROTOCOL rule 5, MATH_SPEC §6, D-026).

Usage::

    python -m hmf.sweep --config configs/fhm/dynamic.yaml --n-trials 20 --work-dir ~/sweeps
    python -m hmf.sweep --config configs/fhm/static.yaml --dataset mami --n-trials 20 \\
        --work-dir ~/sweeps

- Sampler and storage: ``TPESampler(seed=0)``, with the study in an SQLite file outside
  the repo (``--storage``, default ``<work_dir>/<dataset>/<method>/study.db``). Rerun the
  same command to resume an interrupted sweep. The sampler is pickled next to the study
  after each trial's suggestions, so a resumed sweep draws what an uninterrupted one
  would. A trial that never finished is marked ``interrupted``, does not count towards
  the budget, and runs again with the same parameters.
- Budget: ``--n-trials`` finished trials per (dataset, method), at most ``MAX_TRIALS``
  (PROTOCOL rule 5). A trial whose loss turns non-finite counts, as a failed trial.
- Objective: ``hmf.train.train`` with seed 0 on the train and val roles, returning the
  selection metric: val AUROC for binary tasks, val accuracy for multi-class (D-018).
  Each trial runs under ``<work_dir>/<dataset>/<method>/trial<NNN>/``, and its checkpoint
  directory is deleted once the trial ends.
- Search spaces: ``SEARCH_SPACES``, keyed by family: ``router_mode`` for the ``dynamic``
  model, otherwise the model name. ``global`` is not swept; it reuses static's best values.
- ``--dataset`` runs a base config on another dataset, setting ``dataset`` and
  ``num_classes``. The record lists every experiment key that ``--dataset`` or an env
  override changed from the base file under ``overrides``.
- Outputs under ``--out-root`` (default: the current directory, i.e. the repo):
  ``configs/<dataset>/<method>.yaml`` is the frozen best config: the base config's keys
  and runtime values, with the winning trial's experiment settings.
  ``results/<dataset>/sweeps/<method>.json`` holds every trial's params, val metric and
  best epoch, plus the winner's config hash.

The test role is never read: trials go through ``hmf.train.train`` (PROTOCOL rules 1-2).
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import warnings
from pathlib import Path

import optuna
import torch
import yaml
from optuna.trial import TrialState

from hmf.config import config_hash, is_runtime_key, load_config
from hmf.eval.guard import git_sha
from hmf.train import OPTIONAL_RUNTIME, train

MAX_TRIALS = 50  # PROTOCOL rule 5
SEED = 0  # PROTOCOL rule 5: every trial trains with seed 0
SAMPLER_SEED = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CLASSES = {"fhm": 2, "mami": 2, "harm_c": 2, "harm_p": 2, "crisismmd": 2,
                   "mvsa": 3, "food101": 101}  # PROTOCOL splits table


def _float(low: float, high: float, log: bool = False) -> dict:
    return {"type": "float", "low": low, "high": high, "log": log}


def _choice(*choices) -> dict:
    return {"type": "categorical", "choices": list(choices)}


_SHARED = {"lr": _float(1e-5, 5e-4, log=True), "dropout": _float(0.1, 0.5),
           "weight_decay": _float(0.01, 0.05)}
_WARMUP = {"warmup_frac": _choice(0.05, 0.1)}  # every family: baselines get our schedule space
_DYNAMIC = {"lambda_aux": _float(0.1, 1.0), "lambda_rel": _float(0.0, 1.0),
            "tau": _choice(0.25, 0.5, 1.0, 2.0, 4.0), **_SHARED,
            "hidden_dim": _choice(256, 512), **_WARMUP}
SEARCH_SPACES = {  # MATH_SPEC §6 ranges; dimensions per family from the #20 spec
    "dynamic": _DYNAMIC,
    "static": {k: v for k, v in _DYNAMIC.items() if k not in ("tau", "lambda_rel")},  # no router
    "concat_mlp": {**_SHARED, "mlp_hidden": _choice(256, 512), **_WARMUP},
    "linear_probe": {"lr": _SHARED["lr"], "weight_decay": _SHARED["weight_decay"], **_WARMUP},
    "unimodal": {**_SHARED, "hidden_dim": _choice(256, 512), **_WARMUP},
    # #20 step 3.2 (a): V3's widths, dropouts and loss weights stay fixed
    "v3_alpha": {"lr": _SHARED["lr"], "weight_decay": _SHARED["weight_decay"], **_WARMUP},
    "qmf": {**_SHARED, **_WARMUP},  # #23: method constants stay at the official-code defaults
    "pdf": {**_SHARED, **_WARMUP},
}


def family(cfg: dict) -> str:
    name = cfg["router_mode"] if cfg["model"] == "dynamic" else cfg["model"]
    if name == "global":
        raise ValueError("global is not swept: it reuses static's best values (D-026)")
    if name not in SEARCH_SPACES:
        raise ValueError(f"no search space for {name!r}; swept families: {sorted(SEARCH_SPACES)}")
    return name


def for_dataset(cfg: dict, dataset: str) -> dict:
    if dataset not in DATASET_CLASSES:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASET_CLASSES)}")
    return {**cfg, "dataset": dataset, "num_classes": DATASET_CLASSES[dataset]}


def suggest(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec["log"])
        else:
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def trial_config(base: dict, params: dict, trial_dir: Path) -> dict:
    return {**base, **params, "seed": SEED, "resume": False,
            "results_dir": str(trial_dir / "results"),
            "checkpoint_dir": str(trial_dir / "checkpoints")}


def _outside_repo(path: Path, what: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{what} {path} is inside the repo; keep sweep state outside it")
    return path


def _counted(study: optuna.Study) -> list:
    """Finished trials that use the budget; interrupted ones are retried instead."""
    return [t for t in study.get_trials(deepcopy=False)
            if t.state.is_finished() and not t.user_attrs.get("interrupted")]


def _recover(study: optuna.Study, storage, sweep_dir: Path) -> None:
    """Fail trials left RUNNING by a crash, and queue every unfinished trial's params again."""
    study_id = storage.get_study_id_from_name(study.study_name)
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, t.number)
        storage.set_trial_user_attr(trial_id, "interrupted", True)
        storage.set_trial_state_values(trial_id, TrialState.FAIL)
    trials = study.get_trials(deepcopy=False)
    retried = {t.user_attrs.get("retry_of") for t in trials}
    for t in trials:
        if t.user_attrs.get("interrupted") and t.number not in retried:
            shutil.rmtree(sweep_dir / f"trial{t.number:03d}" / "checkpoints", ignore_errors=True)
            study.enqueue_trial(t.params, user_attrs={"retry_of": t.number})


def _save_sampler(sampler, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(sampler))
    tmp.replace(path)


def _load_sampler(path: Path, resuming: bool):
    if path.exists():
        return pickle.loads(path.read_bytes())
    if resuming:
        warnings.warn(f"{path} is missing: the resumed sampler restarts at seed {SAMPLER_SEED}",
                      stacklevel=3)
    return optuna.samplers.TPESampler(seed=SAMPLER_SEED)


def _trial_record(t) -> dict:
    attrs = t.user_attrs
    return {"number": t.number, "state": t.state.name, "params": t.params, "value": t.value,
            **{key: attrs.get(key) for key in ("best_epoch", "epochs_run", "train_seconds",
                                               "config_hash", "git_sha", "val", "retry_of",
                                               "interrupted", "error")}}


def run_sweep(config_path: str | Path, *, n_trials: int, work_dir: str | Path,
              dataset: str | None = None, storage: str | Path | None = None,
              out_root: str | Path = ".", device=None, environ=None) -> dict:
    if not 1 <= n_trials <= MAX_TRIALS:
        raise ValueError(f"n_trials must be between 1 and {MAX_TRIALS} (PROTOCOL rule 5)")
    config_path = Path(config_path)
    base = load_config(config_path, environ=environ, defaults=OPTIONAL_RUNTIME)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if dataset is not None:
        base = for_dataset(base, dataset)
    name = family(base)
    space = SEARCH_SPACES[name]
    missing = [key for key in space if key not in raw]
    if missing:
        raise ValueError(f"{config_path} lacks swept keys {missing}; the frozen config "
                         "must keep the base config's keys")
    ds, method = base["dataset"], base["method"]
    sweep_dir = _outside_repo(Path(work_dir), "work_dir") / ds / method
    sweep_dir.mkdir(parents=True, exist_ok=True)
    db = _outside_repo(Path(storage) if storage else sweep_dir / "study.db", "storage")
    resuming = db.exists()
    rdb = optuna.storages.RDBStorage(f"sqlite:///{db}")
    sampler_path = sweep_dir / "sampler.pkl"
    study = optuna.create_study(study_name=f"{ds}/{method}", storage=rdb, direction="maximize",
                                sampler=_load_sampler(sampler_path, resuming),
                                load_if_exists=True)
    _recover(study, rdb, sweep_dir)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        _save_sampler(study.sampler, sampler_path)  # after this trial's draws (resume parity)
        trial_dir = sweep_dir / f"trial{trial.number:03d}"
        shutil.rmtree(trial_dir, ignore_errors=True)
        cfg = trial_config(base, params, trial_dir)
        trial.set_user_attr("config_hash", config_hash(cfg))
        try:
            metrics = train(cfg, device=device)
        except FloatingPointError as exc:  # diverged: a finished, failed trial
            trial.set_user_attr("error", str(exc))
            raise
        except BaseException:  # interrupted (Ctrl-C, OOM, ...): retried on resume
            trial.set_user_attr("interrupted", True)
            raise
        finally:
            shutil.rmtree(trial_dir / "checkpoints", ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for key in ("best_epoch", "epochs_run", "train_seconds", "git_sha", "val"):
            trial.set_user_attr(key, metrics[key])
        value = metrics["val"][metrics["selection_metric"]]
        print(f"{ds}/{method} trial {trial.number}: val {metrics['selection_metric']} "
              f"{value:.4f} (best epoch {metrics['best_epoch']})", flush=True)
        return value

    while len(_counted(study)) < n_trials:
        study.optimize(objective, n_trials=1, catch=(FloatingPointError,))
    counted = _counted(study)
    complete = [t for t in counted if t.state == TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"{ds}/{method}: no trial completed")
    best = max(complete, key=lambda t: t.value)  # ties keep the earliest trial

    best_cfg = trial_config(base, best.params, sweep_dir / f"trial{best.number:03d}")
    best_hash = best.user_attrs["config_hash"]
    frozen = {key: raw[key] if is_runtime_key(key) else best_cfg[key] for key in raw}
    if config_hash(frozen) != best_hash:
        raise RuntimeError("the frozen config does not reproduce the winning trial's hash")
    out_root = Path(out_root)
    config_out = Path("configs") / ds / f"{method}.yaml"
    record_out = Path("results") / ds / "sweeps" / f"{method}.json"
    metric = "accuracy" if base["num_classes"] > 2 else "auroc"
    header = (f"# {ds}, {method}: frozen by hmf.sweep (#20). Best of {len(counted)} trials on "
              f"val {metric}, seed {SEED}.\n# Base config: {config_path.as_posix()}. "
              f"Swept: {', '.join(space)}.\n# Sweep record: {record_out.as_posix()}\n")
    (out_root / config_out).parent.mkdir(parents=True, exist_ok=True)
    (out_root / config_out).write_text(header + yaml.safe_dump(frozen, sort_keys=False),
                                       encoding="utf-8")
    if config_hash(load_config(out_root / config_out, environ={})) != best_hash:
        raise RuntimeError(f"{config_out} does not reload to the winning trial's hash")

    record = {
        "dataset": ds, "method": method, "family": name,
        "base_config": config_path.as_posix(),
        "overrides": {key: base[key] for key in raw  # --dataset and env, experiment keys only
                      if not is_runtime_key(key) and base[key] != raw[key]},
        "selection_metric": metric, "direction": "maximize",
        "sampler": f"TPESampler(seed={SAMPLER_SEED})", "seed": SEED,
        "n_trials": n_trials, "max_trials": MAX_TRIALS, "optuna_version": optuna.__version__,
        "git_sha": git_sha(), "search_space": space,
        "best": {"trial": best.number, "value": best.value, "params": best.params,
                 "best_epoch": best.user_attrs["best_epoch"], "config_hash": best_hash,
                 "config": config_out.as_posix()},
        "trials": [_trial_record(t) for t in study.get_trials(deepcopy=False)],
    }
    (out_root / record_out).parent.mkdir(parents=True, exist_ok=True)
    (out_root / record_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna sweep on val (seed 0) for one config.")
    parser.add_argument("--config", required=True,
                        help="base config, e.g. configs/fhm/dynamic.yaml")
    parser.add_argument("--dataset", help="run the base config on this dataset instead")
    parser.add_argument("--n-trials", type=int, required=True,
                        help=f"finished trials per (dataset, method), at most {MAX_TRIALS}")
    parser.add_argument("--work-dir", required=True,
                        help="trial runs, sampler and study DB; must be outside the repo")
    parser.add_argument("--storage", help="SQLite file (default <work-dir>/<dataset>/<method>/"
                                          "study.db)")
    parser.add_argument("--out-root", default=".",
                        help="where configs/<dataset>/ and results/<dataset>/sweeps/ are written")
    args = parser.parse_args(argv)
    record = run_sweep(args.config, n_trials=args.n_trials, work_dir=args.work_dir,
                       dataset=args.dataset, storage=args.storage, out_root=args.out_root)
    print(json.dumps(record["best"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
