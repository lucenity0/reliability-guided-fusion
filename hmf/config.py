"""YAML configuration, typed environment overrides and reproducible hashes.

ENV_MAP, apply_env_overrides and get_device are adapted from V3 src/config.py:
https://github.com/lucenity0/A-Unified-Adaptive-Framework-for-Multimodal-Data-Fusion-with-Dynamic-Modality-Reweighting
at commit 75c6b9f5424cfd65b04a3c29c89ded40a7ce38cd.
V3 training tricks and dataset-specific filenames are deliberately not ported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import torch

Scalar = str | int | float | bool


def _as_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected true/false, yes/no, on/off or 1/0")


# env var -> (config key, caster); only keys present in the YAML are overridden.
ENV_MAP = {
    "DATA_DIR": ("data_dir", str),
    "TRAIN_PARQUET": ("train_parquet", str),
    "VAL_PARQUET": ("val_parquet", str),
    "CHECKPOINT_DIR": ("checkpoint_dir", str),
    "CHECKPOINT_NAME": ("checkpoint_name", str),
    "CLIP_MODEL": ("clip_model", str),
    "BATCH_SIZE": ("batch_size", int),
    "GRAD_ACCUM": ("grad_accum", int),
    "NUM_WORKERS": ("num_workers", int),
    "FUSION_LR": ("fusion_lr", float),
    "NUM_EPOCHS": ("num_epochs", int),
    "PATIENCE": ("patience", int),
    "CHECKPOINT_EVERY": ("checkpoint_every", int),
    "MAX_SAMPLES": ("max_samples", int),
    "RESUME": ("resume", _as_bool),
    "SEED": ("seed", int),
    "DETERMINISTIC": ("deterministic", _as_bool),
    "SAVE_EPOCH_PREDS": ("save_epoch_preds", _as_bool),
}

# These settings describe a run, not the frozen experiment registered for final
# evaluation. Path/directory suffixes and checkpoint_* are also excluded below.
RUNTIME_KEYS = frozenset(
    {
        "seed",
        "num_workers",
        "resume",
        "save_epoch_preds",
        "train_parquet",
        "val_parquet",
        "data_dir",
        "feature_dir",
        "results_dir",
        "checkpoint_dir",
        "checkpoint_name",
        "checkpoint_every",
        "path",
        "paths",
        "dir",
        "dirs",
        "directory",
        "directories",
        "home",
    }
)
_PATH_SUFFIXES = ("_path", "_paths", "_dir", "_dirs", "_directory", "_directories")
_FLOAT_LITERAL = re.compile(r"[+-]?(?:(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+|\d+\.\d*|\.\d+)")


def _validate(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a flat YAML mapping")
    for key, value in config.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"config key must be lowercase snake_case: {key!r}")
        if type(value) not in (str, int, float, bool):
            raise ValueError(f"{key}: expected a string, integer, float or boolean")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key}: expected a finite float")
        if isinstance(value, str) and _FLOAT_LITERAL.fullmatch(value.strip()):
            raise ValueError(
                f"{key}: float literal parsed as a string; "
                "write a YAML number such as 1.0e-4 without quotes"
            )


def apply_env_overrides(
    cfg: dict[str, Scalar], environ: Mapping[str, str] | None = None
) -> dict[str, Scalar]:
    """Update cfg in place, with nonempty env values taking precedence.

    ENV_MAP keeps V3 spellings/casters. Additional scalar keys use HMF_ plus their
    uppercase name and YAML value type. Unknown env vars and empty strings are ignored.
    Invalid values raise ValueError without partially updating cfg.
    """
    _validate(cfg)
    env = os.environ if environ is None else environ
    updated = cfg.copy()
    mapped_keys = {key for key, _ in ENV_MAP.values()}
    env_map = {
        f"HMF_{key.upper()}": (key, _as_bool if type(value) is bool else type(value))
        for key, value in cfg.items()
        if key not in mapped_keys
    }
    env_map.update(ENV_MAP)
    for env_name, (key, caster) in env_map.items():
        if key not in cfg:
            continue
        value = env.get(env_name)
        if value is None or value == "":
            continue
        try:
            updated[key] = caster(value)
        except ValueError as exc:
            raise ValueError(f"invalid {env_name} override for {key}: {exc}") from exc
    _validate(updated)
    cfg.update(updated)
    return cfg


def load_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    defaults: Mapping[str, Scalar] | None = None,
) -> dict[str, Scalar]:
    """Load one complete YAML config, then apply env overrides (no implicit merge).

    ``defaults`` adds optional keys the YAML omits, before the overrides, so their env
    variables apply too (overrides only reach keys present). YAML values win.
    """
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if defaults and isinstance(cfg, dict):
        cfg = {**defaults, **cfg}
    return apply_env_overrides(cfg, environ)


def is_runtime_key(key: str) -> bool:
    """True for keys that config_hash ignores: RUNTIME_KEYS, checkpoint_* and paths."""
    return key in RUNTIME_KEYS or key.startswith("checkpoint_") or key.endswith(_PATH_SUFFIXES)


def config_hash(cfg: dict[str, Scalar]) -> str:
    """SHA-256 of experiment settings in canonical JSON, shared across runs.

    Exclude RUNTIME_KEYS, checkpoint_* and path/directory keys. Keep all other
    values, including deterministic, max_samples, model and training settings.
    YAML formatting, source filename and key order do not affect the hash.
    cfg stays intact for the full run record; store the digest separately.
    """
    _validate(cfg)
    experiment = {key: value for key, value in cfg.items() if not is_runtime_key(key)}
    payload = json.dumps(
        experiment, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_device() -> torch.device:
    """Select an available PyTorch device in MPS -> CUDA -> CPU order."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ImportError('device selection requires PyTorch; install hmf with ".[train]"') from exc

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
