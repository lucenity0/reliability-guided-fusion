# Configs

One YAML per `<dataset>/<method>.yaml`. Every value can be overridden by env var
(pattern ported from V3 `src/config.py`). Frozen LMM prompt lives in `lmm_prompt.txt`.

`fhm/qmf.yaml` and `fhm/pdf.yaml` implement #23 on the same `v1` tokens and
D-017 image/text experts as `fhm/dynamic.yaml`. Only lr, dropout, weight_decay and
warmup_frac are swept; method-specific constants are explicit in YAML and validated against
the pinned official-code values. QMF uses unnormalized confidence fusion and
does not emit routing columns or router-health metrics. PDF uses L1 confidence
regression and applies relative calibration only in eval mode. Source pins and
paper/code differences are in `hmf/models/dynamic_baselines.py`.

For the review smoke run on FHM train/val caches, run each with `MAX_SAMPLES=64
NUM_EPOCHS=2 python -m hmf.train --config configs/fhm/qmf.yaml` (then `pdf.yaml`).
Use separate `HMF_RESULTS_DIR` and `CHECKPOINT_DIR` paths for smoke artifacts.
For another dataset, copy the config and set `dataset` and `num_classes` (3 for
MVSA, 101 for Food-101); runtime paths and seed overrides work as below.

`fhm/concat_mlp.yaml`, `linear_probe.yaml`, `unimodal_img.yaml`, `unimodal_txt.yaml`
and `v3_alpha.yaml` are the #20 base configs for design notes §5 rows 1-3 and 7. Their
data and training settings are `fhm/dynamic.yaml`'s base values, and the sweep tunes
only the keys in `hmf.sweep.SEARCH_SPACES` for each family. Every family, baselines
included, sweeps `warmup_frac` over {0.05, 0.1} as dynamic and static do, so the base
value 0.05 is only a starting point. `v3_alpha` sweeps lr, weight_decay and warmup_frac;
its widths, dropouts, loss weights and eps values are V3's at `75c6b9f`.

`python -m hmf.sweep` (#20) writes each frozen config as `<dataset>/<method>.yaml`:
the base config's keys and runtime values, with the winning val trial's swept
values. Its header names the base config and the sweep record
`results/<dataset>/sweeps/<method>.json`, which lists every trial.

`default.yaml` contains shared runtime defaults. Copy and extend it for a dataset
and method; `load_config(path)` loads one complete file without implicit merging.
Use a flat mapping with lowercase snake_case keys and string, integer, finite
float or boolean values. Quote strings that YAML might interpret as dates or booleans.
Strings that look like float literals are rejected: write learning rates as
`lr: 1.0e-4`, without quotes. PyYAML interprets `lr: 1e-4` as a string.

```python
from hmf.config import config_hash, get_device, load_config
from hmf.utils.seed import seed_everything

cfg = load_config("configs/default.yaml")
seed_everything(cfg["seed"], deterministic=cfg["deterministic"])
device = get_device()  # Requires the train extra; prefers MPS, then CUDA, then CPU.
digest = config_hash(cfg)
```

Environment values take precedence over YAML: `SEED=3`, `NUM_WORKERS=2` or
`DETERMINISTIC=false`. `ENV_MAP` preserves the relevant V3 variable names and
casters without a prefix. Other keys require `HMF_` followed by their uppercase
name and use their YAML type, e.g. `HMF_FEATURE_DIR=/path/to/cache`. Ambient
variables such as `PATH` and `HOME` do not override `path` and `home`; intentional
overrides use `HMF_PATH` and `HMF_HOME`. Keys in `ENV_MAP` use only their listed
unprefixed names. Only keys already in the config are overridden;
unknown variables and empty strings are ignored. `load_config(path, defaults=...)`
adds optional keys the YAML omits before the overrides; `python -m hmf.train` passes
`OPTIONAL_RUNTIME` (`splits_dir`, `save_epoch_preds`, `resume`, `checkpoint_every`), so
`HMF_SPLITS_DIR`, `SAVE_EPOCH_PREDS`, `RESUME` and `CHECKPOINT_EVERY` work with every
training config. None of these keys is in the hash, so each run's `metrics.json`
records the ID lists it read under `splits` (path, count, SHA-256). Boolean overrides accept
`true/false`, `yes/no`, `on/off` and `1/0` (case insensitive). Invalid overrides
raise `ValueError`. Paths remain literal: no cwd-dependent resolution or implicit
joining with `data_dir`; dataset loaders will interpret their own path fields.

Hash the effective config **after** all overrides. The full SHA-256 digest is
computed from sorted, compact JSON, independent of YAML key order, comments and
source path. It identifies the frozen experiment across machines and seeds.
`RUNTIME_KEYS` excludes seed, `num_workers`, `resume`, `save_epoch_preds`, `train_parquet`,
`val_parquet`, the standard data/feature/results/checkpoint directories, and
bare path/directory keys (`path`, `paths`, `dir`, `dirs`, `directory`,
`directories`, `home`). All `checkpoint_*` keys and keys ending in `_path`,
`_paths`, `_dir`, `_dirs`, `_directory` or `_directories` are excluded too;
use these suffixes for additional filesystem locations. `deterministic`,
`max_samples`, `clip_model` and every other model/training setting remain in the
hash. Types are significant (e.g. `1` and `1.0` differ). Store the hash separately
and save the **full effective config**, including seed and paths, with
`yaml.safe_dump(cfg)` for each run record; hashing does not remove these values.

Seeding works with the base install; if PyTorch is installed it also seeds its
device RNGs, configures cuDNN and requests deterministic algorithms (unsupported
operations warn). Call it before initializing CUDA or constructing a model.
It preserves an explicitly set `CUBLAS_WORKSPACE_CONFIG`, otherwise selecting
`:4096:8` in deterministic mode. Independent NumPy generators and DataLoader
workers require separate seeds. Set `PYTHONHASHSEED` before starting Python if
needed. Identical results across platforms or library versions are not promised;
see [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
