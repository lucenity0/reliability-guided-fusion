from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("optuna")

from hmf import sweep  # noqa: E402
from hmf.config import load_config  # noqa: E402
from hmf.models.registry import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {  # method: (model, sweep family)
    "concat_mlp": ("concat_mlp", "concat_mlp"),
    "linear_probe": ("linear_probe", "linear_probe"),
    "unimodal_img": ("unimodal", "unimodal"),
    "unimodal_txt": ("unimodal", "unimodal"),
    "v3_alpha": ("v3_alpha", "v3_alpha"),
}
# V3 @ 75c6b9f: src/model.py L120-L247, src/train_dynamic_v3.py L55-L61,
# src/regularization.py L27 (#20 step 3.2: kept fixed; lr, weight_decay and warmup_frac swept).
V3_CONSTANTS = {
    "gate_query_dim": 128, "gate_hidden_dim_1": 512, "gate_hidden_dim_2": 128,
    "gate_dropout_1": 0.4, "gate_dropout_2": 0.3,
    "classifier_hidden_dim_1": 512, "classifier_hidden_dim_2": 256, "classifier_hidden_dim_3": 64,
    "classifier_dropout_1": 0.4, "classifier_dropout_2": 0.3, "classifier_dropout_3": 0.2,
    "lambda_aux": 0.3, "lambda_entropy": 0.002, "lambda_diversity": 0.05,
    "normalize_eps": 1e-12, "entropy_eps": 1e-8, "alpha_entropy_eps": 1e-7,
}


def load(name):
    return load_config(ROOT / f"configs/fhm/{name}.yaml", environ={})


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baseline_configs_build_and_share_dynamics_settings(name):
    cfg, dynamic = load(name), load("dynamic")
    model, family = BASELINES[name]
    assert (cfg["method"], cfg["model"]) == (name, model)
    assert sweep.family(cfg) == family
    assert set(sweep.SEARCH_SPACES[family]) <= cfg.keys()
    build_model(cfg)
    # dynamic.yaml's own swept keys take its sweep's values once frozen, so skip them.
    swept = set(sweep.SEARCH_SPACES["dynamic"])
    for key in cfg.keys() & dynamic.keys() - {"method", "model"} - swept:
        assert cfg[key] == dynamic[key], key


def test_unimodal_configs_differ_only_in_modality():
    img, txt = load("unimodal_img"), load("unimodal_txt")
    assert (img["modality"], txt["modality"]) == ("img", "txt")
    # Each modality's own sweep freezes its unimodal search-space keys, so skip them.
    swept = set(sweep.SEARCH_SPACES["unimodal"])
    assert {k for k in img if img[k] != txt[k]} - swept == {"method", "modality"}


def test_v3_alpha_keeps_v3s_architecture_and_loss_weights():
    cfg = load("v3_alpha")
    assert {key: cfg[key] for key in V3_CONSTANTS} == V3_CONSTANTS
    assert not V3_CONSTANTS.keys() & sweep.SEARCH_SPACES["v3_alpha"].keys()
