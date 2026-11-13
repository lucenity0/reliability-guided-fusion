"""#30 code: LoRA adapters and end-to-end fine-tuning (code only; dropped from the paper)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from hmf.config import load_config  # noqa: E402
from hmf.data.common import Example  # noqa: E402
from hmf.models.lora import LoRALinear, apply_lora, lora_state_dict  # noqa: E402


def tiny_clip():
    torch.manual_seed(0)
    config = transformers.CLIPConfig(
        text_config={"hidden_size": 32, "intermediate_size": 37, "num_attention_heads": 4,
                     "num_hidden_layers": 3, "vocab_size": 99, "max_position_embeddings": 77,
                     "bos_token_id": 97, "eos_token_id": 98, "pad_token_id": 1},
        vision_config={"hidden_size": 48, "intermediate_size": 37, "num_attention_heads": 4,
                       "num_hidden_layers": 3, "image_size": 32, "patch_size": 8},
        projection_dim=16)
    return transformers.CLIPModel(config).eval()


def test_lora_starts_as_frozen_clip_and_trains_only_adapters():
    model = tiny_clip()
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model.vision_model(pixel_values=pixels).last_hidden_state
    adapted = apply_lora(model, last_n=2, rank=4, alpha=8)
    assert len(adapted) == 2 * 2 * 2  # two towers × last two blocks × q and v
    with torch.no_grad():
        after = model.vision_model(pixel_values=pixels).last_hidden_state
    torch.testing.assert_close(before, after)  # B = 0 at initialisation
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)
    assert set(lora_state_dict(model)) == {n for n in trainable}
    with pytest.raises(ValueError, match="already"):
        apply_lora(model, last_n=1)
    with pytest.raises(ValueError, match="fewer"):
        apply_lora(tiny_clip(), last_n=9)


def test_lora_update_is_low_rank_and_scaled():
    base = torch.nn.Linear(6, 5)
    layer = LoRALinear(base, rank=2, alpha=4)
    with torch.no_grad():
        layer.lora_b.normal_()
    x = torch.randn(3, 6)
    expected = base(x) + 2.0 * (x @ layer.lora_a.T @ layer.lora_b.T)
    torch.testing.assert_close(layer(x), expected)


def fake_preprocess(batch):
    rows, images, texts = zip(*batch, strict=True)
    pixels = torch.stack([torch.from_numpy(np.asarray(im.resize((32, 32)), dtype=np.float32)
                                           ).permute(2, 0, 1) / 255 for im in images])
    ids = torch.full((len(texts), 6), 1)
    mask = torch.zeros_like(ids)
    for i, text in enumerate(texts):
        n = min(len(text.split()), 5)
        ids[i, :n] = torch.tensor([2 + (hash(w) % 80) for w in text.split()[:n]])
        ids[i, n] = 98
        mask[i, : n + 1] = 1
    return torch.tensor(rows), pixels, ids, mask, 0


def test_finetune_trains_fusion_and_adapters_on_images(tmp_path):
    from PIL import Image

    from hmf.finetune import finetune

    rng = np.random.default_rng(0)
    examples = []
    for i in range(24):
        path = tmp_path / f"{i}.png"
        colour = (200, 40, 40) if i % 2 else (40, 40, 200)
        Image.fromarray((np.ones((40, 40, 3)) * colour + rng.integers(0, 20, (40, 40, 3)))
                        .astype(np.uint8)).save(path)
        examples.append(Example(str(i), str(path), "red meme" if i % 2 else "blue",
                                i % 2, "train"))
    cfg = {**load_config("configs/fhm/dynamic.yaml", environ={}),
           "feature_dim": 16, "hidden_dim": 8, "num_heads": 2, "cross_fusion_dim": 4,
           "batch_size": 8, "num_epochs": 2, "patience": 2, "max_samples": 0,
           "lora_rank": 2, "lora_alpha": 4, "lora_last_n": 1, "lora_lr": 1e-3,
           "results_dir": str(tmp_path / "results"), "checkpoint_dir": str(tmp_path / "ckpt")}
    clip = tiny_clip()
    metrics = finetune(cfg, examples[:16], examples[16:], clip=clip,
                       preprocess=fake_preprocess, device="cpu")
    assert metrics["n_val"] == 8 and len(metrics["lora_modules"]) == 4
    assert metrics["lora_params"] > 0 and 1 <= metrics["best_epoch"] <= 2
    run = tmp_path / "results/fhm/dynamic/seed0"
    saved = json.loads((run / "metrics.json").read_text())
    assert saved["val"]["auroc"] == metrics["val"]["auroc"]
    assert (run / "preds_val.csv").read_text().count("\n") == 9
    state = torch.load(tmp_path / "ckpt/fhm/dynamic/seed0/best.pt")
    assert set(state["lora"]) == set(lora_state_dict(clip))


@pytest.mark.parametrize("group", ["lora", "full_tokens", "siglip"])
def test_paid_ablation_configs_build_and_are_not_cleared_for_test(group):
    from pathlib import Path

    from hmf.config import config_hash
    from hmf.models.registry import build_model

    root = Path(__file__).resolve().parents[1]
    listed = (root / "results/FINAL_CONFIGS.md").read_text()
    for method in ("dynamic", "static"):
        cfg = load_config(root / f"configs/fhm/{group}/{method}.yaml", environ={})
        assert cfg["method"] == f"{group}_{method}"
        assert config_hash(cfg) not in listed
        build_model(cfg)
