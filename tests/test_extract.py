import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmf.data.common import Example  # noqa: E402
from hmf.features.cache import load_cache  # noqa: E402
from hmf.features.extract import (  # noqa: E402
    DATASETS,
    MAX_REL,
    MIN_COS,
    SPECS,
    CacheSpec,
    Preprocess,
    default_resize_filter,
    encode_batch,
    extract_split,
    main,
    pool_patch_grid,
    pooled_errors,
    pooler,
    resize_meta,
    source_splits,
    spot_check,
)


def test_pool_patch_grid_keeps_cls_and_pools_row_major():
    grid = torch.arange(16.0).reshape(4, 4)  # patch value = its row-major index
    tokens = torch.cat([torch.tensor([[-1.0]]), grid.reshape(16, 1)])[None]  # [1, 17, 1]
    pooled = pool_patch_grid(tokens)
    assert pooled.shape == (1, 5, 1)
    assert pooled[0, :, 0].tolist() == [-1.0, 2.5, 4.5, 10.5, 12.5]


def test_pool_patch_grid_gives_145_tokens_for_clip_l_336():
    assert pool_patch_grid(torch.zeros(2, 577, 8)).shape == (2, 145, 8)
    with pytest.raises(ValueError, match="square grid"):
        pool_patch_grid(torch.zeros(1, 1 + 15, 8))
    with pytest.raises(ValueError, match="square grid"):
        pool_patch_grid(torch.zeros(1, 1 + 9, 8))


def test_roles_map_to_source_splits():
    assert source_splits("fhm", ["train", "val"]) == ["train"]  # Phase 1 carve-out (D-012)
    with pytest.raises(PermissionError):
        source_splits("fhm", ["train", "test"])
    assert source_splits("fhm", ["train", "val", "test"], allow_test=True) == ["train", "dev_seen"]
    assert source_splits("crisismmd", ["val", "train"]) == ["dev", "train"]
    assert source_splits("harm_p", ["train", "val"]) == ["train", "val"]
    assert source_splits("mvsa", ["train", "val"]) == ["train", "dev"]
    assert source_splits("food101", ["val", "test"], allow_test=True) == ["dev", "test"]
    assert source_splits("mami", ["train", "val"]) == ["train"]  # 10% carve-out of train (#6)
    assert source_splits("mami", ["train", "val", "test"], allow_test=True) == ["train", "test"]
    assert set(DATASETS) == {"crisismmd", "fhm", "food101", "harm_c", "harm_p", "mami", "mvsa"}
    with pytest.raises(ValueError, match="no loader"):
        source_splits("unknown", ["train"])


def test_pooled_errors_and_d020_thresholds():
    ref = np.array([[3.0, 4.0], [1.0, 0.0]])
    got = np.array([[3.0, 4.005], [1.0, 0.0]])  # row 0: |d| = 0.005, ‖ref‖ = 5
    errors = pooled_errors(ref, got)
    assert errors["max_abs"] == pytest.approx(0.005)
    assert errors["max_rel"] == pytest.approx(1e-3)
    assert 0.9999 < errors["min_cos"] < 1.0
    assert (MAX_REL, MIN_COS) == (1e-3, 0.9999)


def test_cli_rejects_test_role_without_allow_test():
    with pytest.raises(PermissionError):
        main(["--dataset", "fhm", "--root", "/nonexistent", "--roles", "train", "test"])


def test_specs_follow_math_spec_and_hateclipper():
    assert SPECS["v1"] == CacheSpec("openai/clip-vit-large-patch14-336", 336, pooled_only=False)
    assert SPECS["hc224"] == CacheSpec("openai/clip-vit-large-patch14", 224, pooled_only=True,
                                       pre_resize=(224, 224))


class _RecordingImages:
    def __init__(self):
        self.seen = []

    def __call__(self, images, return_tensors):
        self.seen += [(image.size, image.mode) for image in images]
        return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}


def _tokenizer(texts, padding=None, truncation=None, max_length=None, return_tensors=None,
               verbose=True):
    if return_tensors:
        ones = torch.ones(len(texts), max_length, dtype=torch.long)
        return {"input_ids": ones, "attention_mask": ones}
    return {"input_ids": [[1] * len(t.split()) for t in texts]}


def test_square_resize_applies_to_hc224_only():
    # #13 parity: Hate-CLIPper squashes to 224x224 before the processor (datasets.py L47).
    from PIL import Image

    batch = [(0, Image.new("RGB", (300, 120)), "a b"), (1, Image.new("L", (90, 400)), "c")]
    hc = _RecordingImages()
    Preprocess(hc, _tokenizer, SPECS["hc224"].pre_resize)(batch)
    assert hc.seen == [((224, 224), "RGB"), ((224, 224), "RGB")]
    v1 = _RecordingImages()
    Preprocess(v1, _tokenizer, SPECS["v1"].pre_resize)(batch)
    assert v1.seen == [((300, 120), "RGB"), ((90, 400), "L")]  # CLIP's own resize and crop


def test_resize_meta_records_filter_and_pillow():
    import PIL
    from PIL import Image

    name = default_resize_filter()
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (50, 70, 3), np.uint8))
    assert np.array_equal(np.asarray(image.resize((224, 224))),
                          np.asarray(image.resize((224, 224), Image.Resampling[name])))
    assert resize_meta(SPECS["hc224"]) == {"pre_resize": [224, 224], "pre_resize_filter": name,
                                           "pil_version": PIL.__version__}
    assert resize_meta(SPECS["v1"]) == {"pre_resize": None, "pre_resize_filter": None,
                                        "pil_version": PIL.__version__}


def _fake_batches(n, batch_size, length=6):
    for start in range(0, n, batch_size):
        rows = torch.arange(start, min(n, start + batch_size))
        mask = torch.zeros(len(rows), length, dtype=torch.long)
        for i, row in enumerate(rows.tolist()):
            mask[i, : 1 + row % 3] = 1
        yield rows, rows.float()[:, None], rows[:, None].repeat(1, length), mask, 1


def _fake_encode(max_batch):
    calls = []

    def encode(pixels, input_ids, mask):
        calls.append(len(pixels))
        if len(pixels) > max_batch:
            raise torch.cuda.OutOfMemoryError("fake")
        value = pixels[:, 0]  # = the row index
        return {
            "image_tokens": value[:, None, None].repeat(1, 3, 4),
            "text_tokens": value[:, None, None].repeat(1, mask.shape[1], 4),
            "text_mask": mask.bool(),
            "text_pooled": value[:, None].repeat(1, 4),
            "similarity": value / 100,
        }

    return encode, calls


def test_extract_split_buffers_rows_in_order_and_halves_on_oom(tmp_path):
    examples = [Example(id=f"{i:03d}", image="", text="", label=None if i == 4 else i % 2,
                        split="train") for i in range(10)]
    encode, calls = _fake_encode(max_batch=2)
    out = extract_split(examples, _fake_batches(10, 4), encode, tmp_path / "fhm" / "v1" / "train",
                        meta={"backbone": "fake", "cache_version": "v1", "git_sha": "abc"},
                        batch_size=4)
    cache = load_cache(out)
    assert cache.ids == [e.id for e in examples]
    np.testing.assert_array_equal(cache.image_pooled, np.repeat(np.arange(10.0), 4).reshape(10, 4))
    np.testing.assert_allclose(cache.similarity, np.arange(10) / 100, rtol=1e-6)
    assert cache.labels.tolist() == [0, 1, 0, 1, -1, 1, 0, 1, 0, 1]
    assert cache.text_tokens.shape == (10, 3, 4)  # trimmed to the longest mask
    assert calls[:3] == [4, 2, 2] and max(calls[3:]) == 2  # halved once, then kept
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["batch_size"] == 4 and manifest["batch_size_used"] == 2
    assert manifest["texts_truncated"] == 3 and manifest["num_image_tokens"] == 3
    assert not (tmp_path / "fhm" / "v1" / ".train.partial").exists()


def test_extract_split_cleans_scratch_on_failure(tmp_path):
    encode, _ = _fake_encode(max_batch=0)
    examples = [Example(id="a", image="", text="", label=0, split="train")]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        extract_split(examples, _fake_batches(1, 1), encode, tmp_path / "c",
                      meta={"backbone": "fake", "cache_version": "v1", "git_sha": "abc"},
                      batch_size=1)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def tiny_clip():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.CLIPConfig(
        text_config={"hidden_size": 32, "intermediate_size": 37, "num_attention_heads": 4,
                     "num_hidden_layers": 2, "vocab_size": 99, "max_position_embeddings": 77,
                     "bos_token_id": 97, "eos_token_id": 98, "pad_token_id": 1},
        vision_config={"hidden_size": 48, "intermediate_size": 37, "num_attention_heads": 4,
                       "num_hidden_layers": 2, "image_size": 32, "patch_size": 8},
        projection_dim=16,
    )
    model = transformers.CLIPModel(config).eval()
    pixels = torch.randn(3, 3, 32, 32)
    input_ids = torch.randint(2, 90, (3, 9))
    mask = torch.ones_like(input_ids)
    for i, length in enumerate((9, 5, 2)):  # EOS last, padding after it
        input_ids[i, length - 1] = 98
        input_ids[i, length:] = 1
        mask[i, length:] = 0
    return model, pixels, input_ids, mask


def test_v1_rows_equal_clip_embeddings(tiny_clip):
    model, pixels, input_ids, mask = tiny_clip
    with torch.inference_mode():
        out = encode_batch(model, SPECS["v1"], pixels, input_ids, mask)
        image = pooler(model.get_image_features(pixel_values=pixels))
        text = pooler(model.get_text_features(input_ids=input_ids, attention_mask=mask))
    assert out["image_tokens"].shape == (3, 1 + 4, 16)  # 4×4 grid pooled to 2×2
    torch.testing.assert_close(out["image_tokens"][:, 0], image)
    torch.testing.assert_close(out["text_pooled"], text)
    eos = mask.sum(1) - 1
    torch.testing.assert_close(out["text_tokens"][torch.arange(3), eos], text)
    assert out["text_mask"].tolist() == mask.bool().tolist()
    torch.testing.assert_close(out["similarity"], torch.cosine_similarity(image, text, dim=-1))


def test_hc224_rows_are_pre_projection_pooler_outputs(tiny_clip):
    model, pixels, input_ids, mask = tiny_clip
    with torch.inference_mode():
        out = encode_batch(model, SPECS["hc224"], pixels, input_ids, mask)
        vision = model.vision_model(pixel_values=pixels).pooler_output
        text = model.text_model(input_ids=input_ids, attention_mask=mask).pooler_output
        image_embed = pooler(model.get_image_features(pixel_values=pixels))
        text_embed = pooler(model.get_text_features(input_ids=input_ids, attention_mask=mask))
    assert out["image_tokens"].shape == (3, 1, 48) and out["text_tokens"].shape == (3, 1, 32)
    torch.testing.assert_close(out["image_tokens"][:, 0], vision)
    torch.testing.assert_close(out["text_pooled"], text)
    assert out["text_mask"].all() and out["text_mask"].shape == (3, 1)
    torch.testing.assert_close(out["similarity"],
                               torch.cosine_similarity(image_embed, text_embed, dim=-1))


@pytest.mark.skipif(not os.environ.get("HMF_FEATURE_CACHE"),
                    reason="set HMF_FEATURE_CACHE to one split's cache dir (needs CLIP weights)")
def test_real_cache_pooled_rows_match_clip():
    # #11 acceptance (D-020): on the first 100 rows, against fp32 CLIP, per-row relative
    # error <= 1e-3 and cosine >= 0.9999 for the image and text pooled vectors.
    pytest.importorskip("transformers")
    result = spot_check(os.environ["HMF_FEATURE_CACHE"], n=100)
    assert result["n"] == 100
    for name in ("image", "text"):
        assert result[f"{name}_max_rel"] <= 1e-3, result
        assert result[f"{name}_min_cos"] >= 0.9999, result
    assert result["passed"]


def test_v1_full_keeps_every_patch(tiny_clip):
    model, pixels, input_ids, mask = tiny_clip
    with torch.inference_mode():
        full = encode_batch(model, SPECS["v1_full"], pixels, input_ids, mask)
        pooled = encode_batch(model, SPECS["v1"], pixels, input_ids, mask)
    assert full["image_tokens"].shape == (3, 1 + 16, 16)  # every token of the 4×4 grid
    torch.testing.assert_close(full["image_tokens"][:, 0], pooled["image_tokens"][:, 0])
    grid = full["image_tokens"][:, 1:].reshape(3, 4, 4, 16)
    torch.testing.assert_close(grid[:, :2, :2].mean(dim=(1, 2)), pooled["image_tokens"][:, 1])


def test_siglip_rows_are_pooled_head_then_pooled_patches():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.SiglipConfig(
        text_config={"hidden_size": 32, "intermediate_size": 37, "num_attention_heads": 4,
                     "num_hidden_layers": 2, "vocab_size": 99, "max_position_embeddings": 64},
        vision_config={"hidden_size": 32, "intermediate_size": 37, "num_attention_heads": 4,
                       "num_hidden_layers": 2, "image_size": 36, "patch_size": 6},
    )
    model = transformers.SiglipModel(config).eval()
    pixels, input_ids = torch.randn(2, 3, 36, 36), torch.randint(2, 90, (2, 64))
    mask = torch.ones_like(input_ids)
    mask[1, 10:] = 0
    spec = SPECS["siglip"]
    with torch.inference_mode():
        out = encode_batch(model, spec, pixels, input_ids, mask)
        image = pooler(model.get_image_features(pixel_values=pixels))
        text = pooler(model.get_text_features(input_ids=input_ids))
        patches = model.vision_model(pixel_values=pixels).last_hidden_state
    assert out["image_tokens"].shape == (2, 1 + 4, 32)  # 6×6 patches pooled 3×3 -> 2×2
    torch.testing.assert_close(out["image_tokens"][:, 0], image)
    torch.testing.assert_close(out["image_tokens"][:, 1],
                               patches.reshape(2, 6, 6, 32)[:, :3, :3].mean(dim=(1, 2)))
    torch.testing.assert_close(out["text_pooled"], text)
    assert out["text_tokens"].shape == (2, 64, 32)
    assert out["text_mask"][1].sum() == 10
    torch.testing.assert_close(out["similarity"], torch.cosine_similarity(image, text, dim=-1))
    assert (spec.max_text_tokens, spec.patch_pool, spec.image_size) == (64, 3, 384)
