import json

import numpy as np
import pytest

from hmf.features.cache import load_cache, write_cache


def test_roundtrip_trims_text_and_memory_maps(cache_arrays, tmp_path):
    data = cache_arrays()
    out = write_cache(tmp_path / "fhm" / "v1" / "train", **data)
    cache = load_cache(out)
    longest = int(data["text_mask"].sum(1).max())
    assert cache.text_tokens.shape == (12, longest, 8)
    assert isinstance(cache.image_tokens, np.memmap)
    assert cache.image_tokens.dtype == np.float16
    np.testing.assert_allclose(cache.image_pooled, data["image_tokens"][:, 0], atol=1e-2)
    np.testing.assert_array_equal(cache.text_mask, data["text_mask"][:, :longest])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["num_image_tokens"] == 5 and manifest["backbone"] == "clip-vit-l14-336"
    assert manifest["max_text_tokens"] == longest


def test_indices_follow_requested_order(cache_arrays, tmp_path):
    cache = load_cache(write_cache(tmp_path / "c", **cache_arrays()))
    assert cache.indices(["00003", "00000"]).tolist() == [3, 0]
    with pytest.raises(KeyError):
        cache.indices(["99999"])


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(ids=d["ids"][:-1] + [d["ids"][0]]), "unique"),
        (lambda d: d.update(similarity=d["similarity"][:-1]), "similarity has shape"),
        (lambda d: d["meta"].pop("git_sha"), "missing"),
        (lambda d: d["text_mask"].__setitem__(0, False), "unmasked"),
    ],
)
def test_invalid_inputs_are_rejected(cache_arrays, tmp_path, change, message):
    data = cache_arrays()
    change(data)
    with pytest.raises(ValueError, match=message):
        write_cache(tmp_path / "c", **data)


def test_tampered_ids_are_detected(cache_arrays, tmp_path):
    out = write_cache(tmp_path / "c", **cache_arrays())
    ids = json.loads((out / "ids.json").read_text())
    (out / "ids.json").write_text(json.dumps(ids[::-1]))
    with pytest.raises(ValueError, match="does not match"):
        load_cache(out)


def test_pooled_only_cache_with_different_image_and_text_widths(tmp_path, cache_arrays):
    # Hate-CLIPper features (#13): pre-projection pooler outputs, 1024-d image vs 768-d text.
    arrays = cache_arrays(n=6, t_img=1, d=6)
    arrays["text_tokens"] = arrays["text_pooled"][:, None, :4].copy()
    arrays["text_mask"] = np.ones((6, 1), dtype=bool)
    arrays["text_pooled"] = arrays["text_pooled"][:, :4].copy()
    cache = load_cache(write_cache(tmp_path / "hc", **arrays))
    assert (cache.manifest["dim"], cache.manifest["text_dim"]) == (6, 4)
    assert cache.image_pooled.shape == (6, 6) and cache.text_pooled.shape == (6, 4)

    arrays["text_pooled"] = np.zeros((6, 5), dtype=np.float32)  # disagrees with text_tokens
    with pytest.raises(ValueError, match="text_pooled"):
        write_cache(tmp_path / "bad", **arrays)


def test_fp16_memmaps_are_written_without_a_copy(cache_arrays, tmp_path, monkeypatch):
    # Food-101 train image tokens are ~14 GB: write_cache must stream fp16 memmaps (#87).
    arrays = cache_arrays()
    for name in ("image_tokens", "text_tokens", "text_pooled"):
        mm = np.lib.format.open_memmap(tmp_path / f"in_{name}.npy", mode="w+",
                                       dtype=np.float16, shape=arrays[name].shape)
        mm[:] = arrays[name]
        arrays[name] = mm
    saved, real_save = {}, np.save

    def spy(path, array, *args, **kwargs):
        saved[path.name] = array
        return real_save(path, array, *args, **kwargs)

    monkeypatch.setattr(np, "save", spy)
    out = write_cache(tmp_path / "c", **arrays)
    for name in ("image_tokens", "text_tokens", "text_pooled"):
        assert np.shares_memory(saved[f"{name}.npy"], arrays[name]), name
    cache = load_cache(out)
    longest = int(arrays["text_mask"].sum(1).max())
    np.testing.assert_array_equal(cache.text_tokens, arrays["text_tokens"][:, :longest])
    np.testing.assert_array_equal(cache.image_tokens, arrays["image_tokens"])
