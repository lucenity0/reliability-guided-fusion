import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmf.eval.corruptions import (  # noqa: E402
    CONDITIONS,
    CorruptedFeatures,
    StagedFeatures,
    StreamingFeatures,
    swap_permutation,
    train_statistics,
)
from hmf.features.cache import write_cache  # noqa: E402
from hmf.features.dataset import CachedFeatureDataset  # noqa: E402


@pytest.fixture
def source(tmp_path, cache_arrays):
    arrays = cache_arrays(n=9, t_img=3, length=5, d=4)
    # CLS/EOS representations match, as in real projected caches.
    eos = arrays["text_mask"].sum(axis=1) - 1
    arrays["text_pooled"] = arrays["text_tokens"][np.arange(9), eos]
    arrays["text_tokens"][~arrays["text_mask"]] = 60000
    write_cache(tmp_path / "cache", **arrays)
    return CachedFeatureDataset(tmp_path / "cache")


def test_statistics_use_only_train_ids_and_ignore_padding(source):
    # A deliberately reordered carve-out shares its backing cache with held-out rows.
    ids = [source.ids[i] for i in (7, 1, 4)]
    train = CachedFeatureDataset(source.cache, ids)
    before = train_statistics(train, batch_size=1)
    rows = train.__getitems__([0, 1, 2])
    for modality in ("image", "text"):
        tokens = rows[f"{modality}_tokens"].numpy()
        values = (tokens.reshape(-1, 4) if modality == "image"
                  else tokens[rows["text_mask"].numpy()]).astype(np.float64)
        np.testing.assert_allclose(before[modality]["mean"], values.mean(axis=0))
        np.testing.assert_allclose(before[modality]["std"], values.std(axis=0))
        assert before[modality]["count"] == len(values)
    # Poison every other source row, including val/test sentinels.
    for key in ("image_tokens", "text_tokens"):
        array = np.load(source.cache.path / f"{key}.npy", mmap_mode="r+")
        array[[0, 2, 3, 5, 6, 8]] = -60000
        array.flush()
    after = train_statistics(train, batch_size=3)
    for modality in before:
        np.testing.assert_allclose(before[modality]["mean"], after[modality]["mean"])
        np.testing.assert_allclose(before[modality]["std"], after[modality]["std"])


@pytest.mark.parametrize("n", [2, 3, 9, 500])
def test_swap_is_deterministic_bijective_and_has_no_fixed_points(n):
    p = swap_permutation(n)
    np.testing.assert_array_equal(p, swap_permutation(n))
    np.testing.assert_array_equal(np.sort(p), np.arange(n))
    assert np.all(p != np.arange(n))


def test_singleton_swap_fails():
    with pytest.raises(ValueError, match="at least two"):
        swap_permutation(1)


def test_streaming_copies_selected_rows_without_faulting_source_mmaps(source, monkeypatch):
    positions = [8, 2, 3, 0]
    expected = source.__getitems__(positions)
    with monkeypatch.context() as patch:
        patch.setattr(np.memmap, "__getitem__",
                      lambda *a: pytest.fail("faulted source mmap instead of buffered read"))
        with StreamingFeatures(source) as streamed:
            actual = streamed.__getitems__(positions)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    # The borrowed view must not close or alter the original source cache.
    for key in expected:
        torch.testing.assert_close(source.__getitems__(positions)[key], expected[key])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_corruptions_preserve_labels_intact_modality_and_source(source, condition, tmp_path):
    stats, permutation = train_statistics(source), swap_permutation(len(source))
    directory = tmp_path / "stage"
    directory.mkdir()
    staged = StagedFeatures(source, directory, 4)
    corrupted = CorruptedFeatures(staged, condition, stats, permutation)
    positions = [0, 3, 8]
    clean = source.__getitems__(positions)
    changed = corrupted.__getitems__(positions)
    modality = condition.split("-")[1]
    other = "text" if modality == "image" else "image"
    for key in ("label", "index", f"{other}_tokens", f"{other}_pooled"):
        torch.testing.assert_close(clean[key], changed[key], rtol=0, atol=0)
    if condition != "swap-text":
        torch.testing.assert_close(clean["text_mask"], changed["text_mask"])
    # No mutation of original arrays or staged clean inputs.
    for key in clean:
        torch.testing.assert_close(clean[key], staged.__getitems__(positions)[key], rtol=0, atol=0)
    torch.testing.assert_close(changed["similarity"], torch.nn.functional.cosine_similarity(
        changed["image_pooled"], changed["text_pooled"], dim=-1))
    if condition.startswith("drop"):
        expected = torch.tensor(stats[modality]["mean"], dtype=torch.float32)
        torch.testing.assert_close(changed[f"{modality}_tokens"],
                                   expected.expand_as(changed[f"{modality}_tokens"]))
        torch.testing.assert_close(changed[f"{modality}_pooled"],
                                   expected.expand_as(changed[f"{modality}_pooled"]))
    elif condition.startswith("swap"):
        donor = source.__getitems__(permutation[positions])
        for key in (f"{modality}_tokens", f"{modality}_pooled"):
            torch.testing.assert_close(changed[key], donor[key])
        if modality == "text":
            torch.testing.assert_close(changed["text_mask"], donor["text_mask"])
    else:
        # Independent of access order, chunk size and repeated invocations.
        for i, position in enumerate(positions):
            alone = corrupted.__getitems__([position])
            for key in changed:
                torch.testing.assert_close(changed[key][i], alone[key][0], rtol=0, atol=0)
        eos = changed["text_mask"].long().sum(1) - 1
        torch.testing.assert_close(changed["image_pooled"], changed["image_tokens"][:, 0])
        torch.testing.assert_close(changed["text_pooled"],
                                   changed["text_tokens"][torch.arange(3), eos])


def test_noise_scales_with_sigma_and_train_std(source):
    stats = train_statistics(source)
    stats["image"]["std"][0] = 0  # constant train dimension gets no noise
    p, positions = swap_permutation(len(source)), [2, 4, 8]
    original = source.__getitems__(positions)["image_tokens"]
    low = CorruptedFeatures(source, "noise-image-0.5", stats, p).__getitems__(positions)
    high = CorruptedFeatures(source, "noise-image-1.0", stats, p).__getitems__(positions)
    torch.testing.assert_close(high["image_tokens"] - original,
                               2 * (low["image_tokens"] - original))
    torch.testing.assert_close(high["image_tokens"][:, :, 0], original[:, :, 0])
