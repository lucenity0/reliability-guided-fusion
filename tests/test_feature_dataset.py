import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmf.features.cache import load_cache, write_cache  # noqa: E402
from hmf.features.dataset import CachedFeatureDataset, make_loader  # noqa: E402


@pytest.fixture
def cache_dir(tmp_path, cache_arrays):
    return write_cache(tmp_path / "c", **cache_arrays(n=20))


def test_item_shapes_and_values(cache_dir):
    ds = CachedFeatureDataset(cache_dir)
    item = ds[3]
    cache = load_cache(cache_dir)
    assert item["image_tokens"].shape == (5, 8) and item["image_tokens"].dtype == torch.float32
    assert item["text_mask"].dtype == torch.bool
    assert item["label"].item() == int(cache.labels[3])
    np.testing.assert_allclose(item["image_pooled"].numpy(), cache.image_tokens[3, 0], atol=1e-3)


def test_role_subset_and_batch_order(cache_dir):
    ids = ["00007", "00002", "00011"]
    ds = CachedFeatureDataset(cache_dir, ids=ids)
    assert len(ds) == 3
    batch = ds.__getitems__([2, 0, 1])  # unsorted request
    cache = load_cache(cache_dir)
    expect = cache.labels[[11, 7, 2]]
    np.testing.assert_array_equal(batch["label"].numpy(), expect)
    assert batch["index"].tolist() == [2, 0, 1]
    np.testing.assert_allclose(batch["similarity"].numpy(), cache.similarity[[11, 7, 2]])


def test_loader_is_seeded_and_covers_every_row(cache_dir):
    ds = CachedFeatureDataset(cache_dir)

    def epoch(seed):
        loader = make_loader(ds, batch_size=6, shuffle=True, seed=seed)
        return torch.cat([b["index"] for b in loader]).tolist()

    first = epoch(0)
    assert sorted(first) == list(range(20))
    assert epoch(0) == first and epoch(1) != first
    batch = next(iter(make_loader(ds, batch_size=6, shuffle=False, seed=0)))
    assert batch["text_tokens"].shape[0] == 6 and batch["text_tokens"].ndim == 3


def test_unlabeled_rows_need_opt_in(tmp_path, cache_arrays):
    out = write_cache(tmp_path / "u", **cache_arrays(n=4, labels=[-1, -1, -1, -1]))
    with pytest.raises(ValueError, match="unlabeled"):
        CachedFeatureDataset(out)
    assert len(CachedFeatureDataset(out, require_labels=False)) == 4
