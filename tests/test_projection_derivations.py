"""A-1 linear algebra and payload arithmetic; no empirical model-quality claims."""

import numpy as np
from numpy.testing import assert_allclose


def test_projecting_and_pooling_commute_after_normalization():
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(3, 4, 8))
    normalized = (tokens - tokens.mean(axis=-1, keepdims=True)) / np.sqrt(
        tokens.var(axis=-1, keepdims=True) + 1e-5
    )
    projection = rng.normal(size=(6, 8))
    assert_allclose(
        (normalized @ projection.T).mean(axis=1),
        normalized.mean(axis=1) @ projection.T, atol=1e-14,
    )
    pooled_raw = tokens.mean(axis=1)
    normalize_after_pool = (pooled_raw - pooled_raw.mean(axis=-1, keepdims=True)) / np.sqrt(
        pooled_raw.var(axis=-1, keepdims=True) + 1e-5
    )
    assert not np.allclose(normalized.mean(axis=1), normalize_after_pool)


def test_raw_map_can_match_projected_stream_at_initialization():
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(9, 8))
    projection, learned = rng.normal(size=(6, 8)), rng.normal(size=(4, 6))
    bias = rng.normal(size=4)
    projected = (raw @ projection.T) @ learned.T + bias
    raw_map = learned @ projection
    direct = raw @ raw_map.T + bias
    assert_allclose(direct, projected, atol=1e-13)
    # Equal preactivations also remain equal after the same nonlinear head.
    assert_allclose(np.tanh(direct), np.tanh(projected), atol=1e-13)


def test_frozen_projection_nullspace_is_unrecoverable_downstream():
    rng = np.random.default_rng(17)
    projection = rng.normal(size=(6, 8))
    _, _, right = np.linalg.svd(projection, full_matrices=True)
    null_vector = right[-1]
    learned = rng.normal(size=(4, 6))
    state = rng.normal(size=8)
    assert_allclose(projection @ null_vector, 0, atol=1e-14)
    assert_allclose(learned @ projection @ (state + null_vector), learned @ projection @ state)
    # A free raw-token row can detect precisely that lost direction.
    assert_allclose(null_vector @ (state + null_vector) - null_vector @ state, 1, atol=1e-14)


def test_raw_projection_cost_and_cache_payload():
    tokens, hidden, raw_width, projected_width, bytes_per_scalar = 145, 512, 1024, 768, 2
    projected_cost = tokens * hidden * projected_width * 2
    raw_cost = tokens * hidden * raw_width * 2
    assert raw_cost - projected_cost == 38_010_880
    assert hidden * raw_width - hidden * projected_width == 131_072
    projected_payload = tokens * projected_width * bytes_per_scalar
    raw_payload = tokens * raw_width * bytes_per_scalar
    assert projected_payload == 222_720
    assert raw_payload == 296_960
    assert raw_payload - projected_payload == 74_240
    global_vector = projected_width * bytes_per_scalar
    assert global_vector == 1_536
    assert raw_payload + global_vector - projected_payload == 75_776
    assert 2 * 576 * raw_width * projected_width == 905_969_664
