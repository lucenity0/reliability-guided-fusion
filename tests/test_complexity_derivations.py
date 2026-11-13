"""A-6 synthetic shape/accounting checks; these are not measured profiler FLOPs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import softmax


def matmul(rows, inner, columns):
    """A6.1 convention: two FLOPs for each multiply-accumulate."""
    return 2 * rows * inner * columns


def direction_ledger(queries, context, hidden, heads, ffn):
    assert hidden % heads == 0
    return {
        "query": matmul(queries, hidden, hidden),
        "key": matmul(context, hidden, hidden),
        "value": matmul(context, hidden, hidden),
        "scores": sum(matmul(queries, hidden // heads, context) for _ in range(heads)),
        "weighted_values": sum(matmul(queries, context, hidden // heads) for _ in range(heads)),
        "output": matmul(queries, hidden, hidden),
        "ffn_expand": matmul(queries, hidden, ffn),
        "ffn_contract": matmul(queries, ffn, hidden),
    }


def mlp(widths):
    return sum(matmul(1, a, b) for a, b in zip(widths[:-1], widths[1:], strict=True))


def pool_ledger(tokens, hidden, reassociate=False):
    dense_tokens = 1 if reassociate else tokens
    return (
        2 * matmul(dense_tokens, hidden, hidden)
        + matmul(tokens, hidden, 1)
        + matmul(1, tokens, hidden)
    )


@pytest.mark.parametrize("p,length,h,heads,f,layers", [
    (7, 3, 8, 1, 12, 1), (7, 3, 8, 4, 12, 3), (145, 77, 512, 8, 1024, 2),
])
def test_bidirectional_closed_form_from_operation_shapes(p, length, h, heads, f, layers):
    image = direction_ledger(p, length, h, heads, f)
    text = direction_ledger(length, p, h, heads, f)
    explicit = layers * (sum(image.values()) + sum(text.values()))
    assert explicit == layers * (
        8 * (p + length) * h**2 + 8 * p * length * h + 4 * (p + length) * h * f
    )
    attention = image["scores"] + image["weighted_values"]
    attention += text["scores"] + text["weighted_values"]
    assert attention == 8 * p * length * h
    assert sum(image.values()) == sum(direction_ledger(p, length, h, 1, f).values())
    assert sum(text.values()) == sum(direction_ledger(length, p, h, h, f).values())


def default_ledger(reassociate=False, outputs=1):
    """A6.11 / D-017 approved MLP shapes."""
    p, length, d, h, heads, f, layers, n, k = 145, 77, 768, 512, 8, 1024, 2, 32, 3
    return {
        "input": matmul(p, d, h) + matmul(length, d, h),
        "cross": layers * (
            sum(direction_ledger(p, length, h, heads, f).values())
            + sum(direction_ledger(length, p, h, heads, f).values())
        ),
        "cross_fusion": 2 * matmul(1, d, n) + n * n,
        "pool": 2 * pool_ledger(p, h, reassociate) + 2 * pool_ledger(length, h, reassociate),
        "expert_mlps": 2 * mlp([2 * h, h, h]) + mlp([2 * h + n * n, h, h]),
        "router_projections": 2 * matmul(1, d, h),
        "router_mlp": mlp([2 * h + k + 1, h, k]),
        "heads": k * matmul(1, h, outputs),
        "mix": matmul(1, k, outputs),
    }


def test_worked_default_ledger():
    assert default_ledger() == {
        "input": 174_587_904,
        "cross": 1_953_734_656,
        "cross_fusion": 99_328,
        "pool": 466_477_056,
        "expert_mlps": 5_767_168,
        "router_projections": 1_572_864,
        "router_mlp": 1_055_744,
        "heads": 3_072,
        "mix": 6,
    }
    assert sum(default_ledger().values()) == 2_603_297_798
    assert default_ledger(reassociate=True)["pool"] == 5_103_616
    assert sum(default_ledger(reassociate=True).values()) == 2_141_924_358
    image = direction_ledger(145, 77, 512, 8, 1024)
    text = direction_ledger(77, 145, 512, 8, 1024)
    projections = sum(image[key] + text[key] for key in ["query", "key", "value", "output"])
    assert projections == 465_567_744
    assert image["scores"] + image["weighted_values"] == 45_731_840 // 2
    assert sum(image[key] + text[key] for key in ["ffn_expand", "ffn_contract"]) == 465_567_744


@pytest.mark.parametrize("outputs", [3, 101])
def test_head_output_cost_increment(outputs):
    increment = sum(default_ledger(outputs=outputs).values()) - sum(default_ledger().values())
    assert increment == 2 * 3 * (512 + 1) * (outputs - 1)


@pytest.mark.parametrize("tokens,hidden", [(7, 4), (13, 8)])
def test_attention_pool_reassociation_preserves_masked_output(tokens, hidden):
    rng = np.random.default_rng(tokens)
    x = rng.normal(size=(tokens, hidden))
    key, output = rng.normal(size=(2, hidden, hidden))
    query = rng.normal(size=hidden)
    mask = np.arange(tokens) < tokens - 2
    literal_scores = (x @ key.T) @ query / np.sqrt(hidden)
    literal_weights = softmax(np.where(mask, literal_scores, -np.inf))
    literal = literal_weights @ (x @ output.T)
    associated_scores = x @ (key.T @ query) / np.sqrt(hidden)
    associated_weights = softmax(np.where(mask, associated_scores, -np.inf))
    associated = (associated_weights @ x) @ output.T
    assert_allclose(associated_scores, literal_scores, atol=1e-13)
    assert_allclose(associated, literal, atol=1e-13)
    assert np.all(associated_weights[~mask] == 0)
    assert pool_ledger(tokens, hidden) == 4 * tokens * hidden**2 + 4 * tokens * hidden
    assert pool_ledger(tokens, hidden, True) == 4 * hidden**2 + 4 * tokens * hidden


def self_attention_block(tokens, width, heads, ffn):
    return (
        4 * matmul(tokens, width, width)
        + heads * matmul(tokens, width // heads, tokens)
        + heads * matmul(tokens, tokens, width // heads)
        + matmul(tokens, width, ffn) + matmul(tokens, ffn, width)
    )


def test_separate_frozen_backbone_and_dense_cache_ledger():
    # Architecture dimensions are sourced in A6.13; no weights are downloaded.
    patches, text_tokens, d = (336 // 14) ** 2, 77, 768
    vision = matmul(patches, 3 * 14**2, 1024)
    vision += 24 * self_attention_block(patches + 1, 1024, 16, 4096)
    vision += matmul(1, 1024, d)
    text = 12 * self_attention_block(text_tokens, 768, 12, 3072) + matmul(1, 768, d)
    assert vision == 381_919_789_056
    assert text == 13_299_683_328
    assert vision + text == 395_219_472_384
    overhead_percent = 100 * sum(default_ledger(reassociate=True).values()) / (vision + text)
    assert round(overhead_percent, 2) == 0.54
    assert matmul(patches, 1024, d) == 905_969_664
    assert matmul(text_tokens - 1, 768, d) == 89_653_248
    assert 144 * d * (3 + 1) == 442_368


def test_approved_feature_fusion_head_cost():
    hidden, k, outputs = 512, 3, 1
    feature_mix = matmul(1, k, hidden)
    final_head = mlp([hidden, hidden, outputs])
    total = sum(default_ledger(reassociate=True).values()) - 2 * k * outputs
    assert total + feature_mix + final_head == 2_142_452_736
