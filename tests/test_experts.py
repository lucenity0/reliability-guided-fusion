import math

import pytest

torch = pytest.importorskip("torch")

from hmf.features.cache import write_cache  # noqa: E402
from hmf.features.dataset import CachedFeatureDataset, make_loader  # noqa: E402
from hmf.models.experts import AttentionPool, Experts  # noqa: E402


@pytest.fixture
def batch():
    torch.manual_seed(17)
    image = torch.randn(3, 5, 16)
    return {
        "image_tokens": image,
        "text_tokens": torch.randn(3, 7, 16),
        "image_pooled": image[:, 0],
        "text_pooled": torch.randn(3, 16),
        "text_mask": torch.tensor([[True] * 7, [True] * 3 + [False] * 4,
                                   [False] * 3 + [True] + [False] * 3]),
    }


def make_experts(num_outputs=1, dropout=0.0):
    return Experts(d=16, h=8, heads=2, n_cross_layers=2, cross_fusion_dim=3,
                   dropout=dropout, num_outputs=num_outputs)


@pytest.mark.parametrize("num_outputs", [1, 3])
def test_output_shapes(batch, num_outputs):
    model = make_experts(num_outputs)
    output = model(batch)
    assert tuple(output) == model.expert_names
    for hidden, logits in output.values():
        assert hidden.shape == (3, 8)
        assert logits.shape == (3, num_outputs)
        assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
    assert model.img.mlp[0].in_features == model.txt.mlp[0].in_features == 16
    assert model.interaction.mlp[0].in_features == 25
    assert model.interaction.layers[0].ffn_v[0].out_features == 16


def test_masked_weights_are_exactly_zero(batch):
    model = make_experts().eval()
    mask = batch["text_mask"]
    v, t = model.phi_v(batch["image_tokens"]), model.phi_t(batch["text_tokens"])
    for layer in model.interaction.layers:
        vn, tn = layer.norm_v(v), layer.norm_t(t)
        _, weights = layer.attn_v(vn, tn, tn, key_padding_mask=~mask,
                                 average_attn_weights=False)
        assert torch.count_nonzero(weights.masked_select(~mask[:, None, None, :])) == 0
        torch.testing.assert_close(weights[2, :, :, 3], torch.ones_like(weights[2, :, :, 3]))
        v, t = layer(v, t, mask)
    for pool in (model.txt.pool, model.interaction.pool_t):
        weights = pool.attention_weights(t, mask)
        assert torch.count_nonzero(weights[~mask]) == 0
        assert weights[2, 3].item() == 1.0
        torch.testing.assert_close(weights.sum(-1), torch.ones(3))


def test_appending_garbage_padding_preserves_all_outputs(batch):
    model = make_experts(dropout=0.2).eval()
    extended = dict(batch)
    tokens = batch["text_tokens"].clone()
    tokens[~batch["text_mask"]] = 1e6
    extended["text_tokens"] = torch.cat((tokens, torch.randn(3, 4, 16) * 1e6), dim=1)
    extended["text_mask"] = torch.cat((batch["text_mask"], torch.zeros(3, 4, dtype=torch.bool)),
                                      dim=1)
    with torch.no_grad():
        expected, actual = model(batch), model(extended)
    for name in model.expert_names:
        for before, after in zip(expected[name], actual[name], strict=True):
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("masked", [False, True])
def test_reassociated_pool_matches_literal_formula_float64(batch, masked):
    pool = AttentionPool(8).double()
    x = torch.randn(3, 7, 8, dtype=torch.float64)
    mask = batch["text_mask"] if masked else None
    scores = (pool.key(x) * pool.query).sum(-1) / math.sqrt(8)
    if mask is not None:
        scores = scores + mask.double().log()
    literal = (scores.softmax(-1).unsqueeze(-1) * pool.output(x)).sum(1)
    torch.testing.assert_close(pool(x, mask), literal, rtol=1e-12, atol=1e-12)


def test_simultaneous_two_layer_updates_and_raw_cross_fusion(batch):
    model = make_experts().double().eval()
    batch = {key: value.double() if value.is_floating_point() else value
             for key, value in batch.items()}
    v, t = model.phi_v(batch["image_tokens"]), model.phi_t(batch["text_tokens"])
    expert = model.interaction
    for layer in expert.layers:
        previous_v, previous_t = v, t
        vn, tn = layer.norm_v(previous_v), layer.norm_t(previous_t)
        # Compute both directions from the saved states, independently of layer.forward.
        dv = layer.attn_v(vn, tn, tn, key_padding_mask=~batch["text_mask"])[0]
        dt = layer.attn_t(tn, vn, vn)[0]
        v = previous_v + dv
        t = previous_t + dt
        v = v + layer.ffn_v(layer.ffn_norm_v(v))
        t = t + layer.ffn_t(layer.ffn_norm_t(t))
        actual_v, actual_t = layer(previous_v, previous_t, batch["text_mask"])
        torch.testing.assert_close(actual_v, v, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual_t, t, rtol=1e-12, atol=1e-12)
    cross = torch.stack([torch.outer(pv, pt).flatten() for pv, pt in zip(
        expert.cross_v(batch["image_pooled"]), expert.cross_t(batch["text_pooled"]), strict=True)])
    hidden = expert.mlp(torch.cat((expert.pool_v(v, None),
                                   expert.pool_t(t, batch["text_mask"]), cross), dim=-1))
    actual_h, actual_z = model(batch)["int"]
    torch.testing.assert_close(actual_h, hidden, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_z, expert.head(hidden), rtol=1e-12, atol=1e-12)


def test_token_projections_are_shared_including_dropout(batch):
    model = make_experts(dropout=0.3).train()
    seen = {}

    def capture(name):
        def hook(module, args):
            seen[name] = args
        return hook

    handles = [module.register_forward_pre_hook(capture(name)) for name, module in
               (("img", model.img), ("txt", model.txt), ("int", model.interaction))]
    try:
        model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert seen["img"][0] is seen["int"][0]
    assert seen["txt"][0] is seen["int"][1]


@pytest.mark.parametrize("num_outputs", [1, 3])
def test_gradients_reach_every_parameter(batch, num_outputs):
    model = make_experts(num_outputs, dropout=0.1).train()
    output = model(batch)
    sum(logits.square().sum() for _, logits in output.values()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for name in model.expert_names:
        model.zero_grad(set_to_none=True)
        model(batch)[name][0].square().sum().backward()
        projections = (model.phi_v, model.phi_t) if name == "int" else (
            model.phi_v if name == "img" else model.phi_t,)
        for projection in projections:
            assert projection[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable"))])
def test_cached_batch_forward(tmp_path, cache_arrays, device):
    cache = write_cache(tmp_path / "synthetic", **cache_arrays(n=3, d=16, length=7))
    loader = make_loader(CachedFeatureDataset(cache), batch_size=3, shuffle=False, seed=0)
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    model = make_experts().to(device).eval()
    with torch.no_grad():
        output = model(batch)
    for hidden, logits in output.values():
        assert hidden.shape == (3, 8) and logits.shape == (3, 1)
        assert logits.device.type == device
        assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
