"""Numerical checks of A-4/A-5 equations, independent of future model modules."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def softmax(x):
    weights = np.exp(x - np.max(x))
    return weights / weights.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def central_gradient(function, x, step=1e-5):
    basis = np.eye(len(x)) * step
    return np.array([(function(x + e) - function(x - e)) / (2 * step) for e in basis])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("label", [0, 1])
def test_logit_and_probability_router_gradients(seed, label):
    rng = np.random.default_rng(seed)
    logits, gate = rng.normal(size=(2, 3))
    pi, probabilities = softmax(gate), sigmoid(logits)
    mixed_logit, mixed_probability = pi @ logits, pi @ probabilities

    def loss_l(g):
        z = softmax(g) @ logits
        return np.logaddexp(0, z) - label * z

    def loss_p(g):
        p = softmax(g) @ probabilities
        return -label * np.log(p) - (1 - label) * np.log1p(-p)

    expected_l = (sigmoid(mixed_logit) - label) * pi * (logits - mixed_logit)
    expected_p = (
        (mixed_probability - label)
        / (mixed_probability * (1 - mixed_probability))
        * pi
        * (probabilities - mixed_probability)
    )
    assert_allclose(central_gradient(loss_l, gate), expected_l, atol=1e-9)
    assert_allclose(central_gradient(loss_p, gate), expected_p, atol=1e-9)
    likelihoods = probabilities**label * (1 - probabilities) ** (1 - label)
    posterior = pi * likelihoods / (pi @ likelihoods)
    assert_allclose(expected_p, pi - posterior, atol=1e-14)
    assert_allclose(expected_l.sum(), 0, atol=1e-14)
    assert_allclose(expected_p.sum(), 0, atol=1e-14)


def test_detached_reliability_gradient_and_combined_loss():
    gate = np.array([0.3, -0.8, 1.2])
    logits = np.array([-1.0, 0.5, 2.0])
    target = softmax(-np.logaddexp(0, -logits) / 0.7)

    def reliability(g):
        return np.sum(target * (np.log(target) - np.log(softmax(g))))

    pi = softmax(gate)
    assert_allclose(central_gradient(reliability, gate), pi - target, atol=1e-9)
    weight = 0.3

    def total(g):
        return np.logaddexp(0, -(softmax(g) @ logits)) + weight * reliability(g)

    z = pi @ logits
    expected = (sigmoid(z) - 1) * pi * (logits - z) + weight * (pi - target)
    assert_allclose(central_gradient(total, gate), expected, atol=1e-9)


def test_agreement_and_individual_zero_gradient_are_distinct():
    gate = np.zeros(3)

    def gradient(logits):
        def loss(g):
            return np.logaddexp(0, softmax(g) @ logits)

        return central_gradient(loss, gate)

    assert_allclose(gradient(np.ones(3)), 0, atol=1e-10)
    partial = gradient(np.array([-1.0, 0.0, 1.0]))
    assert abs(partial[1]) < 1e-10
    assert partial[0] < 0 < partial[2]


def test_probability_clamp_has_flat_region():
    probabilities = np.array([1e-9, 2e-9, 3e-9])

    def loss(g):
        return -np.log(np.clip(softmax(g) @ probabilities, 1e-6, 1 - 1e-6))

    assert_allclose(central_gradient(loss, np.zeros(3)), 0, atol=1e-12)


def test_interpretation_identities():
    pi = softmax(np.array([-0.6, 0.4, 0.7]))
    logits = np.array([-1.3, 0.2, 2.1])
    probabilities = sigmoid(logits)
    p_l, p_p = sigmoid(pi @ logits), pi @ probabilities
    assert_allclose(p_l / (1 - p_l), np.prod((probabilities / (1 - probabilities)) ** pi))
    assert probabilities.min() <= p_l <= probabilities.max()
    assert probabilities.min() <= p_p <= probabilities.max()
    assert not np.isclose(p_l, p_p)
    features = np.arange(12).reshape(3, 4) / 10
    weights = np.array([-0.2, 0.4, 0.5, -0.1])
    bias = 0.8
    assert_allclose((pi @ features) @ weights + bias, pi @ (features @ weights + bias))


def test_constant_gate_scale_absorption():
    rng = np.random.default_rng(7)
    image, text = rng.normal(size=(2, 8, 4))
    w_v, w_u = rng.normal(size=(2, 5, 4))
    w_alpha, bias = rng.normal(size=(2, 5))
    c, d = 0.2, 0.7
    original = c * image @ w_v.T + (1 - c) * text @ w_u.T + c * w_alpha + bias
    transformed = (
        d * image @ (c / d * w_v).T
        + (1 - d) * text @ ((1 - c) / (1 - d) * w_u).T
        + d * w_alpha
        + bias
        + (c - d) * w_alpha
    )
    assert_allclose(original, transformed, atol=1e-14)
    # The same constant-gate transformation does not absorb a varying gate.
    varying = np.linspace(0.1, 0.9, 8)[:, None]
    varying_original = (
        varying * (image @ w_v.T) + (1 - varying) * (text @ w_u.T) + varying * w_alpha + bias
    )
    assert not np.allclose(varying_original, transformed)


def penalty(alpha, entropy_weight, diversity_weight):
    negative_entropy = np.mean(alpha * np.log(alpha) + (1 - alpha) * np.log1p(-alpha))
    return entropy_weight * negative_entropy - diversity_weight * np.var(alpha, ddof=1)


def test_entropy_variance_penalty_gradient():
    alpha = np.array([0.2, 0.4, 0.7, 0.8])
    entropy_weight, diversity_weight = 0.3, 0.2
    expected = entropy_weight / len(alpha) * np.log(alpha / (1 - alpha)) - 2 * diversity_weight / (
        len(alpha) - 1
    ) * (alpha - alpha.mean())
    def function(a):
        return penalty(a, entropy_weight, diversity_weight)

    assert_allclose(central_gradient(function, alpha), expected, atol=1e-9)
    gate = np.log(alpha / (1 - alpha))
    assert_allclose(
        central_gradient(lambda s: function(sigmoid(s)), gate),
        expected * alpha * (1 - alpha),
        atol=1e-9,
    )


@pytest.mark.parametrize("diversity_weight", [0.5, 2.0])
def test_half_gate_penalty_curvature(diversity_weight):
    alpha = np.full(4, 0.5)
    step = 1e-4
    direction = np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2)
    entropy_weight = 1.0
    curvature = (
        penalty(alpha + step * direction, entropy_weight, diversity_weight)
        - 2 * penalty(alpha, entropy_weight, diversity_weight)
        + penalty(alpha - step * direction, entropy_weight, diversity_weight)
    ) / step**2
    expected = 4 * entropy_weight / 4 - 2 * diversity_weight / 3
    assert_allclose(curvature, expected, atol=1e-7)
    assert (curvature > 0) == (diversity_weight < 2 * entropy_weight * 3 / 4)
