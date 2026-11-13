"""Synthetic checks of A-7, independent of model code and dataset splits."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, logsumexp, softmax


def central_gradient(function, x, step=1e-5):
    result = np.empty_like(x)
    for index in np.ndindex(x.shape):
        offset = np.zeros_like(x)
        offset[index] = step
        result[index] = (function(x + offset) - function(x - offset)) / (2 * step)
    return result


def ce(logits, label):
    return logsumexp(logits, axis=-1) - logits[..., label]


def target(logits, label, temperature):
    return softmax(-ce(logits, label) / temperature)


def kl(p, q):
    nonzero = p > 0
    return np.sum(p[nonzero] * np.log(p[nonzero] / q[nonzero]))


def loss_p(gate, logits, label):
    log_pi = gate - logsumexp(gate)
    return -logsumexp(log_pi - ce(logits, label))


@pytest.mark.parametrize("classes", [3, 101])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_router_and_expert_gradients(classes, seed):
    rng = np.random.default_rng(seed)
    logits, gate = rng.normal(size=(3, classes)), rng.normal(size=3)
    label = int(rng.integers(classes))
    pi, probabilities = softmax(gate), softmax(logits, axis=-1)
    mixed = pi @ logits
    residual = softmax(mixed) - np.eye(classes)[label]
    posterior = pi * probabilities[:, label] / (pi @ probabilities[:, label])
    target_fixed = target(logits, label, 0.7)
    expected_l = pi * ((logits - mixed) @ residual)
    expected_p = pi - posterior
    expected_rel = pi - target_fixed

    assert_allclose(
        central_gradient(lambda g: ce(softmax(g) @ logits, label), gate),
        expected_l, atol=2e-9,
    )
    assert_allclose(
        central_gradient(lambda g: loss_p(g, logits, label), gate), expected_p, atol=2e-9,
    )
    assert_allclose(
        central_gradient(lambda g: kl(target_fixed, softmax(g)), gate),
        expected_rel, atol=2e-9,
    )
    for variant, expected in [("L", expected_l), ("P", expected_p)]:
        def total(g, variant=variant):
            main = ce(softmax(g) @ logits, label) if variant == "L" else loss_p(g, logits, label)
            return main + 0.5 * ce(logits, label).mean() + 0.3 * kl(target_fixed, softmax(g))

        assert_allclose(
            central_gradient(total, gate), expected + 0.3 * expected_rel, atol=2e-9,
        )
        assert_allclose(expected.sum(), 0, atol=1e-14)

    assert_allclose(
        central_gradient(lambda z: ce(pi @ z, label), logits),
        pi[:, None] * residual, atol=2e-9,
    )
    assert_allclose(
        central_gradient(lambda z: loss_p(gate, z, label), logits),
        posterior[:, None] * (probabilities - np.eye(classes)[label]), atol=2e-9,
    )


@pytest.mark.parametrize("classes", [3, 101])
@pytest.mark.parametrize("likelihoods,limit", [
    ([0.2, 0.8, 0.4], [0, 1, 0]),
    ([0.2, 0.8, 0.8], [0, 0.5, 0.5]),
    ([0.4, 0.4, 0.4], [1 / 3, 1 / 3, 1 / 3]),
])
def test_temperature_limits(classes, likelihoods, limit):
    likelihoods = np.array(likelihoods)
    probabilities = np.repeat(((1 - likelihoods) / (classes - 1))[:, None], classes, axis=1)
    probabilities[:, 0] = likelihoods
    logits = np.log(probabilities)
    assert_allclose(target(logits, 0, 1e-5), limit, atol=1e-10)
    assert_allclose(target(logits, 0, 1e10), np.full(3, 1 / 3), atol=1e-9)


@pytest.mark.parametrize("classes", [3, 101])
@pytest.mark.parametrize("temperature", [0.25, 1, 4])
def test_population_risk_gap(classes, temperature):
    rng = np.random.default_rng(classes)
    logits = rng.normal(size=(3, classes))
    targets = np.array([target(logits, c, temperature) for c in range(classes)])
    eta = rng.dirichlet(np.ones(classes))
    optimum = eta @ targets

    def risk(pi):
        return eta @ np.array([kl(t, pi) for t in targets])

    for pi in rng.dirichlet(np.ones(3), size=10):
        assert_allclose(risk(pi) - risk(optimum), kl(optimum, pi), atol=1e-13)


@pytest.mark.parametrize("classes", [3, 101])
def test_exact_mixture_and_uniform_prior_identity(classes):
    rng = np.random.default_rng(classes)
    logits = rng.normal(size=(3, classes))
    probabilities = softmax(logits, axis=-1)
    pi = np.array([0.7, 0.2, 0.1])
    mixture = pi @ probabilities
    assert_allclose(mixture.sum(), 1)
    for label in [0, classes - 1]:
        assert_allclose(np.exp(-loss_p(np.log(pi), logits, label)), mixture[label])
        q = probabilities[:, label]
        assert_allclose(target(logits, label, 1), q / q.sum(), atol=1e-14)
        assert not np.allclose(target(logits, label, 1), pi * q / (pi @ q))


def test_multiclass_zero_gradients_do_not_require_agreement():
    gate = np.zeros(3)
    logits = np.array([[0.0, 1, -1], [0, -1, 1], [0, 0, 0]])
    assert_allclose(central_gradient(lambda g: ce(softmax(g) @ logits, 0), gate), 0, atol=1e-10)
    assert not np.allclose(softmax(logits[0]), softmax(logits[1]))
    logits = np.log([[0.4, 0.5, 0.1], [0.4, 0.1, 0.5]])
    assert_allclose(central_gradient(lambda g: loss_p(g, logits, 0), np.zeros(2)), 0, atol=1e-10)
    assert not np.allclose(softmax(logits[0]), softmax(logits[1]))


@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("temperature", [0.25, 1, 4])
def test_binary_reduction(label, temperature):
    rng = np.random.default_rng(12)
    logits, gate = rng.normal(size=(3, 2)), rng.normal(size=3)
    scalar = logits[:, 1] - logits[:, 0]
    pi, p = softmax(gate), expit(scalar)
    mixed_z, mixed_p = pi @ scalar, pi @ p
    bce = np.logaddexp(0, scalar) - label * scalar
    assert_allclose(softmax(logits, axis=1)[:, 1], p, atol=1e-14)
    assert_allclose(ce(logits, label), bce, atol=1e-14)
    assert_allclose(target(logits, label, temperature), softmax(-bce / temperature), atol=1e-14)
    assert_allclose(softmax(pi @ logits)[1], expit(mixed_z), atol=1e-14)
    assert_allclose((pi @ softmax(logits, axis=1))[1], mixed_p, atol=1e-14)
    assert_allclose(ce(pi @ logits, label), np.logaddexp(0, mixed_z) - label * mixed_z)
    assert_allclose(loss_p(gate, logits, label), -np.log(mixed_p if label else 1 - mixed_p))
    residual = softmax(pi @ logits) - np.eye(2)[label]
    assert_allclose(
        pi * ((logits - pi @ logits) @ residual),
        (expit(mixed_z) - label) * pi * (scalar - mixed_z), atol=1e-14,
    )
    q = softmax(logits, axis=1)[:, label]
    assert_allclose(
        pi - pi * q / (pi @ q),
        (mixed_p - label) / (mixed_p * (1 - mixed_p)) * pi * (p - mixed_p), atol=1e-14,
    )
    binary_entropy = -(p * np.log(p) + (1 - p) * np.log1p(-p)) / np.log(2)
    categorical = softmax(logits, axis=1)
    assert_allclose(-np.sum(categorical * np.log(categorical), axis=1) / np.log(2), binary_entropy)
    # F reduces through the two-logit difference even with a nonlinear head.
    features, weights = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    output = np.tanh(pi @ features) @ weights
    assert_allclose(softmax(output)[1], expit(output[1] - output[0]), atol=1e-14)
