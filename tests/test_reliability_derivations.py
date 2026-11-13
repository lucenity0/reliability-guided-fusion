"""Synthetic checks for A-2; no training modules or datasets are needed."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def target(losses, temperature):
    weights = np.exp(-(losses - np.min(losses)) / temperature)
    return weights / weights.sum()


def kl(p, q):
    nonzero = p > 0
    return np.sum(p[nonzero] * np.log(p[nonzero] / q[nonzero]))


@pytest.mark.parametrize(
    "losses,limit",
    [
        ([0.2, 1.0, 3.0], [1.0, 0.0, 0.0]),
        ([1.0, 0.2, 0.2], [0.0, 0.5, 0.5]),
        ([0.2, 0.2, 0.2], [1 / 3, 1 / 3, 1 / 3]),
    ],
)
def test_low_temperature_unique_and_tied_minima(losses, limit):
    assert_allclose(target(np.array(losses), 1e-4), limit, atol=1e-12)


def test_high_temperature_is_uniform():
    assert_allclose(target(np.array([0.2, 1.0, 3.0]), 1e9), np.full(3, 1 / 3), atol=1e-8)


@pytest.mark.parametrize("temperature", [0.25, 1.0, 4.0])
def test_population_optimum_and_exact_risk_gap(temperature):
    probabilities = np.array([0.8, 0.2, 0.6])
    targets = np.array(
        [
            target(-np.log1p(-probabilities), temperature),
            target(-np.log(probabilities), temperature),
        ]
    )
    label_distribution = np.array([0.3, 0.7])
    optimum = label_distribution @ targets

    def risk(pi):
        return label_distribution @ np.array([kl(t, pi) for t in targets])

    rng = np.random.default_rng(3)
    for pi in rng.dirichlet(np.ones(3), size=20):
        gap = risk(pi) - risk(optimum)
        assert gap >= -1e-14
        assert_allclose(gap, kl(optimum, pi), atol=1e-13)
    assert kl(optimum, optimum) == 0


def test_hard_targets_can_have_soft_population_optimum():
    probabilities = np.array([0.9, 0.1])
    positive = target(-np.log(probabilities), 1e-4)
    negative = target(-np.log1p(-probabilities), 1e-4)
    assert_allclose(positive, [1, 0], atol=1e-12)
    assert_allclose(negative, [0, 1], atol=1e-12)
    assert_allclose(0.5 * (positive + negative), [0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("label", [0, 1])
def test_uniform_prior_em_identity_and_learned_prior_difference(label):
    probabilities = np.array([0.8, 0.2, 0.6])
    likelihoods = probabilities**label * (1 - probabilities) ** (1 - label)
    t = target(-np.log(likelihoods), 1.0)
    uniform = np.full(3, 1 / 3)
    uniform_posterior = uniform * likelihoods / (uniform @ likelihoods)
    assert_allclose(t, uniform_posterior, atol=1e-14)
    prior = np.array([0.7, 0.1, 0.2])
    posterior = prior * likelihoods / (prior @ likelihoods)
    assert not np.allclose(t, posterior)
    assert_allclose(
        posterior[0] / posterior[1], prior[0] / prior[1] * likelihoods[0] / likelihoods[1]
    )


@pytest.mark.parametrize("weight", [0.1, 1.0, 10.0])
def test_finite_weight_uniform_target_does_not_force_uniform_router(weight):
    probabilities = np.array([0.9, 0.1])
    uniform = np.array([0.5, 0.5])
    step = 1e-5
    def objective(p):
        pi = np.array([p, 1 - p])
        return -np.log(pi @ probabilities) + weight * kl(uniform, pi)

    derivative = (objective(0.5 + step) - objective(0.5 - step)) / (2 * step)
    assert_allclose(derivative, -1.6, atol=1e-8)


def test_normalizing_powered_bernoulli_changes_tempered_target():
    probabilities = np.array([0.8, 0.3, 0.6])
    temperature = 0.5
    target_weights = target(-np.log(probabilities), temperature)
    powered = probabilities ** (1 / temperature)
    normalized_likelihoods = powered / (powered + (1 - probabilities) ** (1 / temperature))
    posterior = normalized_likelihoods / normalized_likelihoods.sum()
    assert not np.allclose(target_weights, posterior)
