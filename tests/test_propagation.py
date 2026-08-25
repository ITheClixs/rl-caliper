"""The covariance recursion and the dependence-aware intervals."""

import numpy as np
import pytest

from caliper.analysis.uncertainty import (
    compatible,
    loglog_fit,
    pairwise_mean_interval_from_matrix,
)
from caliper.exact.pool import build
from caliper.exact.propagation import propagate, seed_memory, update_jacobian
from caliper.objectives.advantages import weight_table


@pytest.fixture(scope="module")
def pool():
    return build(3, 3, 24, (0.05, 0.95), 1.0, seed=0)


def test_jacobian_matches_a_directional_derivative(pool):
    """The finite-difference Jacobian must reproduce a directional derivative it did not fit."""
    from caliper.exact.propagation import mean_gradient

    policy, accepts = pool
    weights = weight_table("rloo", 8)
    jacobian = update_jacobian(policy, accepts, weights)
    rng = np.random.default_rng(0)
    direction = rng.normal(size=policy.dim)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    numerical = (
        mean_gradient(policy.perturbed(step * direction), accepts, weights)
        - mean_gradient(policy.perturbed(-step * direction), accepts, weights)
    ) / (2 * step)
    np.testing.assert_allclose(jacobian @ direction, numerical, rtol=2e-3, atol=1e-10)


def test_covariance_is_symmetric_positive_semidefinite(pool):
    policy, accepts = pool
    result = propagate(policy, accepts, "rloo", 8, 8, 0.5, steps=4)
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(result.covariance).min() > -1e-10


def test_kernel_sums_to_the_predicted_divergence(pool):
    """Per-update contributions must account for the whole of the final number."""
    policy, accepts = pool
    result = propagate(policy, accepts, "rloo", 8, 8, 0.5, steps=5)
    assert sum(result.kernel) == pytest.approx(result.predicted_kl, rel=1e-9)


def test_more_prompts_inject_less_noise(pool):
    policy, accepts = pool
    small = propagate(policy, accepts, "rloo", 8, 8, 0.5, steps=4).predicted_kl
    large = propagate(policy, accepts, "rloo", 8, 32, 0.5, steps=4).predicted_kl
    assert large < small


def test_seed_memory_is_bounded_by_the_run_length():
    assert seed_memory([1.0, 1.0, 1.0, 1.0], 0.95) == 4
    assert seed_memory([0.0, 0.0, 0.0, 1.0], 0.95) == 1


def test_run_level_interval_is_wider_than_the_naive_one():
    """Pairs built from the same runs are dependent; ignoring that understates uncertainty."""
    rng = np.random.default_rng(0)
    runs = rng.normal(size=(6, 3))
    matrix = np.linalg.norm(runs[:, None, :] - runs[None, :, :], axis=-1)
    result = pairwise_mean_interval_from_matrix(matrix, n_boot=2000, seed=1)
    assert result["bootstrap_se"] > result["naive_se_over_pairs"]
    assert result["lo"] < result["mean"] < result["hi"]


def test_loglog_fit_recovers_a_known_exponent():
    rng = np.random.default_rng(3)
    p = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    response = 3.0 * p**-0.5 * np.exp(rng.normal(0, 0.02, size=p.size))
    fit = loglog_fit({"P": p}, response, n_boot=1000)
    assert compatible(fit["P"], -0.5)
    assert fit["_fit"]["r2"] > 0.98


def test_the_resolved_spread_subtracts_the_evaluation_noise():
    """With no seed spread at all, what survives the subtraction must be zero."""
    from caliper.analysis.uncertainty import resolved_spread_interval

    rng = np.random.default_rng(0)
    runs, questions, samples = 8, 48, 16
    identical = np.full(runs, 0.35)
    per_prompt = rng.binomial(samples, identical[:, None] * np.ones((1, questions))) / samples
    result = resolved_spread_interval(per_prompt, samples, n_boot=800, seed=1)
    assert result["resolved"] < 0.01
    assert result["binomial"] > 0.0


def test_the_resolved_spread_recovers_a_real_one():
    """A seed spread well above the evaluation noise must come back close to its true size."""
    from caliper.analysis.uncertainty import resolved_spread_interval

    rng = np.random.default_rng(3)
    runs, questions, samples, true_sd = 12, 48, 16, 0.08
    centres = 0.35 + rng.normal(0.0, true_sd, size=runs)
    per_prompt = (
        rng.binomial(samples, np.clip(centres[:, None] * np.ones((1, questions)), 0, 1)) / samples
    )
    result = resolved_spread_interval(per_prompt, samples, n_boot=800, seed=2)
    assert result["lo"] <= true_sd <= result["hi"]
    assert result["resolved"] < result["observed"]


def test_the_resolved_spread_needs_more_than_one_run():
    from caliper.analysis.uncertainty import resolved_spread_interval

    with pytest.raises(ValueError):
        resolved_spread_interval(np.full((1, 4), 0.3), 8)
