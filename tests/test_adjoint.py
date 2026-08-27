"""The adjoint form of the reported spread, and the structure that makes it cheap."""

import numpy as np
import pytest

from caliper.exact.adjoint import (
    forecast,
    hessian_vector_product,
    mean_trajectory,
    metric_gradient,
    metric_value,
    transposed_jacobian_product,
)
from caliper.exact.pool import build
from caliper.exact.propagation import injected_covariance, mean_gradient, update_jacobian
from caliper.objectives.advantages import weight_table


@pytest.fixture(scope="module")
def pool():
    return build(3, 3, 16, (0.05, 0.95), 1.0, seed=0)


@pytest.mark.parametrize("estimator", ["rloo", "grpo_mean", "grpo_std"])
def test_the_mean_update_has_a_symmetric_jacobian(pool, estimator):
    """Proposition 3: the mean update is a gradient field, so its Jacobian is a Hessian."""
    policy, accepts = pool
    jacobian = update_jacobian(policy, accepts, weight_table(estimator, 8))
    asymmetry = np.abs(jacobian - jacobian.T).max() / np.abs(jacobian).max()
    assert asymmetry < 1e-6


def test_rloo_ascends_the_reported_metric(pool):
    """With lambda = 1 the potential is the pass rate itself."""
    policy, accepts = pool
    np.testing.assert_allclose(
        mean_gradient(policy, accepts, weight_table("rloo", 8)),
        metric_gradient(policy, accepts),
        rtol=0,
        atol=1e-14,
    )


def test_metric_gradient_matches_a_finite_difference(pool):
    policy, accepts = pool
    rng = np.random.default_rng(0)
    direction = rng.normal(size=policy.dim)
    direction /= np.linalg.norm(direction)
    step = 1e-5
    numerical = (
        metric_value(policy.perturbed(step * direction), accepts)
        - metric_value(policy.perturbed(-step * direction), accepts)
    ) / (2 * step)
    assert metric_gradient(policy, accepts) @ direction == pytest.approx(numerical, rel=1e-5)


def test_the_two_jacobian_routes_agree(pool):
    """J b by two gradient evaluations must equal J^T b taken coordinate by coordinate."""
    policy, accepts = pool
    weights = weight_table("rloo", 8)
    rng = np.random.default_rng(1)
    direction = rng.normal(size=policy.dim)
    fast = hessian_vector_product(policy, accepts, weights, direction)
    slow = transposed_jacobian_product(policy, accepts, weights, direction)
    np.testing.assert_allclose(fast, slow, rtol=2e-4, atol=1e-12)


def test_the_adjoint_equals_the_explicit_quadratic_form(pool):
    """Theorem 2 must reproduce grad M^T S_T grad M built the expensive way."""
    policy, accepts = pool
    weights = weight_table("rloo", 8)
    eta, prompts, steps = 0.5, 16, 4
    states = mean_trajectory(policy, accepts, weights, eta, steps)

    dim = policy.dim
    covariance = np.zeros((dim, dim))
    for t in range(steps):
        transfer = np.eye(dim) + eta * update_jacobian(states[t], accepts, weights)
        covariance = transfer @ covariance @ transfer.T + eta**2 * injected_covariance(
            states[t], accepts, weights, prompts
        )
    gradient = metric_gradient(states[steps], accepts)
    explicit = float(gradient @ covariance @ gradient)

    predicted = forecast(policy, accepts, "rloo", 8, prompts, eta, steps, states=states)
    assert predicted.variance == pytest.approx(explicit, rel=1e-6)


def test_the_kernel_accounts_for_the_whole_variance(pool):
    policy, accepts = pool
    result = forecast(policy, accepts, "rloo", 8, 16, 0.5, 5)
    assert sum(result.kernel) == pytest.approx(result.variance, rel=1e-12)


def test_the_two_sources_partition_the_variance(pool):
    """Prompt sampling and rollout sampling are the whole of the batch covariance."""
    policy, accepts = pool
    common = (policy, accepts, "rloo", 8, 16, 0.5, 5)
    whole = forecast(*common).variance
    parts = forecast(*common, source="prompts").variance + forecast(
        *common, source="rollouts"
    ).variance
    assert parts == pytest.approx(whole, rel=1e-10)


def test_more_prompts_narrow_the_forecast(pool):
    policy, accepts = pool
    small = forecast(policy, accepts, "rloo", 8, 8, 0.5, 5).variance
    large = forecast(policy, accepts, "rloo", 8, 32, 0.5, 5).variance
    assert large == pytest.approx(small / 4, rel=0.05)


def test_an_unknown_source_is_rejected(pool):
    policy, accepts = pool
    with pytest.raises(KeyError):
        forecast(policy, accepts, "rloo", 8, 16, 0.5, 3, source="cosmic rays")


def test_the_injected_term_is_the_variance_of_a_projected_batch_gradient(pool):
    """Theorem 2's second equality, against sampled batches rather than against the algebra.

    b^T Cov(g_hat) b has to be the variance of b . g_hat over batches. Everything else in the
    forecast is exact given this term, so it is the one worth checking by simulation.
    """
    policy, accepts = pool
    weights = weight_table("rloo", 8)
    n_prompts, group_size, draws = 12, 8, 20_000
    rng = np.random.default_rng(0)
    direction = rng.normal(size=policy.dim)
    direction /= np.linalg.norm(direction)

    probs, scores = policy.enumerate()
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    projected_scores = scores @ direction  # only the projection is ever needed

    sampled = np.empty(draws)
    for i in range(draws):
        prompt_ids = rng.integers(0, accepts.shape[0], size=n_prompts)
        outcomes = np.searchsorted(cdf, rng.random((n_prompts, group_size)))
        rewards = accepts[prompt_ids[:, None], outcomes].astype(int)
        advantage = weights[rewards.sum(axis=1)[:, None], rewards]
        sampled[i] = (advantage * projected_scores[outcomes]).sum() / (n_prompts * group_size)

    predicted = float(
        direction @ injected_covariance(policy, accepts, weights, n_prompts) @ direction
    )
    measured = float(sampled.var(ddof=1))
    # 20k draws give the sample variance a relative standard error of about 1%
    assert measured == pytest.approx(predicted, rel=0.06)


def test_the_forecast_scales_as_one_over_the_prompt_count():
    """`Var = sum_t (eta^2 / P) Var_i(b' z_i)`, so the forecast is inversely proportional to P.

    This pins down which of the two equivalent forms the implementation uses. Writing the same
    term as `Var(b' g_hat)` over the batch mean would carry no explicit `P`, and using both
    conventions at once would be wrong by a factor of `P`.
    """
    from caliper.exact.adjoint import forecast
    from caliper.exact.pool import build

    policy, accepts = build(3, 2, 12, (0.05, 0.95), 1.0, seed=0)
    counts = [4, 8, 16, 32]
    variances = [
        forecast(policy, accepts, "rloo", 8, p, 0.5, 6).variance for p in counts
    ]
    products = [v * p for v, p in zip(variances, counts, strict=True)]
    assert max(products) / min(products) < 1.02, f"not 1/P: {products}"
