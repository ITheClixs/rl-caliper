"""The closed forms in caliper.exact.moments must match brute-force enumeration exactly."""

import numpy as np
import pytest

from caliper.exact import brute, moments
from caliper.exact.policy import TabularPolicy, accept_set_for_pass_rate, make_prompt
from caliper.objectives.advantages import available, weight_table

ESTIMATORS = available()


def build_prompt(seed: int, vocab: int = 2, length: int = 3, target: float = 0.4):
    rng = np.random.default_rng(seed)
    policy = TabularPolicy.random(vocab, length, rng)
    probs, _ = policy.enumerate()
    accept = accept_set_for_pass_rate(probs, target, rng)
    return make_prompt(policy, accept)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("group_size", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_mean_and_covariance_match_enumeration(name, group_size, seed):
    prompt = build_prompt(seed)
    m = prompt.moments()
    w = weight_table(name, group_size)

    ref_mean, ref_cov = brute.enumerate_group_moments(
        w, prompt.probs, prompt.rewards, prompt.scores
    )
    got_mean = moments.mean(w, m["p"], m["u1"])
    got_cov = moments.covariance(w, m["p"], m["s0"], m["s1"], m["u1"])

    np.testing.assert_allclose(got_mean, ref_mean, atol=1e-12, rtol=0)
    np.testing.assert_allclose(got_cov, ref_cov, atol=1e-12, rtol=0)


@pytest.mark.parametrize("group_size", [2, 3, 5, 8])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.5, 0.9])
def test_difficulty_weights_have_the_predicted_closed_forms(group_size, p):
    assert moments.difficulty_weight(weight_table("rloo", group_size), p) == pytest.approx(1.0)
    assert moments.difficulty_weight(
        weight_table("grpo_mean", group_size), p
    ) == pytest.approx((group_size - 1) / group_size)


@pytest.mark.parametrize("group_size", [2, 3, 5, 8])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.5, 0.9])
def test_rloo_second_moments_give_bernoulli_variance(group_size, p):
    """p Q_1 + (1-p) Q_0 = p(1-p) G/(G-1) exactly, for every G. Equation (8)."""
    w = weight_table("rloo", group_size)
    q0, q1 = moments.second_moments(w, p)
    expected = p * (1 - p) * group_size / (group_size - 1)
    assert p * q1 + (1 - p) * q0 == pytest.approx(expected)


@pytest.mark.parametrize("group_size", [2, 3, 5, 8])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.5, 0.9])
def test_mean_baseline_is_rloo_rescaled(group_size, p):
    """GRPO with a mean baseline is RLOO times (G-1)/G, so its covariance is that squared."""
    scale = (group_size - 1) / group_size
    w_rloo = weight_table("rloo", group_size)
    w_mean = weight_table("grpo_mean", group_size)
    q0r, q1r = moments.second_moments(w_rloo, p)
    q0m, q1m = moments.second_moments(w_mean, p)
    assert q0m == pytest.approx(q0r * scale**2)
    assert q1m == pytest.approx(q1r * scale**2)
