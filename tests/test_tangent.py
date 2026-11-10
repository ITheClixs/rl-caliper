"""Seeds simulated in the tangent model: exact where they should be, and close where they can be."""

import numpy as np
import pytest

from caliper.exact.adjoint import forecast
from caliper.exact.policy import TabularPolicy
from caliper.exact.pool import build
from caliper.exact.tangent import reference_model
from caliper.objectives.advantages import weight_table
from caliper.tangent import Recipe, TangentModel, draw, simulate, update


def true_update(policy, accepts, prompts, outcomes, recipe):
    """The enumerable trainer's update on given draws, written out independently of it."""
    _, scores = policy.enumerate()
    rewards = accepts[prompts[:, None], outcomes].astype(int)
    advantages = recipe.weights[rewards.sum(axis=1)[:, None], rewards]
    gradient = np.einsum("pg,pgd->d", advantages, scores[outcomes])
    return policy.perturbed(recipe.step_size * gradient / outcomes.size)


@pytest.mark.parametrize("estimator", ["rloo", "grpo_mean"])
def test_a_one_token_policy_is_its_own_tangent_model(estimator):
    rng = np.random.default_rng(0)
    policy = TabularPolicy.random(5, 1, rng)
    accepts = (rng.random((6, 5)) < 0.4).astype(float)
    recipe = Recipe(2.0, 4, 8, 12, weight_table(estimator, 8))
    model = reference_model(policy, accepts)
    theta = np.zeros((1, model.dim))
    for _ in range(recipe.steps):
        prompts, outcomes = draw(model, theta, np.arange(6), recipe, rng)
        theta = update(model, theta, prompts, outcomes, recipe)
        policy = true_update(policy, accepts, prompts[0], outcomes[0], recipe)
        surrogate = model.probabilities(theta, np.zeros((1, 1), dtype=int))[0, 0]
        assert np.allclose(surrogate, policy.sequence_probs(), atol=1e-12)


def test_the_first_update_is_exact_for_any_policy():
    policy, accepts = build(3, 3, 12, (0.05, 0.95), 1.0, 3)
    recipe = Recipe(1.5, 8, 8, 1, weight_table("rloo", 8))
    model = reference_model(policy, accepts)
    rng = np.random.default_rng(1)
    theta = np.zeros((1, model.dim))
    prompts, outcomes = draw(model, theta, np.arange(12), recipe, rng)
    moved = update(model, theta, prompts, outcomes, recipe)[0]
    exact = true_update(policy, accepts, prompts[0], outcomes[0], recipe)
    assert np.allclose(moved, (exact.logits - policy.logits).reshape(-1), atol=1e-12)


def test_small_steps_reproduce_the_linear_forecast():
    policy, accepts = build(3, 3, 12, (0.05, 0.95), 1.0, 3)
    recipe = Recipe(0.5, 8, 8, 10, weight_table("rloo", 8))
    seeds = simulate(reference_model(policy, accepts), recipe, np.arange(12), np.arange(12),
                     4000, np.random.default_rng(2))
    linear = forecast(policy, accepts, "rloo", 8, 8, 0.5, 10)
    assert seeds.std == pytest.approx(linear.std, rel=0.1)


def test_a_saturated_run_still_has_a_spread():
    # after enough large steps every run sits on one sequence and the metric gradient is zero;
    # which sequence differs by seed, and the simulated seeds keep that
    policy, accepts = build(3, 3, 12, (0.05, 0.95), 1.0, 3)
    recipe = Recipe(60.0, 8, 8, 50, weight_table("rloo", 8))
    seeds = simulate(reference_model(policy, accepts), recipe, np.arange(12), np.arange(12),
                     400, np.random.default_rng(4))
    settled = np.minimum(seeds.pass_rates, 1 - seeds.pass_rates).max(axis=1) < 1e-2
    assert settled.mean() > 0.5
    assert seeds.std > 0.02


def test_batches_without_replacement_hold_distinct_prompts():
    policy, accepts = build(3, 2, 10, (0.05, 0.95), 1.0, 0)
    model = reference_model(policy, accepts)
    recipe = Recipe(1.0, 6, 4, 1, weight_table("rloo", 4), replace=False)
    prompts, _ = draw(model, np.zeros((50, model.dim)), np.arange(10), recipe,
                      np.random.default_rng(0))
    assert all(len(set(row)) == 6 for row in prompts)
    with pytest.raises(ValueError):
        draw(model, np.zeros((2, model.dim)), np.arange(4), recipe, np.random.default_rng(0))


def test_candidates_a_prompt_does_not_have_are_never_drawn():
    base = np.array([[np.log(0.5), np.log(0.5), -np.inf]])
    model = TangentModel(base, np.zeros((1, 3, 2)), np.array([[1.0, 0.0, 0.0]]))
    recipe = Recipe(1.0, 1, 64, 1, weight_table("rloo", 64))
    _, outcomes = draw(model, np.zeros((20, 2)), np.arange(1), recipe, np.random.default_rng(0))
    assert outcomes.max() <= 1


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError):
        TangentModel(np.zeros((2, 3)), np.zeros((2, 4, 5)), np.zeros((2, 3)))
    model = TangentModel(np.zeros((1, 2)), np.zeros((1, 2, 3)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        simulate(model, Recipe(1.0, 1, 2, 1, weight_table("rloo", 2)), np.arange(1),
                 np.arange(1), 1, np.random.default_rng(0))
