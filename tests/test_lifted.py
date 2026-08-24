"""Lifting the recursion to Adam's state."""

import numpy as np
import pytest

from caliper.exact.lifted import (
    AdamSettings,
    advance_state,
    initial_state,
    transfer_and_injection,
)
from caliper.exact.pool import build
from caliper.exact.propagation import mean_gradient
from caliper.objectives.advantages import weight_table


@pytest.fixture(scope="module")
def warmed():
    policy, accepts = build(3, 2, 12, (0.05, 0.95), 1.0, seed=0)
    weights = weight_table("rloo", 8)
    settings = AdamSettings(step_size=3e-2)
    state = initial_state(policy)
    for _ in range(4):
        state = advance_state(state, accepts, weights, settings)
    return state, accepts, weights, settings


def test_the_mean_update_matches_adam_applied_by_hand(warmed):
    state, accepts, weights, settings = warmed
    grad = mean_gradient(state.policy, accepts, weights)
    moment = settings.beta1 * state.moment + (1 - settings.beta1) * grad
    second = settings.beta2 * state.second + (1 - settings.beta2) * grad**2
    hat_m = moment / (1 - settings.beta1 ** (state.step + 1))
    hat_v = second / (1 - settings.beta2 ** (state.step + 1))
    expected = settings.step_size * hat_m / (np.sqrt(hat_v) + settings.epsilon)

    advanced = advance_state(state, accepts, weights, settings)
    move = (advanced.policy.logits - state.policy.logits).reshape(-1)
    np.testing.assert_allclose(move, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(advanced.moment, moment, rtol=1e-12, atol=0)
    assert advanced.step == state.step + 1


def test_the_injected_covariance_is_positive_semidefinite(warmed):
    state, accepts, weights, settings = warmed
    _, injection = transfer_and_injection(state, accepts, weights, 16, settings)
    np.testing.assert_allclose(injection, injection.T, atol=1e-14)
    assert np.linalg.eigvalsh(injection).min() > -1e-12


def test_the_transfer_operator_moves_the_parameter_block(warmed):
    """A perturbation to the first moment must reach the parameters, or the lift is pointless."""
    state, accepts, weights, settings = warmed
    dim = state.policy.dim
    transfer, _ = transfer_and_injection(state, accepts, weights, 16, settings)
    coupling = np.abs(transfer[:dim, dim : 2 * dim]).max()
    assert coupling > 0.0


def test_freezing_the_second_moment_removes_its_rows(warmed):
    state, accepts, weights, settings = warmed
    dim = state.policy.dim
    frozen, injection = transfer_and_injection(
        state, accepts, weights, 16, settings, carry_second_moment=False
    )
    assert np.abs(frozen[2 * dim :, :]).max() == 0.0
    assert np.abs(injection[:, 2 * dim :]).max() == 0.0


def test_more_prompts_inject_less(warmed):
    state, accepts, weights, settings = warmed
    _, small = transfer_and_injection(state, accepts, weights, 8, settings)
    _, large = transfer_and_injection(state, accepts, weights, 32, settings)
    assert np.trace(large) < np.trace(small)
