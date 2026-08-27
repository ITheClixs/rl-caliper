"""Carrying the optimiser state backwards on a real model, without the curvature term."""

import numpy as np

from caliper.real.adjoint import adam_adjoint_step, adam_sensitivity


def test_the_injected_direction_reduces_to_the_sensitivity_at_the_start():
    """With no accumulated adjoint in m or v, only the direct term survives.

    The backward pass begins with the metric gradient in the theta slot and zeros in the two
    optimiser slots, so the first direction it projects is exactly the diagonal sensitivity
    applied to that gradient. Later steps add the parts that arrive through m and v.
    """
    dim = 8
    rng = np.random.default_rng(0)
    grad = rng.normal(scale=1e-4, size=dim)
    m = np.zeros(dim)
    v = np.abs(rng.normal(scale=1e-8, size=dim))
    metric = rng.normal(size=dim)

    direction, _, _ = adam_adjoint_step(
        metric, np.zeros(dim), np.zeros(dim), m, v, grad, 2e-5
    )
    expected = adam_sensitivity(m, v, grad, 2e-5) * metric
    np.testing.assert_allclose(np.abs(direction), np.abs(expected), rtol=1e-9)


def test_the_optimiser_slots_accumulate_backwards():
    """A perturbation to one gradient keeps acting through m and v on later updates."""
    dim = 6
    rng = np.random.default_rng(1)
    grad = rng.normal(scale=1e-4, size=dim)
    m = rng.normal(scale=1e-5, size=dim)
    v = np.abs(rng.normal(scale=1e-8, size=dim)) + 1e-12
    metric = np.ones(dim)

    _, b1, e1 = adam_adjoint_step(metric, np.zeros(dim), np.zeros(dim), m, v, grad, 2e-5)
    assert np.any(b1 != 0.0), "the m slot must pick up the metric gradient"
    assert np.any(e1 != 0.0), "the v slot must pick up the metric gradient"

    # a second step backwards keeps what the first put there, damped by the betas
    _, b2, e2 = adam_adjoint_step(metric, b1, e1, m, v, grad, 2e-5)
    assert np.all(np.abs(b2) > np.abs(b1) * 0.5)


def test_dropping_the_optimiser_slots_changes_the_direction():
    """The whole point: ignoring m and v is not a small correction."""
    dim = 10
    rng = np.random.default_rng(2)
    grad = rng.normal(scale=1e-4, size=dim)
    m = rng.normal(scale=1e-5, size=dim)
    v = np.abs(rng.normal(scale=1e-8, size=dim)) + 1e-12
    metric = rng.normal(size=dim)

    carried = np.zeros(dim) + 1e-3
    with_state, _, _ = adam_adjoint_step(metric, carried, carried, m, v, grad, 2e-5)
    without, _, _ = adam_adjoint_step(metric, np.zeros(dim), np.zeros(dim), m, v, grad, 2e-5)
    assert not np.allclose(with_state, without)
