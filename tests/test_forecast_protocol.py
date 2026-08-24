"""The framework-agnostic backward pass, against systems whose answer is known."""

import numpy as np
import pytest

from caliper.forecast import Forecast, LinearUpdate, run_backward


def symmetric_negative(dim, rng):
    matrix = -rng.random((dim, dim))
    return (matrix + matrix.T) / 2


def spd(dim, rng):
    root = rng.normal(size=(dim, dim))
    return root @ root.T


def explicit(jacobians, covariance, step, prompts, gradient):
    """Build S_T the expensive way and take the quadratic form."""
    dim = gradient.size
    state = np.zeros((dim, dim))
    for jacobian in jacobians:
        transfer = np.eye(dim) + step * jacobian
        state = transfer @ state @ transfer.T + step**2 * covariance / prompts
    return float(gradient @ state @ gradient)


def test_the_protocol_reproduces_the_explicit_recursion():
    rng = np.random.default_rng(0)
    dim, steps, eta, prompts = 6, 12, 0.1, 8
    jacobian, covariance = symmetric_negative(dim, rng), spd(dim, rng)
    gradient = rng.normal(size=dim)
    updates = [LinearUpdate(jacobian, covariance, eta, prompts) for _ in range(steps)]
    got = run_backward(gradient, updates).variance
    assert got == pytest.approx(
        explicit([jacobian] * steps, covariance, eta, prompts, gradient), rel=1e-12
    )


def test_it_handles_a_jacobian_that_changes_every_step():
    rng = np.random.default_rng(1)
    dim, steps, eta, prompts = 5, 9, 0.2, 4
    jacobians = [symmetric_negative(dim, rng) for _ in range(steps)]
    covariance = spd(dim, rng)
    gradient = rng.normal(size=dim)
    updates = [LinearUpdate(j, covariance, eta, prompts) for j in jacobians]
    assert run_backward(gradient, updates).variance == pytest.approx(
        explicit(jacobians, covariance, eta, prompts, gradient), rel=1e-12
    )


def test_a_zero_jacobian_gives_plain_accumulation():
    """With no filtering the recursion must reduce to summing the injected terms."""
    rng = np.random.default_rng(2)
    dim, steps, eta, prompts = 4, 7, 0.3, 16
    covariance = spd(dim, rng)
    gradient = rng.normal(size=dim)
    updates = [LinearUpdate(np.zeros((dim, dim)), covariance, eta, prompts) for _ in range(steps)]
    expected = steps * eta**2 * float(gradient @ covariance @ gradient) / prompts
    assert run_backward(gradient, updates).variance == pytest.approx(expected, rel=1e-12)


def test_the_kernel_sums_to_the_variance():
    rng = np.random.default_rng(3)
    updates = [LinearUpdate(symmetric_negative(4, rng), spd(4, rng), 0.1, 8) for _ in range(6)]
    result = run_backward(rng.normal(size=4), updates)
    assert sum(result.kernel) == pytest.approx(result.variance, rel=1e-12)


def test_an_empty_run_forecasts_no_spread():
    result = run_backward(np.ones(3), [])
    assert result.variance == 0.0
    assert result.memory() == 0


def test_the_seed_gap_is_larger_by_root_two():
    result = Forecast(variance=2.0)
    assert result.seed_gap_std == pytest.approx(np.sqrt(2) * result.std)


def test_the_interval_widens_with_the_forecast():
    narrow = Forecast(variance=1e-4).interval(0.8)
    wide = Forecast(variance=4e-4).interval(0.8)
    assert wide[0] < narrow[0] < 0.8 < narrow[1] < wide[1]


def test_memory_reads_the_kernel_from_the_end():
    assert Forecast(variance=1.0, kernel=[0.0, 0.0, 0.0, 1.0]).memory(0.95) == 1
    assert Forecast(variance=1.0, kernel=[0.25] * 4).memory(0.95) == 4
