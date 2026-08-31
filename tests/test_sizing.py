import numpy as np
import pytest

from caliper.analysis.sizing import (
    decision_curve,
    forecast_deviation,
    running_deviation,
    running_difference,
    runs_for_error,
    stopping_point,
)


def test_running_difference_matches_the_prefix_means():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(3, 5, 7)), rng.normal(size=(3, 5, 7))
    got = running_difference(a, b)
    for k in range(7):
        want = b[..., : k + 1].mean(-1) - a[..., : k + 1].mean(-1)
        assert np.allclose(got[..., k], want)


def test_running_deviation_matches_the_prefix_variances():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 4, 6)), rng.normal(size=(2, 4, 6))
    got = running_deviation(a, b)
    assert np.all(np.isinf(got[..., 0]))
    for k in range(1, 6):
        want = np.sqrt(a[..., : k + 1].var(-1, ddof=1) + b[..., : k + 1].var(-1, ddof=1))
        assert np.allclose(got[..., k], want)


def test_a_rule_driven_by_the_seeds_cannot_stop_after_one_seed():
    a = np.zeros((1, 1, 4))
    b = np.full((1, 1, 4), 10.0)
    assert stopping_point(running_difference(a, b), running_deviation(a, b), 1.0)[0, 0] == 1


def test_a_forecast_lets_the_same_comparison_stop_immediately():
    a = np.zeros((1, 1, 4))
    b = np.full((1, 1, 4), 10.0)
    deviation = np.full((1, 1, 4), 1.0)
    assert stopping_point(running_difference(a, b), deviation, 1.0)[0, 0] == 0


def test_stopping_falls_back_to_the_cap_when_the_rule_never_fires():
    a = np.zeros((1, 1, 5))
    b = np.zeros((1, 1, 5))
    assert stopping_point(running_difference(a, b), np.ones((1, 1, 5)), 1.0)[0, 0] == 4


def test_forecast_deviation_adds_the_two_arms_in_quadrature():
    assert forecast_deviation(3.0, 4.0) == pytest.approx(5.0)


def test_decision_curve_counts_the_mistakes_it_makes():
    scores = np.array([[[1.0, 1.0]], [[-1.0, -1.0]]])
    zeros = np.zeros_like(scores)
    difference = running_difference(zeros, scores)
    deviation = np.full(difference.shape, 0.1)
    seeds, wrong = decision_curve(difference, deviation, np.array([1.0, 1.0]), np.array([1.0]))
    assert np.allclose(seeds, 1.0)
    assert np.allclose(wrong[0], [0.0, 1.0])


def test_runs_for_error_interpolates_between_the_bracketing_settings():
    seeds = np.array([1.0, 2.0, 3.0])
    wrong = np.array([0.30, 0.10, 0.05])
    assert runs_for_error(seeds, wrong, 0.20) == pytest.approx(3.0)
    assert runs_for_error(seeds, wrong, 0.10) == pytest.approx(4.0)


def test_runs_for_error_reports_no_finite_cost_when_the_target_is_out_of_reach():
    assert not np.isfinite(runs_for_error(np.array([1.0, 2.0]), np.array([0.5, 0.4]), 0.01))
