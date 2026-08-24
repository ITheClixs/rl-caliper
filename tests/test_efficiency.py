"""The shared efficiency curve and the group size it selects."""

import numpy as np
import pytest

from caliper.analysis.efficiency import critical_batch, efficiency, optimal_group_size

TAU_B, TAU_W, SIGNAL = 0.42, 1.31, 0.87


def plain(group_sizes, rollouts):
    """The formula as written in the paper, with no finite-corpus term."""
    g = np.asarray(group_sizes, dtype=float)
    bcrit = (TAU_B + TAU_W / (g - 1.0)) / SIGNAL
    return 1.0 / (1.0 + g * bcrit / rollouts)


def test_an_unbounded_corpus_reduces_to_the_plain_formula():
    grid = np.array([2.0, 3.0, 5.0, 8.0, 16.0])
    np.testing.assert_allclose(
        efficiency(grid, TAU_B, TAU_W, SIGNAL, 256), plain(grid, 256), rtol=0, atol=1e-15
    )


def test_a_finite_corpus_lowers_the_critical_batch():
    """Sampling prompts without replacement removes part of the between-prompt term."""
    grid = np.array([2.0, 4.0, 8.0])
    infinite = critical_batch(grid, TAU_B, TAU_W, SIGNAL, 256)
    finite = critical_batch(grid, TAU_B, TAU_W, SIGNAL, 256, corpus=512)
    assert np.all(finite < infinite)


def test_drawing_the_whole_corpus_removes_the_between_term():
    """With P = N the between-prompt variance is not sampled at all."""
    g, rollouts, corpus = 4.0, 256, 64  # P = 256/4 = 64 = N
    got = critical_batch(np.array([g]), TAU_B, TAU_W, SIGNAL, rollouts, corpus=corpus)
    expected = (TAU_W / (g - 1.0)) / SIGNAL
    assert got[0] == pytest.approx(expected, rel=1e-12)


def test_efficiency_lies_in_the_unit_interval():
    grid = np.linspace(2.0, 64.0, 64)
    rho = efficiency(grid, TAU_B, TAU_W, SIGNAL, 256, corpus=4096)
    assert np.all(rho > 0.0) and np.all(rho <= 1.0)


def test_the_closed_form_optimum_matches_a_numerical_search():
    """G* = 1 + sqrt(tau_w / tau_b) must maximise rho where prefill cost is absent.

    The closed form drops the G/R term, so it is the large-budget limit; the search is run at a
    budget large enough for that term to be negligible.
    """
    grid = np.linspace(2.0, 200.0, 200_000)
    rho = efficiency(grid, TAU_B, TAU_W, SIGNAL, 10**9)
    assert grid[int(np.argmax(rho))] == pytest.approx(optimal_group_size(TAU_B, TAU_W), rel=1e-3)


def test_a_vanishing_between_term_sends_the_optimum_to_infinity():
    assert optimal_group_size(0.0, TAU_W) == float("inf")


def test_more_within_prompt_noise_calls_for_larger_groups():
    assert optimal_group_size(TAU_B, 4 * TAU_W) > optimal_group_size(TAU_B, TAU_W)
