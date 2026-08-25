"""Recovering the seed term from a spread that also contains evaluation noise."""

import numpy as np

from caliper.analysis.uncertainty import resolved_spread_interval, spread_bounds


def test_the_binomial_term_is_unbiased_for_the_evaluation_variance():
    """`p_hat (1 - p_hat)` is a biased estimate of `p (1 - p)`, and the bias matters.

    For `p_hat` the mean of `S` Bernoulli draws, `E[p_hat (1 - p_hat)] = (S-1)/S p (1-p)`, so
    dividing by `S` understates the variance of `p_hat` by that factor and the seed term left
    after subtraction is correspondingly too large. Dividing by `S - 1` is unbiased.
    """
    rng = np.random.default_rng(0)
    samples, questions, runs = 8, 40, 300
    truth = rng.uniform(0.15, 0.85, size=questions)
    # the quantity the subtraction is meant to remove: the variance of one run's mean score
    target = float(np.mean(truth * (1.0 - truth)) / (samples * questions))

    estimates = []
    for _ in range(runs):
        rates = rng.binomial(samples, truth, size=(2, questions)) / samples
        estimates.append(spread_bounds(rates, samples)["binomial"] ** 2)
    assert np.mean(estimates) == np.float64(np.mean(estimates))  # finite
    ratio = float(np.mean(estimates)) / target
    assert 0.96 < ratio < 1.04, f"binomial term biased by {ratio:.3f} of the truth"


def test_the_resolved_interval_uses_the_same_correction():
    rng = np.random.default_rng(1)
    samples, questions = 8, 40
    truth = rng.uniform(0.15, 0.85, size=questions)
    target = float(np.mean(truth * (1.0 - truth)) / (samples * questions))
    rates = rng.binomial(samples, truth, size=(6, questions)) / samples
    out = resolved_spread_interval(rates, samples, n_boot=200, seed=0)
    assert out["binomial"] ** 2 > target * 0.85
