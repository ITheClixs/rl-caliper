"""The reconvergence fits recover exponents and slopes they are handed."""

import numpy as np

from caliper.analysis.reconvergence import level_fit, spread_fit, tail_slopes


def test_level_fit_recovers_the_exponents_in_batch_and_drift():
    rows = [
        {"prompts": p, "drift": d, "kl": 3e-3 * p**-0.5 * (d / 1e-4) ** 0.5, "spread": 0.01}
        for p in (8, 16, 32, 64) for d in (1e-4, 1e-3)
    ]
    fit = level_fit(rows, n_boot=200)
    assert np.allclose(fit["beta"][1:], [-0.5, 0.5])
    assert fit["r2"] > 0.999
    # with no noise every resample that can be fitted gives the same answer
    assert np.allclose([fit["lo"][1], fit["hi"][1]], -0.5)


def test_spread_fit_reports_the_slope_on_divergence():
    kl = np.geomspace(1e-4, 1e-2, 6)
    rows = [{"kl": k, "spread": 0.2 * k**0.5} for k in kl] + [{"kl": 0.0, "spread": 0.1}]
    fit = spread_fit(rows, n_boot=200)
    assert fit["n"] == 6  # the zero-divergence setting cannot sit on a log axis and is dropped
    assert np.isclose(fit["beta"][1], 0.5)


def test_tail_slope_uses_the_second_half_of_the_learning_updates():
    steps = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=float)
    # grows like a random walk early, then flat: only the flat half should be seen
    kl = np.where(steps <= 4, 1e-4 * np.maximum(steps, 1), 4e-4)
    cell = {"history": {"step": steps.tolist(), "pairwise_kl": kl.tolist()}}
    keep = np.ones(steps.size, dtype=bool)
    slopes = tail_slopes([(cell, keep)])
    assert np.isclose(slopes[0], 0.0)
