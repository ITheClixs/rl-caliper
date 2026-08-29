"""Cross-fitting the metric gradient out of the injected term."""

import numpy as np

from caliper.real.adjoint import cross_projected_variance


def _projections(direction, contributions):
    return np.array([float(direction @ z) for z in contributions])


def test_the_cross_estimate_is_unbiased_where_the_plain_one_is_not():
    """`Var(b_hat' z)` carries the estimation error of `b_hat`; the cross term does not.

    With `b_hat = b + eps` and `eps` independent of the batch noise, the plain quadratic
    estimates `b' Sigma b + E[eps' Sigma eps]`, which is larger than the truth by a term that
    does not vanish with more prompts. Two independent estimates of `b` have independent errors,
    so their cross quadratic is unbiased for `b' Sigma b`.
    """
    rng = np.random.default_rng(0)
    dim, prompts, trials, scale = 12, 400, 300, 0.1
    truth_direction = rng.normal(size=dim)
    truth_direction /= np.linalg.norm(truth_direction)
    # per-prompt contributions are isotropic with covariance scale^2 I, so the quantity the
    # injected term wants is exactly scale^2 for a unit direction
    exact = scale**2

    plain, crossed = [], []
    for _ in range(trials):
        z = rng.normal(size=(prompts, dim)) * scale
        noise = 0.6  # the metric gradient is badly estimated, as it is on a real model
        b1 = truth_direction + rng.normal(size=dim) * noise
        b2 = truth_direction + rng.normal(size=dim) * noise
        plain.append(np.var(_projections(b1, z), ddof=1))
        crossed.append(cross_projected_variance(_projections(b1, z), _projections(b2, z)))

    assert np.mean(plain) > 2.0 * exact, "the plain estimator should be badly biased up here"
    assert 0.9 * exact < np.mean(crossed) < 1.1 * exact, (
        f"cross estimate {np.mean(crossed):.5f} against truth {exact:.5f}"
    )


def test_the_cross_estimate_reduces_to_the_variance_when_the_directions_agree():
    rng = np.random.default_rng(1)
    a = rng.normal(size=64)
    assert cross_projected_variance(a, a) == np.var(a, ddof=1)
