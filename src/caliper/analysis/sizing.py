"""Choosing how many repeats to run before calling a comparison.

Two recipes are compared by training each for a few seeds and stopping once one of them leads by
enough to be believed. "Enough" is measured in standard errors, so the rule needs a standard
deviation of the across-seed score. After a single seed per arm there is no empirical one, which
is why the common practice is either to fix the repeat count up front or to spend seeds building
a variance estimate before the comparison can start. Section 3 predicts that deviation from one
run, so the rule is available from the first seed onwards.

Everything here works on scores that have already been collected. The policies differ only in
where the deviation comes from, so the same seeds drive all of them and the comparison is paired.
"""

from __future__ import annotations

import numpy as np


def running_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Mean lead of `second` over `first` after each successive seed.

    Both arrays are indexed `(..., seed)` and share their seed order, so entry `k` is the lead
    a practitioner would have seen with `k + 1` seeds per arm in hand.
    """
    steps = np.arange(1, first.shape[-1] + 1, dtype=float)
    return (np.cumsum(second, -1) - np.cumsum(first, -1)) / steps


def running_deviation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Deviation of the lead estimated from the seeds themselves, seed by seed.

    The first entry is infinite: one seed per arm carries no information about spread, and a rule
    driven by this estimator cannot stop there. That gap is the cost the forecast removes.
    """
    steps = np.arange(1, first.shape[-1] + 1, dtype=float)

    def spread(x):
        mean = np.cumsum(x, -1) / steps
        return (np.cumsum(x**2, -1) - steps * mean**2) / np.maximum(steps - 1.0, 1.0)

    out = np.sqrt(np.maximum(spread(first) + spread(second), 0.0))
    out[..., 0] = np.inf
    return out


def forecast_deviation(first: float, second: float) -> float:
    """Deviation of the lead between two independent arms, from a per-arm forecast."""
    return float(np.hypot(first, second))


def stopping_point(difference: np.ndarray, deviation: np.ndarray, threshold: float) -> np.ndarray:
    """Index of the first seed at which the lead clears `threshold` standard errors.

    Falls back to the last available seed when the rule never fires, which is what a practitioner
    working to a cap would do.
    """
    steps = np.arange(1, difference.shape[-1] + 1, dtype=float)
    with np.errstate(invalid="ignore"):
        fired = np.abs(difference) >= threshold * deviation / np.sqrt(steps)
    return np.where(fired.any(-1), fired.argmax(-1), difference.shape[-1] - 1)


def decision_curve(
    difference: np.ndarray,
    deviation: np.ndarray,
    truth: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeds consumed and mistakes made, per comparison, at each threshold.

    `difference` and `deviation` are indexed `(comparison, ordering, seed)` and `truth` holds the
    sign of the true difference. Averaging is over orderings only, so the caller keeps one number
    per comparison and can resample comparisons afterwards.
    """
    truth = np.asarray(truth)[:, None]
    seeds = np.empty((len(thresholds), difference.shape[0]))
    wrong = np.empty_like(seeds)
    for i, threshold in enumerate(thresholds):
        stop = stopping_point(difference, deviation, threshold)
        seeds[i] = (stop + 1).mean(-1)
        called = np.take_along_axis(difference, stop[:, :, None], -1)[:, :, 0]
        wrong[i] = (np.sign(called) != truth).mean(-1)
    return seeds, wrong


def runs_for_error(seeds: np.ndarray, wrong: np.ndarray, target: float) -> float:
    """Total runs the policy needs to hold its mistake rate at `target`, both arms counted.

    A policy is a curve rather than a point, since the threshold trades seeds against mistakes.
    Reading each curve at a common mistake rate is what makes two policies comparable. The reading
    is linear between the two settings that bracket the target, and infinite if none reaches it.
    """
    cost = 2.0 * np.asarray(seeds, dtype=float)
    rate = np.asarray(wrong, dtype=float)
    order = np.argsort(cost)
    cost, rate = cost[order], rate[order]
    reached = np.flatnonzero(rate <= target)
    if not reached.size:
        return float("inf")
    i = reached[0]
    if i == 0 or rate[i - 1] <= target:
        return float(cost[i])
    share = (rate[i - 1] - target) / (rate[i - 1] - rate[i])
    return float(cost[i - 1] + share * (cost[i] - cost[i - 1]))
