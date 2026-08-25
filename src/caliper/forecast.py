"""A framework-agnostic backward pass for the spread of a reported number.

Theorem 2 needs three things from a training run and nothing else:

  * the gradient of the reported metric at the endpoint,
  * per update, the variance across prompts of the batch gradient projected onto one direction,
  * per update, a Jacobian-vector product of the mean update field with that same direction.

Nothing here knows what a policy is, which array library holds it, or how a rollout is sampled.
Supply an object per update that answers those two questions and this accumulates the forecast.
`caliper.exact.adjoint` and `caliper.real.adjoint` are the two implementations of that protocol in
this repository; a trainer in another framework needs only a third.

The updates are consumed in reverse order, which is the order the recursion runs in and the order
a stored trajectory is usually most convenient to read back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Update(Protocol):
    """One update of a training run, as the backward pass needs to see it."""

    step_size: float
    n_prompts: int

    def projected_variance(self, direction: np.ndarray) -> float:
        """Var over prompts of the per-prompt gradient projected onto `direction`."""

    def jacobian_vector(self, direction: np.ndarray) -> np.ndarray:
        """J v for the mean update field at this update's parameters.

        The field is the *ascent* direction, the one in `theta <- theta + eta g`. A trainer whose
        autodiff returns the gradient of a loss must negate it before differentiating again, or
        the adjoint is carried through `I - eta J` and the sign of every correction flips.

        By Proposition 3 the Jacobian is symmetric for count-based estimators, so this is also
        the transposed product the recursion asks for. An optimiser carrying state breaks that;
        see `caliper.exact.lifted`.
        """


@dataclass
class Forecast:
    variance: float
    kernel: list[float] = field(default_factory=list)
    adjoint_norm: list[float] = field(default_factory=list)

    @property
    def std(self) -> float:
        """Forecast standard deviation of the reported metric across seeds."""
        return float(np.sqrt(max(self.variance, 0.0)))

    @property
    def seed_gap_std(self) -> float:
        """Scale of the difference between two independent runs, which is larger by sqrt(2)."""
        return float(np.sqrt(max(2.0 * self.variance, 0.0)))

    def interval(self, value: float, level: float = 0.95) -> tuple[float, float]:
        """A normal interval around a reported `value`, at the forecast spread."""
        from scipy.stats import norm

        half = float(norm.ppf(0.5 + level / 2.0)) * self.std
        return value - half, value + half

    def memory(self, quantile: float = 0.95) -> int:
        """Number of final updates holding `quantile` of the forecast variance."""
        values = np.asarray(self.kernel)
        if values.size == 0 or values.sum() <= 0:
            return values.size
        tail = np.cumsum(values[::-1]) / values.sum()
        return int(np.searchsorted(tail, quantile) + 1)


def run_backward(metric_gradient: np.ndarray, updates: Sequence[Update]) -> Forecast:
    """Accumulate Var[M] backwards along a stored trajectory.

    `updates` is in forward order; it is consumed from the last update to the first. The adjoint
    starts at the metric gradient and each step contributes one scalar, so no covariance matrix
    exists at any point.
    """
    adjoint = np.asarray(metric_gradient, dtype=float)
    kernel = [0.0] * len(updates)
    norms = [0.0] * len(updates)
    for index in range(len(updates) - 1, -1, -1):
        update = updates[index]
        projected = update.projected_variance(adjoint)
        kernel[index] = float(update.step_size**2 * projected / update.n_prompts)
        norms[index] = float(np.linalg.norm(adjoint))
        adjoint = adjoint + update.step_size * np.asarray(
            update.jacobian_vector(adjoint), dtype=float
        )
    return Forecast(variance=float(sum(kernel)), kernel=kernel, adjoint_norm=norms)


@dataclass
class LinearUpdate:
    """An update whose mean field is linear and whose noise is a fixed covariance.

    Useful for checking an implementation of the protocol against a system whose answer is known
    in closed form, and for unit tests that do not need a policy.
    """

    jacobian: np.ndarray
    covariance: np.ndarray
    step_size: float
    n_prompts: int

    def projected_variance(self, direction: np.ndarray) -> float:
        return float(direction @ self.covariance @ direction)

    def jacobian_vector(self, direction: np.ndarray) -> np.ndarray:
        return self.jacobian @ direction
