"""How far the mean update field bends over one update's worth of noise.

Every forecast in this work linearises the mean update field between runs that seed randomness
has pushed apart. Section 6 measures the residual of that assumption between two runs, which
diagnoses a pair after the fact. It cannot certify a forecast made from one run, which is the
thing a practitioner actually needs before deciding whether to trust a number.

What one run can do is perturb its own stored state along a direction drawn from its own gradient
noise, at a radius matched to one update, and ask whether the mean field is straight over that
distance. That is a symmetric second difference:

    C = || g(x + r d) + g(x - r d) - 2 g(x) ||  /  || g(x + r d) - g(x - r d) ||

zero on an affine field and growing as the field bends. It needs no second run, no replica, and
no knowledge of the answer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from caliper.exact.propagation import injected_covariance, mean_gradient


def curvature_residual(
    field: Callable[[np.ndarray], np.ndarray],
    direction: np.ndarray,
    radius: float,
) -> float:
    """The normalised second difference of `field` along `direction` at `radius`.

    `field` maps a displacement to the mean update there, so `field(0)` is the update at the
    state itself. The denominator is the first difference over the same span, which makes the
    quantity a ratio of bending to travel rather than an absolute curvature.
    """
    unit = np.asarray(direction, dtype=float)
    scale = np.linalg.norm(unit)
    if scale == 0.0:
        raise ValueError("direction must be non-zero")
    unit = unit / scale
    plus = field(radius * unit)
    minus = field(-radius * unit)
    centre = field(np.zeros_like(unit))
    bend = np.linalg.norm(plus + minus - 2.0 * centre)
    travel = np.linalg.norm(plus - minus)
    if travel == 0.0:
        return 0.0
    return float(bend / travel)


def noise_radius(
    policy,
    accepts: np.ndarray,
    weights: np.ndarray,
    n_prompts: int,
    step_size: float,
) -> float:
    """The distance one update of gradient noise moves the parameters.

    This is `eta * sqrt(tr Cov(g_hat))`, the root mean square displacement a single batch's
    departure from the mean update produces. Probing the field at any other radius answers a
    question the run is not asking.
    """
    covariance = injected_covariance(policy, accepts, weights, n_prompts)
    return float(step_size * np.sqrt(max(np.trace(covariance), 0.0)))


def _batch_gradient(policy, accepts, weights, n_prompts, group_size, rng):
    """One stochastic batch gradient, the same draw the trainer takes."""
    probs, scores = policy.enumerate()
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    prompt_ids = rng.integers(0, accepts.shape[0], size=n_prompts)
    outcomes = np.searchsorted(cdf, rng.random((n_prompts, group_size)))
    rewards = accepts[prompt_ids[:, None], outcomes].astype(int)
    counts = rewards.sum(axis=1)
    advantage = weights[counts[:, None], rewards]
    return np.einsum("pg,pgd->d", advantage, scores[outcomes]) / (n_prompts * group_size)


def noise_directions(
    policy,
    accepts: np.ndarray,
    weights: np.ndarray,
    n_prompts: int,
    group_size: int,
    rng: np.random.Generator,
    count: int = 4,
) -> list[np.ndarray]:
    """Directions along which seed noise actually displaces this run.

    Each is the difference of two independent batch gradients, which has zero mean and the
    covariance of the noise itself, so probing along it asks about the part of the field the
    forecast will be integrating over.
    """
    out = []
    for _ in range(count):
        first = _batch_gradient(policy, accepts, weights, n_prompts, group_size, rng)
        second = _batch_gradient(policy, accepts, weights, n_prompts, group_size, rng)
        delta = first - second
        if np.linalg.norm(delta) > 0:
            out.append(delta)
    return out


def run_residual(
    policy,
    accepts: np.ndarray,
    weights: np.ndarray,
    n_prompts: int,
    group_size: int,
    step_size: float,
    rng: np.random.Generator,
    directions: int = 4,
) -> float:
    """The curvature residual at one stored state, averaged over noise directions."""
    radius = noise_radius(policy, accepts, weights, n_prompts, step_size)
    if radius == 0.0:
        return 0.0

    def field(delta: np.ndarray) -> np.ndarray:
        return mean_gradient(policy.perturbed(delta), accepts, weights)

    probes = noise_directions(
        policy, accepts, weights, n_prompts, group_size, rng, directions
    )
    if not probes:
        return 0.0
    return float(np.mean([curvature_residual(field, d, radius) for d in probes]))
