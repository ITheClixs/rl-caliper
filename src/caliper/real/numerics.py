"""The parts of the real-model backward pass that are plain array arithmetic.

They are kept out of `caliper.real.adjoint`, which imports MLX at module level, so that the
arithmetic can be tested on any machine: the Adam lift and the cross-fitted injected term are
checked against numerical derivatives and closed forms in `tests/`, and none of that needs a
model or Apple Silicon.
"""

from __future__ import annotations

import numpy as np


def cross_projected_variance(first: np.ndarray, second: np.ndarray) -> float:
    """The covariance of two projections of the same per-prompt gradients.

    The injected term wants `b' Sigma b` for the true metric gradient `b`. On a real model only
    an estimate `b_hat = b + eps` is available, and

        E[Var_i(b_hat' z_i)] = b' Sigma b + E[eps' Sigma eps],

    so the plain quadratic is biased upward by a term that more prompts do not remove: it is the
    metric gradient that is noisy, not the sample. Two estimates built from independent
    evaluation rollouts have independent errors, so the covariance of their projections is
    unbiased for the quantity wanted.
    """
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("projections must be of the same prompts")
    if a.size < 2:
        return 0.0
    return float(np.sum((a - a.mean()) * (b - b.mean())) / (a.size - 1))


def adam_sensitivity(
    moment: np.ndarray,
    second: np.ndarray,
    mean_gradient: np.ndarray,
    step_size: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """`|d theta' / d g_hat|` for one Adam update, evaluated at the mean gradient.

    Adam applies `theta' = theta - lr m' / (sqrt(v') + eps)` coordinate-wise, so its sensitivity
    to the gradient it was handed is diagonal and the injected covariance is `B Sigma B` with
    `B` that diagonal. Two terms contribute: the moment picks the gradient up directly, and the
    second moment moves the denominator underneath it.

    `moment` and `second` are the optimiser state *before* the update. The magnitude is what the
    forecast needs, since a variance does not see the sign.
    """
    # The moments arrive from the optimiser in single precision, where the product below
    # underflows to zero on any coordinate with no gradient and no accumulated second moment,
    # turning the ratio into 0/0. Do the arithmetic in double.
    moment = np.asarray(moment, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    mean_gradient = np.asarray(mean_gradient, dtype=np.float64)

    m_new = beta1 * moment + (1.0 - beta1) * mean_gradient
    v_new = beta2 * second + (1.0 - beta2) * mean_gradient**2
    root = np.sqrt(np.maximum(v_new, 0.0))
    denom = root + epsilon
    direct = (1.0 - beta1) / denom
    # d/dg of 1/(sqrt(v') + eps), which is where Adam's normalisation enters. The term carries a
    # factor of the gradient, so it tends to zero as the coordinate goes quiet even though the
    # ratio that expresses it does not; take that limit rather than evaluating 0/0.
    through_v = np.zeros_like(m_new)
    alive = root > 0.0
    through_v[alive] = (
        m_new[alive] * (1.0 - beta2) * mean_gradient[alive]
        / (root[alive] * denom[alive] ** 2)
    )
    return np.abs(step_size * (direct - through_v))


def adam_adjoint_step(
    metric: np.ndarray,
    carried_moment: np.ndarray,
    carried_second: np.ndarray,
    moment: np.ndarray,
    second: np.ndarray,
    mean_gradient: np.ndarray,
    step_size: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One backward step of the adjoint on Adam's state `z = (theta, m, v)`.

    A gradient perturbation does not stop acting when the step that received it is over: it stays
    in the moments and keeps moving the parameters for as long as the betas remember it. Carrying
    only the parameter slot ignores that, which on the exact tier costs a factor of sixteen even
    where the dynamics are perfectly local.

    Every block of the transfer operator that needs the Jacobian of the mean update sits in the
    column that differentiates with respect to theta. Those are dropped here. On the exact tier
    that costs nothing at all, and on a real model it is the difference between an analytic
    diagonal recursion and one resting on a Jacobian whose independent estimates agree to a
    cosine of about a tenth.

    Returns the direction to project the batch gradient onto, and the two optimiser slots to
    carry to the previous update.
    """
    moment = np.asarray(moment, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    mean_gradient = np.asarray(mean_gradient, dtype=np.float64)

    m_new = beta1 * moment + (1.0 - beta1) * mean_gradient
    v_new = beta2 * second + (1.0 - beta2) * mean_gradient**2
    root = np.sqrt(np.maximum(v_new, 0.0))
    denom = root + epsilon

    # B = dPhi/dg has three blocks; the direction is B^T applied to the carried adjoint
    sensitivity = adam_sensitivity(
        moment, second, mean_gradient, step_size, beta1, beta2, epsilon
    )
    direction = (
        sensitivity * metric
        + (1.0 - beta1) * carried_moment
        + 2.0 * (1.0 - beta2) * mean_gradient * carried_second
    )

    # the two optimiser rows of A^T, both diagonal once the curvature blocks are dropped
    next_moment = step_size * beta1 * metric / denom + beta1 * carried_moment
    through_v = np.zeros_like(metric)
    alive = root > 0.0
    through_v[alive] = (
        step_size * m_new[alive] * beta2 * metric[alive]
        / (2.0 * root[alive] * denom[alive] ** 2)
    )
    next_second = beta2 * carried_second - through_v
    return direction, next_moment, next_second
