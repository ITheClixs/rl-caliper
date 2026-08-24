"""Predicting the spread of a reported number from one training run.

The covariance recursion in `propagation` describes uncertainty in the parameters. What a reader
of an RL paper actually wants is the spread of the reported score. Linearising a metric `M` about
the trajectory the run took,

    Var_seed[M(theta_T)] = grad M(theta_T)^T S_T grad M(theta_T)

and substituting the unrolled `S_T` turns the quadratic form into a sum of scalars,

    Var[M] = sum_t  b_{t+1}^T Q_t b_{t+1},    b_T = grad M,   b_s = A_s^T b_{s+1}

so the covariance never has to be formed. Each term is the variance of one batch's gradient
projected onto a single direction, which is a scalar a trainer can accumulate.

The backward pass needs `A_s^T b`, and the transpose is free: the mean update of a count-based
group estimator is a gradient field. Its per-prompt mean is `lambda(p_x, G) grad p_x`, so with
`Lambda' = lambda` the field is `grad_theta E_x[Lambda(p_x)]` and its Jacobian is a Hessian,
hence symmetric. `A_s^T b = A_s b = b + eta J_s b` is therefore one Hessian-vector product, which
finite differences deliver from two gradient evaluations and autodiff from one double backward.
For RLOO, where `lambda = 1`, the potential is the pass rate itself.

This module implements the recursion in the enumerable policy, where every term is exact and the
prediction can be checked against independently trained runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caliper.exact.policy import TabularPolicy, fisher, make_prompt
from caliper.exact.propagation import batch_model, injected_covariance, mean_gradient
from caliper.objectives.advantages import weight_table


def metric_value(policy: TabularPolicy, accepts: np.ndarray) -> float:
    """The reported number: pass rate averaged over the prompt pool."""
    return float(np.mean(accepts @ policy.sequence_probs()))


def metric_gradient(policy: TabularPolicy, accepts: np.ndarray) -> np.ndarray:
    """grad of the pass rate, exactly. This is the policy gradient of the reward on the pool."""
    probs, scores = policy.enumerate()
    weighted = accepts * probs  # (n_prompts, n_seq)
    return (weighted @ scores).mean(axis=0)


def transposed_jacobian_product(
    policy: TabularPolicy,
    accepts: np.ndarray,
    weights: np.ndarray,
    direction: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """J^T b = grad_theta (b . g_bar), by central differences on that scalar field.

    On a real model this is a vector-Jacobian product and costs one backward pass; here the
    policy is small enough to take the gradient coordinate by coordinate, which keeps the exact
    check independent of any autodiff implementation.
    """
    dim = policy.dim
    out = np.zeros(dim)
    for index in range(dim):
        shift = np.zeros(dim)
        shift[index] = epsilon
        forward = direction @ mean_gradient(policy.perturbed(shift), accepts, weights)
        backward = direction @ mean_gradient(policy.perturbed(-shift), accepts, weights)
        out[index] = (forward - backward) / (2 * epsilon)
    return out


@dataclass
class MetricForecast:
    variance: float  # Var over seeds of the reported number
    std: float
    seed_gap_std: float  # expected |M_A - M_B| scale between two independent runs
    kernel: list[float]  # each update's share of the final variance
    adjoint_norm: list[float]  # ||b_t||, how strongly update t still matters at the end
    trajectory: list[float]  # the metric along the single run the forecast is made from


def hessian_vector_product(
    policy: TabularPolicy,
    accepts: np.ndarray,
    weights: np.ndarray,
    direction: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """J b by central differences on the mean gradient: two evaluations, whatever dim is.

    Equal to `transposed_jacobian_product` because J is symmetric, and this is the route that
    survives on a model whose parameters cannot be enumerated.
    """
    scale = epsilon / max(np.linalg.norm(direction), 1e-12)
    forward = mean_gradient(policy.perturbed(scale * direction), accepts, weights)
    backward = mean_gradient(policy.perturbed(-scale * direction), accepts, weights)
    return (forward - backward) / (2 * scale)


def mean_trajectory(
    policy: TabularPolicy, accepts: np.ndarray, weights: np.ndarray, step_size: float, steps: int
) -> list[TabularPolicy]:
    states = [policy]
    for _ in range(steps):
        states.append(
            states[-1].perturbed(step_size * mean_gradient(states[-1], accepts, weights))
        )
    return states


def forecast(
    policy: TabularPolicy,
    accepts: np.ndarray,
    estimator: str,
    group_size: int,
    n_prompts: int,
    step_size: float,
    steps: int,
    states: list[TabularPolicy] | None = None,
    matrix_free: bool = True,
    source: str = "all",
    transport: bool = True,
) -> MetricForecast:
    """Carry the metric gradient backwards along one trajectory.

    `states` are the policies a single run actually visited; when omitted the mean trajectory is
    used. Only that one trajectory enters, which is what makes this a forecast rather than a
    description of an ensemble. `source` restricts the injected covariance to one origin of
    randomness, which attributes the final spread to prompt sampling or to rollout sampling.

    `transport=False` holds the adjoint at `grad M` instead of carrying it back, which is the
    ablation that says what the backward pass is worth: it keeps the injected term and discards
    everything the trajectory does to it.
    """
    weights = weight_table(estimator, group_size)
    if states is None:
        states = mean_trajectory(policy, accepts, weights, step_size, steps)
    if len(states) < steps + 1:
        raise ValueError("need one stored state per update, plus the final one")
    trajectory = [metric_value(state, accepts) for state in states[: steps + 1]]

    # backward pass: b_T = grad M, b_s = A_s^T b_{s+1} = b_{s+1} + eta J_s b_{s+1}
    carry = hessian_vector_product if matrix_free else transposed_jacobian_product
    adjoint = metric_gradient(states[steps], accepts)
    kernel = [0.0] * steps
    norms = [0.0] * steps
    for t in range(steps - 1, -1, -1):
        injected = injected_covariance(states[t], accepts, weights, n_prompts, source)
        kernel[t] = float(step_size**2 * adjoint @ injected @ adjoint)
        norms[t] = float(np.linalg.norm(adjoint))
        if transport:
            adjoint = adjoint + step_size * carry(states[t], accepts, weights, adjoint)

    variance = float(sum(kernel))
    return MetricForecast(
        variance=variance,
        std=float(np.sqrt(max(variance, 0.0))),
        seed_gap_std=float(np.sqrt(max(2.0 * variance, 0.0))),
        kernel=kernel,
        adjoint_norm=norms,
        trajectory=trajectory,
    )


def projected_injection(
    policy: TabularPolicy,
    accepts: np.ndarray,
    weights: np.ndarray,
    n_prompts: int,
    direction: np.ndarray,
) -> float:
    """b^T Cov(g_hat) b without forming the covariance: the variance of a projected gradient.

    This is the form a trainer can accumulate online, and the reason the forecast costs one extra
    scalar per update rather than a matrix.
    """
    model = batch_model(policy, accepts)
    return float(model.projected_variance(weights, direction) / n_prompts)


def divergence_metric(policy: TabularPolicy, accepts: np.ndarray) -> np.ndarray:
    """The Fisher information averaged over the pool, for comparison against the metric route."""
    return np.mean([fisher(make_prompt(policy, a)) for a in accepts], axis=0)
