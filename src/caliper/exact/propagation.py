"""Finite-time propagation of training noise through the linearised RL dynamics.

A stochastic update perturbs the policy, and every later update acts on that perturbation. The
covariance of a run around its own mean trajectory therefore satisfies

    S_{t+1} = A_t S_t A_t^T + Q_t,          S_0 = 0

with `A_t = I + eta_t J_t` the Jacobian of the mean update and `Q_t = eta_t^2 Cov(g_hat)` the
covariance injected by one batch. Noise is not accumulated; it is filtered. Directions that the
dynamics contract are forgotten, and directions it leaves alone survive.

In the enumerable policy every term is computable exactly and the whole recursion can be checked
against independent stochastic trajectories, which is what this module is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caliper.exact.aggregate import ExactBatchModel
from caliper.exact.policy import TabularPolicy, fisher, make_prompt
from caliper.objectives.advantages import weight_table


def batch_model(policy: TabularPolicy, accepts: np.ndarray) -> ExactBatchModel:
    prompts = [make_prompt(policy, a) for a in accepts]
    return ExactBatchModel([p.moments() for p in prompts], [fisher(p) for p in prompts])


def mean_gradient(policy: TabularPolicy, accepts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return batch_model(policy, accepts).gradient(weights)


def update_jacobian(
    policy: TabularPolicy,
    accepts: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """J = d g_bar / d theta, by central differences on the exact mean gradient."""
    dim = policy.dim
    jacobian = np.zeros((dim, dim))
    for index in range(dim):
        step = np.zeros(dim)
        step[index] = epsilon
        forward = mean_gradient(policy.perturbed(step), accepts, weights)
        backward = mean_gradient(policy.perturbed(-step), accepts, weights)
        jacobian[:, index] = (forward - backward) / (2 * epsilon)
    return jacobian


def injected_covariance(
    policy: TabularPolicy,
    accepts: np.ndarray,
    weights: np.ndarray,
    n_prompts: int,
    source: str = "all",
) -> np.ndarray:
    """Cov(g_hat) for one batch at the current policy.

    The two terms are the two places randomness enters a batch: which prompts were drawn, and
    which responses were sampled from them. Restricting to one of them answers what share of the
    final uncertainty that source is responsible for.
    """
    model = batch_model(policy, accepts)
    if source == "prompts":
        matrix = model.sigma_b(weights)
    elif source == "rollouts":
        matrix = model.sigma_w(weights)
    elif source == "all":
        matrix = model.sigma_b(weights) + model.sigma_w(weights)
    else:
        raise KeyError(source)
    return matrix / n_prompts


@dataclass
class Propagation:
    covariance: np.ndarray  # S_T
    predicted_kl: float  # E[KL] between two independent seeds
    trace: list[float]  # tr(F S_t) at each step
    kernel: list[float]  # contribution of each update to the final predicted KL


def propagate(
    policy: TabularPolicy,
    accepts: np.ndarray,
    estimator: str,
    group_size: int,
    n_prompts: int,
    step_size: float,
    steps: int,
    kernel_metric: np.ndarray | None = None,
) -> Propagation:
    """Run the deterministic trajectory and the covariance recursion alongside it.

    `kernel_metric` is the quadratic form used to score the final covariance; the Fisher
    information gives predicted policy divergence between two seeds. The per-update kernel
    reports how much of that final number each update is responsible for, which is the quantity
    a scalar timescale hides.
    """
    weights = weight_table(estimator, group_size)
    current = policy
    dim = policy.dim
    covariance = np.zeros((dim, dim))
    transfers = []
    injections = []
    trace = []

    for _ in range(steps):
        jacobian = update_jacobian(current, accepts, weights)
        transfer = np.eye(dim) + step_size * jacobian
        injection = step_size**2 * injected_covariance(current, accepts, weights, n_prompts)
        covariance = transfer @ covariance @ transfer.T + injection
        transfers.append(transfer)
        injections.append(injection)
        current = current.perturbed(step_size * mean_gradient(current, accepts, weights))
        metric = kernel_metric if kernel_metric is not None else np.mean(
            [fisher(make_prompt(current, a)) for a in accepts], axis=0
        )
        trace.append(float(np.trace(metric @ covariance)))

    metric = kernel_metric
    if metric is None:
        metric = np.mean([fisher(make_prompt(current, a)) for a in accepts], axis=0)

    # how much of the final number each update is responsible for, propagated forward
    kernel = []
    for index, injection in enumerate(injections):
        carried = injection
        for transfer in transfers[index + 1 :]:
            carried = transfer @ carried @ transfer.T
        kernel.append(float(np.trace(metric @ carried)))

    # two independent seeds differ by twice the covariance of one around the mean
    return Propagation(
        covariance=covariance,
        predicted_kl=float(np.trace(metric @ covariance)),
        trace=trace,
        kernel=kernel,
    )


def seed_memory(kernel: list[float], quantile: float = 0.95) -> int:
    """Number of final updates holding `quantile` of the predicted variance."""
    values = np.array(kernel)
    if values.sum() <= 0:
        return len(values)
    tail = np.cumsum(values[::-1]) / values.sum()
    return int(np.searchsorted(tail, quantile) + 1)
