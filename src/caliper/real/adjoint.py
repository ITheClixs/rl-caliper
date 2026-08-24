"""The backward pass of Theorem 2 on a model whose parameters cannot be enumerated.

Three quantities are needed per update, and each is one pass a trainer already knows how to make.

  grad M      the gradient of the reported score. For a pass rate this is the policy gradient of
              the reward on the held-out set, so it is a rollout and a backward pass.
  J_t b       a Hessian-vector product, taken by central differences on the mean update field at
              the stored parameters. The two evaluations share a random key, so they sample the
              same responses wherever the perturbation does not change which token wins, and the
              difference is a directional derivative rather than a difference of two noise draws.
  b^T Q_t b   the variance across prompts of the batch gradient projected onto b, times eta^2 / P.
              One scalar.

Nothing here forms a matrix. The transpose Theorem 2 asks for is not taken, because
Proposition 3 says the Jacobian is symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from caliper.objectives.advantages import weight_table


def flatten(tree) -> np.ndarray:
    return np.concatenate([np.asarray(v).reshape(-1) for _, v in tree_flatten(tree)])


def unflatten_like(vector: np.ndarray, tree):
    """Rebuild a parameter tree from a flat vector, in the order `tree_flatten` produced."""
    out, offset = [], 0
    for key, value in tree_flatten(tree):
        size = int(np.asarray(value).size)
        out.append((key, mx.array(vector[offset : offset + size].reshape(value.shape))))
        offset += size
    if offset != vector.size:
        raise ValueError("vector does not match the parameter tree")
    return tree_unflatten(out)


def displaced(trainer, base_state: dict, direction: np.ndarray, scale: float):
    """Set the adapters to base + scale * direction, in place."""
    flat = flatten(base_state) + scale * direction
    trainer.model.update(unflatten_like(flat, base_state))


def metric_gradient(trainer, items, rng, key) -> tuple[np.ndarray, float, mx.array]:
    """grad of the held-out pass rate: the reward-weighted score, with no baseline."""
    sequences, masks, rewards, key = trainer.rollout(items, key)
    advantages = [rewards[i].astype(float) for i in range(len(items))]
    _, grads = trainer._gradient(sequences, masks, advantages)
    return flatten(grads), float(rewards.mean()), key


def projected_batch_variance(trainer, corpus, rng, key, direction: np.ndarray):
    """Var over prompts of the projected per-prompt gradient, and the batch's pass rate."""
    cfg = trainer.config
    picks = rng.choice(len(corpus), size=cfg.prompts, replace=False)
    items = [corpus[i] for i in picks]
    sequences, masks, rewards, key = trainer.rollout(items, key)
    counts = rewards.sum(axis=1).astype(int)
    weights = weight_table(cfg.estimator, cfg.group_size)
    projections = []
    for i in range(len(items)):
        advantage = weights[counts[i], rewards[i].astype(int)]
        _, grads = trainer._gradient([sequences[i]], [masks[i]], [advantage])
        projections.append(float(direction @ flatten(grads)))
    return float(np.var(projections, ddof=1)), float(rewards.mean()), key


def mean_update(trainer, corpus, rng_seed: int, key, batches: int = 1) -> np.ndarray:
    """An estimate of g_bar at the current parameters, using a fixed draw of prompts."""
    cfg = trainer.config
    weights = weight_table(cfg.estimator, cfg.group_size)
    total = None
    for b in range(batches):
        rng = np.random.default_rng(rng_seed + b)
        picks = rng.choice(len(corpus), size=cfg.prompts, replace=False)
        items = [corpus[i] for i in picks]
        sequences, masks, rewards, _ = trainer.rollout(items, mx.random.split(key, batches)[b])
        counts = rewards.sum(axis=1).astype(int)
        advantages = [
            weights[counts[i], rewards[i].astype(int)] for i in range(len(items))
        ]
        _, grads = trainer._gradient(sequences, masks, advantages)
        flat = flatten(grads)
        total = flat if total is None else total + flat
    return total / batches


def hessian_vector(
    trainer, base_state: dict, corpus, direction: np.ndarray, key,
    rng_seed: int, epsilon: float, batches: int = 1,
) -> np.ndarray:
    """J b by central differences with common random numbers.

    The same prompt draw and the same sampling key are used on both sides, so the two rollouts
    differ only where the perturbation changes which token is drawn.
    """
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.zeros_like(direction)
    scale = epsilon / norm
    displaced(trainer, base_state, direction, scale)
    forward = mean_update(trainer, corpus, rng_seed, key, batches)
    displaced(trainer, base_state, direction, -scale)
    backward = mean_update(trainer, corpus, rng_seed, key, batches)
    displaced(trainer, base_state, direction, 0.0)
    return (forward - backward) / (2 * scale)


@dataclass
class RealForecast:
    variance: float
    std: float
    kernel: list[float]
    adjoint_norm: list[float]
    metric_value: float


def forecast(
    trainer,
    states: list[dict],
    corpus,
    held_out,
    step_size: float,
    n_prompts: int,
    key,
    seed: int = 0,
    epsilon: float = 1e-3,
    batches: int = 1,
) -> RealForecast:
    """Carry the metric gradient back along the stored states of one run.

    `states[t]` are the adapters after update t, so `states[-1]` is where the run ended.
    """
    steps = len(states) - 1
    displaced(trainer, states[-1], np.zeros(flatten(states[-1]).size), 0.0)
    adjoint, metric, key = metric_gradient(
        trainer, held_out, np.random.default_rng(seed), key
    )

    kernel = [0.0] * steps
    norms = [0.0] * steps
    for t in range(steps - 1, -1, -1):
        displaced(trainer, states[t], np.zeros(adjoint.size), 0.0)
        projected, _, key = projected_batch_variance(
            trainer, corpus, np.random.default_rng(seed + 100 + t), key, adjoint
        )
        kernel[t] = step_size**2 * projected / n_prompts
        norms[t] = float(np.linalg.norm(adjoint))
        adjoint = adjoint + step_size * hessian_vector(
            trainer, states[t], corpus, adjoint, key, seed + 500 + t, epsilon, batches
        )
    variance = float(sum(kernel))
    return RealForecast(
        variance=variance,
        std=float(np.sqrt(max(variance, 0.0))),
        kernel=kernel,
        adjoint_norm=norms,
        metric_value=metric,
    )
