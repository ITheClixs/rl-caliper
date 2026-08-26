"""The backward pass of Theorem 2 on a model whose parameters cannot be enumerated.

Three quantities are needed per update, and each is one pass a trainer already knows how to make.

  grad M      the gradient of the reported score. For a pass rate this is the policy gradient of
              the reward on the held-out set, so it is a rollout and a backward pass.
  J_t b       a Hessian-vector product, taken by central differences on the mean update field at
              the stored parameters. The two evaluations share a random key, so they sample the
              same responses wherever the perturbation does not change which token wins, and the
              difference is a directional derivative rather than a difference of two noise draws.
              The difference step has a floor: model weights are float32 here, so below about
              1e-4 the two evaluations cancel into rounding error rather than into a derivative.
              The default of 1e-3 sits above that floor and well below the scale on which the
              gradient field bends.
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


def ascent_direction(grads) -> np.ndarray:
    """The trainer's `_gradient` returns the gradient of a loss; the update is its negative.

    Everything in this module works in the ascent convention, the one the recursion is written
    in: the update is `theta <- theta + eta g`, so `g` is minus the loss gradient. Getting this
    backwards would transport the adjoint through `I - eta J` instead of `I + eta J`.
    """
    return -flatten(grads)


def metric_gradient(trainer, items, rng, key) -> tuple[np.ndarray, float, mx.array]:
    """grad of the held-out pass rate: the reward-weighted score, with no baseline."""
    sequences, masks, rewards, key = trainer.rollout(items, key)
    advantages = [rewards[i].astype(float) for i in range(len(items))]
    _, grads = trainer._gradient(sequences, masks, advantages)
    return ascent_direction(grads), float(rewards.mean()), key


def projected_batch_variance(
    trainer, corpus, rng, key, direction: np.ndarray, prompts: int | None = None
):
    """Var over prompts of the projected per-prompt gradient, the pass rate, and the live share.

    The live share is the fraction of prompts whose group is not unanimous. A count-based
    advantage is identically zero on a unanimous group, so once the policy has become
    deterministic on the corpus no rollout noise enters at all and the injected term is exactly
    zero. That is a real property of the run, not a failure of the estimate, and reporting it
    separately is what tells the two apart.
    """
    cfg = trainer.config
    # the variance is over the prompt distribution, so it is estimated better with more prompts
    # than the training batch happens to use; the 1/P that divides it stays the training P
    count = min(prompts or cfg.prompts, len(corpus))
    picks = rng.choice(len(corpus), size=count, replace=False)
    items = [corpus[i] for i in picks]
    sequences, masks, rewards, key = trainer.rollout(items, key)
    counts = rewards.sum(axis=1).astype(int)
    weights = weight_table(cfg.estimator, cfg.group_size)
    projections = []
    live = 0
    for i in range(len(items)):
        advantage = weights[counts[i], rewards[i].astype(int)]
        if np.any(advantage != 0.0):
            live += 1
        _, grads = trainer._gradient([sequences[i]], [masks[i]], [advantage])
        projections.append(float(direction @ ascent_direction(grads)))
    return (
        float(np.var(projections, ddof=1)),
        float(rewards.mean()),
        live / max(len(items), 1),
        key,
    )


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
        flat = ascent_direction(grads)
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
    m_new = beta1 * moment + (1.0 - beta1) * mean_gradient
    v_new = beta2 * second + (1.0 - beta2) * mean_gradient**2
    root = np.sqrt(np.maximum(v_new, 0.0))
    denom = root + epsilon
    direct = (1.0 - beta1) / denom
    # d/dg of 1/(sqrt(v') + eps), which is where Adam's normalisation enters
    through_v = m_new * (1.0 - beta2) * mean_gradient / (np.maximum(root, 1e-30) * denom**2)
    return np.abs(step_size * (direct - through_v))


@dataclass
class RealForecast:
    variance: float
    std: float
    kernel: list[float]
    adjoint_norm: list[float]
    metric_value: float
    jvp_agreement: list[float]  # cosine between two independent estimates of J b
    live_share: list[float]  # fraction of prompts still sampling more than one answer
    second_moment_floor: list[float]  # share of coordinates with v = 0, where Adam is singular
    init_checksum: float  # identifies the adapter initialisation this run started from


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
    check_agreement: bool = True,
    transport: bool = True,
    variance_prompts: int | None = None,
    optimiser_states: list[dict] | None = None,
    adam: tuple[float, float, float] | None = None,
) -> RealForecast:
    """Carry the metric gradient back along the stored states of one run.

    `states[t]` are the adapters after update t, so `states[-1]` is where the run ended.

    `variance_prompts` sets how many prompts the injected term is estimated from. Once most
    groups have gone unanimous a small batch mostly contains zeros, and the sample variance over
    eight prompts is a poor and low-biased estimate of the variance over the corpus.

    `transport=False` holds the adjoint at `grad M` and skips the Hessian-vector products
    entirely, which is the cheap form: one pass per update instead of three.

    `optimiser_states[t]` are Adam's moments before update t. Given them, the update's
    sensitivity to the gradient it was handed is the diagonal of `adam_sensitivity` rather than
    the plain `eta`, so the injected term becomes `Var_i(b' B z_i) / P` with `B` that diagonal.
    Without them the plain-ascent form `eta^2 Var_i(b' z_i) / P` is used.

    With `check_agreement`, each Hessian-vector product is estimated a second time from an
    independent draw of prompts and the cosine between the two is recorded. A forecast whose
    backward pass is dominated by sampling noise will show cosines near zero, and that is worth
    knowing before the number it produces is believed.
    """
    steps = len(states) - 1
    # Two forecasts are only comparable if they started from the same adapters. Recording the
    # checksum of state zero makes that checkable afterwards instead of assumed.
    checksum = float(np.abs(flatten(states[0])).sum())
    displaced(trainer, states[-1], np.zeros(flatten(states[-1]).size), 0.0)
    adjoint, metric, key = metric_gradient(
        trainer, held_out, np.random.default_rng(seed), key
    )

    kernel = [0.0] * steps
    norms = [0.0] * steps
    live_share = [0.0] * steps
    floor = [0.0] * steps
    agreement = []
    for t in range(steps - 1, -1, -1):
        displaced(trainer, states[t], np.zeros(adjoint.size), 0.0)
        if optimiser_states is not None:
            # the update the run actually took was Adam's, so the noise enters through its
            # diagonal preconditioner rather than through eta alone
            store = optimiser_states[t]
            beta1, beta2, eps = adam if adam is not None else (0.9, 0.999, 1e-8)
            mean = mean_update(trainer, corpus, seed + 700 + t, key, batches)
            if not np.all(np.isfinite(mean)):
                raise FloatingPointError(
                    f"mean update at step {t} is not finite; the run has diverged"
                )
            scale = adam_sensitivity(
                store["m"], store["v"], mean, step_size, beta1, beta2, eps
            )
            # Adam is singular at v = 0: the step is then eta m / eps, so the sensitivity is
            # bounded only by epsilon and the linearisation says nothing. Record how much of the
            # state is still there rather than reporting a number that rests on it.
            floor[t] = float(np.mean(store["v"] <= 0.0))
            direction = adjoint * scale
        else:
            direction = adjoint
        projected, _, live, key = projected_batch_variance(
            trainer, corpus, np.random.default_rng(seed + 100 + t), key, direction,
            prompts=variance_prompts,
        )
        kernel[t] = (
            projected / n_prompts
            if optimiser_states is not None
            else step_size**2 * projected / n_prompts
        )
        live_share[t] = live
        norms[t] = float(np.linalg.norm(adjoint))
        if not transport:
            continue
        product = hessian_vector(
            trainer, states[t], corpus, adjoint, key, seed + 500 + t, epsilon, batches
        )
        if check_agreement:
            other = hessian_vector(
                trainer, states[t], corpus, adjoint, key, seed + 9000 + t, epsilon, batches
            )
            scale = np.linalg.norm(product) * np.linalg.norm(other)
            agreement.append(float(product @ other / scale) if scale > 0 else float("nan"))
        adjoint = adjoint + step_size * product
    variance = float(sum(kernel))
    return RealForecast(
        variance=variance,
        std=float(np.sqrt(max(variance, 0.0))),
        kernel=kernel,
        adjoint_norm=norms,
        metric_value=metric,
        jvp_agreement=agreement,
        live_share=live_share,
        second_moment_floor=floor,
        init_checksum=checksum,
    )
