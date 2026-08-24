"""The same recursion when the optimiser carries state.

Plain ascent makes the parameters the whole state, so the transfer operator is `I + eta J`. Adam
does not: the update depends on the moment estimates as well, and a perturbation to a gradient
survives in them long after the step that produced it. The state that closes is

    z = (theta, m, v)

with the mean map

    m' = b1 m + (1 - b1) g_bar(theta)
    v' = b2 v + (1 - b2) g_bar(theta)^2
    theta' = theta + eta m' / (sqrt(v') + eps)

Writing that map as `z' = Phi(z)`, the transfer operator is `A = dPhi/dz` and the injected
covariance is `Q = B Cov(g_hat) B^T` with `B = dPhi/dg`, the sensitivity of one update to the
gradient it was given. Both are taken by central differences on the exact mean map, so nothing
about the derivation has to be trusted.

The second-moment estimate makes the map non-linear in the gradient, so `B` is evaluated at the
mean gradient; that is the same local linearisation the rest of the theory rests on, applied one
level up.

The map is singular at `v = 0`: the derivative of `1/(sqrt(v) + eps)` diverges there, so a
recursion started at initialisation is not linearisable at all. The moments have to be warmed
before the covariance is propagated, which is what a real run does anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caliper.exact.policy import TabularPolicy
from caliper.exact.propagation import injected_covariance, mean_gradient


@dataclass(frozen=True)
class AdamSettings:
    step_size: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    bias_correction: bool = True


@dataclass
class LiftedState:
    policy: TabularPolicy
    moment: np.ndarray
    second: np.ndarray
    step: int = 0

    def vector(self) -> np.ndarray:
        return np.concatenate([np.zeros(self.policy.dim), self.moment, self.second])


def _advance(
    logits_delta: np.ndarray,
    moment: np.ndarray,
    second: np.ndarray,
    policy: TabularPolicy,
    accepts: np.ndarray,
    weights: np.ndarray,
    settings: AdamSettings,
    step: int,
    gradient: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One mean Adam update from a state displaced by `logits_delta`.

    Returns the displacement of the new parameters, and the new moments. `gradient` overrides the
    mean gradient, which is how the sensitivity to the injected noise is taken.
    """
    shifted = policy.perturbed(logits_delta)
    grad = mean_gradient(shifted, accepts, weights) if gradient is None else gradient
    new_moment = settings.beta1 * moment + (1.0 - settings.beta1) * grad
    new_second = settings.beta2 * second + (1.0 - settings.beta2) * grad**2
    if settings.bias_correction:
        hat_m = new_moment / (1.0 - settings.beta1 ** (step + 1))
        hat_v = new_second / (1.0 - settings.beta2 ** (step + 1))
    else:
        hat_m, hat_v = new_moment, new_second
    move = settings.step_size * hat_m / (np.sqrt(np.maximum(hat_v, 0.0)) + settings.epsilon)
    return logits_delta + move, new_moment, new_second


def transfer_and_injection(
    state: LiftedState,
    accepts: np.ndarray,
    weights: np.ndarray,
    n_prompts: int,
    settings: AdamSettings,
    epsilon: float = 1e-6,
    carry_second_moment: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """A = dPhi/dz and Q = B Cov(g_hat) B^T for one Adam update, by central differences.

    `carry_second_moment` decides whether v is part of the propagated state. It is a slow
    variable -- beta2 is 0.999 -- and the derivative of its normalisation is the term that makes
    the map hard to linearise, so holding it at its mean trajectory is worth measuring against
    carrying it.
    """
    dim = state.policy.dim
    size = 3 * dim
    base = np.concatenate([np.zeros(dim), state.moment, state.second])

    def apply(vector: np.ndarray) -> np.ndarray:
        delta, moment, second = vector[:dim], vector[dim : 2 * dim], vector[2 * dim :]
        out = _advance(
            delta, moment, second, state.policy, accepts, weights, settings, state.step
        )
        return np.concatenate(out)

    # the three blocks of z live on very different scales -- v is of order g^2 -- so the
    # difference step is taken relative to the coordinate it perturbs
    scale = np.maximum(np.abs(base), 1e-12) * epsilon
    scale[:dim] = epsilon
    transfer = np.zeros((size, size))
    for index in range(size):
        shift = np.zeros(size)
        shift[index] = scale[index]
        transfer[:, index] = (apply(base + shift) - apply(base - shift)) / (2 * scale[index])

    # sensitivity of the update to the gradient it was handed
    mean = mean_gradient(state.policy, accepts, weights)
    sensitivity = np.zeros((size, dim))
    for index in range(dim):
        shift = np.zeros(dim)
        shift[index] = epsilon
        plus = np.concatenate(
            _advance(base[:dim], state.moment, state.second, state.policy, accepts, weights,
                     settings, state.step, gradient=mean + shift)
        )
        minus = np.concatenate(
            _advance(base[:dim], state.moment, state.second, state.policy, accepts, weights,
                     settings, state.step, gradient=mean - shift)
        )
        sensitivity[:, index] = (plus - minus) / (2 * epsilon)

    covariance = injected_covariance(state.policy, accepts, weights, n_prompts)
    injection = sensitivity @ covariance @ sensitivity.T
    if not carry_second_moment:
        keep = np.zeros(size)
        keep[: 2 * dim] = 1.0
        transfer = transfer * keep[:, None] * keep[None, :]
        injection = injection * keep[:, None] * keep[None, :]
    return transfer, injection


def advance_state(
    state: LiftedState,
    accepts: np.ndarray,
    weights: np.ndarray,
    settings: AdamSettings,
) -> LiftedState:
    delta, moment, second = _advance(
        np.zeros(state.policy.dim), state.moment, state.second,
        state.policy, accepts, weights, settings, state.step,
    )
    return LiftedState(state.policy.perturbed(delta), moment, second, state.step + 1)


def initial_state(policy: TabularPolicy) -> LiftedState:
    return LiftedState(policy, np.zeros(policy.dim), np.zeros(policy.dim), 0)
