"""Seeds simulated in the tangent model of a policy.

The forecast in `caliper.forecast` linearises the training dynamics about one run. That is exact
while runs stay close to each other and says nothing once they do not: when a run saturates, the
gradient of the reported metric at its endpoint vanishes and so does the forecast, however far
apart the seeds ended.

This module linearises the model instead, once, at reference parameters, and keeps the dynamics.
For every prompt `x` and every response `a` it may give, take the score `s(x, a) = grad log
pi(a | x)` at the reference. The tangent policy is

    pi'(a | x)  proportional to  pi_ref(a | x) exp(s(x, a) . theta'),

an exponential family whose sufficient statistics are the frozen scores. A policy-gradient update
of `theta'` needs the sampled responses, their rewards and those scores, and no pass through the
model, so thousands of seeds cost less than one real update. Sampling, the estimator's weights,
the softmax and the absorption of a group that has gone unanimous are all kept exactly; the one
approximation is that the scores do not move.

If the log-probabilities are affine in the parameters up to normalisation the tangent policy is
the policy and a simulated seed has the law of a real one. A softmax over one token is that case.
Otherwise the error is the drift of the scores along the run, which this module does not measure.

Nothing here knows what a policy is or which library holds it. A caller supplies the reference
log-probabilities, the scores and the rewards of a set of candidate responses per prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TangentModel:
    """A policy linearised at reference parameters, over a candidate set per prompt.

    `base[i, c]` is the reference log-probability of candidate `c` for prompt `i`, with `-inf`
    where a prompt has fewer candidates than the widest. `features[i, c]` is that candidate's
    score at the reference and `rewards[i, c]` whether it passes. Probabilities are renormalised
    over the candidates, so mass the candidates do not cover is treated as unreachable; a caller
    who wants it kept adds one candidate holding it, with a zero score.
    """

    base: np.ndarray  # (n_prompts, n_candidates)
    features: np.ndarray  # (n_prompts, n_candidates, dim)
    rewards: np.ndarray  # (n_prompts, n_candidates)

    def __post_init__(self) -> None:
        if self.features.shape[:2] != self.base.shape or self.rewards.shape != self.base.shape:
            raise ValueError("base, features and rewards disagree on prompts or candidates")

    @property
    def dim(self) -> int:
        return self.features.shape[2]

    def probabilities(self, theta: np.ndarray, prompts: np.ndarray) -> np.ndarray:
        """pi'(. | x) for `theta` of shape (seeds, dim) and `prompts` of shape (seeds, n)."""
        tilt = np.einsum("snck,sk->snc", self.features[prompts], theta)
        return _normalised(self.base[prompts] + tilt)

    def pass_rates(self, theta: np.ndarray, prompts: np.ndarray) -> np.ndarray:
        """Pass probability of each listed prompt under each seed's parameters: (seeds, n).

        The prompts are the same for every seed here, so their scores are not copied per seed.
        """
        tilt = np.einsum("nck,sk->snc", self.features[prompts], theta)
        probs = _normalised(self.base[prompts][None] + tilt)
        return (probs * self.rewards[prompts][None]).sum(axis=2)


def _normalised(logits: np.ndarray) -> np.ndarray:
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class Recipe:
    """What a training run does, as far as the seed is concerned."""

    step_size: float
    n_prompts: int
    group_size: int
    steps: int
    weights: np.ndarray  # the estimator's table w[k, r], as `objectives.weight_table` returns it
    replace: bool = True  # whether a batch may draw the same prompt twice


def update(
    model: TangentModel,
    theta: np.ndarray,
    prompts: np.ndarray,
    outcomes: np.ndarray,
    recipe: Recipe,
) -> np.ndarray:
    """One policy-gradient update of every seed, given what each of them sampled.

    `prompts` is (seeds, P) and `outcomes` (seeds, P, G), indices into the candidates. Sampling is
    left to the caller so that an update can be compared with a real trainer's on the same draws.
    """
    seeds, n_prompts, group = outcomes.shape
    probs = model.probabilities(theta, prompts)
    rewards = np.take_along_axis(model.rewards[prompts], outcomes, axis=2).astype(int)
    counts = rewards.sum(axis=2)
    advantages = recipe.weights[counts[:, :, None], rewards]
    # Sum over the group of advantage times (score of the sample minus the mean score), which is
    # the gradient of log pi' in an exponential family. It is collected per candidate first, so
    # the scores are touched once per prompt and not once per sample.
    load = np.zeros(probs.shape)
    seed_index = np.arange(seeds)[:, None, None]
    prompt_index = np.arange(n_prompts)[None, :, None]
    np.add.at(load, (seed_index, prompt_index, outcomes), advantages)
    load = load - advantages.sum(axis=2, keepdims=True) * probs
    gradient = np.einsum("snc,snck->sk", load, model.features[prompts])
    return theta + recipe.step_size * gradient / (n_prompts * group)


def draw(
    model: TangentModel,
    theta: np.ndarray,
    pool: np.ndarray,
    recipe: Recipe,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """A batch of prompts from `pool` and a group of responses to each, for every seed."""
    seeds = theta.shape[0]
    if recipe.replace:
        picks = rng.integers(0, pool.size, size=(seeds, recipe.n_prompts))
    else:
        if recipe.n_prompts > pool.size:
            raise ValueError("a batch without replacement cannot exceed the pool")
        picks = np.argsort(rng.random((seeds, pool.size)), axis=1)[:, : recipe.n_prompts]
    prompts = pool[picks]
    cumulative = np.cumsum(model.probabilities(theta, prompts), axis=2)
    uniform = rng.random((seeds, recipe.n_prompts, recipe.group_size))
    outcomes = (uniform[:, :, :, None] > cumulative[:, :, None, :]).sum(axis=3)
    # rounding can leave the last cumulative value a hair under one
    return prompts, np.minimum(outcomes, cumulative.shape[2] - 1)


@dataclass(frozen=True)
class SimulatedSeeds:
    scores: np.ndarray  # (seeds,) the reported metric of each simulated run
    pass_rates: np.ndarray  # (seeds, n_eval) per-prompt pass probability at the end of each run

    @property
    def std(self) -> float:
        return float(self.scores.std(ddof=1))


def simulate(
    model: TangentModel,
    recipe: Recipe,
    train_pool: np.ndarray,
    eval_pool: np.ndarray,
    seeds: int,
    rng: np.random.Generator,
    chunk: int = 256,
) -> SimulatedSeeds:
    """Train `seeds` independent runs in the tangent model and report where each ended.

    Runs are independent, so they are advanced `chunk` at a time to bound memory; the result does
    not depend on the chunk size beyond the order random numbers are consumed in.
    """
    if seeds < 2:
        raise ValueError("a spread needs at least two seeds")
    train_pool = np.asarray(train_pool, dtype=int)
    eval_pool = np.asarray(eval_pool, dtype=int)
    rates = []
    for start in range(0, seeds, chunk):
        theta = np.zeros((min(chunk, seeds - start), model.dim))
        for _ in range(recipe.steps):
            prompts, outcomes = draw(model, theta, train_pool, recipe, rng)
            theta = update(model, theta, prompts, outcomes, recipe)
        rates.append(model.pass_rates(theta, eval_pool))
    pass_rates = np.concatenate(rates, axis=0)
    return SimulatedSeeds(scores=pass_rates.mean(axis=1), pass_rates=pass_rates)
