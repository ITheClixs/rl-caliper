"""Vectorised RLVR training in the enumerable testbed.

The policy is small enough that the objective and the drift are computed exactly at every step,
so a full training run yields ground truth rather than an estimate of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caliper.control.drift import DriftController
from caliper.exact.policy import TabularPolicy
from caliper.objectives.advantages import weight_table


@dataclass
class RunConfig:
    estimator: str = "rloo"
    n_prompts: int = 32
    group_size: int = 8
    steps: int = 200
    drift_target: float = 1e-4
    initial_step_size: float = 1.0
    controller_gain: float = 0.5
    measure_every: int = 10


def _kl(old: np.ndarray, new: np.ndarray) -> float:
    mask = new > 0
    return float(np.sum(new[mask] * (np.log(new[mask]) - np.log(old[mask]))))


class ExactTrainer:
    """One RLVR run over a fixed pool of accept sets, with exact objective and drift."""

    def __init__(self, policy: TabularPolicy, accepts: np.ndarray, config: RunConfig):
        self.policy = policy
        self.accepts = accepts  # (n_pool, n_sequences) of 0/1
        self.config = config
        self.controller = DriftController(
            target=config.drift_target,
            step_size=config.initial_step_size,
            gain=config.controller_gain,
        )
        self.weights = weight_table(config.estimator, config.group_size)

    def objective(self) -> float:
        """Exact mean pass rate over the prompt pool."""
        return float(np.mean(self.accepts @ self.policy.sequence_probs()))

    def _sample_outcomes(self, probs: np.ndarray, shape, rng: np.random.Generator) -> np.ndarray:
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        return np.searchsorted(cdf, rng.random(shape))

    def step(self, rng: np.random.Generator) -> dict[str, float]:
        cfg = self.config
        probs, scores = self.policy.enumerate()
        pool = self.accepts.shape[0]

        prompt_ids = rng.integers(0, pool, size=cfg.n_prompts)
        outcomes = self._sample_outcomes(probs, (cfg.n_prompts, cfg.group_size), rng)
        rewards = self.accepts[prompt_ids[:, None], outcomes].astype(int)
        k = rewards.sum(axis=1)
        adv = self.weights[k[:, None], rewards]

        contribution = np.einsum("pg,pgd->d", adv, scores[outcomes])
        grad = contribution / (cfg.n_prompts * cfg.group_size)

        before = probs
        self.policy = self.policy.perturbed(self.controller.step_size * grad)
        drift = _kl(before, self.policy.sequence_probs())
        step_size = self.controller.step_size
        self.controller.update(drift)
        return {
            "drift": drift,
            "step_size": step_size,
            "mean_reward": float(rewards.mean()),
        }

    def run(self, rng: np.random.Generator) -> dict[str, list[float]]:
        history = {"objective": [], "drift": [], "step_size": [], "step": []}
        for t in range(self.config.steps):
            info = self.step(rng)
            if t % self.config.measure_every == 0 or t == self.config.steps - 1:
                history["objective"].append(self.objective())
                history["drift"].append(info["drift"])
                history["step_size"].append(info["step_size"])
                history["step"].append(t)
        history["final_objective"] = self.objective()
        return history
