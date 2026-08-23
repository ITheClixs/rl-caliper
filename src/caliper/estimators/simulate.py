"""Sampling harness that drives the crossed split estimator from an enumerable policy.

Also the reference implementation of the accumulation pattern a real trainer follows.
"""

from __future__ import annotations

import numpy as np

from caliper.estimators.splits import SplitEstimates, decompose
from caliper.exact.policy import ExactPrompt
from caliper.objectives.advantages import weight_table


def sample_group(prompt: ExactPrompt, group_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(prompt.probs.shape[0], size=group_size, p=prompt.probs)


def group_gradient(prompt: ExactPrompt, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    rewards = prompt.rewards[idx].astype(int)
    k = int(rewards.sum())
    adv = w[k, rewards]
    return (adv[:, None] * prompt.scores[idx]).sum(axis=0) / idx.shape[0]


def simulate_split_batch(
    prompts: list[ExactPrompt],
    estimator: str,
    n_prompts: int,
    group_size: int,
    rng: np.random.Generator,
    matrix: np.ndarray | None = None,
    reference: str = "rloo",
) -> SplitEstimates:
    """One batch: draw prompts and rollouts, fill the 2x2 cells, decompose."""
    if n_prompts % 2 or group_size % 2:
        raise ValueError("the crossed split needs even prompt and group counts")
    subgroup = group_size // 2
    w_sub = weight_table(estimator, subgroup)
    w_ref = weight_table(reference, subgroup)

    chosen = rng.choice(len(prompts), size=n_prompts, replace=True)
    dim = prompts[0].scores.shape[1]
    cells = np.zeros((2, 2, dim))
    reference_cells = np.zeros((2, 2, dim))

    for slot, prompt_idx in enumerate(chosen):
        prompt = prompts[prompt_idx]
        idx = sample_group(prompt, group_size, rng)
        half = 0 if slot < n_prompts // 2 else 1
        for sub, piece in enumerate((idx[:subgroup], idx[subgroup:])):
            cells[half, sub] += group_gradient(prompt, piece, w_sub)
            reference_cells[half, sub] += group_gradient(prompt, piece, w_ref)

    cells /= n_prompts // 2
    reference_cells /= n_prompts // 2
    return decompose(cells, n_prompts, group_size, matrix, reference_cells)
