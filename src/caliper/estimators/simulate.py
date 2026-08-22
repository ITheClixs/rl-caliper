"""Sampling harness that drives the split estimator from an enumerable policy.

Used to check the estimator of `caliper.estimators.splits` against the exact quantities, and as
the reference implementation of the four-buffer accumulation pattern a real trainer follows.
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
    """One batch: draw prompts, draw rollouts, fill the four buffers, decompose."""
    if n_prompts % 2:
        raise ValueError("prompt split needs an even number of prompts")
    subgroup = group_size // 2
    w_full = weight_table(estimator, group_size)
    w_sub = weight_table(estimator, subgroup)
    w_ref = weight_table(reference, group_size)

    chosen = rng.choice(len(prompts), size=n_prompts, replace=True)
    dim = prompts[0].scores.shape[1]
    buffers = {k: np.zeros(dim) for k in ("pa", "pb", "ra", "rb", "refa", "refb")}

    for slot, prompt_idx in enumerate(chosen):
        prompt = prompts[prompt_idx]
        idx = sample_group(prompt, group_size, rng)
        full = group_gradient(prompt, idx, w_full)
        ref = group_gradient(prompt, idx, w_ref)
        half = "pa" if slot < n_prompts // 2 else "pb"
        buffers[half] += full
        buffers["refa" if slot < n_prompts // 2 else "refb"] += ref
        buffers["ra"] += group_gradient(prompt, idx[:subgroup], w_sub)
        buffers["rb"] += group_gradient(prompt, idx[subgroup:], w_sub)

    half_n = n_prompts // 2
    return decompose(
        prompt_half_a=buffers["pa"] / half_n,
        prompt_half_b=buffers["pb"] / half_n,
        rollout_half_a=buffers["ra"] / n_prompts,
        rollout_half_b=buffers["rb"] / n_prompts,
        n_prompts=n_prompts,
        group_size=group_size,
        subgroup_size=subgroup,
        matrix=matrix,
        reference_half_a=buffers["refa"] / half_n,
        reference_half_b=buffers["refb"] / half_n,
    )
