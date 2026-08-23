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
    blocks: int = 2,
    replace: bool = False,
    correct_finite_corpus: bool = True,
) -> SplitEstimates:
    """One batch: draw prompts and rollouts, fill the K x 2 cells, decompose."""
    if group_size % 2 or n_prompts % blocks:
        raise ValueError("group size must be even and prompts must divide into blocks")
    subgroup = group_size // 2
    w_sub = weight_table(estimator, subgroup)
    w_ref = weight_table(reference, subgroup)

    # sampling with replacement puts the same prompt in two blocks, which makes the
    # different-block inner product pick up a same-prompt term and biases tau_b down
    chosen = rng.choice(len(prompts), size=n_prompts, replace=replace)
    dim = prompts[0].scores.shape[1]
    cells = np.zeros((blocks, 2, dim))
    reference_cells = np.zeros((blocks, 2, dim))
    per_block = n_prompts // blocks

    for slot, prompt_idx in enumerate(chosen):
        prompt = prompts[prompt_idx]
        idx = sample_group(prompt, group_size, rng)
        block = slot // per_block
        for sub, piece in enumerate((idx[:subgroup], idx[subgroup:])):
            cells[block, sub] += group_gradient(prompt, piece, w_sub)
            reference_cells[block, sub] += group_gradient(prompt, piece, w_ref)

    cells /= per_block
    reference_cells /= per_block
    corpus = None if replace or not correct_finite_corpus else len(prompts)
    return decompose(cells, n_prompts, group_size, matrix, reference_cells, corpus_size=corpus)
