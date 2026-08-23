"""Construction of prompt pools with controlled difficulty and controlled diversity.

Difficulty sets the pass-rate distribution and therefore the within-prompt noise; diversity sets
how much the per-prompt gradients differ and therefore the between-prompt noise. Together they move
the ratio that determines the optimal group size, which is what makes the prediction falsifiable.
"""

from __future__ import annotations

import numpy as np

from caliper.exact.policy import TabularPolicy, accept_set_for_pass_rate


def accept_pool(
    probs: np.ndarray,
    pool: int,
    difficulty: tuple[float, float],
    diversity: float,
    rng: np.random.Generator,
    min_rate: float = 0.02,
    max_rate: float = 0.98,
) -> np.ndarray:
    """Accept sets sharing a common core, resampled with probability `diversity`.

    diversity = 1 gives independent accept sets (large between-prompt noise); diversity -> 0 gives
    a pool of near-identical prompts (small between-prompt noise).
    """
    lo, hi = difficulty
    base = accept_set_for_pass_rate(probs, rng.uniform(lo, hi), rng)
    out = []
    while len(out) < pool:
        candidate = accept_set_for_pass_rate(probs, rng.uniform(lo, hi), rng)
        keep = rng.random(probs.shape[0]) < diversity
        accept = np.where(keep, candidate, base)
        rate = float(probs @ accept)
        if min_rate < rate < max_rate:
            out.append(accept)
    return np.array(out)


def build(
    vocab: int,
    length: int,
    pool: int,
    difficulty: tuple[float, float],
    diversity: float,
    seed: int,
    logit_scale: float = 0.8,
) -> tuple[TabularPolicy, np.ndarray]:
    rng = np.random.default_rng(seed)
    policy = TabularPolicy.random(vocab, length, rng, scale=logit_scale)
    probs = policy.sequence_probs()
    return policy, accept_pool(probs, pool, difficulty, diversity, rng)
