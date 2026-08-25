"""Hierarchical noise estimation from a K x 2 accumulation grid.

A trainer that accumulates its batch gradient over microbatches can index those microbatches by
prompt block and by rollout sub-group, giving

    g[a][b] = mean gradient over prompt block a using rollout sub-group b,
              a = 1..K blocks of P/K prompts,   b = 1, 2 sub-groups of G/2 responses

Blocks are independent, and given a prompt the two sub-groups are independent, so

    signal = mean over a != a' of  g[a][0] . g[a'][1]      (different prompts and responses)
    tau_b  = (P/K) * ( mean over a of g[a][0] . g[a][1]  -  signal )
    tau_w  = (P/K)/2 * mean over a of | g[a][0] - g[a][1] |^2

Each term is an inner product of independent quantities rather than a difference of two large
variances. Raising K averages the block-level statistics over K(K-1) pairs instead of one and
shrinks the P/K multiplier in front of tau_b, so the noisiest term gets cheaper the more
microbatches the trainer already uses. K = 2 recovers the minimal form.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def _quad(x: np.ndarray, matrix: np.ndarray | None) -> float:
    return float(x @ x) if matrix is None else float(x @ matrix @ x)


def _bilinear(x: np.ndarray, y: np.ndarray, matrix: np.ndarray | None) -> float:
    return float(x @ y) if matrix is None else float(x @ matrix @ y)


@dataclass
class SplitEstimates:
    signal: float
    alignment: float
    tau_w: float
    tau_b: float
    tau_w_scaled: float
    tau_total_subtractive: float
    group_size: int
    n_prompts: int

    def tau_total(self) -> float:
        return self.tau_b + self.tau_w

    def critical_batch(self) -> float:
        return self.tau_total() / self.signal

    def efficiency(self) -> float:
        return 1.0 / (1.0 + self.critical_batch() / self.n_prompts)

    def optimal_group_size(self, prefill_ratio: float = 0.0) -> float:
        if self.tau_b <= 0:
            return float("inf")
        return 1.0 + np.sqrt((1.0 + prefill_ratio) * self.tau_w_scaled / self.tau_b)

    def subtractive_group_size(self) -> float:
        """What a trainer would get by differencing total and within-prompt variance instead."""
        tau_b = self.tau_total_subtractive - self.tau_w
        if tau_b <= 0:
            return float("inf")
        return 1.0 + np.sqrt(self.tau_w_scaled / tau_b)


def decompose(
    cells: np.ndarray,
    n_prompts: int,
    group_size: int,
    matrix: np.ndarray | None = None,
    reference_cells: np.ndarray | None = None,
    corpus_size: int | None = None,
) -> SplitEstimates:
    """cells has shape (blocks, 2, dim); each cell is a mean over its own prompts and responses.

    When prompts are drawn without replacement from a corpus of `corpus_size`, two blocks hold
    distinct prompts and are therefore negatively correlated: the different-block inner product has
    expectation `|g_bar|^2 - tau_b / (corpus_size - 1)` rather than `|g_bar|^2`. Passing the corpus
    size applies that correction, which matters whenever tau_b is large relative to the signal --
    that is, whenever the run is far below its critical batch size, which is the usual case.
    """
    blocks = cells.shape[0]
    if cells.shape[1] != 2:
        raise ValueError("expected two rollout sub-groups")
    if blocks < 2:
        raise ValueError("need at least two prompt blocks")
    if group_size % 2 or n_prompts % blocks:
        raise ValueError("group size must be even and prompts must divide into blocks")

    per_block = n_prompts // blocks
    subgroup = group_size // 2

    pairs = list(itertools.permutations(range(blocks), 2))
    across = float(
        np.mean([_bilinear(cells[a, 0], cells[b, 1], matrix) for a, b in pairs])
    )
    within = float(np.mean([_bilinear(cells[a, 0], cells[a, 1], matrix) for a in range(blocks)]))
    tau_b = per_block * (within - across)

    gap = float(np.mean([_quad(cells[a, 0] - cells[a, 1], matrix) for a in range(blocks)]))
    tau_w_sub = 0.5 * per_block * gap
    tau_w_scaled = tau_w_sub * (subgroup - 1) if subgroup > 1 else tau_w_sub
    tau_w = tau_w_scaled / (group_size - 1)

    # the alternative a trainer would reach for: total variance from block-to-block spread
    block_means = cells.mean(axis=1)
    grand = block_means.mean(axis=0)
    spread = float(
        np.mean([_quad(block_means[a] - grand, matrix) for a in range(blocks)])
    ) * blocks / (blocks - 1)
    tau_total_subtractive = spread * per_block

    if corpus_size is not None:
        if corpus_size < 2:
            raise ValueError("corpus_size must be at least 2")
        # Without replacement two blocks hold distinct prompts, so
        #   W - A = N B / (n (N-1)),
        # and the raw n (W - A) overstates the between-prompt term by N / (N - 1). Correct it
        # before it is used to put the signal back, or the correction inherits the same bias.
        tau_b = tau_b * (corpus_size - 1) / corpus_size
        across = across + tau_b / (corpus_size - 1)

    if reference_cells is not None:
        alignment = float(
            np.mean([_bilinear(reference_cells[a, 0], cells[b, 1], matrix) for a, b in pairs])
        )
        if corpus_size is not None:
            alignment = alignment + tau_b / (corpus_size - 1)
    else:
        alignment = across

    return SplitEstimates(
        signal=across,
        alignment=alignment,
        tau_w=tau_w,
        tau_b=tau_b,
        tau_w_scaled=tau_w_scaled,
        tau_total_subtractive=tau_total_subtractive,
        group_size=group_size,
        n_prompts=n_prompts,
    )


def average(estimates: list[SplitEstimates]) -> SplitEstimates:
    """Average raw moments before forming any ratio."""
    if not estimates:
        raise ValueError("no estimates")
    fields = ["signal", "alignment", "tau_w", "tau_b", "tau_w_scaled", "tau_total_subtractive"]
    means = {f: float(np.mean([getattr(e, f) for e in estimates])) for f in fields}
    return SplitEstimates(
        group_size=estimates[0].group_size, n_prompts=estimates[0].n_prompts, **means
    )
