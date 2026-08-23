"""Hierarchical noise estimation from a 2x2 crossing of the batch.

A trainer that already accumulates its batch gradient over microbatches can partition those
microbatches by prompt half and by rollout sub-group, giving four gradients

    g[a][b] = mean gradient over prompt half a using rollout sub-group b

Every term of the decomposition is then an inner product of two *independent* quantities, which is
far better conditioned than differencing two large variances:

    signal  = E[ g[0][0] . g[1][1] ]                      (different prompts, different rollouts)
    tau_b   = P/2 * ( E[ g[a][0] . g[a][1] ] - signal )   (same prompts, different rollouts)
    tau_w   = P/4 * E[ | g[a][0] - g[a][1] |^2 ]          (the prompt term cancels here)

The cost is four gradient buffers instead of one, and no extra rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quad(x: np.ndarray, matrix: np.ndarray | None) -> float:
    return float(x @ x) if matrix is None else float(x @ matrix @ x)


def _bilinear(x: np.ndarray, y: np.ndarray, matrix: np.ndarray | None) -> float:
    return float(x @ y) if matrix is None else float(x @ matrix @ y)


@dataclass
class SplitEstimates:
    signal: float  # g_bar^T M g_bar
    alignment: float  # grad J^T M g_bar
    tau_w: float  # tr(M Sigma_w(G))
    tau_b: float  # tr(M Sigma_b)
    tau_w_scaled: float  # (G - 1) tr(M Sigma_w(G)), the G-independent form
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


def decompose(
    cells: np.ndarray,
    n_prompts: int,
    group_size: int,
    matrix: np.ndarray | None = None,
    reference_cells: np.ndarray | None = None,
) -> SplitEstimates:
    """cells has shape (2, 2, dim): prompt half by rollout sub-group.

    Each cell is the mean gradient over its own P/2 prompts and G/2 rollouts, computed with the
    sub-group's own advantages. Every estimate is unbiased on its own; ratios are formed only after
    the raw moments have been averaged over batches.
    """
    if cells.shape[:2] != (2, 2):
        raise ValueError("expected a 2x2 grid of gradient cells")
    if group_size % 2 or n_prompts % 2:
        raise ValueError("the crossed split needs an even group size and an even prompt count")

    half_prompts = n_prompts // 2
    subgroup = group_size // 2

    across = 0.5 * (
        _bilinear(cells[0, 0], cells[1, 1], matrix) + _bilinear(cells[0, 1], cells[1, 0], matrix)
    )
    within_prompts = 0.5 * (
        _bilinear(cells[0, 0], cells[0, 1], matrix) + _bilinear(cells[1, 0], cells[1, 1], matrix)
    )
    tau_b = half_prompts * (within_prompts - across)

    rollout_gap = 0.5 * (
        _quad(cells[0, 0] - cells[0, 1], matrix) + _quad(cells[1, 0] - cells[1, 1], matrix)
    )
    tau_w_sub = 0.5 * half_prompts * rollout_gap
    tau_w_scaled = tau_w_sub * (subgroup - 1) if subgroup > 1 else tau_w_sub
    tau_w = tau_w_scaled / (group_size - 1)

    if reference_cells is not None:
        alignment = 0.5 * (
            _bilinear(reference_cells[0, 0], cells[1, 1], matrix)
            + _bilinear(reference_cells[1, 1], cells[0, 0], matrix)
        )
    else:
        alignment = across

    return SplitEstimates(
        signal=across,
        alignment=alignment,
        tau_w=tau_w,
        tau_b=tau_b,
        tau_w_scaled=tau_w_scaled,
        group_size=group_size,
        n_prompts=n_prompts,
    )


def average(estimates: list[SplitEstimates]) -> SplitEstimates:
    """Average the raw moments before forming any ratio."""
    if not estimates:
        raise ValueError("no estimates")
    fields = ["signal", "alignment", "tau_w", "tau_b", "tau_w_scaled"]
    means = {f: float(np.mean([getattr(e, f) for e in estimates])) for f in fields}
    return SplitEstimates(
        group_size=estimates[0].group_size, n_prompts=estimates[0].n_prompts, **means
    )
