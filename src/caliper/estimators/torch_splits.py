"""Batched form of the crossed split estimator, for a whole population at once.

Mirrors `caliper.estimators.splits.decompose` term for term; a test pins the two together.
"""

from __future__ import annotations

import torch


def decompose_batched(
    cells: torch.Tensor,
    n_prompts: int,
    group_size: int,
) -> dict[str, torch.Tensor]:
    """cells: (2, 2, population, dim) -> per-member estimates of the noise terms."""
    if cells.shape[:2] != (2, 2):
        raise ValueError("expected a 2x2 grid of gradient cells")
    if group_size % 2 or n_prompts % 2:
        raise ValueError("the crossed split needs an even group size and an even prompt count")

    half_prompts = n_prompts // 2
    subgroup = group_size // 2

    def dot(a, b):
        return (a * b).sum(dim=-1)

    across = 0.5 * (dot(cells[0, 0], cells[1, 1]) + dot(cells[0, 1], cells[1, 0]))
    within = 0.5 * (dot(cells[0, 0], cells[0, 1]) + dot(cells[1, 0], cells[1, 1]))
    tau_b = half_prompts * (within - across)

    gap = 0.5 * (
        dot(cells[0, 0] - cells[0, 1], cells[0, 0] - cells[0, 1])
        + dot(cells[1, 0] - cells[1, 1], cells[1, 0] - cells[1, 1])
    )
    tau_w_sub = 0.5 * half_prompts * gap
    tau_w_scaled = tau_w_sub * (subgroup - 1) if subgroup > 1 else tau_w_sub
    tau_w = tau_w_scaled / (group_size - 1)

    return {
        "signal": across,
        "tau_b": tau_b,
        "tau_w": tau_w,
        "tau_w_scaled": tau_w_scaled,
    }
