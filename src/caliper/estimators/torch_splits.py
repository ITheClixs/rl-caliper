"""Batched form of the K x 2 split estimator, for a whole population at once.

Mirrors `caliper.estimators.splits.decompose` term for term; a test pins the two together.
"""

from __future__ import annotations

import torch


def decompose_batched(
    cells: torch.Tensor,
    n_prompts: int,
    group_size: int,
    corpus_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """cells: (blocks, 2, population, dim) -> per-member estimates of the noise terms."""
    blocks = cells.shape[0]
    if cells.shape[1] != 2:
        raise ValueError("expected two rollout sub-groups")
    if blocks < 2:
        raise ValueError("need at least two prompt blocks")
    if group_size % 2 or n_prompts % blocks:
        raise ValueError("group size must be even and prompts must divide into blocks")

    per_block = n_prompts // blocks
    subgroup = group_size // 2

    def dot(a, b):
        return (a * b).sum(dim=-1)

    across = torch.stack(
        [dot(cells[a, 0], cells[b, 1]) for a in range(blocks) for b in range(blocks) if a != b]
    ).mean(dim=0)
    within = torch.stack([dot(cells[a, 0], cells[a, 1]) for a in range(blocks)]).mean(dim=0)
    tau_b = per_block * (within - across)

    gap = torch.stack(
        [dot(cells[a, 0] - cells[a, 1], cells[a, 0] - cells[a, 1]) for a in range(blocks)]
    ).mean(dim=0)
    tau_w_sub = 0.5 * per_block * gap
    tau_w_scaled = tau_w_sub * (subgroup - 1) if subgroup > 1 else tau_w_sub
    tau_w = tau_w_scaled / (group_size - 1)

    if corpus_size is not None:
        across = across + tau_b / (corpus_size - 1)

    return {
        "signal": across,
        "tau_b": tau_b,
        "tau_w": tau_w,
        "tau_w_scaled": tau_w_scaled,
    }
