"""Hierarchical noise estimation from four gradient accumulations.

A trainer that accumulates its batch gradient into four buffers instead of one -- prompt half A/B
and rollout sub-group A/B -- can recover the full decomposition of Section 6 of docs/theory.md at
no extra rollout cost and no extra backward passes.

Prompt split:  disjoint halves of the prompts, each using all G rollouts.
Rollout split: all prompts, each group cut into two independent sub-groups whose advantages are
computed within the sub-group. The prompt-level component is common to both and cancels in the
difference, which isolates the within-prompt term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quad(x: np.ndarray, matrix: np.ndarray | None) -> float:
    if matrix is None:
        return float(x @ x)
    return float(x @ matrix @ x)


def _bilinear(x: np.ndarray, y: np.ndarray, matrix: np.ndarray | None) -> float:
    if matrix is None:
        return float(x @ y)
    return float(x @ matrix @ y)


@dataclass
class SplitEstimates:
    signal: float  # g_bar^T M g_bar
    alignment: float  # grad J^T M g_bar (equals signal for an unbiased estimator with M = I)
    tau_total: float  # tr(M (Sigma_b + Sigma_w(G)))
    tau_w: float  # tr(M Sigma_w(G))
    tau_b: float  # tr(M Sigma_b)
    tau_w_scaled: float  # (G - 1) tr(M Sigma_w(G)), the G-independent form
    group_size: int
    n_prompts: int

    def critical_batch(self) -> float:
        return self.tau_total / self.signal

    def efficiency(self) -> float:
        return 1.0 / (1.0 + self.critical_batch() / self.n_prompts)

    def optimal_group_size(self, prefill_ratio: float = 0.0) -> float:
        if self.tau_b <= 0:
            return float("inf")
        return 1.0 + np.sqrt((1.0 + prefill_ratio) * self.tau_w_scaled / self.tau_b)


def decompose(
    prompt_half_a: np.ndarray,
    prompt_half_b: np.ndarray,
    rollout_half_a: np.ndarray,
    rollout_half_b: np.ndarray,
    n_prompts: int,
    group_size: int,
    subgroup_size: int,
    matrix: np.ndarray | None = None,
    reference_half_a: np.ndarray | None = None,
    reference_half_b: np.ndarray | None = None,
    between_second_moment: float | None = None,
) -> SplitEstimates:
    """Single-batch estimates of the hierarchical noise terms.

    Every quantity is unbiased on its own; in use they are averaged over batches before being
    combined, since ratios of noisy estimates are not unbiased.
    """
    if group_size % 2 or subgroup_size * 2 != group_size:
        raise ValueError("the rollout split needs an even group size and two equal sub-groups")

    signal = _bilinear(prompt_half_a, prompt_half_b, matrix)

    diff_p = prompt_half_a - prompt_half_b
    tau_total = 0.25 * n_prompts * _quad(diff_p, matrix)

    diff_r = rollout_half_a - rollout_half_b
    tau_w_sub = 0.5 * n_prompts * _quad(diff_r, matrix)
    # convert from the sub-group size actually used to the full group size
    tau_w_scaled = tau_w_sub * (subgroup_size - 1) if subgroup_size > 1 else tau_w_sub
    tau_w = tau_w_scaled / (group_size - 1)

    # E_i[z_A^T M z_B] over two independent sub-groups of the same prompt estimates
    # tr(M(Sigma_b + g_bar g_bar^T)) with no within-prompt contribution, so subtracting the signal
    # leaves Sigma_b. Falling back to tau_total - tau_w is a difference of two large variances and
    # is badly conditioned when the pool is homogeneous. The direct form assumes lambda does not
    # depend on G, which holds for RLOO and for the mean baseline.
    tau_b = (
        tau_total - tau_w if between_second_moment is None else between_second_moment - signal
    )

    if reference_half_a is not None and reference_half_b is not None:
        alignment = 0.5 * (
            _bilinear(reference_half_a, prompt_half_b, matrix)
            + _bilinear(reference_half_b, prompt_half_a, matrix)
        )
    else:
        alignment = signal

    return SplitEstimates(
        signal=signal,
        alignment=alignment,
        tau_total=tau_total,
        tau_w=tau_w,
        tau_b=tau_b,
        tau_w_scaled=tau_w_scaled,
        group_size=group_size,
        n_prompts=n_prompts,
    )


def average(estimates: list[SplitEstimates]) -> SplitEstimates:
    """Average the raw moment estimates before forming any ratio."""
    if not estimates:
        raise ValueError("no estimates")
    fields = ["signal", "alignment", "tau_total", "tau_w", "tau_b", "tau_w_scaled"]
    means = {f: float(np.mean([getattr(e, f) for e in estimates])) for f in fields}
    return SplitEstimates(
        group_size=estimates[0].group_size,
        n_prompts=estimates[0].n_prompts,
        **means,
    )
