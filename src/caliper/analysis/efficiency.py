"""The efficiency of a batch split, and the group size that maximises it.

One implementation of equation (13) of `docs/theory.md`, shared by every experiment that
predicts a group-size curve, so the prediction cannot drift between them.
"""

from __future__ import annotations

import numpy as np


def critical_batch(
    group_sizes: np.ndarray,
    tau_b: float,
    tau_w_scaled: float,
    signal: float,
    rollouts: float,
    corpus: int | None = None,
) -> np.ndarray:
    """B_crit as a function of group size, in prompts.

    `tau_w_scaled` carries the (G-1) convention of the leave-one-out estimators, so it is
    divided by G-1 rather than by G. `corpus` applies the finite-pool correction (7b): drawing
    P of N prompts without replacement removes a (P-1)/(N-1) share of the between-prompt term.
    """
    g = np.asarray(group_sizes, dtype=float)
    prompts = rollouts / g
    share = 0.0 if corpus is None else (prompts - 1.0) / (corpus - 1.0)
    return ((1.0 - share) * tau_b + tau_w_scaled / (g - 1.0)) / signal


def efficiency(
    group_sizes: np.ndarray,
    tau_b: float,
    tau_w_scaled: float,
    signal: float,
    rollouts: float,
    corpus: int | None = None,
) -> np.ndarray:
    """rho(G) = 1 / (1 + B_crit(G) / P) at a fixed rollout budget. No fitted parameters."""
    g = np.asarray(group_sizes, dtype=float)
    bcrit = critical_batch(g, tau_b, tau_w_scaled, signal, rollouts, corpus)
    return 1.0 / (1.0 + g * bcrit / rollouts)


def optimal_group_size(
    tau_b: float,
    tau_w_scaled: float,
    rollouts: float | None = None,
    corpus: int | None = None,
) -> float:
    """The group size that minimises the noise at a fixed rollout budget, ignoring prefill cost.

    With an unbounded corpus this is `1 + sqrt(tau_w / tau_b)`.

    With a corpus of `corpus` prompts drawn without replacement it is not that expression with
    the finite-population factor substituted in, because the budget fixes `R = P G` and so `P`
    moves with `G`: the factor is a function of the quantity being optimised. Minimising
    `G B_crit(G)` at fixed `R` gives

        d/dG [ (G N - R) tau_b / (N - 1) + G tau_w / (G - 1) ] = 0,

    hence `G* = 1 + sqrt((N - 1) tau_w / (N tau_b))`, subject to `P <= N`, that is `G >= R / N`.
    The correction is small, and it lowers the optimum rather than raising it.
    """
    if tau_b <= 0.0:
        return float("inf")
    ratio = max(tau_w_scaled, 0.0) / tau_b
    if corpus is None:
        return 1.0 + np.sqrt(ratio)
    if corpus < 2:
        raise ValueError("corpus must be at least 2")
    interior = 1.0 + np.sqrt(ratio * (corpus - 1.0) / corpus)
    if rollouts is None:
        return float(interior)
    return float(max(rollouts / corpus, interior))
