"""Stochastic support: how much of a corpus can carry gradient noise at all.

A count-based advantage is identically zero on a unanimous group, so a prompt contributes to the
seed noise only when its group contains both a success and a failure. For a prompt whose policy
succeeds with probability p, and a group of G samples,

    live(p, G) = 1 - p^G - (1-p)^G

is the probability of that happening. It is the natural coordinate for choosing a corpus, and it
is a property of the policy and the prompt together rather than of the task: a prompt at p = 0.5
carries almost every group, one at p = 0.99 carries almost none, and which prompts sit where
changes as the model changes.

The batch size a run reports is therefore not the batch size its uncertainty estimator gets. With
mean live probability `lbar`, the informative count is roughly

    P_eff = P * lbar

which is what makes a nominal thirty-two prompts behave like three when support has collapsed.
"""

from __future__ import annotations

import numpy as np


def live_probability(pass_rate, group_size: int):
    """`1 - p^G - (1-p)^G`, the chance a group of G is not unanimous."""
    p = np.clip(np.asarray(pass_rate, dtype=float), 0.0, 1.0)
    return 1.0 - p**group_size - (1.0 - p) ** group_size


def effective_prompts(pass_rates, group_size: int, prompts: int) -> float:
    """`P * mean(live)`, the number of prompts per batch that can carry noise."""
    return float(prompts * np.mean(live_probability(pass_rates, group_size)))


def support_profile(pass_rates, group_size: int) -> dict[str, float]:
    """The summary a corpus is matched on across models."""
    live = live_probability(pass_rates, group_size)
    rates = np.asarray(pass_rates, dtype=float)
    return {
        "mean_live": float(np.mean(live)),
        "median_live": float(np.median(live)),
        "share_live_above_half": float(np.mean(live > 0.5)),
        "mean_pass": float(np.mean(rates)),
        "mean_reward_variance": float(np.mean(rates * (1.0 - rates))),
        "n": int(rates.size),
    }
