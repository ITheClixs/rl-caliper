"""Brute-force moments of the per-prompt estimator by enumerating all group outcomes.

Used only to validate `caliper.exact.moments`; the cost is |Y|^G.
"""

from __future__ import annotations

import itertools

import numpy as np


def enumerate_group_moments(
    w: np.ndarray,
    probs: np.ndarray,
    rewards: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact (E[z], Cov(z)) by summing over every group of G outcomes.

    probs, rewards index the |Y| distinct outcomes; scores has shape (|Y|, d).
    """
    group_size = w.shape[0] - 1
    n_out = probs.shape[0]
    dim = scores.shape[1]

    mean = np.zeros(dim)
    second = np.zeros((dim, dim))
    for combo in itertools.product(range(n_out), repeat=group_size):
        idx = np.asarray(combo)
        prob = float(np.prod(probs[idx]))
        if prob == 0.0:
            continue
        r = rewards[idx].astype(int)
        k = int(r.sum())
        adv = w[k, r]
        z = (adv[:, None] * scores[idx]).sum(axis=0) / group_size
        mean += prob * z
        second += prob * np.outer(z, z)
    return mean, second - np.outer(mean, mean)
