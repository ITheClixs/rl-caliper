"""Exact finite-G moments of count-based group estimators under binary rewards.

Implements Results 1 and 2 of docs/theory.md. Given a weight table w[k, r] and a per-prompt pass
rate p, all first and second moments of the per-prompt estimator reduce to binomial contractions
of that table.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def _binom_pmf(n: int, p: float) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    return binom.pmf(np.arange(n + 1), n, p)


def first_moments(w: np.ndarray, p: float) -> tuple[float, float]:
    """W_0, W_1 = E_m[w(r + m, r)] with m ~ Bin(G - 1, p)."""
    group_size = w.shape[0] - 1
    pmf = _binom_pmf(group_size - 1, p)
    m = np.arange(group_size)
    return float(pmf @ w[m, 0]), float(pmf @ w[m + 1, 1])


def second_moments(w: np.ndarray, p: float) -> tuple[float, float]:
    """Q_0, Q_1 = E_m[w(r + m, r)^2]."""
    group_size = w.shape[0] - 1
    pmf = _binom_pmf(group_size - 1, p)
    m = np.arange(group_size)
    return float(pmf @ w[m, 0] ** 2), float(pmf @ w[m + 1, 1] ** 2)


def cross_moments(w: np.ndarray, p: float) -> tuple[float, float, float]:
    """C_00, C_10, C_11 = E_m'[w(a + b + m', a) w(a + b + m', b)], m' ~ Bin(G - 2, p)."""
    group_size = w.shape[0] - 1
    if group_size < 2:
        return 0.0, 0.0, 0.0
    pmf = _binom_pmf(group_size - 2, p)
    m = np.arange(group_size - 1)
    c00 = float(pmf @ (w[m, 0] * w[m, 0]))
    c10 = float(pmf @ (w[m + 1, 0] * w[m + 1, 1]))
    c11 = float(pmf @ (w[m + 2, 1] * w[m + 2, 1]))
    return c00, c10, c11


def difficulty_weight(w: np.ndarray, p: float) -> float:
    """lambda(p, G) = W_1 - W_0.  E[z | x] = lambda * h(x)."""
    w0, w1 = first_moments(w, p)
    return w1 - w0


def covariance(
    w: np.ndarray,
    p: float,
    s0: np.ndarray,
    s1: np.ndarray,
    u1: np.ndarray,
) -> np.ndarray:
    """Exact Cov(z | x), equation (6) of docs/theory.md.

    s0, s1 are the score covariances conditional on reward 0 and 1; u1 is the conditional mean
    score given reward 1.
    """
    group_size = w.shape[0] - 1
    q0, q1 = second_moments(w, p)
    c00, c10, c11 = cross_moments(w, p)
    lam = difficulty_weight(w, p)

    cov = (p * q1 * s1 + (1.0 - p) * q0 * s0) / group_size
    coef = (p * q1 + p**2 * q0 / (1.0 - p)) / group_size
    coef += p**2 * (((group_size - 1) / group_size) * (c11 - 2 * c10 + c00) - lam**2)
    return cov + coef * np.outer(u1, u1)


def projected_covariance(
    w: np.ndarray,
    p: float,
    s0: np.ndarray,
    s1: np.ndarray,
    u1: np.ndarray,
    direction: np.ndarray,
) -> float:
    """b^T Cov(z | x) b, the same expression contracted against one direction.

    The outer-product term collapses to a scalar and the conditional score covariances enter
    only through their quadratic forms, so nothing of size dim x dim is needed.
    """
    group_size = w.shape[0] - 1
    q0, q1 = second_moments(w, p)
    c00, c10, c11 = cross_moments(w, p)
    lam = difficulty_weight(w, p)

    quad = (p * q1 * (direction @ s1 @ direction) + (1.0 - p) * q0 * (direction @ s0 @ direction))
    coef = (p * q1 + p**2 * q0 / (1.0 - p)) / group_size
    coef += p**2 * (((group_size - 1) / group_size) * (c11 - 2 * c10 + c00) - lam**2)
    return float(quad / group_size + coef * (direction @ u1) ** 2)


def mean(w: np.ndarray, p: float, u1: np.ndarray) -> np.ndarray:
    """Exact E[z | x] = lambda(p, G) * p * u1."""
    return difficulty_weight(w, p) * p * u1
