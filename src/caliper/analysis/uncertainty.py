"""Interval estimation that respects how the measurements are actually correlated.

Two places in this work invite the same mistake. Pairwise divergences between N runs look like
N(N-1)/2 observations but are built from N objects, so a standard error computed as though the
pairs were independent is too small. And a scaling exponent fitted over a handful of settings has
uncertainty that a point estimate and an R^2 do not convey.
"""

from __future__ import annotations

import numpy as np


def pairwise_mean_interval_from_matrix(
    matrix: np.ndarray,
    n_boot: int = 4000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Interval for a mean pairwise statistic given the full matrix of pair values.

    Resampling is over runs. Pairs that share a run move together, which is the dependence a
    per-pair standard error ignores. Resampled indices that collide contribute no pair, matching
    the fact that a run carries no information about its distance from itself.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if n < 3:
        raise ValueError("need at least three runs to bootstrap at the run level")
    upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
    point = float(np.mean([matrix[i, j] for i, j in upper]))

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        values = [
            matrix[pick[i], pick[j]]
            for i in range(n)
            for j in range(i + 1, n)
            if pick[i] != pick[j]
        ]
        if values:
            draws.append(float(np.mean(values)))
    lo, hi = np.percentile(draws, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    naive = float(
        np.std([matrix[i, j] for i, j in upper], ddof=1) / np.sqrt(len(upper))
    )
    return {
        "mean": point,
        "lo": float(lo),
        "hi": float(hi),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "naive_se_over_pairs": naive,
        "n_runs": n,
    }


def pairwise_mean_interval(
    per_run: list[np.ndarray] | np.ndarray,
    statistic,
    n_boot: int = 4000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Bootstrap a mean pairwise statistic by resampling runs, not pairs.

    `per_run[i]` describes run i; `statistic(a, b)` is the quantity computed for a pair. Resampling
    at the level of runs keeps the dependence between pairs that share a run, which resampling the
    pairs themselves would destroy.
    """
    runs = list(per_run)
    n = len(runs)
    if n < 3:
        raise ValueError("need at least three runs to bootstrap at the run level")

    def mean_over_pairs(index: np.ndarray) -> float:
        values = [
            statistic(runs[index[i]], runs[index[j]])
            for i in range(len(index))
            for j in range(i + 1, len(index))
            if index[i] != index[j]
        ]
        return float(np.mean(values)) if values else float("nan")

    point = mean_over_pairs(np.arange(n))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        value = mean_over_pairs(rng.integers(0, n, size=n))
        if np.isfinite(value):
            draws.append(value)
    lo, hi = np.percentile(draws, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    naive = np.std(
        [statistic(runs[i], runs[j]) for i in range(n) for j in range(i + 1, n)], ddof=1
    ) / np.sqrt(n * (n - 1) / 2)
    return {
        "mean": point,
        "lo": float(lo),
        "hi": float(hi),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "naive_se_over_pairs": float(naive),
        "n_runs": n,
    }


def spread_interval(
    scores: np.ndarray, n_boot: int = 4000, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Across-seed standard deviation of a reported number, with a bootstrap interval.

    Resampling is over runs, which are the independent units; the interval is wide at the seed
    counts anyone actually trains, and saying so is the point of reporting it.
    """
    values = np.asarray(scores, dtype=float)
    if values.size < 2:
        raise ValueError("need at least two runs to speak of a spread")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, values.size, size=values.size)
        draws[i] = values[pick].std(ddof=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.percentile(draws, [100 * tail, 100 * (1.0 - tail)])
    return {
        "std": float(values.std(ddof=1)),
        "lo": float(lo),
        "hi": float(hi),
        "mean": float(values.mean()),
        "n_runs": int(values.size),
    }


def loglog_fit(
    predictors: dict[str, np.ndarray],
    response: np.ndarray,
    n_boot: int = 4000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Fit log(response) against log of each predictor, with bootstrap intervals on the exponents.

    Returns one entry per predictor holding the exponent, its interval, and the residual fit
    quality. With few settings the intervals are wide, which is the point of reporting them.
    """
    names = list(predictors)
    design = np.column_stack(
        [np.log(np.asarray(predictors[k], dtype=float)) for k in names]
        + [np.ones(len(response))]
    )
    target = np.log(np.asarray(response, dtype=float))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    r2 = 1.0 - (residual**2).sum() / ((target - target.mean()) ** 2).sum()

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(target), size=len(target))
        if np.linalg.matrix_rank(design[pick]) < design.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(design[pick], target[pick], rcond=None)
        draws.append(beta)
    draws = np.array(draws)

    out = {}
    for index, name in enumerate(names):
        lo, hi = np.percentile(
            draws[:, index], [100 * (1 - level) / 2, 100 * (1 + level) / 2]
        )
        out[name] = {
            "exponent": float(coefficients[index]),
            "lo": float(lo),
            "hi": float(hi),
        }
    out["_fit"] = {"r2": float(r2), "n": int(len(target))}
    return out


def compatible(interval: dict[str, float], value: float) -> bool:
    """Whether a hypothesised exponent lies inside the estimated interval."""
    return interval["lo"] <= value <= interval["hi"]
