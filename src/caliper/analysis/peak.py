"""Locating the peak of a noisy sweep, with a bootstrap interval over seeds."""

from __future__ import annotations

import numpy as np


def quadratic_peak(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Vertex of a weighted quadratic fit; nan if the fit is not concave."""
    w = np.ones_like(x) if weights is None else weights
    coef = np.polyfit(x, y, 2, w=np.sqrt(w))
    if coef[0] >= 0:
        return float("nan")
    return float(-coef[1] / (2 * coef[0]))


def bootstrap_peak(
    grid: np.ndarray,
    samples: list[np.ndarray],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Peak of mean(samples) over log(grid), with a percentile interval over seed resamples.

    `samples[i]` holds the per-seed outcomes at grid point i.
    """
    rng = np.random.default_rng(seed)
    logx = np.log(np.asarray(grid, dtype=float))
    point = quadratic_peak(logx, np.array([s.mean() for s in samples]))
    draws = []
    n_seeds = min(len(s) for s in samples)
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        means = np.array([np.asarray(s)[idx].mean() for s in samples])
        peak = quadratic_peak(logx, means)
        if np.isfinite(peak):
            draws.append(peak)
    if not draws:
        return float(np.exp(point)), float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.exp(point)), float(np.exp(lo)), float(np.exp(hi))


def efficiency_curve(group_sizes: np.ndarray, alpha: float, ratio: float) -> np.ndarray:
    """rho(G) for a fixed rollouts-per-step budget, equation (13).

    alpha absorbs tau_b / (R * Gcal); ratio is tau_w / tau_b.
    """
    g = np.asarray(group_sizes, dtype=float)
    return 1.0 / (1.0 + alpha * g * (1.0 + ratio / (g - 1.0)))


def fit_ratio(
    group_sizes: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None = None,
    starts: list[list[float]] | None = None,
) -> dict[str, float]:
    """Fit y = c0 + c1 sqrt(rho(G)) and report the implied optimal group size.

    Using the shape the theory predicts, rather than a generic parabola, keeps the fit stable in
    the flat region around the optimum where the curve carries little curvature.
    """
    from scipy.optimize import least_squares

    g = np.asarray(group_sizes, dtype=float)
    y = np.asarray(values, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)

    def residual(theta):
        c0, c1, log_alpha, log_ratio = theta
        rho = efficiency_curve(g, np.exp(log_alpha), np.exp(log_ratio))
        return (c0 + c1 * np.sqrt(rho) - y) * np.sqrt(w)

    if starts is None:
        starts = [
            [y.min(), max(float(np.ptp(y)), 1e-6), a, r]
            for a in (-6.0, -3.0, 0.0)
            for r in (0.0, 2.0, 4.0)
        ]
    best = None
    for start in starts:
            try:
                fit = least_squares(
                    residual,
                    start,
                    bounds=([-np.inf, 0.0, -20.0, -5.0], [np.inf, np.inf, 5.0, 12.0]),
                    max_nfev=4000,
                )
            except ValueError:
                continue
            if best is None or fit.cost < best.cost:
                best = fit
    if best is None:
        return {"ratio": float("nan"), "g_star": float("nan"), "cost": float("nan")}
    ratio = float(np.exp(best.x[3]))
    return {
        "ratio": ratio,
        "g_star": 1.0 + np.sqrt(ratio),
        "alpha": float(np.exp(best.x[2])),
        "cost": float(best.cost),
        "theta": [float(v) for v in best.x],
    }


def bootstrap_fit(
    grid: np.ndarray,
    samples: list[np.ndarray],
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Shape-constrained peak with a percentile interval over seed resamples."""
    rng = np.random.default_rng(seed)
    means = np.array([np.asarray(s).mean() for s in samples])
    reference = fit_ratio(grid, means)
    point = reference["g_star"]
    warm = [reference["theta"]] if "theta" in reference else None
    n_seeds = min(len(s) for s in samples)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        resampled = np.array([np.asarray(s)[idx].mean() for s in samples])
        value = fit_ratio(grid, resampled, starts=warm)["g_star"]
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)
