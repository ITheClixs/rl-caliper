"""Seed reconvergence on the from-scratch transformers (`s1_contraction`): the quantities the paper
quotes about whether divergence settles and how its level scales.

These used to live in the figure script, and the fits the text quotes were made by hand from it.
Both the figure and the paper's macros now read them from here.
"""

from __future__ import annotations

import numpy as np

from caliper.runtime import io

# updates past this pass rate, or past three times the drift target, have stopped learning
SATURATED = 0.85
MATCHED_PASS_RATE = 0.60


def contraction_cells() -> list[tuple[dict, np.ndarray]]:
    """Every stored contraction setting, with the mask of updates that are still learning.

    Runs that measured spread without the noise correction (no `true_spread`) are skipped, and so
    is any setting with fewer than four learning updates.
    """
    cells = []
    for run in io.load_all("s1_contraction"):
        for cell in run["result"]["cells"]:
            history = cell["history"]
            if "true_spread" not in history:
                continue
            pass_rate = np.array(history["pass_mean"])
            drift = np.array(history["drift"])
            keep = (pass_rate < SATURATED) & (drift < 3 * cell["drift_target"])
            if keep.sum() >= 4:
                cells.append((cell, keep))
    if not cells:
        raise SystemExit("no contraction runs")
    return cells


def matched_progress(cells, target: float = MATCHED_PASS_RATE) -> list[dict]:
    """One summary per setting: divergence and outcome spread at a common pass rate.

    Trajectory points within a setting cannot be pooled -- the divergence is already at its
    stationary value and what varies along a trajectory is measurement noise, not signal. Settings
    also differ in their noise scale, so they are compared where the policies are equally good.
    """
    rows = []
    for cell, keep in cells:
        pass_rate = np.array(cell["history"]["pass_mean"])[keep]
        kl = np.array(cell["history"]["pairwise_kl"])[keep]
        spread = np.array(cell["history"]["true_spread"])[keep]
        if not (pass_rate.min() <= target <= pass_rate.max()):
            continue
        rows.append({
            "prompts": cell["prompts"],
            "drift": cell["drift_target"],
            "critical_batch": cell.get("critical_batch", float("nan")),
            "kl": float(np.interp(target, pass_rate, kl)),
            "spread": float(np.interp(target, pass_rate, spread)),
        })
    return rows


def tail_slopes(cells) -> np.ndarray:
    """Log-log slope of divergence against update count over the second half of each setting's
    learning updates. Accumulation would give +1."""
    slopes = []
    for cell, keep in cells:
        steps = np.array(cell["history"]["step"], dtype=float)
        kl = np.array(cell["history"]["pairwise_kl"])
        idx = np.where(keep & (steps > 0))[0]
        idx = idx[len(idx) // 2:]
        slopes.append(np.polyfit(np.log(steps[idx]), np.log(kl[idx]), 1)[0])
    return np.array(slopes)


def _bootstrap(design: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Coefficients refitted on settings resampled with replacement; degenerate draws dropped."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if np.linalg.matrix_rank(design[i]) == design.shape[1]:
            draws.append(np.linalg.lstsq(design[i], y[i], rcond=None)[0])
    return np.array(draws)


def _fit(design: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    r2 = float(1 - resid @ resid / ((y - y.mean()) ** 2).sum())
    lo, hi = np.percentile(_bootstrap(design, y, n_boot, seed), [2.5, 97.5], axis=0)
    return {"beta": beta, "lo": lo, "hi": hi, "r2": r2, "n": len(y)}


def level_fit(rows, n_boot: int = 4000, seed: int = 0) -> dict:
    """log divergence on log P and log drift target, with 95% intervals over settings.

    `Q_t ~ eta^2 Sigma / P` predicts exponents of -1/2 and +1/2.
    """
    design = np.column_stack([
        np.ones(len(rows)),
        np.log([r["prompts"] for r in rows]),
        np.log([r["drift"] for r in rows]),
    ])
    return _fit(design, np.log([r["kl"] for r in rows]), n_boot, seed)


def spread_fit(rows, n_boot: int = 4000, seed: int = 0) -> dict:
    """log outcome spread on log divergence. The withdrawn prediction was an exponent of 1/2."""
    rows = [r for r in rows if r["kl"] > 0 and r["spread"] > 0]
    design = np.column_stack([np.ones(len(rows)), np.log([r["kl"] for r in rows])])
    return _fit(design, np.log([r["spread"] for r in rows]), n_boot, seed)
