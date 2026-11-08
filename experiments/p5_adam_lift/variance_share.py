"""P5, variance share: how much of Adam's mean second moment the gradient variance supplies.

The mean of Adam's second-moment update is b2 v + (1 - b2) (g_bar^2 + diag Cov(g_hat)). An
earlier version of the lift squared the mean gradient and dropped the variance. This measures
what that dropped, on the same grid as `run.py`: along each setting's mean trajectory (warm-up
included), the ratio of the summed gradient variance to the summed squared mean gradient, and its
median over updates. No sampling is involved; every term is exact on the enumerable tier.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.exact.lifted import AdamSettings, advance_state, initial_state
from caliper.exact.pool import build
from caliper.exact.propagation import injected_covariance, mean_gradient
from caliper.objectives.advantages import weight_table
from caliper.runtime import io

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--prompts", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--step-size", type=float, nargs="+", default=[0.03, 0.1])
    ap.add_argument("--bands", nargs="+", default=list(BANDS))
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weights = weight_table(args.estimator, args.group_size)
    cells = []
    for band, steps, prompts, eta in itertools.product(
        args.bands, args.steps, args.prompts, args.step_size
    ):
        policy, accepts = build(args.vocab, args.length, args.pool, BANDS[band], 1.0, args.seed)
        settings = AdamSettings(step_size=eta)
        state = initial_state(policy)
        ratios = []
        for _ in range(args.warmup + steps):
            grad = mean_gradient(state.policy, accepts, weights)
            variance = np.diag(injected_covariance(state.policy, accepts, weights, prompts))
            ratios.append(float(variance.sum() / (grad**2).sum()))
            state = advance_state(state, accepts, weights, settings, prompts)
        cells.append({"band": band, "steps": steps, "prompts": prompts, "step_size": eta,
                      "median_ratio": float(np.median(ratios)),
                      "min_ratio": min(ratios), "max_ratio": max(ratios)})
        print(f"{band:>6s} T={steps:3d} P={prompts:3d} eta={eta:.2f}: variance over squared mean "
              f"{cells[-1]['median_ratio']:6.2f} (median over updates)")
    medians = np.array([c["median_ratio"] for c in cells])
    print(f"per-setting medians span {medians.min():.1f} to {medians.max():.1f}")
    path = io.save("p5_variance_share", vars(args), {"cells": cells})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
