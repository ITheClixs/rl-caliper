"""What size of seed spread the real-model design can actually resolve.

The reported spread is a difference of two variances: the observed spread across runs, minus the
binomial noise of a finite evaluation. Below some true spread the difference is indistinguishable
from zero, and reporting a point estimate without knowing where that boundary is says nothing.

This simulates the design exactly as run -- so many runs, so many held-out questions, so many
samples per question -- at a range of true seed spreads, and reports how often the subtraction
recovers something and how close it lands.
"""

from __future__ import annotations

import argparse

import numpy as np

from caliper.analysis.uncertainty import resolved_spread_interval
from caliper.runtime import io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=int, default=[6, 8, 16])
    ap.add_argument("--questions", type=int, default=48)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--centre", type=float, default=0.35)
    ap.add_argument("--true-spreads", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"evaluation: {args.questions} questions x {args.samples} samples, "
          f"pass rate near {args.centre}")
    print(f"{'true s.d.':>9s} {'runs':>5s} {'resolved > 0':>13s} {'median estimate':>16s} "
          f"{'interval covers':>16s}")
    for true_sd in args.true_spreads:
        for runs in args.runs:
            estimates, covered, nonzero = [], 0, 0
            for trial in range(args.trials):
                centres = args.centre + rng.normal(0.0, true_sd, size=runs)
                rates = np.clip(centres[:, None] * np.ones((1, args.questions)), 0.0, 1.0)
                per_prompt = rng.binomial(args.samples, rates) / args.samples
                result = resolved_spread_interval(
                    per_prompt, args.samples, n_boot=args.boot, seed=trial
                )
                estimates.append(result["resolved"])
                nonzero += result["resolved"] > 0.0
                covered += result["lo"] <= true_sd <= result["hi"]
            rows.append({
                "true_sd": true_sd, "runs": runs,
                "nonzero_rate": nonzero / args.trials,
                "median_estimate": float(np.median(estimates)),
                "coverage": covered / args.trials,
            })
            print(f"{true_sd:9.3f} {runs:5d} {nonzero / args.trials:13.0%} "
                  f"{np.median(estimates):16.4f} {covered / args.trials:16.0%}")

    io.save("s7_power", vars(args), {"rows": rows})


if __name__ == "__main__":
    main()
