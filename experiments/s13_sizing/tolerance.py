"""How wrong the forecast may be before sizing stops paying.

The forecast is accurate on the exact models and only sometimes accurate on a pretrained one, so
the question that decides whether any of this is usable is how much of the saving survives a
forecast that is off. Here the deviations of s13 are corrupted on purpose, by a spread of random
multiplicative error and by a systematic scale error, and the sizing benchmark is rerun on the
corrupted values.

Error is drawn as a lognormal, so the median error factor of a corruption of width `s` is
`exp(0.6745 s)`, and the sweep is reported in those units rather than in `s`.
"""

from __future__ import annotations

import argparse

import numpy as np
from run import collect, comparisons, load_grid

from caliper.analysis.sizing import (
    decision_curve,
    forecast_deviation,
    running_difference,
    runs_for_error,
)
from caliper.runtime import io

NORMAL_QUARTILE = 0.674489750196


def corrupt(arms, spread, bias, rng):
    """Perturb every forecast, independently per configuration, plus a shared scale error."""
    noise = rng.normal(0.0, spread, size=len(arms))
    return {id(a): a["adjoint"] * float(np.exp(b + bias))
            for a, b in zip(arms, noise, strict=True)}


def cost(pairs, difference, deviations, thresholds, truth, target):
    spread = np.array([forecast_deviation(deviations[id(a)], deviations[id(b)])
                       for a, b, _ in pairs])
    seeds, wrong = decision_curve(
        difference, np.broadcast_to(spread[:, None, None], difference.shape), truth, thresholds)
    return runs_for_error(seeds.mean(1), wrong.mean(1), target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--cells", type=int, default=192)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--orders", type=int, default=400)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--holdout", type=int, default=32,
                    help="seeds reserved for the true ordering, as in the benchmark")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.01, 0.001])
    ap.add_argument("--repeats", type=int, default=8, help="draws of the random corruption")
    ap.add_argument("--errors", nargs="+", type=float,
                    default=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0])
    ap.add_argument("--biases", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    record = load_grid(args.cells, args.pool)
    arms = collect(record, args.pool)
    seeds = min(len(a["scores"]) for a in arms)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(seeds)
    truth_seeds, available = (
        (shuffled[: args.holdout], shuffled[args.holdout:]) if args.holdout
        else (shuffled, shuffled))
    pairs = comparisons(arms, args.margin, truth_seeds)
    orders = np.array([rng.permutation(available)[: args.cap] for _ in range(args.orders)])
    first = np.stack([a["scores"][orders] for a, _, _ in pairs])
    second = np.stack([b["scores"][orders] for _, b, _ in pairs])
    truth = np.array([t for _, _, t in pairs])
    difference = running_difference(first, second)
    thresholds = np.geomspace(1e-3, 1e2, 161)

    steps = np.arange(1, args.cap + 1, dtype=float)
    fixed_wrong = (np.sign(difference) != truth[:, None, None]).mean(1).T.mean(1)
    print(f"{len(pairs)} comparisons, cap {args.cap} seeds per arm\n")

    grids, summary = {}, {}
    for target in args.targets:
        clean = cost(pairs, difference, {id(a): a["adjoint"] for a in arms},
                     thresholds, truth, target)
        fixed = runs_for_error(steps, fixed_wrong, target)
        print(f"target mistake rate {100 * target:.1f}%:  forecast as measured {clean:.2f} runs, "
              f"fixed budget {fixed:.2f} runs")
        print(f"{'median error':>14s}" + "".join(f"{b:>9.3g}x" for b in args.biases))
        rows = {}
        for factor in args.errors:
            spread = np.log(factor) / NORMAL_QUARTILE
            row = []
            for bias in args.biases:
                draws = [cost(pairs, difference,
                              corrupt(arms, spread, np.log(bias), np.random.default_rng(r)),
                              thresholds, truth, target)
                         for r in range(args.repeats if spread > 0 else 1)]
                row.append(float(np.median(draws)))
            rows[factor] = row
            print(f"{factor:>13.2f}x" + "".join(
                f"{v:>10.2f}" if np.isfinite(v) else f"{'over cap':>10s}" for v in row))

        # Where a corrupted forecast stops being worth more than simply fixing the run count.
        column = args.biases.index(1.0)
        ordered = sorted(rows)
        breakeven = None
        for lower, upper in zip(ordered, ordered[1:], strict=False):
            below, above = rows[lower][column], rows[upper][column]
            if below is None or above is None or not (np.isfinite(below) and np.isfinite(above)):
                continue
            if below <= fixed <= above:
                breakeven = lower + (fixed - below) / (above - below) * (upper - lower)
        tag = ("beyond the range tested" if breakeven is None
               else f"{breakeven:.1f}x relative error")
        print(f"  break-even against a fixed budget: {tag}\n")
        grids[target] = rows
        summary[target] = {"clean": clean, "fixed": fixed, "breakeven": breakeven}

    path = io.save("s13_tolerance", vars(args), {
        "pairs": len(pairs),
        "errors": args.errors,
        "biases": args.biases,
        "targets": args.targets,
        "summary": {str(k): {kk: (None if vv is None or not np.isfinite(vv) else float(vv))
                             for kk, vv in v.items()} for k, v in summary.items()},
        "runs": {str(k): {str(kk): [None if not np.isfinite(x) else x for x in row]
                          for kk, row in g.items()} for k, g in grids.items()},
    })
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
