"""S13, held out: the sizing benchmark with every policy's threshold chosen before it is graded.

`run.py` reads each policy at the threshold that meets a mistake-rate target on the very
comparisons it is then scored on. That is fair between policies, since each gets the same
in-sample choice of its one knob, but it is not what a practitioner can do, and it is what makes
a constant scale error look free. Here the knob is chosen in two ways a practitioner could use:

  held out   for each of the twelve tasks, the threshold (or, for the fixed budget, the seed
             count) is the cheapest one meeting the target on the other eleven, then applied
             unchanged to the held-out task;
  nominal    no tuning at all: stop once the lead clears the one-sided normal quantile of the
             target, z = Phi^-1(1 - target), in units of the policy's own standard error.

Intervals resample tasks rather than comparisons, since comparisons within a task share arms
and are not independent. A fixed budget with no cap is added from the normal approximation, so
a saving cannot come from the cap alone.
"""

from __future__ import annotations

import argparse

import numpy as np
from run import collect, comparisons, evaluate, load_grid
from scipy.stats import norm

from caliper.analysis.sizing import (
    decision_curve,
    forecast_deviation,
    running_deviation,
    running_difference,
)
from caliper.runtime import io

TUNED = ("fixed", "constant", "empirical", "frozen", "calibrated", "adjoint", "oracle")
NOMINAL = ("constant", "empirical", "frozen", "calibrated", "adjoint", "oracle")


def choose(seeds: np.ndarray, wrong: np.ndarray, target: float) -> int:
    """The cheapest setting whose mean mistake rate meets `target`, else the most cautious."""
    cost, rate = seeds.mean(1), wrong.mean(1)
    met = np.flatnonzero(rate <= target)
    if not met.size:
        return int(np.argmax(cost))
    return int(met[np.argmin(cost[met])])


def held_out(curves, tasks: np.ndarray, target: float) -> dict[str, np.ndarray]:
    """Per pair, seeds used and mistakes made when the knob is chosen on the other tasks."""
    out = {}
    for name in TUNED:
        seeds, wrong = curves[name]
        used, missed = np.empty(seeds.shape[1]), np.empty(seeds.shape[1])
        for task in np.unique(tasks):
            test = tasks == task
            pick = choose(seeds[:, ~test], wrong[:, ~test], target)
            used[test], missed[test] = seeds[pick, test], wrong[pick, test]
        out[name] = np.stack([used, missed])
    return out


def nominal(pairs, orders, target: float) -> dict[str, np.ndarray]:
    """Per pair, seeds and mistakes at the textbook threshold, with no tuning."""
    index = np.asarray(orders)
    first = np.stack([a["scores"][index] for a, _, _ in pairs])
    second = np.stack([b["scores"][index] for _, b, _ in pairs])
    truth = np.array([t for _, _, t in pairs])
    difference = running_difference(first, second)
    z = np.array([norm.ppf(1.0 - target)])
    out = {}
    for name in NOMINAL:
        if name == "empirical":
            deviation = running_deviation(first, second)
        else:
            spread = np.array([forecast_deviation(a[name], b[name]) for a, b, _ in pairs])
            deviation = np.broadcast_to(spread[:, None, None], difference.shape)
        seeds, wrong = decision_curve(difference, deviation, truth, z)
        out[name] = np.stack([seeds[0], wrong[0]])
    return out


def uncapped_fixed(pairs, truth_seeds: np.ndarray, target: float, most: int = 4096) -> float:
    """Runs a fixed budget needs with no cap, from the normal approximation on held-out seeds."""
    gaps, spreads = [], []
    for first, second, _ in pairs:
        left, right = first["scores"][truth_seeds], second["scores"][truth_seeds]
        gaps.append(abs(right.mean() - left.mean()))
        spreads.append(np.hypot(left.std(ddof=1), right.std(ddof=1)))
    gaps, spreads = np.array(gaps), np.array(spreads)
    for n in range(1, most + 1):
        if norm.cdf(-gaps * np.sqrt(n) / spreads).mean() <= target:
            return 2.0 * n
    return float("inf")


def summarise(per_pair: dict[str, np.ndarray], tasks, draws: int, rng) -> dict:
    """Pooled runs and realised mistake rate, with task-resampled intervals on cost ratios."""
    unique = np.unique(tasks)
    members = [np.flatnonzero(tasks == task) for task in unique]
    out = {name: {"runs": float(2 * v[0].mean()), "mistakes": float(v[1].mean())}
           for name, v in per_pair.items()}
    ratios = {name: [] for name in per_pair}
    for _ in range(draws):
        chosen = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        reference = per_pair["adjoint"][0, chosen].mean()
        for name, values in per_pair.items():
            ratios[name].append(values[0, chosen].mean() / reference)
    for name, values in ratios.items():
        out[name]["ratio_to_adjoint"] = float(
            per_pair[name][0].mean() / per_pair["adjoint"][0].mean())
        out[name]["ratio_interval"] = [float(np.percentile(values, 2.5)),
                                       float(np.percentile(values, 97.5))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--cells", type=int, default=192)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--orders", type=int, default=400)
    ap.add_argument("--margins", nargs="+", type=float, default=[4.0, 2.0])
    ap.add_argument("--holdout", type=int, default=32)
    ap.add_argument("--targets", nargs="+", type=float, default=[0.01, 0.001])
    ap.add_argument("--grid", type=int, default=161)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    record = load_grid(args.cells, args.pool)
    arms = collect(record, args.pool)
    seeds = min(len(a["scores"]) for a in arms)
    thresholds = np.geomspace(1e-3, 1e2, args.grid)
    results = {}
    for margin in args.margins:
        # the same draw order as run.py, so at its margin the pairs and orderings are identical
        rng = np.random.default_rng(args.seed)
        shuffled = rng.permutation(seeds)
        truth_seeds, available = shuffled[: args.holdout], shuffled[args.holdout:]
        pairs = comparisons(arms, margin, truth_seeds)
        orders = [rng.permutation(available)[: args.cap] for _ in range(args.orders)]
        names = sorted({first["task"] for first, _, _ in pairs})
        tasks = np.array([names.index(first["task"]) for first, _, _ in pairs])
        curves = evaluate(pairs, orders, thresholds)
        boot = np.random.default_rng(args.seed + 1)
        for target in args.targets:
            key = f"margin={margin:g},target={target:g}"
            nominal_runs = nominal(pairs, orders, target)
            results[key] = {
                "pairs": len(pairs),
                "tasks": len(names),
                "held_out": summarise(held_out(curves, tasks, target), tasks, args.bootstrap, boot),
                "nominal": summarise(nominal_runs, tasks, args.bootstrap, boot),
                "uncapped_fixed_runs": uncapped_fixed(pairs, truth_seeds, target),
                # how often the nominal adjoint rule ran into the cap rather than stopping
                "adjoint_cap_share": float((nominal_runs["adjoint"][0] >= args.cap).mean()),
            }
            row = results[key]
            print(f"\n{key}: {row['pairs']} pairs over {row['tasks']} tasks")
            for mode in ("held_out", "nominal"):
                for name, value in row[mode].items():
                    band = value["ratio_interval"]
                    print(f"  {mode:>8s} {name:>10s}: {value['runs']:6.2f} runs, "
                          f"mistakes {100 * value['mistakes']:6.3f}%, "
                          f"x{value['ratio_to_adjoint']:5.2f} [{band[0]:.2f}, {band[1]:.2f}]")
            print(f"  uncapped fixed budget: {row['uncapped_fixed_runs']:.0f} runs")

    path = io.save("s15_sizing_heldout", vars(args), {
        "results": results, "source": record["manifest"]["config_hash"]})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
