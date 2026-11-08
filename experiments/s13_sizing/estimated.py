"""S13, estimated: the closed form with its noise traces estimated from one run, not computed.

The held-out benchmark (`heldout.py`) found the closed form within about 1.1x of the adjoint
forecast once its threshold is learned on other tasks. But it fed the closed form the exact
traces tau_b and tau_w of the enumerable policy, which no practitioner has. Here each
configuration gets the estimate a single run could make of them: the split estimator
(`caliper.estimators.splits`) averaged over as many batches as the run has updates, each with the
run's own prompt count and group size, drawn at the initial policy. Everything else, the pairs,
the orderings, the leave-one-task-out choice of threshold and the task-resampled intervals, is
`heldout.py` unchanged.
"""

from __future__ import annotations

import argparse

import heldout
import numpy as np
import run
from run import BANDS, collect, comparisons, evaluate, load_grid

from caliper.estimators.simulate import simulate_split_batch
from caliper.estimators.splits import average
from caliper.exact.policy import make_prompt
from caliper.exact.pool import build
from caliper.runtime import io

FLOOR = 1e-12  # a noisy estimate can go negative; the deviation needs a positive spread


def estimated_deviation(cell: dict, pool: int, seed: int) -> float:
    """The closed form with tau_b and tau_w estimated from one run's worth of batches."""
    policy, accepts = build(cell["vocab"], cell["length"], pool, BANDS[cell["band"]],
                            cell["diversity"], 0)
    prompts = [make_prompt(policy, a) for a in accepts]
    rng = np.random.default_rng(seed)
    estimate = average([
        simulate_split_batch(prompts, "rloo", cell["prompts"], cell["group_size"], rng,
                             matrix=None, blocks=2)
        for _ in range(cell["steps"])
    ])
    spread = max(estimate.tau_b + estimate.tau_w / (cell["group_size"] - 1), FLOOR)
    return cell["step_size"] * np.sqrt(cell["steps"] / cell["prompts"]) * np.sqrt(spread)


def calibrate_estimated(arms: list[dict]) -> None:
    """One scale per task, learned on the other tasks, as `run.calibrate` does for the exact one."""
    for arm in arms:
        others = [a for a in arms if a["task"] != arm["task"]]
        scale = np.exp(np.mean([np.log(a["oracle"] / a["estimated"]) for a in others]))
        arm["estimated_calibrated"] = arm["estimated"] * float(scale)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--cells", type=int, default=192)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--orders", type=int, default=400)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--holdout", type=int, default=32)
    ap.add_argument("--targets", nargs="+", type=float, default=[0.01, 0.001])
    ap.add_argument("--grid", type=int, default=161)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    record = load_grid(args.cells, args.pool)
    arms = collect(record, args.pool)
    for index, (arm, cell) in enumerate(zip(arms, record["result"]["cells"], strict=True)):
        arm["estimated"] = estimated_deviation(cell, args.pool, args.seed + 1009 * index)
    calibrate_estimated(arms)
    relative = np.array([
        max(a["estimated"] / a["frozen"], a["frozen"] / a["estimated"]) for a in arms])
    print(f"estimated against exact closed form: median {np.median(relative):.2f}x, "
          f"worst {relative.max():.2f}x")

    names = ("estimated", "estimated_calibrated")
    run.POLICIES = run.POLICIES + names
    heldout.TUNED = heldout.TUNED + names
    heldout.NOMINAL = heldout.NOMINAL + names

    rng = np.random.default_rng(args.seed)
    seeds = min(len(a["scores"]) for a in arms)
    shuffled = rng.permutation(seeds)
    truth_seeds, available = shuffled[: args.holdout], shuffled[args.holdout:]
    pairs = comparisons(arms, args.margin, truth_seeds)
    orders = [rng.permutation(available)[: args.cap] for _ in range(args.orders)]
    groups = sorted({first["task"] for first, _, _ in pairs})
    tasks = np.array([groups.index(first["task"]) for first, _, _ in pairs])
    curves = evaluate(pairs, orders, np.geomspace(1e-3, 1e2, args.grid))
    boot = np.random.default_rng(args.seed + 1)
    results = {}
    for target in args.targets:
        key = f"margin={args.margin:g},target={target:g}"
        results[key] = {
            "held_out": heldout.summarise(heldout.held_out(curves, tasks, target), tasks,
                                          args.bootstrap, boot),
            "nominal": heldout.summarise(heldout.nominal(pairs, orders, target), tasks,
                                         args.bootstrap, boot),
        }
        print(f"\n{key}: {len(pairs)} pairs")
        for mode in ("held_out", "nominal"):
            for name in ("frozen", "calibrated", *names, "adjoint"):
                value = results[key][mode].get(name)
                if value is None:
                    continue
                lo, hi = value["ratio_interval"]
                print(f"  {mode:>8s} {name:>20s}: {value['runs']:6.2f} runs, mistakes "
                      f"{100 * value['mistakes']:6.3f}%, x{value['ratio_to_adjoint']:5.2f} "
                      f"[{lo:.2f}, {hi:.2f}]")

    path = io.save("s18_sizing_estimated", vars(args), {
        "results": results,
        "estimated_vs_exact_median": float(np.median(relative)),
        "estimated_vs_exact_worst": float(relative.max()),
        "source": record["manifest"]["config_hash"],
    })
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
