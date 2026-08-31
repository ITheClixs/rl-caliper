"""S13: how many runs it takes to decide which of two configurations is better.

The forecast of section 3 is a prediction about a quantity nobody measures for its own sake. What
practitioners do with an across-seed deviation is size their comparisons: train two recipes for a
few seeds each and stop once the lead is convincing. This experiment asks what that costs.

Every policy sees the same seeds in the same order and differs only in where its deviation comes
from. The scores are the 48 per-seed scores already stored for each of the 192 cells of p2, so no
new training is required and the comparison is exactly paired.

Comparisons are restricted to pairs whose ordering is unambiguous over all 48 seeds, so a
disagreement is a mistake rather than a coin landing the other way up.
"""

from __future__ import annotations

import argparse

import numpy as np

from caliper.analysis.sizing import (
    decision_curve,
    forecast_deviation,
    running_deviation,
    running_difference,
    runs_for_error,
)
from caliper.exact.aggregate import ExactBatchModel
from caliper.exact.policy import fisher, make_prompt
from caliper.exact.pool import build
from caliper.runtime import io

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}
POLICIES = ("fixed", "constant", "empirical", "calibrated", "adjoint", "oracle")
LABEL = {
    "fixed": "fixed budget",
    "constant": "constant deviation",
    "empirical": "deviation from the seeds",
    "calibrated": "calibrated frozen forecast",
    "adjoint": "adjoint forecast",
    "oracle": "measured deviation",
}


def load_grid(cells_wanted: int, pool: int):
    for record in io.load_all("p2_metric_forecast"):
        cells = record["result"].get("cells", [])
        if (
            len(cells) == cells_wanted
            and record["manifest"]["config"].get("pool") == pool
            and "seed_scores" in cells[0]
        ):
            return record
    raise SystemExit("no forecast grid with stored per-seed scores")


def frozen_deviation(cell, pool: int, cache: dict) -> float:
    """The predictor available before any run: exact noise terms and the step count, no fit."""
    task = (cell["vocab"], cell["length"], cell["diversity"], cell["band"])
    key = (task, cell["group_size"])
    if key not in cache:
        policy, accepts = build(cell["vocab"], cell["length"], pool, BANDS[cell["band"]],
                                cell["diversity"], 0)
        prompts = [make_prompt(policy, a) for a in accepts]
        model = ExactBatchModel([p.moments() for p in prompts], [fisher(p) for p in prompts])
        cache[key] = model.noise_terms("rloo", cell["group_size"], curvature="identity")
    terms = cache[key]
    spread = terms.tau_b + terms.tau_w / (cell["group_size"] - 1)
    return cell["step_size"] * np.sqrt(cell["steps"] / cell["prompts"]) * np.sqrt(spread)


def calibrate(arms: list[dict]) -> None:
    """Give the frozen predictor an absolute scale, fitted without the task it is applied to.

    Ranking configurations needs the deviation only up to a common factor, so the frozen
    predictor serves unchanged. Sizing a comparison needs the deviation itself, and a predictor
    off by a constant either stops far too early or never stops at all. One factor per task,
    fitted on the other tasks, is the least that closes that gap.
    """
    for arm in arms:
        others = [a for a in arms if a["task"] != arm["task"]]
        scale = np.exp(np.mean([np.log(a["oracle"] / a["frozen"]) for a in others]))
        arm["calibrated"] = arm["frozen"] * float(scale)


def collect(record, pool: int) -> list[dict]:
    cache: dict = {}
    arms = []
    for cell in record["result"]["cells"]:
        scores = np.asarray(cell["seed_scores"], dtype=float)
        arms.append({
            "task": (cell["vocab"], cell["length"], cell["diversity"], cell["band"]),
            "scores": scores,
            "oracle": float(scores.std(ddof=1)),
            "adjoint": float(cell["predicted_std"]),
            "frozen": frozen_deviation(cell, pool, cache),
        })
    typical = float(np.median([a["oracle"] for a in arms]))
    for arm in arms:
        arm["constant"] = typical
    calibrate(arms)
    return arms


def comparisons(arms: list[dict], margin: float, seeds: int) -> list[tuple]:
    out = []
    for i, first in enumerate(arms):
        for second in arms[i + 1:]:
            if first["task"] != second["task"]:
                continue
            gap = second["scores"].mean() - first["scores"].mean()
            noise = np.hypot(first["oracle"], second["oracle"]) / np.sqrt(seeds)
            if abs(gap) > margin * noise:
                out.append((first, second, float(np.sign(gap))))
    return out


def evaluate(pairs, orders, thresholds):
    """Seeds used and mistakes made per comparison, for each policy."""
    index = np.asarray(orders)
    first = np.stack([a["scores"][index] for a, _, _ in pairs])
    second = np.stack([b["scores"][index] for _, b, _ in pairs])
    truth = np.array([t for _, _, t in pairs])
    difference = running_difference(first, second)

    out = {}
    for name in POLICIES:
        if name == "fixed":
            steps = np.arange(1, difference.shape[-1] + 1, dtype=float)
            seeds = np.broadcast_to(steps[:, None], (len(steps), len(pairs))).copy()
            wrong = (np.sign(difference) != truth[:, None, None]).mean(1).T
        elif name == "empirical":
            seeds, wrong = decision_curve(
                difference, running_deviation(first, second), truth, thresholds)
        else:
            spread = np.array([forecast_deviation(a[name], b[name]) for a, b, _ in pairs])
            deviation = np.broadcast_to(spread[:, None, None], difference.shape)
            seeds, wrong = decision_curve(difference, deviation, truth, thresholds)
        out[name] = (seeds, wrong)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--cells", type=int, default=192)
    ap.add_argument("--cap", type=int, default=16, help="most seeds per arm a policy may spend")
    ap.add_argument("--orders", type=int, default=400, help="seed orderings per comparison")
    ap.add_argument("--margin", type=float, default=4.0, help="how unambiguous a pair must be")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.01, 0.005, 0.002, 0.001])
    ap.add_argument("--bootstrap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    record = load_grid(args.cells, args.pool)
    arms = collect(record, args.pool)
    seeds = min(len(a["scores"]) for a in arms)
    pairs = comparisons(arms, args.margin, seeds)
    if not pairs:
        raise SystemExit("no unambiguous comparisons at this margin")

    rng = np.random.default_rng(args.seed)
    orders = [rng.permutation(seeds)[: args.cap] for _ in range(args.orders)]
    thresholds = np.arange(0.5, 4.01, 0.125)
    curves = evaluate(pairs, orders, thresholds)

    accuracy = np.array(
        [max(a["adjoint"] / a["oracle"], a["oracle"] / a["adjoint"]) for a in arms])

    print(f"{len(pairs)} unambiguous comparisons over {len(arms)} configurations, "
          f"{seeds} seeds each")
    print(f"forecast accuracy: median {np.median(accuracy):.2f}x, worst {accuracy.max():.2f}x\n")
    print("runs needed to hold the mistake rate at a target\n")
    header = "".join(f"{100 * t:>9.1f}%" for t in args.targets)
    print(f"{'':>30s}{header}")
    weight = np.ones(len(pairs))
    table = {}
    for name in POLICIES:
        seeds_used, wrong = curves[name]
        row = [runs_for_error(np.average(seeds_used, 1, weights=weight),
                              np.average(wrong, 1, weights=weight), t) for t in args.targets]
        table[name] = row
        print(f"{LABEL[name]:>30s}" +
              "".join(f"{v:>10.2f}" if np.isfinite(v) else f"{'over cap':>10s}" for v in row))

    resample = np.random.default_rng(args.seed + 1)
    draws = {name: [] for name in POLICIES}
    for _ in range(args.bootstrap):
        weight = np.bincount(resample.integers(0, len(pairs), len(pairs)),
                             minlength=len(pairs)).astype(float)
        for name in POLICIES:
            seeds_used, wrong = curves[name]
            draws[name].append(runs_for_error(
                np.average(seeds_used, 1, weights=weight),
                np.average(wrong, 1, weights=weight), args.targets[-1]))

    reference = np.array(draws["adjoint"])
    print(f"\nagainst the adjoint forecast at {100 * args.targets[-1]:.1f}%, "
          f"{args.bootstrap} resamples of the comparisons\n")
    ratios = {}
    for name in POLICIES:
        values = np.array(draws[name])
        keep = np.isfinite(values) & np.isfinite(reference)
        ratio = values[keep] / reference[keep]
        ratios[name] = [float(np.median(ratio)), float(np.percentile(ratio, 2.5)),
                        float(np.percentile(ratio, 97.5))]
        print(f"  {LABEL[name]:>28s}: {np.median(values[keep]):5.2f} runs    "
              f"{np.median(ratio):5.2f}x [{np.percentile(ratio, 2.5):.2f}, "
              f"{np.percentile(ratio, 97.5):.2f}]")

    config = {k: v for k, v in vars(args).items()}
    path = io.save("s13_sizing", config, {
        "pairs": len(pairs),
        "arms": len(arms),
        "seeds": int(seeds),
        "thresholds": thresholds.tolist(),
        "targets": args.targets,
        "runs_at_target": {k: [None if not np.isfinite(v) else v for v in row]
                           for k, row in table.items()},
        "ratio_at_finest": ratios,
        "forecast_accuracy": {"median": float(np.median(accuracy)), "worst": float(accuracy.max())},
        "curves": {name: {"seeds": (2 * curves[name][0].mean(1)).tolist(),
                          "error": curves[name][1].mean(1).tolist()} for name in POLICIES},
        "source": record["manifest"]["config_hash"],
    })
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
