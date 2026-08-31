"""The other decision: pick the best recipe of several, from one run each.

Sizing a comparison is one use of a forecast. The other is choosing among configurations after a
single run of each, ranking them by gain per unit of noise and compute rather than by gain alone.
The two decisions do not reward the forecast equally, and reporting only the flattering one would
misrepresent what it buys, so both are measured on the same grid.

Seeds are split three ways and never reused across roles: one pilot supplies the observed gain, a
disjoint pool supplies whatever deviation the policy is entitled to, and a third reserved set
scores the choice. Letting the scoring seeds also set the deviation is what makes an oracle look
unbeatable, since its denominator would then share noise with the denominator it is graded on.
"""

from __future__ import annotations

import argparse

import numpy as np
from run import BANDS, collect, load_grid

from caliper.exact.policy import make_prompt
from caliper.exact.pool import build
from caliper.runtime import io


def base_rate(task, pool: int, cache: dict) -> float:
    """Mean pass rate of the untrained policy on the task's pool.

    Gains are measured from here, so it has to be the same reference for every recipe
    sharing a task.
    """
    if task not in cache:
        vocab, length, diversity, band = task
        policy, accepts = build(vocab, length, pool, BANDS[band], diversity, 0)
        cache[task] = float(np.mean([make_prompt(policy, a).pass_rate for a in accepts]))
    return cache[task]


def utility(scores: np.ndarray, base: float, cost: float) -> float:
    """Gain per unit of noise and rollout, the quantity a chooser is trying to maximise."""
    gain = scores.mean() - base
    spread = scores.std(ddof=1)
    return gain / (spread * np.sqrt(cost)) if gain > 0.0 and spread > 0.0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--cells", type=int, default=192)
    ap.add_argument("--holdout", type=int, default=24, help="seeds reserved for scoring")
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    record = load_grid(args.cells, args.pool)
    arms = collect(record, args.pool)
    cells = {id(a): c for a, c in zip(arms, record["result"]["cells"], strict=True)}
    bases: dict = {}
    for arm in arms:
        cell = cells[id(arm)]
        arm["cost"] = float(cell["prompts"] * cell["group_size"] * cell["steps"])
        arm["base"] = base_rate(arm["task"], args.pool, bases)
    seeds = min(len(a["scores"]) for a in arms)
    tasks = sorted({a["task"] for a in arms})
    rng = np.random.default_rng(args.seed)

    sources = ("constant", "frozen", "adjoint", "oracle")
    kept = {name: np.zeros((args.trials, len(tasks))) for name in sources}
    for trial in range(args.trials):
        order = rng.permutation(seeds)
        score_on, pool = order[: args.holdout], order[args.holdout:]
        pilot, read = pool[0], pool[1:]
        for t, task in enumerate(tasks):
            same = [a for a in arms if a["task"] == task]
            best = max(utility(a["scores"][score_on], a["base"], a["cost"]) for a in same)
            for name in sources:
                def deviation(arm, name=name, read=read):
                    return arm["scores"][read].std(ddof=1) if name == "oracle" else arm[name]
                picked = max(same, key=lambda a: (a["scores"][pilot] - a["base"])
                             / (deviation(a) * np.sqrt(a["cost"])))
                got = utility(picked["scores"][score_on], picked["base"], picked["cost"])
                kept[name][trial, t] = got / best if best > 0.0 else np.nan

    LABEL = {"constant": "one deviation for every recipe", "frozen": "frozen forecast",
             "adjoint": "adjoint forecast", "oracle": "measured deviation"}
    print(f"{len(tasks)} tasks, {len(arms)} configurations, {args.trials} pilots, "
          f"{args.holdout} seeds reserved for scoring\n")
    print("share of the best achievable value the choice keeps\n")
    summary = {}
    for name in sources:
        per_trial = np.nanmedian(kept[name], axis=1)
        summary[name] = [float(np.median(per_trial)), float(np.percentile(per_trial, 10)),
                         float(per_trial.min())]
        print(f"  {LABEL[name]:>30s}: median {100 * summary[name][0]:3.0f}%"
              f"   tenth percentile {100 * summary[name][1]:3.0f}%"
              f"   worst {100 * summary[name][2]:3.0f}%")

    resample = np.random.default_rng(args.seed + 1)
    gaps = []
    for _ in range(args.bootstrap):
        pick = resample.integers(0, len(tasks), len(tasks))
        adjoint = np.median(np.nanmedian(kept["adjoint"][:, pick], axis=1))
        constant = np.median(np.nanmedian(kept["constant"][:, pick], axis=1))
        oracle = np.median(np.nanmedian(kept["oracle"][:, pick], axis=1))
        gaps.append([adjoint - constant, oracle - constant])
    gaps = np.array(gaps)
    share = gaps[:, 0] / np.where(gaps[:, 1] != 0, gaps[:, 1], np.nan)
    print(f"\n  over a constant deviation the forecast gains "
          f"{100 * np.median(gaps[:, 0]):.1f} points "
          f"[{100 * np.percentile(gaps[:, 0], 2.5):.1f}, "
          f"{100 * np.percentile(gaps[:, 0], 97.5):.1f}]")
    print(f"  of the {100 * np.median(gaps[:, 1]):.1f} points a measured deviation gains, "
          f"a share of {np.nanmedian(share):.2f}")

    path = io.save("s13_selection", vars(args), {
        "tasks": len(tasks), "arms": len(arms),
        "kept": {k: v for k, v in summary.items()},
        "gain_points": float(np.median(gaps[:, 0])),
        "gain_low": float(np.percentile(gaps[:, 0], 2.5)),
        "gain_high": float(np.percentile(gaps[:, 0], 97.5)),
        "oracle_points": float(np.median(gaps[:, 1])),
        "share_of_oracle": float(np.nanmedian(share)),
        "source": record["manifest"]["config_hash"],
    })
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
