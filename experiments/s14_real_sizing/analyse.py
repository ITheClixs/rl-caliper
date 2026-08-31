"""S14: the sizing benchmark on a pretrained model.

Six recipes are trained on one corpus at a matched rollout budget, each with an anchor run that
produces the forecast and then validation seeds. The benchmark of s13 is then run on the scores.

What a practitioner watches is the held-out score, whose spread across seeds carries two terms: the
seed part the forecast predicts, and the evaluation part from scoring a finite prompt set. Neither
alone is the right denominator for a stopping rule, so the forecast is completed with the binomial
term, which is a closed-form quantity needing no runs. The baselines see the same observed scores,
so nothing is given to the forecast that is withheld from them.

With sixteen seeds per recipe there are not enough to both fix an ordering on reserved seeds and
leave a useful pool, so the ordering is read from all of them. Section 6 of the paper shows that
choice favours whichever policy draws the most seeds, which is the fixed budget, so the numbers
here understate the forecast rather than flattering it.
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
from caliper.runtime import io

POLICIES = ("fixed", "empirical", "constant", "adjoint", "oracle")
LABEL = {
    "fixed": "fixed budget, no stopping rule",
    "empirical": "deviation from the seeds in hand",
    "constant": "one deviation for every recipe",
    "adjoint": "adjoint forecast, from one run",
    "oracle": "measured deviation",
}


def collect(prefix: str) -> list[dict]:
    arms = {}
    for record in io.load_all("s11_prospective"):
        label = record["manifest"]["config"].get("label", "")
        result = record["result"]
        if not label.startswith(prefix) or not result.get("held_out_scores"):
            continue
        scores = np.asarray(result["held_out_scores"], dtype=float)
        binomial = float(result.get("binomial_variance") or 0.0)
        arms[label] = {
            "label": label,
            "scores": scores,
            "cost": result["prompts"] * result["group_size"] * result["steps"],
            "oracle": float(scores.std(ddof=1)),
            "adjoint": float(np.sqrt(result["predicted_std"] ** 2 + binomial)),
            "seed_part": float(result["predicted_std"]),
            "binomial": float(np.sqrt(binomial)),
        }
    return [arms[k] for k in sorted(arms)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sizing-")
    ap.add_argument("--margins", nargs="+", type=float, default=[2.0, 2.5, 3.0],
                    help="how unambiguous a pair must be; reported at each")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--orders", type=int, default=600)
    ap.add_argument("--targets", nargs="+", type=float, default=[0.05, 0.02, 0.01])
    ap.add_argument("--bootstrap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = collect(args.prefix)
    if len(arms) < 2:
        raise SystemExit(f"found {len(arms)} recipes under {args.prefix!r}; need at least two")
    seeds = min(len(a["scores"]) for a in arms)
    typical = float(np.median([a["oracle"] for a in arms]))
    for arm in arms:
        arm["constant"] = typical

    print(f"{len(arms)} recipes, {seeds} seeds each\n")
    print(f"{'recipe':>26s} {'mean':>7s} {'measured':>9s} {'forecast':>9s} "
          f"{'seed':>8s} {'eval':>8s} {'error':>7s}")
    for a in arms:
        error = max(a["adjoint"] / a["oracle"], a["oracle"] / a["adjoint"])
        print(f"{a['label']:>26s} {a['scores'].mean():7.4f} {a['oracle']:9.4f} "
              f"{a['adjoint']:9.4f} {a['seed_part']:8.4f} {a['binomial']:8.4f} {error:6.2f}x")

    def order_pairs(margin):
        out = []
        for i, first in enumerate(arms):
            for second in arms[i + 1:]:
                gap = second["scores"].mean() - first["scores"].mean()
                noise = np.hypot(first["oracle"], second["oracle"]) / np.sqrt(seeds)
                if abs(gap) > margin * noise:
                    out.append((first, second, float(np.sign(gap))))
        return out

    def evaluate(pairs):
        rng = np.random.default_rng(args.seed)
        orders = np.array([rng.permutation(seeds)[: args.cap] for _ in range(args.orders)])
        first = np.stack([a["scores"][orders] for a, _, _ in pairs])
        second = np.stack([b["scores"][orders] for _, b, _ in pairs])
        truth = np.array([t for _, _, t in pairs])
        difference = running_difference(first, second)
        thresholds = np.geomspace(1e-3, 1e2, 161)
        out = {}
        for name in POLICIES:
            if name == "fixed":
                steps = np.arange(1, args.cap + 1, dtype=float)
                used = np.broadcast_to(steps[:, None], (args.cap, len(pairs))).copy()
                wrong = (np.sign(difference) != truth[:, None, None]).mean(1).T
            elif name == "empirical":
                used, wrong = decision_curve(
                    difference, running_deviation(first, second), truth, thresholds)
            else:
                spread = np.array([forecast_deviation(a[name], b[name]) for a, b, _ in pairs])
                used, wrong = decision_curve(
                    difference, np.broadcast_to(spread[:, None, None], difference.shape),
                    truth, thresholds)
            out[name] = (used, wrong)
        return out

    total = len(arms) * (len(arms) - 1) // 2
    table, ratios, counts = {}, {}, {}
    for margin in args.margins:
        pairs = order_pairs(margin)
        counts[margin] = len(pairs)
        print(f"\n{'=' * 62}\nordering unambiguous at {margin:g} standard errors: "
              f"{len(pairs)} of {total} pairs")
        if not pairs:
            print("  none; nothing to benchmark at this margin")
            continue
        if len(pairs) < 5:
            print("  few pairs: the resampled interval below is not meaningful")
        curves = evaluate(pairs)
        print(f"\n{'':>32s}" + "".join(f"{100 * t:>9.1f}%" for t in args.targets))
        weight = np.ones(len(pairs))
        table[margin] = {}
        for name in POLICIES:
            used, wrong = curves[name]
            row = [runs_for_error(np.average(used, 1, weights=weight),
                                  np.average(wrong, 1, weights=weight), t) for t in args.targets]
            table[margin][name] = row
            print(f"{LABEL[name]:>32s}" +
                  "".join(f"{v:>10.2f}" if np.isfinite(v) else f"{'over cap':>10s}" for v in row))

        resample = np.random.default_rng(args.seed + 1)
        draws = {name: [] for name in POLICIES}
        for _ in range(args.bootstrap):
            w = np.bincount(resample.integers(0, len(pairs), len(pairs)),
                            minlength=len(pairs)).astype(float)
            for name in POLICIES:
                used, wrong = curves[name]
                draws[name].append(runs_for_error(np.average(used, 1, weights=w),
                                                  np.average(wrong, 1, weights=w),
                                                  args.targets[-1]))
        reference = np.array(draws["adjoint"])
        ratios[margin] = {}
        print(f"\n  against the adjoint forecast at {100 * args.targets[-1]:.1f}%")
        for name in POLICIES:
            values = np.array(draws[name])
            keep = np.isfinite(values) & np.isfinite(reference)
            ratio = values[keep] / reference[keep]
            ratios[margin][name] = [float(np.median(ratio)), float(np.percentile(ratio, 2.5)),
                                    float(np.percentile(ratio, 97.5))]
            print(f"  {LABEL[name]:>32s}: {np.median(values[keep]):5.2f} runs   "
                  f"{np.median(ratio):5.2f}x [{np.percentile(ratio, 2.5):.2f}, "
                  f"{np.percentile(ratio, 97.5):.2f}]")

    if not table:
        raise SystemExit("no margin produced enough unambiguous pairs to benchmark")

    accuracy = [max(a["adjoint"] / a["oracle"], a["oracle"] / a["adjoint"]) for a in arms]
    path = io.save("s14_real_sizing", vars(args), {
        "recipes": len(arms), "seeds": int(seeds),
        "targets": args.targets,
        "pairs": {str(m): n for m, n in counts.items()},
        "runs_at_target": {str(m): {k: [None if not np.isfinite(v) else v for v in row]
                                    for k, row in rows.items()} for m, rows in table.items()},
        "ratio_at_finest": {str(m): v for m, v in ratios.items()},
        "forecast_error": {"median": float(np.median(accuracy)), "worst": float(max(accuracy))},
        "arms": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in a.items()} for a in arms],
    })
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
