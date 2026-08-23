"""T1: the closed form for the optimal split against exact numerical optimisation.

Theorem 1 gives a closed form that is exact when rollouts cost the same regardless of how they are
grouped, and a small-batch limit once a group shares its prompt's prefill. This checks both claims
against direct minimisation of the full cost-aware objective, over a grid of prefill ratios and
rollout budgets.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from caliper.runtime import io


def objective(group_size, prefill_ratio, rollouts, beta_b, beta_w):
    """Cost per unit of squared progress: (a + G)(R/G + Bcrit(G)), up to a constant."""
    bcrit = beta_b + beta_w / (group_size - 1.0)
    return (prefill_ratio + group_size) * (rollouts / group_size + bcrit)


def exact_optimum(prefill_ratio, rollouts, beta_b, beta_w):
    result = minimize_scalar(
        objective,
        bounds=(1.0 + 1e-6, 1e5),
        args=(prefill_ratio, rollouts, beta_b, beta_w),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def stationary_optimum(prefill_ratio, rollouts, beta_b, beta_w):
    """Unique root of the stationarity condition of Theorem 1(ii)."""

    def derivative(g):
        return (
            beta_b
            - prefill_ratio * rollouts / g**2
            - (1.0 + prefill_ratio) * beta_w / (g - 1.0) ** 2
        )

    return float(brentq(derivative, 1.0 + 1e-9, 1e6, xtol=1e-12))


def closed_form(prefill_ratio, beta_b, beta_w):
    return 1.0 + np.sqrt((1.0 + prefill_ratio) * beta_w / beta_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-b", type=float, default=40.07)
    ap.add_argument("--tau-w", type=float, default=146.0)
    ap.add_argument("--signal", type=float, default=1.144)
    ap.add_argument("--prefill-ratios", type=float, nargs="+",
                    default=[0.0, 0.04, 0.70, 2.63, 10.92])
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 256, 1024, 8192])
    args = ap.parse_args()

    beta_b = args.tau_b / args.signal
    beta_w = args.tau_w / args.signal
    critical = beta_b + beta_w / 7.0
    print(f"beta_b {beta_b:.1f}  beta_w {beta_w:.1f}  Bcrit at G=8 {critical:.1f} prompts\n")

    rows = []
    header = f"{'c_pre/c_dec':>12s} {'R':>6s} {'P at opt':>9s} {'exact':>8s} "
    print(header + f"{'stationary':>11s} {'closed form':>12s} {'cf error %':>11s}")
    for ratio in args.prefill_ratios:
        for rollouts in args.budgets:
            exact = exact_optimum(ratio, rollouts, beta_b, beta_w)
            stat = (
                closed_form(0.0, beta_b, beta_w)
                if ratio == 0
                else stationary_optimum(ratio, rollouts, beta_b, beta_w)
            )
            approx = closed_form(ratio, beta_b, beta_w)
            rows.append(
                {
                    "prefill_ratio": ratio,
                    "rollouts": rollouts,
                    "prompts_at_optimum": rollouts / exact,
                    "exact": exact,
                    "stationary": stat,
                    "closed_form": approx,
                    "closed_form_error": 100 * abs(approx - exact) / exact,
                    "stationary_error": 100 * abs(stat - exact) / exact,
                }
            )
            print(
                f"{ratio:12.2f} {rollouts:6d} {rollouts / exact:9.1f} {exact:8.2f} "
                f"{stat:11.2f} {approx:12.2f} {100 * abs(approx - exact) / exact:11.1f}"
            )

    uniform = [r for r in rows if r["prefill_ratio"] == 0.0]
    worst_uniform = max(r["closed_form_error"] for r in uniform)
    worst_stationary = max(r["stationary_error"] for r in rows)
    below = [r for r in rows if r["prompts_at_optimum"] < critical]
    print(
        f"\nuniform cost: closed form exact to {worst_uniform:.2e}% over "
        f"{len(uniform)} budgets\n"
        f"stationarity form: matches exact optimisation to {worst_stationary:.2e}% everywhere\n"
        f"closed form below the critical batch size ({len(below)} cells): "
        f"worst error {max(r['closed_form_error'] for r in below):.1f}%\n"
        f"closed form above it ({len(rows) - len(below)} cells): "
        f"worst error {max(r['closed_form_error'] for r in rows if r not in below):.1f}%"
    )
    io.save("t1_optimum", vars(args), {"rows": rows, "critical_batch": critical})


if __name__ == "__main__":
    main()
