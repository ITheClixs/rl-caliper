"""A1: does the closed-form optimal group size match the exact optimum?

Everything is computed in closed form over an enumerable policy, so the "true" optimum is found by
evaluating the exact per-step progress for every G rather than by training.
"""

from __future__ import annotations

import argparse

import numpy as np

from caliper.exact.aggregate import ExactBatchModel, predicted_optimal_group_size
from caliper.exact.policy import (
    TabularPolicy,
    accept_set_for_pass_rate,
    fisher,
    make_prompt,
)
from caliper.runtime import io


def build_population(
    n_prompts: int, vocab: int, length: int, difficulty: tuple[float, float], seed: int
):
    rng = np.random.default_rng(seed)
    policy = TabularPolicy.random(vocab, length, rng)
    probs, _ = policy.enumerate()
    prompts, fishers = [], []
    lo, hi = difficulty
    for _ in range(n_prompts):
        target = rng.uniform(lo, hi)
        accept = accept_set_for_pass_rate(probs, target, rng)
        prompt = make_prompt(policy, accept)
        p = prompt.pass_rate
        if p <= 1e-6 or p >= 1 - 1e-6:
            continue
        prompts.append(prompt.moments())
        fishers.append(fisher(prompt))
    return ExactBatchModel(prompts, fishers)


def sweep(model: ExactBatchModel, estimator: str, rollouts: int, group_sizes, curvature: str):
    rows = []
    for g in group_sizes:
        terms = model.noise_terms(estimator, g, curvature)
        rows.append(
            {
                "group_size": int(g),
                "progress": model.progress_per_rollout_budget(estimator, g, rollouts, curvature),
                "tau_b": terms.tau_b,
                "tau_w": terms.tau_w,
                "tau_w_scaled": terms.tau_w * (g - 1),
                "signal": terms.signal,
                "critical_batch": terms.critical_batch(),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--rollouts", type=int, default=4096)
    ap.add_argument("--curvature", default="fisher")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    group_sizes = list(range(2, 33))
    difficulties = {
        "broad": (0.02, 0.98),
        "hard": (0.02, 0.30),
        "easy": (0.70, 0.98),
        "mid": (0.35, 0.65),
    }

    results = {}
    for est in ["rloo", "grpo_mean", "grpo_std"]:
        for label, band in difficulties.items():
            model = build_population(args.prompts, args.vocab, args.length, band, args.seed)
            rows = sweep(model, est, args.rollouts, group_sizes, args.curvature)
            best = max(rows, key=lambda r: r["progress"])
            ref = rows[len(rows) // 2]
            predicted = predicted_optimal_group_size(ref["tau_b"], ref["tau_w_scaled"])
            mean_pq = float(np.mean([m["p"] * (1 - m["p"]) for m in model.prompts]))
            results[f"{est}/{label}"] = {
                "rows": rows,
                "exact_best_g": best["group_size"],
                "predicted_g": predicted,
                "tau_ratio": ref["tau_w_scaled"] / ref["tau_b"],
                "mean_p_one_minus_p": mean_pq,
                "n_prompts": len(model.prompts),
            }
            ratio = results[f"{est}/{label}"]["tau_ratio"]
            print(
                f"{est:10s} {label:6s} exact G*={best['group_size']:3d} "
                f"predicted={predicted:6.2f}  tau_w/tau_b={ratio:8.2f} "
                f"E[p(1-p)]={mean_pq:.3f}"
            )

    config = vars(args)
    path = io.save("a1_group_size", config, results)
    print("saved", path)


if __name__ == "__main__":
    main()
