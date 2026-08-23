"""A3: end-to-end test of the closed-form optimal group size.

Rollouts per step, number of steps and the per-step drift target are all held fixed; the only thing
that varies is how the rollouts are split into prompts and group members. The objective is exact,
so the empirical optimum is not an estimate.
"""

from __future__ import annotations

import argparse

import numpy as np

from caliper.estimators.simulate import simulate_split_batch
from caliper.estimators.splits import average
from caliper.exact.aggregate import ExactBatchModel
from caliper.exact.policy import fisher, make_prompt
from caliper.exact.pool import build as build_pool
from caliper.exact.train import ExactTrainer, RunConfig
from caliper.runtime import io


def measured_optimum(policy, accepts, estimator, group_size, batches, seed):
    """Predict G* from the split estimator, using batches drawn at the initial policy."""
    prompts = [make_prompt(policy, a) for a in accepts]
    rng = np.random.default_rng(seed)
    matrix = np.mean([fisher(p) for p in prompts], axis=0)
    est = average(
        [
            simulate_split_batch(prompts, estimator, 16, group_size, rng, matrix=matrix)
            for _ in range(batches)
        ]
    )
    exact = ExactBatchModel([p.moments() for p in prompts], [fisher(p) for p in prompts])
    terms = exact.noise_terms(estimator, group_size, curvature="fisher")
    return {
        "measured_g_star": est.optimal_group_size(),
        "exact_g_star": 1.0 + np.sqrt(terms.tau_w * (group_size - 1) / terms.tau_b),
        "exact_tau_ratio": terms.tau_w * (group_size - 1) / terms.tau_b,
        "measured_tau_ratio": est.tau_w_scaled / est.tau_b,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--rollouts-per-step", type=int, default=256)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--drift-target", type=float, default=2e-4)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--difficulty", default="broad")
    ap.add_argument("--diversity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bands = {"broad": (0.05, 0.95), "hard": (0.05, 0.30), "easy": (0.70, 0.95)}
    policy, accepts = build_pool(
        args.vocab, args.length, args.pool, bands[args.difficulty], args.diversity, args.seed
    )
    group_sizes = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60]
    group_sizes = [g for g in group_sizes if args.rollouts_per_step % g == 0]

    prediction = measured_optimum(policy, accepts, args.estimator, 8, 600, args.seed + 1)
    print(
        f"predicted G*: exact {prediction['exact_g_star']:.2f}, "
        f"measured {prediction['measured_g_star']:.2f}"
    )

    rows = []
    for g in group_sizes:
        finals = []
        for s in range(args.seeds):
            cfg = RunConfig(
                estimator=args.estimator,
                n_prompts=args.rollouts_per_step // g,
                group_size=g,
                steps=args.steps,
                drift_target=args.drift_target,
                initial_step_size=1.0,
            )
            trainer = ExactTrainer(policy, accepts, cfg)
            history = trainer.run(np.random.default_rng(10_000 + 97 * s + g))
            finals.append(history["final_objective"])
        arr = np.array(finals)
        rows.append(
            {
                "group_size": g,
                "n_prompts": args.rollouts_per_step // g,
                "final_mean": float(arr.mean()),
                "final_se": float(arr.std(ddof=1) / np.sqrt(arr.size)),
                "finals": arr.tolist(),
            }
        )
        print(
            f"G={g:3d}  P={args.rollouts_per_step // g:4d}  "
            f"final J = {arr.mean():.4f} +/- {arr.std(ddof=1) / np.sqrt(arr.size):.4f}"
        )

    best = max(rows, key=lambda r: r["final_mean"])
    print(f"empirical best G = {best['group_size']}")
    io.save(
        "a3_split",
        vars(args),
        {"rows": rows, "prediction": prediction, "empirical_best_g": best["group_size"]},
    )


if __name__ == "__main__":
    main()
