"""A4: does the measured noise ratio predict how the rollout budget should be split?

For each prompt pool the theory predicts an entire efficiency curve rho(G), not merely its argmax.
The test compares that curve, computed with no free parameters, against the improvement actually
obtained by training at every split of a fixed rollout budget. Pools whose predicted curve is flat
are a prediction too, and are scored as such rather than discarded.
"""

from __future__ import annotations

import argparse
import itertools
import zlib

import numpy as np
from scipy.stats import spearmanr

from caliper.estimators.simulate import simulate_split_batch
from caliper.estimators.splits import average
from caliper.exact.aggregate import ExactBatchModel
from caliper.exact.policy import fisher, make_prompt
from caliper.exact.pool import build
from caliper.exact.train import ExactTrainer, RunConfig
from caliper.runtime import io

BANDS = {
    "broad": (0.05, 0.95),
    "hard": (0.05, 0.35),
    "easy": (0.65, 0.95),
    "mid": (0.40, 0.60),
}


def efficiency(group_sizes, tau_b, tau_w_scaled, signal, rollouts):
    """rho(G) = 1 / (1 + G * Bcrit(G) / R), with no fitted parameters."""
    g = np.asarray(group_sizes, dtype=float)
    bcrit = (tau_b + tau_w_scaled / (g - 1.0)) / signal
    return 1.0 / (1.0 + g * bcrit / rollouts)


def pool_terms(policy, accepts, estimator, probe_group, batches, seed):
    prompts = [make_prompt(policy, a) for a in accepts]
    fishers = [fisher(p) for p in prompts]
    matrix = np.mean(fishers, axis=0)
    exact = ExactBatchModel([p.moments() for p in prompts], fishers).noise_terms(
        estimator, probe_group, curvature="fisher"
    )
    rng = np.random.default_rng(seed)
    est = average(
        [
            simulate_split_batch(prompts, estimator, 16, probe_group, rng, matrix=matrix)
            for _ in range(batches)
        ]
    )
    return {
        "exact": {
            "tau_b": exact.tau_b,
            "tau_w_scaled": exact.tau_w * (probe_group - 1),
            "signal": exact.signal,
            "g_star": 1.0 + np.sqrt(exact.tau_w * (probe_group - 1) / exact.tau_b),
        },
        "measured": {
            "tau_b": est.tau_b,
            "tau_w_scaled": est.tau_w_scaled,
            "signal": est.signal,
            "g_star": est.optimal_group_size(),
        },
    }


def train_sweep(policy, accepts, estimator, group_sizes, rollouts, steps, drift, seeds):
    start = float(np.mean(accepts @ policy.sequence_probs()))
    out = []
    for g in group_sizes:
        gains = []
        for s in range(seeds):
            cfg = RunConfig(
                estimator=estimator,
                n_prompts=rollouts // g,
                group_size=g,
                steps=steps,
                drift_target=drift,
            )
            trainer = ExactTrainer(policy, accepts, cfg)
            history = trainer.run(np.random.default_rng(7919 * s + 31 * g + 13 * rollouts))
            gains.append(history["final_objective"] - start)
        out.append(np.array(gains))
    return start, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--total-rollouts", type=int, default=19200)
    ap.add_argument("--total-drift", type=float, default=4e-3)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--probe-batches", type=int, default=2000)
    ap.add_argument("--pools", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    budgets = [48, 240]
    diversities = [0.05, 0.15, 0.4, 1.0]
    candidates = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 48, 60, 80, 120]
    cells = []

    print(
        f"{'band':6s} {'div':>5s} {'R':>4s} {'exact G*':>9s} {'meas G*':>8s} "
        f"{'range':>7s} {'spearman':>9s} {'R2':>6s}"
    )
    for band, div, rep in itertools.product(BANDS, diversities, range(args.pools)):
        tag = f"{band}:{div}".encode()
        seed = args.seed + 1000 * rep + zlib.crc32(tag) % 977
        policy, accepts = build(args.vocab, args.length, args.pool, BANDS[band], div, seed)
        terms = pool_terms(policy, accepts, args.estimator, 8, args.probe_batches, seed + 7)

        for rollouts in budgets:
            group_sizes = [g for g in candidates if rollouts % g == 0 and g <= rollouts // 2]
            steps = args.total_rollouts // rollouts
            drift = args.total_drift / steps
            start, gains = train_sweep(
                policy,
                accepts,
                args.estimator,
                group_sizes,
                rollouts,
                steps,
                drift,
                args.seeds,
            )
            observed = np.array([g.mean() for g in gains])
            se = np.array([g.std(ddof=1) / np.sqrt(g.size) for g in gains])
            predicted = efficiency(
                group_sizes,
                terms["exact"]["tau_b"],
                terms["exact"]["tau_w_scaled"],
                terms["exact"]["signal"],
                rollouts,
            )
            rho = float(spearmanr(predicted, observed).statistic)
            design = np.vstack([np.ones_like(predicted), np.sqrt(predicted)]).T
            coef, *_ = np.linalg.lstsq(design, observed, rcond=None)
            resid = observed - design @ coef
            ss_tot = float(((observed - observed.mean()) ** 2).sum())
            r2 = float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan")
            pred_range = float(predicted.max() / predicted.min())

            cells.append(
                {
                    "band": band,
                    "diversity": div,
                    "replicate": rep,
                    "rollouts_per_step": rollouts,
                    "steps": steps,
                    "drift_target": drift,
                    "start_objective": start,
                    "group_sizes": group_sizes,
                    "observed": observed.tolist(),
                    "observed_se": se.tolist(),
                    "predicted_efficiency": predicted.tolist(),
                    "spearman": rho,
                    "r2": r2,
                    "predicted_range": pred_range,
                    **{f"exact_{k}": v for k, v in terms["exact"].items()},
                    **{f"measured_{k}": v for k, v in terms["measured"].items()},
                }
            )
            print(
                f"{band:6s} {div:5.2f} {rollouts:4d} {terms['exact']['g_star']:9.2f} "
                f"{terms['measured']['g_star']:8.2f} {pred_range:7.2f} {rho:9.3f} {r2:6.3f}"
            )

    io.save("a4_law", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
