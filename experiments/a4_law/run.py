"""A4: does the measured noise ratio predict the empirical optimal group size?

The prompt pool's diversity and difficulty are varied to move tau_w/tau_b over a wide range. For
each pool the predicted optimum is computed from the split estimator, and the empirical optimum is
located by training at every split of a fixed rollout budget and fitting the peak.
"""

from __future__ import annotations

import argparse
import itertools
import zlib

import numpy as np

from caliper.analysis.peak import bootstrap_fit
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


def predictions(policy, accepts, estimator, probe_group, batches, seed):
    prompts = [make_prompt(policy, a) for a in accepts]
    fishers = [fisher(p) for p in prompts]
    matrix = np.mean(fishers, axis=0)
    rng = np.random.default_rng(seed)
    est = average(
        [
            simulate_split_batch(prompts, estimator, 16, probe_group, rng, matrix=matrix)
            for _ in range(batches)
        ]
    )
    exact = ExactBatchModel([p.moments() for p in prompts], fishers)
    terms = exact.noise_terms(estimator, probe_group, curvature="fisher")
    exact_ratio = terms.tau_w * (probe_group - 1) / terms.tau_b
    return {
        "exact_ratio": float(exact_ratio),
        "exact_g_star": float(1 + np.sqrt(exact_ratio)),
        "measured_ratio": float(est.tau_w_scaled / est.tau_b),
        "measured_g_star": float(est.optimal_group_size()),
        "critical_batch": float(terms.critical_batch()),
    }


def sweep(policy, accepts, args, group_sizes):
    per_g = []
    for g in group_sizes:
        finals = []
        for s in range(args.seeds):
            cfg = RunConfig(
                estimator=args.estimator,
                n_prompts=args.rollouts_per_step // g,
                group_size=g,
                steps=args.steps,
                drift_target=args.drift_target,
            )
            trainer = ExactTrainer(policy, accepts, cfg)
            finals.append(trainer.run(np.random.default_rng(7919 * s + 31 * g))["final_objective"])
        per_g.append(np.array(finals))
    return per_g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--rollouts-per-step", type=int, default=240)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--drift-target", type=float, default=5e-5)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--probe-batches", type=int, default=3000)
    ap.add_argument("--pools", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    candidates = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 48, 60, 80, 120]
    group_sizes = [g for g in candidates if args.rollouts_per_step % g == 0]
    diversities = [0.05, 0.15, 0.4, 1.0]
    cells = []

    print(f"{'band':6s} {'div':>5s} {'exact G*':>9s} {'meas G*':>8s} {'empirical G*':>24s}")
    for band, div, rep in itertools.product(BANDS, diversities, range(args.pools)):
        tag = f"{band}:{div}".encode()
        seed = args.seed + 1000 * rep + zlib.crc32(tag) % 977
        policy, accepts = build(args.vocab, args.length, args.pool, BANDS[band], div, seed)
        pred = predictions(policy, accepts, args.estimator, 8, args.probe_batches, seed + 7)
        per_g = sweep(policy, accepts, args, group_sizes)
        peak, lo, hi = bootstrap_fit(np.array(group_sizes), per_g, seed=seed)
        cells.append(
            {
                "band": band,
                "diversity": div,
                "replicate": rep,
                **pred,
                "empirical_g_star": peak,
                "empirical_lo": lo,
                "empirical_hi": hi,
                "curve": [float(x.mean()) for x in per_g],
                "curve_se": [float(x.std(ddof=1) / np.sqrt(x.size)) for x in per_g],
                "group_sizes": group_sizes,
            }
        )
        print(
            f"{band:6s} {div:5.2f} {pred['exact_g_star']:9.2f} {pred['measured_g_star']:8.2f} "
            f"{peak:8.2f} [{lo:6.2f}, {hi:6.2f}]"
        )

    io.save("a4_law", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
