"""P4: where the linearisation the forecast rests on stops holding.

Everything in this paper assumes two runs stay close enough that the mean update field is linear
between them. That assumption is checkable. For two runs at theta_A and theta_B,

    actual   = || g_bar(theta_B) - g_bar(theta_A) ||
    linear   = || J(theta_A) (theta_B - theta_A) ||

and their ratio is one where the linearisation holds. Reporting the forecast error against that
ratio says when the forecast can be trusted, using a quantity a practitioner can measure without
knowing the answer.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.analysis.uncertainty import spread_interval
from caliper.exact.adjoint import forecast
from caliper.exact.pool import build
from caliper.exact.propagation import mean_gradient, update_jacobian
from caliper.exact.train import ExactTrainer, RunConfig
from caliper.objectives.advantages import weight_table
from caliper.runtime import io

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}


def run_pair(policy, accepts, cfg, seed):
    """Two runs, keeping the states of the first and the endpoints of both at every step."""
    trainers = [ExactTrainer(policy, accepts, cfg) for _ in range(2)]
    rngs = [np.random.default_rng(seed + 7919 * i) for i in range(2)]
    states, pairs = [trainers[0].policy], []
    for _ in range(cfg.steps):
        for trainer, rng in zip(trainers, rngs, strict=True):
            trainer.step(rng)
        states.append(trainers[0].policy)
        pairs.append((trainers[0].policy, trainers[1].policy))
    return states, pairs, [t.objective() for t in trainers]


def nonlinearity(pairs, accepts, weights):
    """Median over the run of the gap between the actual and the linearised gradient difference."""
    ratios = []
    for left, right in pairs:
        delta = (right.logits - left.logits).reshape(-1)
        if np.linalg.norm(delta) < 1e-14:
            continue
        actual = mean_gradient(right, accepts, weights) - mean_gradient(left, accepts, weights)
        linear = update_jacobian(left, accepts, weights) @ delta
        residual = np.linalg.norm(actual - linear) / max(np.linalg.norm(actual), 1e-30)
        ratios.append(float(residual))
    return float(np.median(ratios)), float(np.max(ratios))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--steps", nargs="+", type=int, default=[10, 25])
    ap.add_argument("--prompts", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--step-size", nargs="+", type=float, default=[0.5, 1.5, 4.0])
    ap.add_argument("--bands", nargs="+", default=["easy", "mixed", "hard"])
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weights = weight_table(args.estimator, args.group_size)
    cells = []
    print(f"{'band':>6s} {'T':>4s} {'P':>4s} {'eta':>5s} {'residual':>9s} {'worst':>7s} "
          f"{'pred/meas':>10s}")
    for band, steps, prompts, eta in itertools.product(
        args.bands, args.steps, args.prompts, args.step_size
    ):
        policy, accepts = build(args.vocab, args.length, args.pool, BANDS[band], 1.0, args.seed)
        cfg = RunConfig(
            estimator=args.estimator, n_prompts=prompts, group_size=args.group_size,
            steps=steps, initial_step_size=eta, controller_gain=0.0, measure_every=steps,
        )
        states, pairs, first_scores = run_pair(policy, accepts, cfg, args.seed + 1)
        median_residual, worst_residual = nonlinearity(pairs, accepts, weights)
        prediction = forecast(
            policy, accepts, args.estimator, args.group_size, prompts, eta, steps, states=states
        )

        scores = list(first_scores)
        for s in range(2, args.seeds):
            trainer = ExactTrainer(policy, accepts, cfg)
            rng = np.random.default_rng(args.seed + 1 + 7919 * s)
            for _ in range(steps):
                trainer.step(rng)
            scores.append(trainer.objective())
        measured = spread_interval(np.array(scores), n_boot=3000, seed=args.seed)
        ratio = prediction.std / measured["std"]

        cells.append({
            "band": band, "steps": steps, "prompts": prompts, "step_size": eta,
            "median_residual": median_residual, "worst_residual": worst_residual,
            "predicted_std": prediction.std, "measured_std": measured["std"],
            "ratio": float(ratio), "final_metric": prediction.trajectory[-1],
        })
        print(f"{band:>6s} {steps:4d} {prompts:4d} {eta:5.2f} {median_residual:9.3f} "
              f"{worst_residual:7.3f} {ratio:10.2f}", flush=True)

    residual = np.array([c["median_residual"] for c in cells])
    error = np.abs(np.log(np.array([c["ratio"] for c in cells])))
    order = np.argsort(residual)
    half = len(order) // 2
    low, high = error[order[:half]], error[order[half:]]
    corr = float(np.corrcoef(residual, error)[0, 1])
    print(f"\ncorrelation between the residual and |log(predicted/measured)|: {corr:.2f}")
    print(f"forecast error, settings with the smaller residual: {np.exp(np.median(low)):.2f}x")
    print(f"forecast error, settings with the larger residual:  {np.exp(np.median(high)):.2f}x")
    io.save("p4_linearity", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
