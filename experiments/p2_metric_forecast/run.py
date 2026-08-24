"""P2: can one run predict how far a rerun would move the number it reports?

The protocol is prospective by construction. A single run is trained and frozen; the states it
visited are carried backwards to produce a forecast of the across-seed standard deviation of the
final pass rate; only then are the remaining seeds trained and the realised spread measured. The
forecast never sees them.

Because the policy is enumerable the reported number is the exact pass rate rather than an
estimate of it, so the measured spread is the spread of the quantity itself and not of a
finite evaluation set.

Each setting is also forecast with the adjoint held fixed at grad M, which keeps the injected term
and discards everything the trajectory does to it. That ablation is what says whether the backward
pass is earning its cost.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.analysis.uncertainty import spread_interval
from caliper.exact.adjoint import forecast
from caliper.exact.pool import build
from caliper.exact.train import ExactTrainer, RunConfig
from caliper.runtime import io

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}


def train_one(policy, accepts, cfg, seed):
    """One run, keeping every state it passes through."""
    trainer = ExactTrainer(policy, accepts, cfg)
    rng = np.random.default_rng(seed)
    states = [trainer.policy]
    for _ in range(cfg.steps):
        trainer.step(rng)
        states.append(trainer.policy)
    return states, trainer.objective()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["3x3"],
                    help="policy shapes as vocab x length, e.g. 3x3 4x2")
    ap.add_argument("--diversity", nargs="+", type=float, default=[1.0])
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--steps", nargs="+", type=int, default=[10, 25])
    ap.add_argument("--prompts", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--group-size", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--step-size", nargs="+", type=float, default=[0.5, 1.5])
    ap.add_argument("--bands", nargs="+", default=["easy", "mixed", "hard"])
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shapes = [tuple(int(v) for v in shape.split("x")) for shape in args.shapes]
    cells = []
    grid = itertools.product(
        shapes, args.diversity, args.bands, args.steps, args.prompts,
        args.group_size, args.step_size,
    )
    print(f"{'shape':>6s} {'div':>4s} {'band':>6s} {'T':>4s} {'P':>4s} {'G':>3s} {'eta':>5s} "
          f"{'predicted':>10s} {'measured':>10s} {'95% interval':>22s}")
    for (vocab, length), diversity, band, steps, prompts, group, eta in grid:
        policy, accepts = build(vocab, length, args.pool, BANDS[band], diversity, args.seed)
        cfg = RunConfig(
            estimator=args.estimator,
            n_prompts=prompts,
            group_size=group,
            steps=steps,
            initial_step_size=eta,
            controller_gain=0.0,  # the theory is written at a fixed step size
            measure_every=steps,
        )

        # (1) one run, frozen, and a forecast made from it alone
        states, first_score = train_one(policy, accepts, cfg, args.seed + 1)
        prediction = forecast(
            policy, accepts, args.estimator, group, prompts, eta, steps, states=states
        )
        # the ablation: keep the injected term, discard what the trajectory does to it
        untransported = forecast(
            policy, accepts, args.estimator, group, prompts, eta, steps, states=states,
            transport=False,
        )

        # (2) only now, the independent seeds
        scores = [first_score]
        for s in range(1, args.seeds):
            _, score = train_one(policy, accepts, cfg, args.seed + 1 + 7919 * s)
            scores.append(score)
        scores = np.array(scores)
        measured = spread_interval(scores, n_boot=4000, seed=args.seed)

        cells.append(
            {
                "vocab": vocab,
                "length": length,
                "diversity": diversity,
                "band": band,
                "steps": steps,
                "prompts": prompts,
                "group_size": group,
                "step_size": eta,
                "predicted_std": prediction.std,
                "untransported_std": untransported.std,
                "measured_std": measured["std"],
                "measured_lo": measured["lo"],
                "measured_hi": measured["hi"],
                "kernel": prediction.kernel,
                "adjoint_norm": prediction.adjoint_norm,
                "forecast_trajectory": prediction.trajectory,
                "mean_score": float(scores.mean()),
                "seeds": int(scores.size),
            }
        )
        shape = f"{vocab}x{length}"
        print(f"{shape:>6s} {diversity:4.1f} {band:>6s} {steps:4d} {prompts:4d} "
              f"{group:3d} {eta:5.2f} {prediction.std:10.3e} {measured['std']:10.3e} "
              f"[{measured['lo']:.3e}, {measured['hi']:.3e}]", flush=True)

    ratios = np.array([c["predicted_std"] / c["measured_std"] for c in cells])
    covered = [c["measured_lo"] <= c["predicted_std"] <= c["measured_hi"] for c in cells]
    print(f"\nratio predicted/measured: median {np.median(ratios):.2f}x  "
          f"range {ratios.min():.2f}--{ratios.max():.2f}")
    print(f"forecast inside the measured 95% interval in {sum(covered)}/{len(covered)} settings")
    for key, label in (("predicted_std", "with transport"), ("untransported_std", "without")):
        error = np.abs(np.log(np.array([c[key] / c["measured_std"] for c in cells])))
        inside = sum(
            c["measured_lo"] <= c[key] <= c["measured_hi"] for c in cells
        )
        print(f"  {label:>15s}: median {np.exp(np.median(error)):.2f}x  "
              f"worst {np.exp(error.max()):.2f}x  inside {inside}/{len(cells)}")
    io.save("p2_metric_forecast", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
