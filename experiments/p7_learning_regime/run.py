"""P7: the forecast when the run it is made from actually learns.

The settings behind the headline of P2 gain at most a few hundredths in pass rate. This raises the
step size until runs gain several times that and collapse onto one sequence, with every input still
exact, and grades three things against seeds trained afterwards:

  * the forecast of Theorem 2, carried back along one run;
  * the same with the adjoint held at grad M, the cheap form;
  * the spread of seeds simulated in the tangent model at the starting policy, which uses no
    training run at all (`caliper.tangent`).

The one-run curvature residual is recorded too, so that whether it tells the regimes apart can be
read from the same record. The protocol is the one P2 uses: the anchor first, everything computed
from it frozen, then the seeds it is graded against, with the anchor kept out of them.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.exact.adjoint import forecast, metric_value
from caliper.exact.curvature import run_residual
from caliper.exact.pool import build
from caliper.exact.tangent import reference_model
from caliper.exact.train import ExactTrainer, RunConfig
from caliper.objectives.advantages import weight_table
from caliper.runtime import io
from caliper.tangent import Recipe, simulate

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}
DECILES = np.linspace(0.1, 0.9, 9)


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
    ap.add_argument("--shapes", nargs="+", default=["3x3", "4x2"])
    ap.add_argument("--diversity", nargs="+", type=float, default=[1.0, 0.3])
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=48, help="including the anchor")
    ap.add_argument("--steps", nargs="+", type=int, default=[12, 50])
    ap.add_argument("--prompts", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--group-size", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--step-size", nargs="+", type=float, default=[1.5, 6.0, 20.0, 60.0])
    ap.add_argument("--bands", nargs="+", default=list(BANDS))
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--simulated-seeds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=991)
    args = ap.parse_args()

    shapes = [tuple(int(v) for v in shape.split("x")) for shape in args.shapes]
    grid = itertools.product(
        shapes, args.diversity, args.bands, args.steps, args.prompts, args.group_size,
        args.step_size,
    )
    cells = []
    print(f"{'shape':>5s} {'div':>4s} {'band':>5s} {'T':>3s} {'P':>3s} {'G':>2s} {'eta':>5s} "
          f"{'gain':>6s} {'measured':>9s} {'carried':>8s} {'held':>6s} {'tangent':>8s}")
    for index, ((vocab, length), diversity, band, steps, prompts, group, eta) in enumerate(grid):
        policy, accepts = build(vocab, length, args.pool, BANDS[band], diversity, args.seed)
        weights = weight_table(args.estimator, group)
        cfg = RunConfig(
            estimator=args.estimator, n_prompts=prompts, group_size=group, steps=steps,
            initial_step_size=eta, controller_gain=0.0, measure_every=steps,
        )

        # (1) the anchor, and the two forecasts and the diagnostic that rest on it alone
        states, first_score = train_one(policy, accepts, cfg, args.seed + 1)
        carried = forecast(policy, accepts, args.estimator, group, prompts, eta, steps,
                           states=states)
        held = forecast(policy, accepts, args.estimator, group, prompts, eta, steps,
                        states=states, transport=False)
        residual = float(np.mean([
            run_residual(state, accepts, weights, prompts, group, eta,
                         np.random.default_rng(args.seed + 4241 + i))
            for i, state in enumerate(states[:-1])
        ]))

        # (2) seeds simulated in the tangent model at the start: no run enters this at all
        pool = np.arange(accepts.shape[0])
        simulated = simulate(
            reference_model(policy, accepts),
            Recipe(eta, prompts, group, steps, weights),
            pool, pool, args.simulated_seeds,
            np.random.default_rng(args.seed + 60_000 + index),
        )

        # (3) only now, the seeds all of the above are graded against
        scores = [first_score]
        for s in range(1, args.seeds):
            scores.append(train_one(policy, accepts, cfg, args.seed + 1 + 7919 * s)[1])
        later = np.array(scores[1:])
        measured = float(later.std(ddof=1))
        start = metric_value(policy, accepts)

        cells.append({
            "vocab": vocab, "length": length, "diversity": diversity, "band": band,
            "steps": steps, "prompts": prompts, "group_size": group, "step_size": eta,
            "start": start,
            "anchor_gain": first_score - start,
            "predicted_std": carried.std,
            "untransported_std": held.std,
            "curvature_residual": residual,
            "tangent_std": simulated.std,
            "tangent_mean": float(simulated.scores.mean()),
            "tangent_deciles": np.quantile(simulated.scores, DECILES).tolist(),
            "tangent_seeds": args.simulated_seeds,
            # the anchor first, then the seeds trained after it
            "seed_scores": scores,
            "seeds": len(scores),
        })
        print(f"{vocab}x{length:<3d} {diversity:4.1f} {band:>5s} {steps:3d} {prompts:3d} "
              f"{group:2d} {eta:5.1f} {first_score - start:+6.3f} {measured:9.2e} "
              f"{carried.std / measured:8.2f} {held.std / measured:6.2f} "
              f"{simulated.std / measured:8.2f}", flush=True)

    path = io.save("p7_learning_regime", vars(args), {"cells": cells})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
