"""B6: does the group-size law survive Adam?

Every result so far is under plain gradient ascent, where the update is the gradient and the noise
terms can be read in the parameter metric. Adam moves in a different metric and is scale invariant,
so the theory only carries over if the noise is measured after preconditioning. This runs the same
sweep as B1 under both optimisers, instrumenting the preconditioned gradient in the Adam case.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from caliper.analysis.efficiency import efficiency
from caliper.envs.modsum import ModSum
from caliper.population.model import ModelConfig, PopulationTransformer
from caliper.population.train import (
    RLConfig,
    RLVRTrainer,
    SFTConfig,
    make_pool,
    measure_noise,
    pass_rate,
    restore,
    snapshot,
    supervised_warmup,
)
from caliper.runtime import io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--rollouts-per-step", type=int, default=256)
    ap.add_argument("--corpus", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--measure-batches", type=int, default=40)
    ap.add_argument("--prime-batches", type=int, default=16)
    ap.add_argument("--measurement-floor", type=float, default=1e-2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = ModSum(modulus=7, chain=4)
    group_sizes = [g for g in [2, 4, 8, 16, 32, 64] if args.rollouts_per_step % g == 0]
    # Adam's update has unit scale, so it needs a much smaller step than the raw gradient
    starts = {"sgd": (0.02, 1e-3), "adam": (2e-3, 1e-3)}
    cells = []

    for optimiser in ("sgd", "adam"):
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        config = ModelConfig(
            vocab=task.vocab,
            context=task.total_len,
            width=args.width,
            depth=2,
            heads=2,
            population=args.population,
        )
        model = PopulationTransformer(config, generator).to(args.device)
        supervised_warmup(
            model,
            task,
            SFTConfig(steps=2500, batch=64, target_pass_rate=0.30, eval_every=4),
            generator,
        )
        base = snapshot(model)
        pool = make_pool(task, args.population, args.corpus, generator)
        start = pass_rate(model, task, 512, generator, pool).cpu().numpy()

        step_size, drift_target = starts[optimiser]
        probe = RLConfig(
            prompts=32,
            group_size=8,
            blocks=8,
            steps=0,
            optimiser=optimiser,
            initial_step_size=step_size,
            measurement_floor=0.0 if optimiser == "sgd" else args.measurement_floor,
        )
        terms = measure_noise(
            model,
            task,
            probe,
            generator,
            args.measure_batches,
            pool=pool,
            prime_batches=0 if optimiser == "sgd" else args.prime_batches,
        )
        restore(model, base)
        tau_b = float(terms["tau_b"].mean())
        tau_w = float(terms["tau_w_scaled"].mean())
        signal = float(terms["signal"].mean())
        share = (args.rollouts_per_step / 8 - 1) / (args.corpus - 1)
        g_star = 1 + np.sqrt(tau_w / max((1 - share) * tau_b, 1e-12))
        print(
            f"\n{optimiser}: start pass {start.mean():.3f}  tau_b {tau_b:.3e}  "
            f"tau_w {tau_w:.3e}  G* {g_star:.2f}"
        )

        curve, ses = [], []
        for g in group_sizes:
            restore(model, base)
            cfg = RLConfig(
                measurement_floor=0.0 if optimiser == "sgd" else args.measurement_floor,
                prompts=args.rollouts_per_step // g,
                group_size=g,
                steps=args.steps,
                blocks=8,
                drift_target=drift_target,
                initial_step_size=step_size,
                optimiser=optimiser,
                instrument_every=10_000,
                eval_every=10_000,
            )
            history = RLVRTrainer(model, task, cfg, generator, pool=pool).run()
            gain = np.array(history["final_pass_rate"]) - start
            curve.append(float(gain.mean()))
            ses.append(float(gain.std(ddof=1) / np.sqrt(gain.size)))
            print(f"  G={g:3d}  gain {gain.mean():+.4f} +/- {ses[-1]:.4f}")

        predicted = efficiency(
            group_sizes, tau_b, tau_w, signal, args.rollouts_per_step, args.corpus
        )
        observed = np.array(curve)
        design = np.vstack([np.ones_like(predicted), np.sqrt(predicted)]).T
        coef, *_ = np.linalg.lstsq(design, observed, rcond=None)
        resid = observed - design @ coef
        r2 = float(1 - (resid**2).sum() / ((observed - observed.mean()) ** 2).sum())
        print(f"  predicted efficiency {np.round(predicted, 3).tolist()}")
        print(f"  R^2 against sqrt(rho): {r2:.3f}   best measured G = "
              f"{group_sizes[int(np.argmax(observed))]}")
        cells.append(
            {
                "optimiser": optimiser,
                "group_sizes": group_sizes,
                "observed": curve,
                "observed_se": ses,
                "predicted_efficiency": predicted.tolist(),
                "tau_b": tau_b,
                "tau_w_scaled": tau_w,
                "signal": signal,
                "g_star": float(g_star),
                "r2": r2,
                "start_pass_rate": float(start.mean()),
            }
        )

    io.save("b6_adam", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
