"""B2: does a recipe transfer across width when written in drift rather than in step size?

Two sweeps at every width. One fixes the learning rate, the way a recipe normally does. The other
fixes the drift target and lets the controller find whatever step size realises it. If drift is the
scale-free quantity, the second sweep's optimum should sit at the same place at every width while
the first sweep's optimum moves.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from caliper.envs.modsum import ModSum
from caliper.population.model import ModelConfig, PopulationTransformer
from caliper.population.train import (
    RLConfig,
    RLVRTrainer,
    SFTConfig,
    measure_noise,
    pass_rate,
    restore,
    snapshot,
    supervised_warmup,
)
from caliper.runtime import io


def prepare(task, width, depth, population, device, seed, target_pass_rate):
    generator = torch.Generator(device=device).manual_seed(seed)
    config = ModelConfig(
        vocab=task.vocab,
        context=task.total_len,
        width=width,
        depth=depth,
        heads=2,
        population=population,
    )
    model = PopulationTransformer(config, generator).to(device)
    supervised_warmup(
        model,
        task,
        SFTConfig(steps=1200, batch=64, target_pass_rate=target_pass_rate, eval_every=4),
        generator,
    )
    return model, generator


def run_one(model, task, generator, base, prompts, group_size, steps, drift, step_size, control):
    restore(model, base)
    config = RLConfig(
        prompts=prompts,
        group_size=group_size,
        steps=steps,
        drift_target=drift,
        initial_step_size=step_size,
        control=control,
        instrument_every=10_000,
        eval_every=10_000,
    )
    trainer = RLVRTrainer(model, task, config, generator)
    history = trainer.run()
    realised = np.array(
        [c.step_size for c in trainer.controllers]
    )
    return np.array(history["final_pass_rate"]), realised


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=5)
    ap.add_argument("--chain", type=int, default=3)
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup-pass-rate", type=float, default=0.30)
    ap.add_argument("--measure-batches", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = ModSum(modulus=args.modulus, chain=args.chain)
    drift_targets = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    step_sizes = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    cells = []

    for width in args.widths:
        model, generator = prepare(
            task, width, args.depth, args.population, args.device, args.seed,
            args.warmup_pass_rate,
        )
        base = snapshot(model)
        probe = RLConfig(prompts=args.prompts, group_size=args.group_size, steps=0)
        terms = measure_noise(model, task, probe, generator, args.measure_batches)
        restore(model, base)
        start = pass_rate(model, task, 512, generator).cpu().numpy()

        tau_b = float(terms["tau_b"].mean())
        tau_w = float(terms["tau_w_scaled"].mean())
        signal = float(terms["signal"].mean())
        print(
            f"\nwidth {width}: start pass {start.mean():.3f}  tau_b {tau_b:.3e}  "
            f"tau_w {tau_w:.3e}  G* {1 + np.sqrt(max(tau_w / tau_b, 0)):.2f}  "
            f"Bcrit {(tau_b + tau_w / (args.group_size - 1)) / signal:.1f}"
        )

        for mode, grid in (("drift", drift_targets), ("step_size", step_sizes)):
            gains, extra = [], []
            for value in grid:
                final, realised = run_one(
                    model,
                    task,
                    generator,
                    base,
                    args.prompts,
                    args.group_size,
                    args.steps,
                    drift=value if mode == "drift" else 1e-3,
                    step_size=0.02 if mode == "drift" else value,
                    control=(mode == "drift"),
                )
                gains.append(final - start)
                extra.append(float(realised.mean()))
            means = np.array([g.mean() for g in gains])
            ses = np.array([g.std(ddof=1) / np.sqrt(g.size) for g in gains])
            best = int(np.argmax(means))
            print(
                f"  {mode:9s} grid {grid}\n"
                f"            gain {np.round(means, 4).tolist()}\n"
                f"            best {grid[best]:.1e}  "
                + (f"realised step sizes {np.round(extra, 4).tolist()}" if mode == "drift" else "")
            )
            cells.append(
                {
                    "width": width,
                    "mode": mode,
                    "grid": grid,
                    "gain": means.tolist(),
                    "gain_se": ses.tolist(),
                    "best": grid[best],
                    "realised_step_size": extra,
                    "tau_b": tau_b,
                    "tau_w_scaled": tau_w,
                    "signal": signal,
                    "start_pass_rate": float(start.mean()),
                }
            )

    io.save("b2_transfer", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
