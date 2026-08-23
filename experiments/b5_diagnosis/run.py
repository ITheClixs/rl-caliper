"""B5: why did the measured recipe lose on the held-out setting?

The two recipes in B4 differ in two ways at once -- the group size and the rule that sets the step
size -- so the loss cannot be attributed from that comparison alone. This crosses the two factors:
each group size is swept over both a learning-rate grid and a drift-target grid, at the held-out
setting, so that the best achievable performance at each group size can be compared directly and
the transferred values can be located against the local optimum.
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
    ap.add_argument("--rollouts", type=int, default=1024)
    ap.add_argument("--corpus", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--measure-batches", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    task = ModSum(modulus=7, chain=4)
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
        model, task, SFTConfig(steps=2500, batch=64, target_pass_rate=0.30, eval_every=4), generator
    )
    base = snapshot(model)
    pool = make_pool(task, args.population, args.corpus, generator)
    start = pass_rate(model, task, 512, generator, pool).cpu().numpy()

    probe = RLConfig(prompts=32, group_size=8, blocks=8, steps=0)
    terms = measure_noise(model, task, probe, generator, args.measure_batches, pool=pool)
    restore(model, base)
    tau_b = float(terms["tau_b"].mean())
    tau_w = float(terms["tau_w_scaled"].mean())
    signal = float(terms["signal"].mean())
    print(
        f"start pass {start.mean():.3f}  tau_b {tau_b:.3e}  "
        f"tau_w {tau_w:.3e}  signal {signal:.3e}"
    )
    for g in args.group_sizes:
        bcrit = (tau_b + tau_w / (g - 1)) / signal
        prompts = args.rollouts // g
        print(f"  G={g:3d}: Bcrit {bcrit:8.1f} prompts, P={prompts:4d}, "
              f"predicted efficiency {1 / (1 + bcrit / prompts):.3f}")

    step_grid = [0.01 * 1.8**i for i in range(6)]
    drift_grid = [3e-4 * 2.2**i for i in range(5)]
    rows = []

    for group_size in args.group_sizes:
        prompts = args.rollouts // group_size
        for mode, grid in (("step_size", step_grid), ("drift", drift_grid)):
            gains, finals = [], []
            for value in grid:
                restore(model, base)
                cfg = RLConfig(
                    prompts=prompts,
                    group_size=group_size,
                    steps=args.steps,
                    blocks=8,
                    instrument_every=10_000,
                    eval_every=10_000,
                    drift_target=value if mode == "drift" else 1e-3,
                    initial_step_size=0.02 if mode == "drift" else value,
                    control=(mode == "drift"),
                )
                history = RLVRTrainer(model, task, cfg, generator, pool=pool).run()
                final = np.array(history["final_pass_rate"])
                gains.append(float((final - start).mean()))
                finals.append(float(final.mean()))
            best = int(np.argmax(gains))
            rows.append(
                {
                    "group_size": group_size,
                    "mode": mode,
                    "grid": grid,
                    "gain": gains,
                    "final_pass_rate": finals,
                    "best_value": grid[best],
                    "best_gain": gains[best],
                    "at_edge": best in (0, len(grid) - 1),
                }
            )
            print(
                f"G={group_size:3d} {mode:9s} best {grid[best]:.4g} gain {gains[best]:+.4f} "
                f"(final pass {finals[best]:.3f}){' [edge]' if rows[-1]['at_edge'] else ''}"
            )
            print(f"          gains {np.round(gains, 4).tolist()}")

    io.save(
        "b5_diagnosis",
        vars(args),
        {
            "rows": rows,
            "tau_b": tau_b,
            "tau_w_scaled": tau_w,
            "signal": signal,
            "start_pass_rate": float(start.mean()),
        },
    )


if __name__ == "__main__":
    main()
