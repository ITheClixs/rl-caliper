"""B1: the group-size law on transformers trained from scratch.

A population of models is warmed up to a common pass rate, the noise terms are measured at that
frozen policy, and then the same starting point is trained at every split of a fixed rollout
budget. Prompt-pool size is the knob that moves the between-prompt noise.
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


def efficiency(group_sizes, tau_b, tau_w_scaled, signal, rollouts, corpus=None):
    """rho(G) with the finite-corpus factor, equations (13) and (7b) of docs/theory.md."""
    g = np.asarray(group_sizes, dtype=float)
    prompts = rollouts / g
    share = 0.0 if corpus is None else (prompts - 1.0) / (corpus - 1.0)
    bcrit = ((1.0 - share) * tau_b + tau_w_scaled / (g - 1.0)) / signal
    return 1.0 / (1.0 + g * bcrit / rollouts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=7)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--rollouts-per-step", type=int, default=256)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--drift-target", type=float, default=1e-3)
    ap.add_argument("--warmup-pass-rate", type=float, default=0.30)
    ap.add_argument("--measure-batches", type=int, default=48)
    ap.add_argument("--optimiser", default="sgd")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = ModSum(modulus=args.modulus, chain=args.chain)
    group_sizes = [g for g in [2, 4, 8, 16, 32, 64] if args.rollouts_per_step % g == 0]
    pools = [args.rollouts_per_step // 2, 192, 512, 4096, None]
    cells = []

    for pool in pools:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        config = ModelConfig(
            vocab=task.vocab,
            context=task.total_len,
            width=args.width,
            depth=args.depth,
            heads=2,
            population=args.population,
        )
        model = PopulationTransformer(config, generator).to(args.device)
        supervised_warmup(
            model,
            task,
            SFTConfig(steps=800, batch=64, target_pass_rate=args.warmup_pass_rate, eval_every=4),
            generator,
        )
        base = snapshot(model)
        prompt_pool = (
            None if pool is None else make_pool(task, args.population, pool, generator)
        )

        probe = RLConfig(
            prompts=32, group_size=8, pool_size=pool, optimiser=args.optimiser, steps=0, blocks=8
        )
        terms = measure_noise(
            model, task, probe, generator, args.measure_batches, pool=prompt_pool
        )
        restore(model, base)
        start = pass_rate(model, task, 512, generator, prompt_pool).cpu().numpy()

        tau_b = terms["tau_b"].cpu().numpy()
        tau_w_scaled = terms["tau_w_scaled"].cpu().numpy()
        signal = terms["signal"].cpu().numpy()
        pooled = {
            "tau_b": float(tau_b.mean()),
            "tau_w_scaled": float(tau_w_scaled.mean()),
            "signal": float(signal.mean()),
        }
        # G* is read at the reference split P = R/8, where the corpus factor is evaluated
        reference_prompts = args.rollouts_per_step / 8
        share = 0.0 if pool is None else (reference_prompts - 1.0) / (pool - 1.0)
        effective_tau_b = max((1.0 - share) * pooled["tau_b"], 1e-12)
        g_star = 1 + np.sqrt(max(pooled["tau_w_scaled"] / effective_tau_b, 0.0))
        print(
            f"\npool={pool}: tau_b {pooled['tau_b']:.3e}  tau_w {pooled['tau_w_scaled']:.3e}  "
            f"G* {g_star:.2f}  start pass {start.mean():.3f}"
        )

        curve = []
        for g in group_sizes:
            restore(model, base)
            rl = RLConfig(
                blocks=8,
                prompts=args.rollouts_per_step // g,
                group_size=g,
                pool_size=pool,
                steps=args.steps,
                drift_target=args.drift_target,
                initial_step_size=0.02,
                optimiser=args.optimiser,
                instrument_every=10_000,
                eval_every=10_000,
            )
            trainer = RLVRTrainer(model, task, rl, generator, pool=prompt_pool)
            history = trainer.run()
            final = np.array(history["final_pass_rate"])
            curve.append(final - start)
            print(
                f"  G={g:3d} P={rl.prompts:4d}  gain {np.mean(final - start):+.4f} "
                f"+/- {np.std(final - start, ddof=1) / np.sqrt(final.size):.4f}"
            )

        observed = np.array([c.mean() for c in curve])
        predicted = efficiency(
            group_sizes,
            pooled["tau_b"],
            pooled["tau_w_scaled"],
            pooled["signal"],
            args.rollouts_per_step,
            corpus=pool,
        )
        cells.append(
            {
                "pool": pool,
                "group_sizes": group_sizes,
                "observed": observed.tolist(),
                "observed_se": [float(c.std(ddof=1) / np.sqrt(c.size)) for c in curve],
                "predicted_efficiency": predicted.tolist(),
                "g_star": float(g_star),
                "start_pass_rate": start.tolist(),
                **pooled,
            }
        )
        print(f"  predicted efficiency: {np.round(predicted, 3).tolist()}")

    io.save("b1_group_size", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
