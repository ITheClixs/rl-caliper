"""S1: does the divergence between two seeds accumulate, or does it settle?

Population members are given identical parameters and then differ only in the randomness of their
rollouts, which is the variance a practitioner sees on rerunning with a new seed. If the policy
performed a random walk driven by gradient noise, the divergence between members would grow
linearly in the number of updates. This measures whether it does.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from caliper.envs.modsum import ModSum
from caliper.population.model import ModelConfig, PopulationTransformer
from caliper.population.train import (
    RLConfig,
    RLVRTrainer,
    SFTConfig,
    clone_member,
    draw_prompts,
    make_pool,
    measure_noise,
    pairwise_policy_divergence,
    resolved_spread,
    restore,
    snapshot,
    supervised_warmup,
)
from caliper.runtime import io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=7)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--corpus", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompt-counts", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--drift-targets", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--warmup-pass-rate", type=float, default=0.25)
    ap.add_argument("--probe-prompts", type=int, default=128)
    ap.add_argument("--eval-samples", type=int, default=4096)
    ap.add_argument("--measure-batches", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = ModSum(modulus=args.modulus, chain=args.chain)
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
        SFTConfig(steps=2500, batch=64, target_pass_rate=args.warmup_pass_rate, eval_every=4),
        generator,
    )
    clone_member(model)
    base = snapshot(model)
    pool = make_pool(task, args.population, args.corpus, generator)
    probe_prompts = draw_prompts(task, pool, args.population, args.probe_prompts, generator)[0]

    checkpoints = sorted({1, 2, 5, 10, 20, 30, 45, 60, 80, 100, args.steps})
    checkpoints = [c for c in checkpoints if c <= args.steps]
    cells = []

    for prompts, drift in itertools.product(args.prompt_counts, args.drift_targets):
        restore(model, base)
        assert pairwise_policy_divergence(model, task, probe_prompts)["pairwise_kl"] == 0.0
        rl = RLConfig(
            prompts=prompts,
            group_size=args.group_size,
            steps=args.steps,
            drift_target=drift,
            initial_step_size=0.02,
            blocks=8,
            instrument_every=10_000,
            eval_every=10_000,
        )
        trainer = RLVRTrainer(model, task, rl, generator, pool=pool)
        probe = RLConfig(prompts=32, group_size=args.group_size, blocks=8, steps=0)
        terms = measure_noise(model, task, probe, generator, args.measure_batches, pool=pool)
        restore(model, base)
        tau_b = float(terms["tau_b"].mean())
        tau_w = float(terms["tau_w_scaled"].mean())
        signal = float(terms["signal"].mean())
        critical = (tau_b + tau_w / (args.group_size - 1)) / signal
        history = {"step": [], "pairwise_kl": [], "pass_mean": [], "true_spread": [],
                   "observed_spread": [], "eval_spread": [], "drift": []}
        realised = []
        for t in range(args.steps):
            info = trainer.step(instrument=False)
            realised.append(float(np.mean(info["drift"])))
            if t + 1 in checkpoints:
                kl = pairwise_policy_divergence(model, task, probe_prompts)["pairwise_kl"]
                spread = resolved_spread(model, task, generator, pool, args.eval_samples)
                history["step"].append(t + 1)
                history["pairwise_kl"].append(kl)
                history["pass_mean"].append(spread["mean_pass_rate"])
                history["true_spread"].append(spread["true_spread"])
                history["observed_spread"].append(spread["observed_spread"])
                history["eval_spread"].append(spread["evaluation_spread"])
                history["drift"].append(float(np.mean(realised[-10:])))

        steps = np.array(history["step"], dtype=float)
        kls = np.array(history["pairwise_kl"])
        tail = steps >= steps.max() / 3
        slope = float(np.polyfit(np.log(steps[tail]), np.log(kls[tail]), 1)[0])
        cells.append(
            {
                "prompts": prompts,
                "drift_target": drift,
                "critical_batch": critical,
                "noise_fraction": critical / (prompts + critical),
                "tau_b": tau_b,
                "tau_w_scaled": tau_w,
                "realised_drift": float(np.mean(realised)),
                "history": history,
                "tail_slope": slope,
                "plateau_kl": float(np.mean(kls[tail])),
                "final_pass": history["pass_mean"][-1],
            }
        )
        print(
            f"P={prompts:3d} D*={drift:.0e}  Bcrit {critical:5.1f}  "
            f"noise frac {critical / (prompts + critical):.3f}  "
            f"final pass {history['pass_mean'][-1]:.3f}  "
            f"plateau KL {np.mean(kls[tail]):.3e}  tail slope {slope:+.2f}  "
            f"true spread {history['true_spread'][-1]:.4f}"
        )
        print("   step  " + " ".join(f"{s:8.0f}" for s in steps))
        print("   KL    " + " ".join(f"{k:8.2e}" for k in kls))
        print("   pass  " + " ".join(f"{v:8.3f}" for v in history["pass_mean"]))
        print("   sprd  " + " ".join(f"{v:8.4f}" for v in history["true_spread"]))

    io.save("s1_contraction", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
