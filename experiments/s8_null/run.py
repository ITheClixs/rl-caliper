"""S8: is the filtering caused by the learning signal, or would any two runs stay close?

Two conditions, identical in every respect that governs noise. Same task, same warmed-up starting
policy, same prompt draws, same group size, same fixed step size, same number of updates. The only
difference is the reward: in one condition it is the verifier's, in the other it is a fair coin
flip that carries no information about the response that earned it.

Gradient magnitudes are comparable between the two, because the advantage of a count-based
estimator depends on the reward pattern within a group and not on whether that pattern is
deserved. If divergence stays flat only in the condition with a signal, the filtering is a property
of the objective rather than of runs happening to start near each other.

The step size is fixed rather than drift-controlled, since a controller would react to the
difference in signal and the two conditions would no longer be taking comparable steps.
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
    clone_member,
    draw_prompts,
    make_pool,
    pairwise_policy_divergence,
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
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--step-size", type=float, default=0.02)
    ap.add_argument("--warmup-pass-rate", type=float, default=0.20)
    ap.add_argument("--probe-prompts", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = ModSum(modulus=args.modulus, chain=args.chain)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    model = PopulationTransformer(
        ModelConfig(
            vocab=task.vocab, context=task.total_len, width=args.width,
            depth=2, heads=2, population=args.population,
        ),
        generator,
    ).to(args.device)
    supervised_warmup(
        model, task,
        SFTConfig(steps=2500, batch=64, target_pass_rate=args.warmup_pass_rate, eval_every=4),
        generator,
    )
    clone_member(model)
    base = snapshot(model)
    pool = make_pool(task, args.population, args.corpus, generator)
    probe_prompts = draw_prompts(task, pool, args.population, args.probe_prompts, generator)[0]

    checkpoints = sorted({1, 5, 10, 20, 40, 60, args.steps})
    checkpoints = [c for c in checkpoints if c <= args.steps]

    conditions = {}
    for mode in ("verifier", "coin"):
        restore(model, base)
        assert pairwise_policy_divergence(model, task, probe_prompts)["pairwise_kl"] == 0.0
        # both conditions draw from the same generator state, so the rollouts are comparable
        run_generator = torch.Generator(device=args.device).manual_seed(args.seed + 17)
        rl = RLConfig(
            prompts=args.prompts, group_size=args.group_size, steps=args.steps,
            initial_step_size=args.step_size, control=False, reward_mode=mode,
            blocks=8, instrument_every=10_000, eval_every=10_000,
        )
        trainer = RLVRTrainer(model, task, rl, run_generator, pool=pool)
        history = {"step": [], "pairwise_kl": [], "pass_rate": [], "drift": []}
        drifts = []
        print(f"\n{mode}")
        print(f"{'update':>7s} {'divergence':>12s} {'pass rate':>10s} {'drift':>10s}")
        for t in range(args.steps):
            info = trainer.step(instrument=False)
            drifts.append(float(np.mean(info["drift"])))
            if t + 1 in checkpoints:
                kl = pairwise_policy_divergence(model, task, probe_prompts)["pairwise_kl"]
                rate = float(np.mean(info["pass_rate"]))
                history["step"].append(t + 1)
                history["pairwise_kl"].append(kl)
                history["pass_rate"].append(rate)
                history["drift"].append(float(np.mean(drifts[-10:])))
                print(f"{t + 1:7d} {kl:12.3e} {rate:10.3f} "
                      f"{np.mean(drifts[-10:]):10.3e}", flush=True)
        conditions[mode] = history

    growth = {
        mode: h["pairwise_kl"][-1] / max(h["pairwise_kl"][0], 1e-30)
        for mode, h in conditions.items()
    }
    print(f"\ndivergence at {args.steps} updates relative to the first update:")
    for mode, value in growth.items():
        print(f"  {mode:>9s}: {value:6.2f}x")
    print(f"mean drift per update, verifier {np.mean(conditions['verifier']['drift']):.3e} "
          f"vs coin {np.mean(conditions['coin']['drift']):.3e}")

    io.save("s8_null", vars(args), {
        "conditions": conditions,
        "growth": growth,
        "checkpoints": checkpoints,
    })


if __name__ == "__main__":
    main()
