"""B4: does the measured recipe beat a tuned default on a setting neither was tuned on?

Two recipes are fixed on a tuning setting and then carried, unchanged, to a held-out setting with a
larger rollout budget, a larger corpus and a different seed.

  default   a common group size, and the learning rate that was best on the tuning setting
  measured  the group size the probe reports on the held-out setting, and the step size the
            controller derives from a drift target that was best on the tuning setting

The default gets the more generous treatment: its learning rate is chosen by a sweep, while the
measured recipe transfers a single scalar and reads the rest off the run it is about to do.
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


def prepare(task, args, seed):
    generator = torch.Generator(device=args.device).manual_seed(seed)
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
    return model, generator


def train(model, task, generator, base, pool, prompts, group_size, steps, start, **kwargs):
    restore(model, base)
    config = RLConfig(
        prompts=prompts,
        group_size=group_size,
        steps=steps,
        blocks=8,
        instrument_every=10_000,
        eval_every=10_000,
        **kwargs,
    )
    trainer = RLVRTrainer(model, task, config, generator, pool=pool)
    history = trainer.run()
    return np.array(history["final_pass_rate"]) - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--warmup-pass-rate", type=float, default=0.30)
    ap.add_argument("--tune-rollouts", type=int, default=256)
    ap.add_argument("--tune-corpus", type=int, default=512)
    ap.add_argument("--held-out-rollouts", type=int, default=1024)
    ap.add_argument("--held-out-corpus", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--default-group-size", type=int, default=16)
    ap.add_argument("--measure-batches", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = ModSum(modulus=7, chain=4)
    step_grid = [0.01 * 1.8**i for i in range(6)]
    drift_grid = [3e-4 * 2.2**i for i in range(5)]

    # ---- tuning setting -------------------------------------------------------------------
    model, generator = prepare(task, args, args.seed)
    base = snapshot(model)
    pool = make_pool(task, args.population, args.tune_corpus, generator)
    start = pass_rate(model, task, 512, generator, pool).cpu().numpy()
    prompts = args.tune_rollouts // args.default_group_size

    step_gains = [
        train(model, task, generator, base, pool, prompts, args.default_group_size,
              args.steps, start, initial_step_size=value, control=False).mean()
        for value in step_grid
    ]
    best_step = step_grid[int(np.argmax(step_gains))]

    drift_gains = [
        train(model, task, generator, base, pool, prompts, args.default_group_size,
              args.steps, start, drift_target=value, initial_step_size=0.02,
              control=True).mean()
        for value in drift_grid
    ]
    best_drift = drift_grid[int(np.argmax(drift_gains))]
    print(f"tuned on R={args.tune_rollouts}, corpus {args.tune_corpus}: "
          f"best step size {best_step:.4f}, best drift target {best_drift:.2e}")

    # ---- held-out setting -----------------------------------------------------------------
    model, generator = prepare(task, args, args.seed + 101)
    base = snapshot(model)
    pool = make_pool(task, args.population, args.held_out_corpus, generator)
    start = pass_rate(model, task, 512, generator, pool).cpu().numpy()

    probe = RLConfig(prompts=32, group_size=8, blocks=8, steps=0)
    terms = measure_noise(model, task, probe, generator, args.measure_batches, pool=pool)
    restore(model, base)
    tau_b = float(terms["tau_b"].mean())
    tau_w = float(terms["tau_w_scaled"].mean())
    share = (args.held_out_rollouts / 8 - 1) / (args.held_out_corpus - 1)
    g_star = 1 + np.sqrt(tau_w / max((1 - share) * tau_b, 1e-12))
    chosen = max(2, int(2 ** round(np.log2(g_star))))
    print(f"probe on the held-out setting: tau_b {tau_b:.3e}, tau_w {tau_w:.3e}, "
          f"G* {g_star:.2f} -> using G = {chosen}")

    default = train(
        model, task, generator, base, pool,
        args.held_out_rollouts // args.default_group_size, args.default_group_size,
        args.steps, start, initial_step_size=best_step, control=False,
    )
    measured = train(
        model, task, generator, base, pool,
        args.held_out_rollouts // chosen, chosen,
        args.steps, start, drift_target=best_drift, initial_step_size=0.02, control=True,
    )

    def report(name, values):
        se = values.std(ddof=1) / np.sqrt(values.size)
        print(f"  {name:24s} gain {values.mean():+.4f} +/- {se:.4f}")
        return {"mean": float(values.mean()), "se": float(se), "values": values.tolist()}

    print(f"\nheld out at R={args.held_out_rollouts}, corpus {args.held_out_corpus}:")
    rows = {
        "default": report(f"default (G={args.default_group_size}, tuned lr)", default),
        "measured": report(f"measured (G={chosen}, drift target)", measured),
    }
    lift = 100 * (rows["measured"]["mean"] - rows["default"]["mean"]) / rows["default"]["mean"]
    paired = measured - default
    t_stat = paired.mean() / (paired.std(ddof=1) / np.sqrt(paired.size))
    print(f"  measured recipe is {lift:+.1f}% better; paired t = {t_stat:.2f} over "
          f"{paired.size} runs")

    io.save(
        "b4_recipe",
        vars(args),
        {
            "best_step_size": best_step,
            "best_drift": best_drift,
            "tau_b": tau_b,
            "tau_w_scaled": tau_w,
            "g_star": float(g_star),
            "chosen_group_size": chosen,
            "results": rows,
            "lift_percent": float(lift),
            "paired_t": float(t_stat),
            "tuning_step_gains": [float(v) for v in step_gains],
            "tuning_drift_gains": [float(v) for v in drift_gains],
        },
    )


if __name__ == "__main__":
    main()
