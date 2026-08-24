"""S5: how long does a perturbation between two runs survive?

The stationary divergence of Proposition 4 is the diffusive drift injected per update times the
number of updates a perturbation survives. The second factor is not visible in a sweep that
checkpoints sparsely, because the approach to the stationary level happens over the first few tens
of updates. This measures the divergence after every update, so that the timescale can be fitted,
and then checks the product against the level the run settles at.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch
from scipy.optimize import curve_fit

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
    pass_rate,
    restore,
    snapshot,
    supervised_warmup,
)
from caliper.runtime import io


def approach(step, level, tau):
    """Variance of an AR(1) process started from zero separation."""
    return level * (1.0 - np.exp(-2.0 * step / tau))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--corpus", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompt-counts", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--drift-targets", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--warmup-pass-rate", type=float, default=0.15)
    ap.add_argument("--probe-prompts", type=int, default=192)
    ap.add_argument("--measure-batches", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = ModSum(modulus=7, chain=4)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    config = ModelConfig(
        vocab=task.vocab, context=task.total_len, width=args.width, depth=2, heads=2,
        population=args.population,
    )
    model = PopulationTransformer(config, generator).to(args.device)
    supervised_warmup(
        model, task,
        SFTConfig(steps=2500, batch=64, target_pass_rate=args.warmup_pass_rate, eval_every=4),
        generator,
    )
    clone_member(model)
    base = snapshot(model)
    pool = make_pool(task, args.population, args.corpus, generator)
    probe_prompts = draw_prompts(task, pool, args.population, args.probe_prompts, generator)[0]

    cells = []
    print(
        f"{'P':>4s} {'D*':>7s} {'tau_c':>7s} {'D_noise':>10s} {'tau*D_noise':>12s} "
        f"{'settled KL':>11s} {'ratio':>6s} {'pass':>6s}"
    )
    for prompts, drift in itertools.product(args.prompt_counts, args.drift_targets):
        restore(model, base)
        probe = RLConfig(prompts=32, group_size=args.group_size, blocks=8, steps=0)
        terms = measure_noise(model, task, probe, generator, args.measure_batches, pool=pool)
        restore(model, base)
        critical = float(
            (terms["tau_b"].mean() + terms["tau_w_scaled"].mean() / (args.group_size - 1))
            / terms["signal"].mean()
        )
        noise_fraction = critical / (prompts + critical)

        rl = RLConfig(
            prompts=prompts, group_size=args.group_size, steps=args.steps,
            drift_target=drift, initial_step_size=0.02, blocks=8,
            instrument_every=10_000, eval_every=10_000,
        )
        trainer = RLVRTrainer(model, task, rl, generator, pool=pool)
        steps, kls, drifts = [], [], []
        for t in range(args.steps):
            info = trainer.step(instrument=False)
            drifts.append(float(np.mean(info["drift"])))
            steps.append(t + 1)
            kls.append(pairwise_policy_divergence(model, task, probe_prompts)["pairwise_kl"])
        final_pass = float(pass_rate(model, task, 2048, generator, pool).mean())

        s = np.array(steps, dtype=float)
        k = np.array(kls)
        (level, tau), covariance = curve_fit(
            approach, s, k, p0=[k[len(k) // 2 :].mean(), 5.0], maxfev=40000,
            bounds=([0.0, 0.2], [np.inf, 1e3]),
        )
        injected = float(np.mean(drifts)) * noise_fraction
        cells.append(
            {
                "prompts": prompts, "drift_target": drift, "critical_batch": critical,
                "noise_fraction": noise_fraction, "tau_c": float(tau),
                "tau_c_se": float(np.sqrt(covariance[1, 1])), "level": float(level),
                "diffusive_drift": injected, "predicted": float(tau * injected),
                "steps": steps, "pairwise_kl": kls, "drift": drifts,
                "final_pass_rate": final_pass,
            }
        )
        print(
            f"{prompts:4d} {drift:7.0e} {tau:7.1f} {injected:10.3e} {tau * injected:12.3e} "
            f"{level:11.3e} {level / (tau * injected):6.2f} {final_pass:6.3f}"
        )

    predicted = np.array([c["predicted"] for c in cells])
    measured = np.array([c["level"] for c in cells])
    slope, intercept = np.polyfit(np.log(predicted), np.log(measured), 1)
    resid = np.log(measured) - np.polyval([slope, intercept], np.log(predicted))
    r2 = 1 - (resid**2).sum() / ((np.log(measured) - np.log(measured).mean()) ** 2).sum()
    print(f"\nsettled KL against tau_c * D_noise: slope {slope:.2f}, R^2 {r2:.3f}, n={len(cells)}")
    io.save("s5_contraction_time", vars(args), {"cells": cells, "slope": slope, "r2": r2})


if __name__ == "__main__":
    main()
