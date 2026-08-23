"""B3: which hyperparameter transfers when the batch size changes?

Drift obeys D = (1/2) eta^2 (Gcal + N/P), so the step size that realises a given drift must grow
with P while the drift itself does not. The coefficient Gcal + N/P is measured directly from a
probe step -- take a step of known size, read the KL -- so the prediction for how the optimal step
size moves with P carries no fitted quantity.
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
    pass_rate,
    restore,
    snapshot,
    supervised_warmup,
)
from caliper.runtime import io


def probe_drift_coefficient(model, task, generator, base, pool, prompts, group_size, step_size,
                            repeats):
    """Return 2 D / eta^2 = Gcal + N/P, measured by taking single steps of known size."""
    values = []
    for _ in range(repeats):
        restore(model, base)
        config = RLConfig(
            prompts=prompts,
            group_size=group_size,
            steps=1,
            initial_step_size=step_size,
            control=False,
            instrument_every=10_000,
            eval_every=10_000,
        )
        trainer = RLVRTrainer(model, task, config, generator, pool=pool)
        info = trainer.step(instrument=False)
        values.append(np.array(info["drift"]))
    drift = np.stack(values).mean(axis=0)
    restore(model, base)
    return 2.0 * drift / step_size**2


def sweep(model, task, generator, base, pool, prompts, group_size, steps, grid, mode, start):
    means, ses, realised, drifts = [], [], [], []
    for value in grid:
        restore(model, base)
        config = RLConfig(
            prompts=prompts,
            group_size=group_size,
            steps=steps,
            drift_target=value if mode == "drift" else 1e-3,
            initial_step_size=0.02 if mode == "drift" else value,
            control=(mode == "drift"),
            instrument_every=10_000,
            eval_every=10_000,
        )
        trainer = RLVRTrainer(model, task, config, generator, pool=pool)
        history = trainer.run()
        gain = np.array(history["final_pass_rate"]) - start
        means.append(float(gain.mean()))
        ses.append(float(gain.std(ddof=1) / np.sqrt(gain.size)))
        realised.append(float(np.mean([c.step_size for c in trainer.controllers])))
        drifts.append(float(np.mean(history["mean_drift"])))
    return means, ses, realised, drifts


def peak(grid, means):
    """Log-quadratic interpolation of the argmax over a geometric grid."""
    x = np.log(np.asarray(grid, dtype=float))
    y = np.asarray(means)
    best = int(np.argmax(y))
    if best in (0, len(y) - 1):
        return float(grid[best]), True
    lo, hi = best - 1, best + 2
    coef = np.polyfit(x[lo:hi], y[lo:hi], 2)
    if coef[0] >= 0:
        return float(grid[best]), True
    return float(np.exp(-coef[1] / (2 * coef[0]))), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=5)
    ap.add_argument("--chain", type=int, default=3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompt-counts", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--corpus", type=int, default=1024)
    ap.add_argument("--probe-repeats", type=int, default=8)
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
        model, task, SFTConfig(steps=1200, batch=64, target_pass_rate=0.30, eval_every=4), generator
    )
    base = snapshot(model)
    pool = make_pool(task, args.population, args.corpus, generator)
    start = pass_rate(model, task, 512, generator, pool).cpu().numpy()
    print(f"start pass rate {start.mean():.3f}")

    step_grid = [0.008 * 1.7**i for i in range(8)]
    drift_grid = [2e-4 * 2.2**i for i in range(7)]
    rows = []

    for prompts in args.prompt_counts:
        coefficient = probe_drift_coefficient(
            model, task, generator, base, pool, prompts, args.group_size, 0.02,
            args.probe_repeats,
        )
        step_means, step_ses, _, step_drifts = sweep(
            model, task, generator, base, pool, prompts, args.group_size, args.steps,
            step_grid, "step_size", start,
        )
        drift_means, drift_ses, realised, drift_drifts = sweep(
            model, task, generator, base, pool, prompts, args.group_size, args.steps,
            drift_grid, "drift", start,
        )
        best_step, step_edge = peak(step_grid, step_means)
        best_drift, drift_edge = peak(drift_grid, drift_means)
        rows.append(
            {
                "prompts": prompts,
                "drift_coefficient": float(coefficient.mean()),
                "step_grid": step_grid,
                "step_gain": step_means,
                "step_gain_se": step_ses,
                "best_step_size": best_step,
                "best_step_at_edge": step_edge,
                "drift_grid": drift_grid,
                "drift_gain": drift_means,
                "drift_gain_se": drift_ses,
                "best_drift": best_drift,
                "best_drift_at_edge": drift_edge,
                "realised_step_size": realised,
                "step_sweep_drift": step_drifts,
                "drift_sweep_drift": drift_drifts,
            }
        )
        at_best = float(np.interp(np.log(best_step), np.log(step_grid), step_drifts))
        print(
            f"P={prompts:3d}  Gcal+N/P = {coefficient.mean():.3e}  "
            f"best eta {best_step:.4f}{' (edge)' if step_edge else ''}  "
            f"best D* {best_drift:.2e}{' (edge)' if drift_edge else ''}  "
            f"drift at best eta {at_best:.2e}"
        )

    reference = rows[0]
    header = f"\n{'P':>4s} {'predicted eta*':>15s} {'measured eta*':>14s}"
    print(header + f" {'ratio':>7s} {'D* / D*_ref':>12s}")
    for row in rows:
        predicted = reference["best_step_size"] * np.sqrt(
            reference["drift_coefficient"] / row["drift_coefficient"]
        )
        row["predicted_step_size"] = float(predicted)
        print(
            f"{row['prompts']:4d} {predicted:15.4f} {row['best_step_size']:14.4f} "
            f"{row['best_step_size'] / predicted:7.2f} "
            f"{row['best_drift'] / reference['best_drift']:12.2f}"
        )

    io.save("b3_batch_transfer", vars(args), {"rows": rows})


if __name__ == "__main__":
    main()
