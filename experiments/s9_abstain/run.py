"""S9: does the one-run diagnostic refuse the runs whose forecast is wrong?

The threshold is fixed on the exact grid of p2 and used here unchanged. Nothing about the real
model's measured spread enters the decision: the residual is computed from one frozen trajectory,
compared against a number decided before this experiment existed, and the run is either accepted
or refused on that alone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

from caliper.envs.real_tasks import FAMILIES
from caliper.real.adjoint import real_curvature_residual
from caliper.real.train import RealRLConfig, RealRLTrainer
from caliper.runtime import io


def adapter_state(model):
    return {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}


def load_adapter(model, state):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--label", default="qwen2.5-0.5b")
    ap.add_argument("--family", default="count_in_string")
    ap.add_argument("--corpus-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--optimiser", default="sgd", choices=("sgd", "adam"))
    ap.add_argument("--checkpoints", type=int, default=4)
    ap.add_argument("--directions", type=int, default=2)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="runs/s9_abstain")
    args = ap.parse_args()

    Path(args.store).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    corpus = FAMILIES[args.family].sample(args.corpus_size, rng)

    config = RealRLConfig(
        group_size=args.group_size, prompts=args.prompts, blocks=8,
        max_tokens=args.max_tokens, lora_layers=args.lora_layers, steps=args.steps,
        learning_rate=args.learning_rate, optimiser=args.optimiser, seed=args.seed,
    )
    model, tokenizer = load(args.model)
    trainer = RealRLTrainer(model, tokenizer, config)
    initial = adapter_state(trainer.model)

    key = mx.random.key(1000 + args.seed)
    train_rng = np.random.default_rng(7919 * (args.seed + 1))
    states = [adapter_state(trainer.model)]
    started = time.time()
    for _ in range(config.steps):
        _, key = trainer.step(corpus, train_rng, key)
        states.append(adapter_state(trainer.model))
    print(f"anchor trained in {(time.time() - started) / 60:.1f} min", flush=True)

    picks = np.linspace(0, len(states) - 2, args.checkpoints).astype(int)
    residuals, radii = [], []
    for i, t in enumerate(picks):
        load_adapter(trainer.model, states[t])
        residual, radius = real_curvature_residual(
            trainer, states[t], corpus, key, args.learning_rate,
            seed=args.seed + 313 * (i + 1), directions=args.directions,
        )
        residuals.append(residual)
        radii.append(radius)
        print(f"update {t:3d}: residual {residual:.4f}  radius {radius:.3e}", flush=True)

    score = float(np.mean(residuals))
    refused = bool(score > args.threshold)
    print(f"\nmean residual {score:.4f} against a threshold of {args.threshold:.4f}")
    print("VERDICT: " + ("refuse the forecast" if refused else "accept the forecast"))

    io.save("s9_abstain", vars(args), {
        "label": args.label,
        "residual": score,
        "per_checkpoint": residuals,
        "radii": radii,
        "threshold": args.threshold,
        "refused": refused,
        "initial_checksum": float(np.abs(np.concatenate(
            [v.reshape(-1) for v in initial.values()])).sum()),
    })


if __name__ == "__main__":
    main()
