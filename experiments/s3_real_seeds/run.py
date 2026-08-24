"""S3: does the seed divergence of a pretrained model settle, as it does for small transformers?

Several runs start from the same policy -- LoRA adapters begin at zero -- and differ only in the
randomness of their rollouts. Adapters are checkpointed, and afterwards every checkpoint is scored
on one fixed batch of sequences so that the divergence between runs is measured on identical
inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

from caliper.real.tasks import FAMILIES
from caliper.real.train import RealRLConfig, RealRLTrainer, policy_divergence, response_log_probs
from caliper.runtime import io


def adapter_state(model):
    return {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}


def load_adapter(model, state):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))


def build_probe_batch(trainer, corpus, count, key):
    """A fixed batch of prompt+response sequences that every checkpoint is scored on."""
    items = corpus[:count]
    sequences, masks, _, key = trainer.rollout(items, key)
    return sequences, masks, key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--family", default="count_letter")
    ap.add_argument("--corpus-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--probe-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-only", action="store_true",
                    help="reuse checkpointed adapters instead of retraining")
    args = ap.parse_args()

    checkpoints = sorted({1, 2, 5, 10, 20, 30, args.steps})
    checkpoints = [c for c in checkpoints if c <= args.steps]
    store = Path("runs/s3_adapters")
    store.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    corpus = FAMILIES[args.family].sample(args.corpus_size, rng)

    config = RealRLConfig(
        group_size=args.group_size,
        prompts=args.prompts,
        blocks=8,
        max_tokens=args.max_tokens,
        lora_layers=args.lora_layers,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    # one fixed batch of sequences, produced by the untouched base policy
    model, tokenizer = load(args.model)
    reference = RealRLTrainer(model, tokenizer, config)
    probe_sequences, probe_masks, _ = build_probe_batch(
        reference, corpus, args.probe_prompts, mx.random.key(12345)
    )
    initial_adapter = adapter_state(reference.model)
    del reference, model

    pass_history = {}
    if args.score_only:
        previous = [
            r for r in io.load_all("s3_real_seeds") if "pass_history" in r["result"]
        ]
        if previous:
            pass_history = {int(k): v for k, v in previous[-1]["result"]["pass_history"].items()}
    for seed in ([] if args.score_only else range(args.seeds)):
        model, tokenizer = load(args.model)
        trainer = RealRLTrainer(model, tokenizer, config)
        load_adapter(trainer.model, initial_adapter)
        key = mx.random.key(1000 + seed)
        run_rng = np.random.default_rng(args.seed + 7919 * (seed + 1))
        rates = []
        for t in range(args.steps):
            info, key = trainer.step(corpus, run_rng, key)
            rates.append(info["pass_rate"])
            if t + 1 in checkpoints:
                np.savez(store / f"seed{seed}_step{t + 1}.npz", **adapter_state(trainer.model))
        pass_history[seed] = rates
        print(f"seed {seed}: pass rate {np.mean(rates[:5]):.3f} -> {np.mean(rates[-5:]):.3f}")
        del trainer, model

    # score every checkpoint on the same fixed batch
    model, tokenizer = load(args.model)
    scorer = RealRLTrainer(model, tokenizer, config)
    rows = []
    print(f"\n{'step':>6s} {'pairwise KL':>13s} {'pass spread':>13s} {'mean pass':>11s}")
    for step in checkpoints:
        logps = []
        for seed in range(args.seeds):
            state = dict(np.load(store / f"seed{seed}_step{step}.npz"))
            load_adapter(scorer.model, state)
            per_prompt = [
                response_log_probs(scorer.model, seq) for seq in probe_sequences
            ]
            logps.append(per_prompt)
        divergences = []
        for a in range(args.seeds):
            for b in range(args.seeds):
                if a == b:
                    continue
                divergences.append(
                    float(
                        np.mean(
                            [
                                policy_divergence(logps[a][i], logps[b][i], probe_masks[i])
                                for i in range(len(probe_sequences))
                            ]
                        )
                    )
                )
        if pass_history:
            window = [
                np.mean(pass_history[s][max(0, step - 5) : step]) for s in range(args.seeds)
            ]
        else:
            window = [float("nan")]
        rows.append(
            {
                "step": step,
                "pairwise_kl": float(np.mean(divergences)),
                "pairwise_kl_se": float(np.std(divergences, ddof=1) / np.sqrt(len(divergences))),
                "pass_mean": float(np.mean(window)),
                "pass_spread": float(np.std(window, ddof=1)) if len(window) > 1 else float("nan"),
            }
        )
        print(
            f"{step:6d} {rows[-1]['pairwise_kl']:13.4e} {rows[-1]['pass_spread']:13.4f} "
            f"{rows[-1]['pass_mean']:11.3f}"
        )

    steps = np.array([r["step"] for r in rows], dtype=float)
    kls = np.array([r["pairwise_kl"] for r in rows])
    tail = steps >= steps.max() / 3
    slope = float(np.polyfit(np.log(steps[tail]), np.log(kls[tail]), 1)[0])
    print(f"\ntail slope of KL against steps: {slope:+.2f}  (+1 = accumulating, <=0 = settling)")
    io.save(
        "s3_real_seeds",
        vars(args),
        {"rows": rows, "tail_slope": slope, "pass_history": pass_history},
    )


if __name__ == "__main__":
    main()
