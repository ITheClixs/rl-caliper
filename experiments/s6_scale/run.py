"""S6: seed spread of a reported benchmark number, at the largest scale this machine holds.

Eight runs differ only in the randomness of their rollouts. LoRA adapters start at zero, so they
begin from an identical policy. Two quantities are measured: the divergence between the policies
they reach, on one fixed batch of sequences, and the spread of the score they would report, on one
fixed held-out set that no run trains on.

The reported score is estimated from a finite number of samples, so its across-run spread contains
evaluation noise that has nothing to do with the seed. That component is binomial and known, and
is subtracted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

from caliper.analysis.uncertainty import pairwise_mean_interval_from_matrix, spread_interval
from caliper.real.generate import sample_group
from caliper.real.tasks import FAMILIES
from caliper.real.tasks import reward as exact_match
from caliper.real.train import RealRLConfig, RealRLTrainer, policy_divergence, response_log_probs
from caliper.runtime import io


def adapter_state(model):
    return {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}


def load_adapter(model, state):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))


def evaluate(trainer, items, samples: int, key):
    """Pass rate on a held-out set, and the per-prompt rates the binomial correction needs."""
    rates = []
    for item in items:
        key, subkey = mx.random.split(key, 2)
        prompt_ids = trainer._chat(item["prompt"])
        responses = sample_group(
            trainer.model, prompt_ids, samples, trainer.config.max_tokens,
            trainer.config.temperature, subkey, eos_id=trainer.tokenizer.eos_token_id,
        )
        mx.eval(responses)
        rows = np.array(responses)
        rates.append(
            float(np.mean([exact_match(trainer._trim(list(r)), item["answer"]) for r in rows]))
        )
    return np.array(rates), key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--label", default="qwen2.5-7b")
    ap.add_argument("--family", default="count_letter")
    ap.add_argument("--corpus-size", type=int, default=256)
    ap.add_argument("--eval-prompts", type=int, default=32)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--probe-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="runs/s6_adapters")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    checkpoints = sorted({c for c in (1, 2, 5, 10, 15, args.steps) if c <= args.steps})
    store = Path(args.store) / args.label
    store.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    corpus = FAMILIES[args.family].sample(args.corpus_size, rng)
    held_out = FAMILIES[args.family].sample(args.eval_prompts, np.random.default_rng(9001))

    config = RealRLConfig(
        group_size=args.group_size, prompts=args.prompts, blocks=8,
        max_tokens=args.max_tokens, lora_layers=args.lora_layers, steps=args.steps,
        learning_rate=args.learning_rate, seed=args.seed,
    )

    model, tokenizer = load(args.model)
    reference = RealRLTrainer(model, tokenizer, config)
    probe_sequences, probe_masks, _, _ = reference.rollout(
        corpus[: args.probe_prompts], mx.random.key(12345)
    )
    initial_adapter = adapter_state(reference.model)
    base_rates, _ = evaluate(reference, held_out, args.eval_samples, mx.random.key(555))
    print(f"base model pass rate on the held-out set: {base_rates.mean():.4f}", flush=True)
    del reference, model

    eval_path = store / "eval_rates.json"
    records = json.loads(eval_path.read_text()) if eval_path.exists() else {}

    for seed in ([] if args.score_only else range(args.seeds)):
        started = time.time()
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
        # every run is scored on the same held-out set with the same evaluation randomness
        # own evaluation randomness per run: the binomial term subtracted below assumes the
        # evaluation noise is independent across runs
        held_rates, _ = evaluate(
            trainer, held_out, args.eval_samples, mx.random.key(555 + 101 * seed)
        )
        records[str(seed)] = {"train": rates, "held_out": held_rates.tolist()}
        eval_path.write_text(json.dumps(records))
        print(f"seed {seed}: train {np.mean(rates[:3]):.3f} -> {np.mean(rates[-3:]):.3f} | "
              f"held out {held_rates.mean():.4f} | {(time.time() - started) / 60:.1f} min",
              flush=True)
        del trainer, model

    # spread of the reported number, with the binomial evaluation noise removed
    scores = np.array([np.mean(records[str(s)]["held_out"]) for s in range(args.seeds)])
    per_prompt = np.array([records[str(s)]["held_out"] for s in range(args.seeds)])
    observed = spread_interval(scores, n_boot=4000, seed=args.seed)
    binomial = float(
        np.mean(per_prompt * (1.0 - per_prompt)) / (args.eval_samples * args.eval_prompts)
    )
    resolved = float(np.sqrt(max(observed["std"] ** 2 - binomial, 0.0)))

    # divergence between the policies, on one fixed batch
    model, tokenizer = load(args.model)
    scorer = RealRLTrainer(model, tokenizer, config)
    rows = []
    for step in checkpoints:
        logps = []
        for seed in range(args.seeds):
            load_adapter(scorer.model, dict(np.load(store / f"seed{seed}_step{step}.npz")))
            logps.append([response_log_probs(scorer.model, seq) for seq in probe_sequences])
        matrix = np.zeros((args.seeds, args.seeds))
        for a in range(args.seeds):
            for b in range(a + 1, args.seeds):
                value = float(np.mean([
                    policy_divergence(logps[a][i], logps[b][i], probe_masks[i])
                    for i in range(len(probe_sequences))
                ]))
                matrix[a, b] = matrix[b, a] = value
        interval = pairwise_mean_interval_from_matrix(matrix, n_boot=4000, seed=args.seed)
        rows.append({"step": step, **interval})
        print(f"step {step:3d}: divergence {interval['mean']:.3e} "
              f"[{interval['lo']:.3e}, {interval['hi']:.3e}]", flush=True)

    print(f"\nheld-out score {observed['mean']:.4f}, across-seed sd {observed['std']:.4f} "
          f"[{observed['lo']:.4f}, {observed['hi']:.4f}]")
    print(f"binomial evaluation component sd {np.sqrt(binomial):.4f}; "
          f"resolved seed sd {resolved:.4f}")

    io.save(
        "s6_scale",
        vars(args),
        {
            "label": args.label,
            "divergence": rows,
            "held_out_scores": scores.tolist(),
            "observed_spread": observed,
            "binomial_variance": binomial,
            "resolved_std": resolved,
            "base_pass_rate": float(base_rates.mean()),
            "train_history": {k: v["train"] for k, v in records.items()},
        },
    )


if __name__ == "__main__":
    main()
