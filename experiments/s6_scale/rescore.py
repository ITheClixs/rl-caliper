"""Re-score the stored final adapters on a larger held-out set.

The spread across seeds of an estimated pass rate contains a binomial term from the finite
evaluation, which is subtracted; the smaller that term is to begin with, the more of the seed term
survives the subtraction. Training is the expensive part and is already done, so this buys
resolution cheaply.

Each run is scored with its own evaluation randomness, because the subtraction assumes the
evaluation noise is independent between runs. This pass is also what makes the numbers reported
for the seven-billion-parameter run correct on that point: the training loop scored as it went,
before that was fixed, and this re-scores from the stored adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm import load

from caliper.analysis.uncertainty import resolved_spread_interval, spread_interval
from caliper.real.generate import sample_group
from caliper.real.tasks import FAMILIES
from caliper.real.tasks import reward as exact_match
from caliper.real.train import RealRLConfig, RealRLTrainer
from caliper.runtime import io


def load_adapter(model, state):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))


def evaluate(trainer, items, samples, key):
    rates = []
    for item in items:
        key, subkey = mx.random.split(key, 2)
        prompt_ids = trainer._chat(item["prompt"])
        responses = sample_group(
            trainer.model, prompt_ids, samples, trainer.config.max_tokens,
            trainer.config.temperature, subkey, eos_id=trainer.tokenizer.eos_token_id,
        )
        mx.eval(responses)
        rates.append(float(np.mean([
            exact_match(trainer._trim(list(r)), item["answer"]) for r in np.array(responses)
        ])))
    return np.array(rates), key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--label", default="qwen2.5-7b")
    ap.add_argument("--family", default="count_letter")
    ap.add_argument("--eval-prompts", type=int, default=48)
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--store", default="runs/s6_adapters")
    args = ap.parse_args()

    store = Path(args.store) / args.label
    held_out = FAMILIES[args.family].sample(args.eval_prompts, np.random.default_rng(9001))
    config = RealRLConfig(
        group_size=args.group_size, prompts=8, blocks=8, max_tokens=args.max_tokens,
        lora_layers=args.lora_layers, steps=args.step,
    )
    model, tokenizer = load(args.model)
    trainer = RealRLTrainer(model, tokenizer, config)

    base, _ = evaluate(trainer, held_out, args.eval_samples, mx.random.key(777))
    print(f"base held-out pass rate {base.mean():.4f}", flush=True)

    rows = []
    for seed in range(args.seeds):
        path = store / f"seed{seed}_step{args.step}.npz"
        if not path.exists():
            print(f"missing {path}, stopping at {seed} seeds", flush=True)
            break
        load_adapter(trainer.model, dict(np.load(path)))
        # independent evaluation randomness per run, so the binomial subtraction is the right one
        rates, _ = evaluate(trainer, held_out, args.eval_samples, mx.random.key(777 + 101 * seed))
        rows.append(rates.tolist())
        print(f"seed {seed}: held out {rates.mean():.4f}", flush=True)

    per_prompt = np.array(rows)
    scores = per_prompt.mean(axis=1)
    observed = spread_interval(scores, n_boot=4000, seed=0)
    interval = resolved_spread_interval(per_prompt, args.eval_samples, n_boot=4000, seed=0)
    binomial = interval["binomial"] ** 2
    resolved = interval["resolved"]
    print(f"\n{len(scores)} seeds, {args.eval_prompts} x {args.eval_samples} samples each")
    print(f"held-out score {observed['mean']:.4f}")
    print(f"observed sd {observed['std']:.5f} [{observed['lo']:.5f}, {observed['hi']:.5f}]")
    print(f"binomial component sd {interval['binomial']:.5f}; resolved seed sd {resolved:.5f} "
          f"[{interval['lo']:.5f}, {interval['hi']:.5f}]")

    io.save("s6_rescore", vars(args), {
        "label": args.label,
        "held_out_scores": scores.tolist(),
        "per_prompt": per_prompt.tolist(),
        "observed_spread": observed,
        "binomial_variance": binomial,
        "resolved_std": resolved,
        "resolved_interval": interval,
        "base_pass_rate": float(base.mean()),
    })
    (store / "rescore.json").write_text(json.dumps({
        "scores": scores.tolist(), "resolved_std": resolved,
        "binomial_sd": float(np.sqrt(binomial)),
    }))


if __name__ == "__main__":
    main()
