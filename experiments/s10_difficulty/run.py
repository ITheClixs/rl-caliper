"""S10: which prompts is the model genuinely uncertain about?

Every real-model run in this paper loses its noise within a few updates: groups go unanimous, the
count-based advantage is identically zero, and there is nothing left to forecast. A mean pass rate
in the interesting band does not prevent that, because a corpus can average 0.4 by holding half
its prompts at 0 and half at 1, and each group is then unanimous on every draw.

What matters is the per-prompt distribution. This measures it at the base policy, with enough
samples per prompt to tell 0.5 from 0 or 1, and writes out the subset that is genuinely mixed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from caliper.real.probe import sample_group
from caliper.real.tasks import FAMILIES
from caliper.real.tasks import reward as exact_match
from caliper.real.train import RealRLConfig, RealRLTrainer
from caliper.runtime import io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--label", default="qwen2.5-0.5b")
    ap.add_argument("--families", nargs="+", default=sorted(FAMILIES))
    ap.add_argument("--per-family", type=int, default=96)
    ap.add_argument("--samples", type=int, default=32,
                    help="rollouts per prompt; enough to separate 0.5 from 0 or 1")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--low", type=float, default=0.15)
    ap.add_argument("--high", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/s10_difficulty")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    config = RealRLConfig(
        group_size=8, prompts=8, blocks=8, max_tokens=args.max_tokens,
        lora_layers=8, steps=1, learning_rate=0.0, optimiser="sgd",
        temperature=args.temperature, seed=args.seed,
    )
    model, tokenizer = load(args.model)
    trainer = RealRLTrainer(model, tokenizer, config)

    key = mx.random.key(4242 + args.seed)
    kept, surveyed, summary = [], [], []
    for family in args.families:
        rng = np.random.default_rng(args.seed + abs(hash(family)) % 10_000)
        items = FAMILIES[family].sample(args.per_family, rng)
        started = time.time()
        rates = []
        for item in items:
            key, sub = mx.random.split(key, 2)
            ids = trainer._chat(item["prompt"])
            responses = sample_group(
                trainer.model, ids, args.samples, config.max_tokens,
                config.temperature, sub, eos_id=tokenizer.eos_token_id,
            )
            mx.eval(responses)
            rate = float(np.mean([
                exact_match(trainer._trim(list(r)), item["answer"])
                for r in np.array(responses)
            ]))
            rates.append(rate)
            surveyed.append({**item, "family": family, "base_rate": rate})
            if args.low <= rate <= args.high:
                kept.append({**item, "family": family, "base_rate": rate})
        rates = np.array(rates)
        # the quantity that decides whether groups stay mixed
        live = float(np.mean((rates > 0) & (rates < 1)))
        band = float(np.mean((rates >= args.low) & (rates <= args.high)))
        summary.append({
            "family": family, "mean": float(rates.mean()), "live_share": live,
            "in_band": band, "kept": int(((rates >= args.low) & (rates <= args.high)).sum()),
            "rates": rates.tolist(),
        })
        print(f"{family:>16s}  mean {rates.mean():.3f}  never-unanimous {live:5.2f}  "
              f"in [{args.low},{args.high}] {band:5.2f}  ({(time.time()-started)/60:.1f} min)",
              flush=True)

    order = sorted(summary, key=lambda r: -r["in_band"])
    print("\nbest families by the share of prompts the model is genuinely unsure of:")
    for row in order[:4]:
        print(f"  {row['family']:>16s}  {row['in_band']:.2f}  ({row['kept']} prompts)")

    out = Path(args.out) / f"{args.label}-mixed.json"
    out.write_text(json.dumps(kept, indent=2))
    # every surveyed prompt with its measured rate, so a corpus can later be built at a chosen
    # live-group probability rather than only at the one band this run happened to keep
    every = Path(args.out) / f"{args.label}-surveyed.json"
    every.write_text(json.dumps(surveyed, indent=2))
    print(f"\nwrote {len(kept)} mixed prompts to {out}")
    print(f"wrote {len(surveyed)} surveyed prompts with rates to {every}")
    io.save("s10_difficulty", vars(args), {"label": args.label, "families": summary,
                                           "kept": len(kept)})


if __name__ == "__main__":
    main()
