"""What happens to a reported number when RLVR collapses the response distribution.

Two runs of Qwen2.5-7B on a task the model can already mostly do. Their policies differ; the
question is whether the metric can tell. Uses adapters already on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm import load

from caliper.real.tasks import FAMILIES
from caliper.real.train import RealRLConfig, RealRLTrainer, policy_divergence, response_log_probs
from caliper.runtime import io


def load_adapter(model, state):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--label", default="qwen2.5-7b-count_letter")
    ap.add_argument("--family", default="count_letter")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 2, 5, 10, 15, 20])
    ap.add_argument("--probe-prompts", type=int, default=8)
    ap.add_argument("--store", default="runs/s6_adapters")
    args = ap.parse_args()

    store = Path(args.store) / args.label
    rates = json.loads((store / "eval_rates.json").read_text())
    per_prompt = np.array([rates[str(s)]["held_out"] for s in range(args.seeds)])
    scores = per_prompt.mean(axis=1)
    degenerate = float(np.mean((per_prompt == 0.0) | (per_prompt == 1.0)))
    within = float(np.mean(per_prompt * (1.0 - per_prompt)))

    config = RealRLConfig(group_size=8, prompts=8, blocks=8, max_tokens=20, lora_layers=8)
    corpus = FAMILIES[args.family].sample(64, np.random.default_rng(0))
    model, tokenizer = load(args.model)
    trainer = RealRLTrainer(model, tokenizer, config)
    sequences, masks, _, _ = trainer.rollout(corpus[: args.probe_prompts], mx.random.key(12345))

    rows = []
    for step in args.steps:
        paths = [store / f"seed{s}_step{step}.npz" for s in range(args.seeds)]
        if not all(p.exists() for p in paths):
            continue
        logps = []
        for path in paths:
            load_adapter(trainer.model, dict(np.load(path)))
            logps.append([response_log_probs(trainer.model, seq) for seq in sequences])
        pairs = [
            float(np.mean([
                policy_divergence(logps[a][i], logps[b][i], masks[i])
                for i in range(len(sequences))
            ]))
            for a in range(args.seeds)
            for b in range(a + 1, args.seeds)
        ]
        rows.append({"step": step, "divergence": float(np.mean(pairs))})
        print(f"step {step:3d}: divergence {np.mean(pairs):.3e}", flush=True)

    print(f"\nheld-out scores {np.round(scores, 4).tolist()}")
    print(f"per-prompt rates that are exactly 0 or 1: {degenerate:.1%}")
    print(f"E[p(1-p)] on the held-out set: {within:.5f}")
    print(f"prompts where the runs disagree at all: "
          f"{int((np.abs(per_prompt[0] - per_prompt[1]) > 0).sum())}/{per_prompt.shape[1]}")

    io.save("s6_saturation", vars(args), {
        "label": args.label,
        "divergence": rows,
        "held_out_scores": scores.tolist(),
        "degenerate_fraction": degenerate,
        "within_prompt_variance": within,
        "disagreeing_prompts": int((np.abs(per_prompt[0] - per_prompt[1]) > 0).sum()),
        "prompts": int(per_prompt.shape[1]),
    })


if __name__ == "__main__":
    main()
