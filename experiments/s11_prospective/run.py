"""S11: the prospective test, on a corpus the model is genuinely unsure about.

One anchor run is trained. From it alone, and before any other seed exists, three things are
computed and written to disk: the forecast of the across-seed spread, the one-run curvature
residual, and the accept/reject decision that follows from comparing that residual against a
threshold fixed on the exact grid of p2. Only then are the validation seeds trained.

The anchor is excluded from the measured spread. Its role is to produce the prediction, and a
predictor that is also one of its own targets is not independent of them.
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

from caliper.analysis.uncertainty import resolved_spread_interval, spread_interval
from caliper.real.adjoint import forecast, real_curvature_residual
from caliper.real.probe import sample_group
from caliper.real.tasks import reward as exact_match
from caliper.real.train import RealRLConfig, RealRLTrainer
from caliper.runtime import io


def adapter_state(model):
    return {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}


def load_adapter(model, state):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))


def optimiser_moments(trainer, order):
    table = dict(tree_flatten(trainer.optimiser.state))
    out = {}
    for name in ("m", "v"):
        pieces = []
        for key in order:
            value = table.get(f"{key}.{name}")
            if value is None:
                return None
            pieces.append(np.asarray(value).reshape(-1))
        out[name] = np.concatenate(pieces)
    return out


def evaluate(trainer, items, samples, key):
    rates = []
    for item in items:
        key, sub = mx.random.split(key, 2)
        ids = trainer._chat(item["prompt"])
        responses = sample_group(
            trainer.model, ids, samples, trainer.config.max_tokens,
            trainer.config.temperature, sub, eos_id=trainer.tokenizer.eos_token_id,
        )
        mx.eval(responses)
        rates.append(float(np.mean([
            exact_match(trainer._trim(list(r)), item["answer"]) for r in np.array(responses)
        ])))
    return np.array(rates), key


def train_run(model_name, config, corpus, initial, seed, keep=False):
    model, tokenizer = load(model_name)
    trainer = RealRLTrainer(model, tokenizer, config)
    load_adapter(trainer.model, initial)
    key = mx.random.key(1000 + seed)
    rng = np.random.default_rng(7919 * (seed + 1))
    states = [adapter_state(trainer.model)] if keep else None
    order = list(states[0]) if keep else None
    moments, live, trace = ([] if keep else None), [], []
    for _ in range(config.steps):
        if keep:
            before = optimiser_moments(trainer, order)
            if before is None:
                width = sum(v.size for v in states[0].values())
                before = {"m": np.zeros(width), "v": np.zeros(width)}
            moments.append(before)
        info, key = trainer.step(corpus, rng, key)
        live.append(info["live_share"])
        trace.append(info)
        if keep:
            states.append(adapter_state(trainer.model))
    return trainer, states, moments, float(np.mean(live)), key, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--label", default="qwen2.5-0.5b-mixed")
    ap.add_argument("--prompts-file", default="runs/s10_difficulty/qwen2.5-0.5b-mixed.json")
    ap.add_argument("--eval-prompts", type=int, default=64)
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=12, help="validation seeds, excluding the anchor")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--optimiser", default="sgd", choices=("sgd", "adam"))
    ap.add_argument("--variance-prompts", type=int, default=32)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--success-band", nargs=2, type=float, default=[2 / 3, 1.5],
                    help="the ratio range declared a success, recorded before the seeds run")
    ap.add_argument("--cross-fit", action="store_true",
                    help="debias the injected term with two independent metric gradients")
    ap.add_argument("--checkpoints", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="runs/s11_prospective")
    args = ap.parse_args()

    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    pool = json.loads(Path(args.prompts_file).read_text())
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    held_out = [pool[i] for i in order[: args.eval_prompts]]
    corpus = [pool[i] for i in order[args.eval_prompts :]]
    print(f"{len(corpus)} training prompts, {len(held_out)} held out, all genuinely mixed "
          f"at the base policy", flush=True)

    config = RealRLConfig(
        group_size=args.group_size, prompts=args.prompts, blocks=8,
        max_tokens=args.max_tokens, lora_layers=args.lora_layers, steps=args.steps,
        learning_rate=args.learning_rate, optimiser=args.optimiser, seed=args.seed,
    )
    model, tokenizer = load(args.model)
    reference = RealRLTrainer(model, tokenizer, config)
    initial = adapter_state(reference.model)
    base, _ = evaluate(reference, held_out, args.eval_samples, mx.random.key(555))
    print(f"base held-out pass rate {base.mean():.4f}", flush=True)
    del reference, model

    # ---- the anchor, and everything decided from it alone ----
    started = time.time()
    trainer, states, moments, live, key, trace = train_run(
        args.model, config, corpus, initial, 0, keep=True
    )
    print("  update  live  P_eff   E[p(1-p)]   |g|", flush=True)
    for t, info in enumerate(trace):
        print(f"  {t:6d}  {info['live_share']:.2f}  {info['effective_prompts']:5.1f}  "
              f"{info['reward_variance']:9.4f}  {info['gradient_norm']:.3e}", flush=True)
    print(f"anchor trained in {(time.time() - started) / 60:.1f} min, "
          f"mean live share {live:.3f}", flush=True)

    prediction = forecast(
        trainer, states, corpus, held_out, args.learning_rate, args.prompts,
        mx.random.key(4242), seed=args.seed, transport=False,
        variance_prompts=args.variance_prompts, check_agreement=False,
        optimiser_states=(moments if args.optimiser == "adam" else None),
        cross_fit=args.cross_fit,
    )
    lo, hi = (float(np.sqrt(max(v, 0.0))) for v in prediction.split_variance)
    print(f"forecast {prediction.std:.5f}  half-samples {lo:.5f} / {hi:.5f}", flush=True)

    picks = np.linspace(0, len(states) - 2, args.checkpoints).astype(int)
    residuals = []
    for i, t in enumerate(picks):
        load_adapter(trainer.model, states[t])
        value, radius = real_curvature_residual(
            trainer, states[t], corpus, key, args.learning_rate,
            seed=args.seed + 313 * (i + 1), directions=2,
        )
        residuals.append(value)
        print(f"  update {t:3d}: residual {value:.4f}"
              f"{'' if radius else '  (collapsed)'}", flush=True)
    graded = [r for r in residuals if r == r]
    score = float(np.mean(graded)) if graded else float("nan")
    accepted = bool(graded and score <= args.threshold)
    print(f"residual {score:.4f} against threshold {args.threshold:.4f}: "
          f"{'ACCEPT' if accepted else 'REFUSE'}", flush=True)

    frozen = {
        "model": args.model, "prompts_file": args.prompts_file,
        "optimiser": args.optimiser, "learning_rate": args.learning_rate,
        "steps": args.steps, "prompts": args.prompts, "group_size": args.group_size,
        "variance_prompts": args.variance_prompts,
        "predicted_std": prediction.std,
        "predicted_half_samples": [lo, hi],
        "curvature_residual": score,
        "threshold": args.threshold,
        "decision": "accept" if accepted else "refuse",
        "anchor_live_share": live,
        "anchor_trace": trace,
        "validation_seeds": args.seeds,
        "success_band": list(args.success_band),
        "frozen_at": time.time(),
    }
    frozen_dir = store / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / f"{args.label}.json").write_text(json.dumps(frozen, indent=2))
    print(f"\nfrozen to {frozen_dir / (args.label + '.json')}; "
          f"training {args.seeds} validation seeds now\n", flush=True)
    del trainer

    # ---- only now, the seeds the prediction will be judged against ----
    scores, per_prompt = [], []
    for s in range(1, args.seeds + 1):
        started = time.time()
        run, _, _, seed_live, _, _ = train_run(args.model, config, corpus, initial, s)
        rates, _ = evaluate(run, held_out, args.eval_samples, mx.random.key(555 + 101 * s))
        scores.append(float(rates.mean()))
        per_prompt.append(rates.tolist())
        print(f"seed {s:2d}: held out {rates.mean():.4f}  live {seed_live:.2f}  "
              f"({(time.time() - started) / 60:.1f} min)", flush=True)
        del run

    scores = np.array(scores)
    observed = spread_interval(scores, n_boot=4000, seed=args.seed)
    resolved = resolved_spread_interval(
        np.array(per_prompt), args.eval_samples, n_boot=4000, seed=args.seed
    )
    ratio = prediction.std / max(resolved["resolved"], 1e-12)
    print(f"\nmeasured spread {observed['std']:.5f}, seed part {resolved['resolved']:.5f} "
          f"[{resolved['lo']:.5f}, {resolved['hi']:.5f}] over {args.seeds} seeds")
    print(f"forecast {prediction.std:.5f}  ratio {ratio:.2f}x  "
          f"(the frozen decision was {frozen['decision']})")

    io.save("s11_prospective", vars(args), {
        "label": args.label, **frozen,
        "base_pass_rate": float(base.mean()),
        "held_out_scores": scores.tolist(),
        "observed_spread": observed,
        "resolved_std": resolved["resolved"],
        "resolved_interval": {"lo": resolved["lo"], "hi": resolved["hi"]},
        "binomial_variance": resolved["binomial"] ** 2,
        "ratio": float(ratio),
        "eval_key_per_seed": True,
    })


if __name__ == "__main__":
    main()
