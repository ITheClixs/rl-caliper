"""S7: the forecast of Theorem 2, made on a pretrained model.

One run is trained with plain gradient ascent -- the update the propagation theory is written for
-- storing its adapters after every update. The metric gradient is then carried back along those
states to forecast the across-seed standard deviation of the held-out pass rate. Only afterwards
are the remaining seeds trained and their spread measured.

Gradient ascent rather than Adam is deliberate. Section p5 shows the lifted Adam recursion is
unreliable where the pass rate saturates, and a forecast we cannot defend is worth less than a
smaller claim we can.
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
from caliper.real.adjoint import RealForecast, forecast
from caliper.real.generate import sample_group
from caliper.real.tasks import FAMILIES
from caliper.real.tasks import reward as exact_match
from caliper.real.train import RealRLConfig, RealRLTrainer
from caliper.runtime import io


def adapter_state(model):
    return {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}


def load_adapter(model, state):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))


def optimiser_moments(trainer, order):
    """Adam's m and v, flattened in the same order as the adapters themselves.

    MLX stores them under `<parameter path>.m` and `.v`, so the two are gathered by walking the
    adapter keys rather than the optimiser tree, which keeps the vectors aligned with the
    gradient and the metric direction.
    """
    table = dict(tree_flatten(trainer.optimiser.state))
    moments = {}
    for name in ("m", "v"):
        pieces = []
        for key in order:
            value = table.get(f"{key}.{name}")
            if value is None:
                return None
            pieces.append(np.asarray(value).reshape(-1))
        moments[name] = np.concatenate(pieces)
    return moments


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


def train_run(model_name, config, corpus, initial, seed, store=None):
    model, tokenizer = load(model_name)
    trainer = RealRLTrainer(model, tokenizer, config)
    load_adapter(trainer.model, initial)
    key = mx.random.key(1000 + seed)
    rng = np.random.default_rng(7919 * (seed + 1))
    states = [adapter_state(trainer.model)] if store is not None else None
    order = list(states[0]) if store is not None else None
    # the moments as they stood *before* each update, which is where the sensitivity is taken
    moments = [] if store is not None else None
    rates = []
    for _ in range(config.steps):
        if store is not None:
            before = optimiser_moments(trainer, order)
            moments.append(
                before
                if before is not None
                else {"m": np.zeros(flatten_like(states[0])), "v": np.zeros(flatten_like(states[0]))}
            )
        info, key = trainer.step(corpus, rng, key)
        rates.append(info["pass_rate"])
        if store is not None:
            states.append(adapter_state(trainer.model))
    return trainer, states, rates, key, moments


def flatten_like(state):
    return sum(np.asarray(v).size for v in state.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--label", default="qwen2.5-0.5b")
    ap.add_argument("--family", default="count_letter")
    ap.add_argument("--corpus-size", type=int, default=256)
    ap.add_argument("--eval-prompts", type=int, default=48)
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=1e-3)
    ap.add_argument("--adjoint-batches", type=int, default=2)
    ap.add_argument("--skip-forecast", action="store_true",
                    help="train the seeds and record their traces without the backward pass")
    ap.add_argument("--variance-prompts", type=int, default=None,
                    help="prompts used to estimate the injected term (default: the batch size)")
    ap.add_argument("--no-transport", action="store_true",
                    help="hold the adjoint at grad M: the cheap form, one pass per update")
    ap.add_argument("--forecast-repeats", type=int, default=2,
                    help="independent forecasts from one run, to see how much the estimate moves")
    ap.add_argument("--optimiser", default="sgd", choices=("sgd", "adam"),
                    help="plain ascent is the update the recursion is written for; adam is what "
                         "a practitioner uses, and carries the noise through its own state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="runs/s7_forecast")
    args = ap.parse_args()

    store = Path(args.store) / args.label
    store.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    corpus = FAMILIES[args.family].sample(args.corpus_size, rng)
    held_out = FAMILIES[args.family].sample(args.eval_prompts, np.random.default_rng(9001))

    config = RealRLConfig(
        group_size=args.group_size, prompts=args.prompts, blocks=8,
        max_tokens=args.max_tokens, lora_layers=args.lora_layers, steps=args.steps,
        learning_rate=args.learning_rate, optimiser=args.optimiser, seed=args.seed,
    )

    model, tokenizer = load(args.model)
    reference = RealRLTrainer(model, tokenizer, config)
    initial = adapter_state(reference.model)
    base_rates, _ = evaluate(reference, held_out, args.eval_samples, mx.random.key(555))
    print(f"base held-out pass rate {base_rates.mean():.4f}", flush=True)
    del reference, model

    # (1) one run, kept in full
    started = time.time()
    trainer, states, rates, key, moments = train_run(
        args.model, config, corpus, initial, 0, store=store
    )
    # every run is scored with its own evaluation randomness. Sharing one key across seeds would
    # correlate the evaluation noise between them, and the binomial term subtracted below assumes
    # that noise is independent across runs; sharing it would make the subtraction over-correct.
    first_rates, _ = evaluate(trainer, held_out, args.eval_samples, mx.random.key(555 + 0))
    print(f"seed 0: train {np.mean(rates[:3]):.3f} -> {np.mean(rates[-3:]):.3f} | "
          f"held out {first_rates.mean():.4f} | {(time.time() - started) / 60:.1f} min", flush=True)

    # (2) the forecast, from that run alone. Repeating it with independent batches says how
    # much of what it reports is the run and how much is the estimate.
    repeats = []
    for repeat in range(0 if args.skip_forecast else max(args.forecast_repeats, 1)):
        started = time.time()
        repeats.append(forecast(
            trainer, states, corpus, held_out, args.learning_rate, args.prompts,
            mx.random.key(4242 + 31 * repeat), seed=args.seed + 991 * repeat,
            epsilon=args.epsilon, batches=args.adjoint_batches,
            transport=not args.no_transport,
            variance_prompts=args.variance_prompts,
            optimiser_states=(moments if config.optimiser == "adam" else None),
            # the diagnostic costs a second backward pass, and has nothing to check without one
            check_agreement=(repeat == 0 and not args.no_transport),
        ))
        print(f"forecast {repeat}: sd {repeats[-1].std:.5f} "
              f"(variance {repeats[-1].variance:.3e}) in {(time.time() - started) / 60:.1f} min",
              flush=True)
    if not repeats:
        prediction = RealForecast(
            variance=0.0, std=0.0, kernel=[], adjoint_norm=[],
            metric_value=float(first_rates.mean()), jvp_agreement=[], live_share=[],
            second_moment_floor=[], split_variance=(float("nan"), float("nan")),
            init_checksum=float("nan"),
        )
        repeats = [prediction]
    prediction = repeats[0]
    spread_of_forecasts = float(np.std([r.std for r in repeats], ddof=1)) if len(repeats) > 1 \
        else float("nan")
    print(f"forecast: sd {prediction.std:.5f}, and across {len(repeats)} independent "
          f"estimates from the same run the sd of that number is {spread_of_forecasts:.5f}",
          flush=True)
    lo, hi = prediction.split_variance
    print(f"two half-sample estimates of the same forecast: "
          f"{np.sqrt(max(lo, 0)):.5f} and {np.sqrt(max(hi, 0)):.5f}", flush=True)
    print(f"kernel {np.round(prediction.kernel, 8).tolist()}", flush=True)
    print(f"prompts still sampling more than one answer, per update: "
          f"{np.round(prediction.live_share, 2).tolist()}", flush=True)
    if prediction.jvp_agreement:
        agree = np.array(prediction.jvp_agreement)
        print(f"Hessian-vector product agreement across independent draws: "
              f"median {np.nanmedian(agree):.3f}, min {np.nanmin(agree):.3f}", flush=True)
    (store / "forecast.json").write_text(json.dumps({
        "variance": prediction.variance, "std": prediction.std,
        "kernel": prediction.kernel, "adjoint_norm": prediction.adjoint_norm,
        "jvp_agreement": prediction.jvp_agreement, "live_share": prediction.live_share,
        "split_variance": list(prediction.split_variance),
            "init_checksum": prediction.init_checksum,
            "split_variance": list(prediction.split_variance),
    }))
    del trainer

    # (3) only now, the other seeds
    scores = [float(first_rates.mean())]
    per_prompt = [first_rates.tolist()]
    traces = [rates]
    for seed in range(1, args.seeds):
        started = time.time()
        trainer, _, rates, _, _ = train_run(args.model, config, corpus, initial, seed)
        held, _ = evaluate(trainer, held_out, args.eval_samples, mx.random.key(555 + 101 * seed))
        scores.append(float(held.mean()))
        per_prompt.append(held.tolist())
        traces.append(rates)
        print(f"seed {seed}: train {np.mean(rates[:3]):.3f} -> {np.mean(rates[-3:]):.3f} | "
              f"held out {held.mean():.4f} | {(time.time() - started) / 60:.1f} min", flush=True)
        del trainer

    if len(scores) < 2:
        print("\nonly one seed: no spread to compare the forecast against")
        io.save("s7_real_forecast", vars(args), {
            "label": args.label,
            "transport": not args.no_transport,
            "predicted_std": prediction.std,
            "predicted_variance": prediction.variance,
            "kernel": prediction.kernel,
            "adjoint_norm": prediction.adjoint_norm,
            "jvp_agreement": prediction.jvp_agreement,
            "live_share": prediction.live_share,
            "init_checksum": prediction.init_checksum,
            "split_variance": list(prediction.split_variance),
            "held_out_scores": scores,
            "base_pass_rate": float(base_rates.mean()),
            "eval_key_per_seed": True,
        })
        return
    observed = spread_interval(np.array(scores), n_boot=4000, seed=args.seed)
    interval = resolved_spread_interval(
        np.array(per_prompt), args.eval_samples, n_boot=4000, seed=args.seed
    )
    binomial = interval["binomial"] ** 2
    resolved = interval["resolved"]
    print(f"\nheld-out score {observed['mean']:.4f}")
    print(f"observed sd {observed['std']:.5f} [{observed['lo']:.5f}, {observed['hi']:.5f}]")
    print(f"binomial component sd {interval['binomial']:.5f}; resolved seed sd {resolved:.5f} "
          f"[{interval['lo']:.5f}, {interval['hi']:.5f}]")
    print(f"bootstrap draws landing at zero: {interval['at_zero']:.0%}")
    print(f"forecast {prediction.std:.5f}  ratio to resolved "
          f"{prediction.std / max(resolved, 1e-12):.2f}x")

    io.save("s7_real_forecast", vars(args), {
        "label": args.label,
        "transport": not args.no_transport,
        "predicted_std": prediction.std,
        "predicted_variance": prediction.variance,
        "kernel": prediction.kernel,
        "adjoint_norm": prediction.adjoint_norm,
        "jvp_agreement": prediction.jvp_agreement,
        "live_share": prediction.live_share,
            "init_checksum": prediction.init_checksum,
            "split_variance": list(prediction.split_variance),
        "repeat_stds": [r.std for r in repeats],
        "forecast_estimation_sd": spread_of_forecasts,
        "held_out_scores": scores,
        "observed_spread": observed,
        "binomial_variance": binomial,
        "resolved_std": resolved,
        "resolved_interval": interval,
        "per_prompt": per_prompt,  # kept so the interval can be recomputed without retraining
        "train_traces": traces,
        "eval_samples": args.eval_samples,
        "base_pass_rate": float(base_rates.mean()),
            "eval_key_per_seed": True,
    })


if __name__ == "__main__":
    main()
