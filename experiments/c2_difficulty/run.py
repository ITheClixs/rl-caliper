"""C2: does the within-prompt noise follow the reward histogram, within one task family?

Corollary 3 predicts tau_w proportional to E[p(1-p)] at a fixed score scale. Comparing across task
families confounds that scale, because families differ in response length and token distribution.
Comparing difficulty buckets inside one family does not.

Prompts are bucketed by a pass rate estimated from a separate, larger sample than the one used in
the measurement, so that the bucket boundaries do not inherit the noise of the quantity being
tested.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
from mlx_lm import load

from caliper.estimators.splits import average
from caliper.real.generate import sample_group
from caliper.real.probe import ProbeConfig, RealNoiseProbe
from caliper.real.tasks import FAMILIES, reward
from caliper.runtime import io


def estimate_pass_rates(model, tokenizer, corpus, samples, max_tokens, seed):
    key = mx.random.key(seed)
    rates = []
    for item in corpus:
        key, subkey = mx.random.split(key, 2)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = sample_group(
            model,
            mx.array(tokenizer.encode(text)),
            samples,
            max_tokens,
            1.0,
            subkey,
            eos_id=tokenizer.eos_token_id,
        )
        mx.eval(out)
        decoded = []
        for row in np.array(out):
            ids = list(row)
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            decoded.append(tokenizer.decode(ids))
        rates.append(float(np.mean([reward(d, item["answer"]) for d in decoded])))
    return np.array(rates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--family", default="count_letter")
    ap.add_argument("--corpus-size", type=int, default=384)
    ap.add_argument("--screen-samples", type=int, default=24)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--buckets", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    rng = np.random.default_rng(args.seed)
    corpus = FAMILIES[args.family].sample(args.corpus_size, rng)

    screened = estimate_pass_rates(
        model, tokenizer, corpus, args.screen_samples, args.max_tokens, args.seed
    )
    order = np.argsort(screened)
    splits = np.array_split(order, args.buckets)
    print(f"screened pass rates: mean {screened.mean():.3f}, "
          f"{(screened == 0).sum()} of {len(screened)} never solved")

    rows = []
    print(f"\n{'bucket':8s} {'screen p':>9s} {'probe p':>9s} {'E[p(1-p)]':>10s} {'tau_w':>11s} "
          f"{'tau_w / E[p(1-p)]':>18s}")
    for index, picks in enumerate(splits):
        subset = [corpus[i] for i in picks]
        config = ProbeConfig(
            group_size=args.group_size,
            prompts=args.prompts,
            blocks=args.blocks,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        fresh_model, fresh_tokenizer = load(args.model)
        probe = RealNoiseProbe(fresh_model, fresh_tokenizer, config)
        estimates, rates = probe.measure(
            subset, args.batches, np.random.default_rng(args.seed + index)
        )
        pooled = average(estimates)
        bernoulli = float(np.mean(rates * (1 - rates)))
        rows.append(
            {
                "bucket": index,
                "screen_pass_rate": float(screened[picks].mean()),
                "probe_pass_rate": float(rates.mean()),
                "bernoulli": bernoulli,
                "tau_w_scaled": pooled.tau_w_scaled,
                "tau_b": pooled.tau_b,
                "signal": pooled.signal,
                "g_star": pooled.optimal_group_size(),
            }
        )
        print(
            f"{index:8d} {screened[picks].mean():9.3f} {rates.mean():9.3f} {bernoulli:10.3f} "
            f"{pooled.tau_w_scaled:11.3e} {pooled.tau_w_scaled / max(bernoulli, 1e-9):18.3e}"
        )

    ratios = np.array([r["tau_w_scaled"] / max(r["bernoulli"], 1e-9) for r in rows])
    print(
        f"\nratio across buckets, relative to the mean: "
        f"{np.round(ratios / ratios.mean(), 3).tolist()}  (Corollary 3 predicts all ones)"
    )
    io.save("c2_difficulty", vars(args), {"rows": rows, "screened": screened.tolist()})


if __name__ == "__main__":
    main()
