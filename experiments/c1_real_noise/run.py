"""C1: the two noise scales measured on a pretrained model.

Corpora are built from short exact-match families whose difficulty differs, and from a mixture of
all of them. Difficulty moves the within-prompt term through the pass-rate histogram; mixing
families moves the between-prompt term by making prompts pull in different directions.
"""

from __future__ import annotations

import argparse

import numpy as np
from mlx_lm import load

from caliper.estimators.splits import average
from caliper.real.probe import ProbeConfig, RealNoiseProbe
from caliper.real.tasks import FAMILIES
from caliper.runtime import io


def build_corpus(name: str, size: int, rng: np.random.Generator) -> list[dict]:
    if name == "mixed":
        families = list(FAMILIES.values())
        items = []
        for index in range(size):
            items.extend(families[index % len(families)].sample(1, rng))
        return items
    return FAMILIES[name].sample(size, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--corpora", nargs="+", default=["count_letter", "nth_word", "add_two", "mixed"])
    ap.add_argument("--corpus-size", type=int, default=256)
    ap.add_argument("--batches", type=int, default=48)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    print(
        f"{'corpus':14s} {'pass':>6s} {'E[p(1-p)]':>10s} {'tau_b':>11s} {'tau_w':>11s} "
        f"{'G*':>6s} {'Bcrit':>9s}"
    )
    for name in args.corpora:
        model, tokenizer = load(args.model)
        rng = np.random.default_rng(args.seed)
        corpus = build_corpus(name, args.corpus_size, rng)
        config = ProbeConfig(
            group_size=args.group_size,
            prompts=args.prompts,
            blocks=args.blocks,
            max_tokens=args.max_tokens,
            lora_layers=args.lora_layers,
            seed=args.seed,
        )
        probe = RealNoiseProbe(model, tokenizer, config)
        estimates, pass_rates = probe.measure(corpus, args.batches, rng)
        pooled = average(estimates)
        bernoulli = float(np.mean(pass_rates * (1 - pass_rates)))
        rows.append(
            {
                "corpus": name,
                "mean_pass_rate": float(pass_rates.mean()),
                "bernoulli": bernoulli,
                "tau_b": pooled.tau_b,
                "tau_w_scaled": pooled.tau_w_scaled,
                "signal": pooled.signal,
                "g_star": pooled.optimal_group_size(),
                "critical_batch": pooled.critical_batch(),
                "pass_rates": pass_rates.tolist(),
                "per_batch_tau_b": [e.tau_b for e in estimates],
                "per_batch_tau_w_scaled": [e.tau_w_scaled for e in estimates],
                "per_batch_signal": [e.signal for e in estimates],
            }
        )
        print(
            f"{name:14s} {pass_rates.mean():6.3f} {bernoulli:10.3f} {pooled.tau_b:11.3e} "
            f"{pooled.tau_w_scaled:11.3e} {pooled.optimal_group_size():6.2f} "
            f"{pooled.critical_batch():9.1f}"
        )

    ratio = np.array([r["tau_w_scaled"] / r["bernoulli"] for r in rows])
    print(
        f"\ntau_w / E[p(1-p)] across corpora: "
        f"{np.round(ratio / ratio.mean(), 3).tolist()} (Corollary 3 predicts a constant)"
    )
    io.save("c1_real_noise", vars(args), {"rows": rows})


if __name__ == "__main__":
    main()
