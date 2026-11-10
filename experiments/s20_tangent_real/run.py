"""S20: seeds simulated in the tangent model of a pretrained policy, graded against real ones.

No run is trained here. At the starting adapters, each prompt's responses are sampled, the distinct
ones become its candidates, and each candidate's log-probability and score are computed by one
teacher-forced pass. That fixes the tangent model of `caliper.tangent`; the recipe of the s11 record
named by --label is then simulated in it for many seeds, and the spread of their held-out pass rate
is compared with the spread that record measured over real seeds.

What the simulated seeds can be wrong about is stated by the construction. The scores are frozen at
the start, and LoRA's are bilinear: with B at zero the gradient with respect to A vanishes, so the
tangent model moves B alone. And a prompt can only give responses that were sampled for it at the
start, or its reference answer, which is added whether sampled or not; the probability the
candidates do not cover stays where it was.

Reads the s11 record, which must have been run with --record-outputs, for the recipe, the split
and the per-prompt pass rates the simulated ones are compared with.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

from caliper.objectives.advantages import weight_table
from caliper.real.tasks import reward as exact_match
from caliper.real.train import RealRLConfig, RealRLTrainer
from caliper.runtime import io
from caliper.tangent import Recipe, TangentModel, simulate


def record(label: str) -> dict:
    hits = [r for r in io.load_all("s11_prospective") if r["result"].get("label") == label]
    if len(hits) != 1:
        raise SystemExit(f"{label}: {len(hits)} s11 records, not one")
    if "per_prompt" not in hits[0]["result"]:
        raise SystemExit(f"{label} was not run with --record-outputs")
    return hits[0]


def response_mask(trainer, prompt_len: int, response: list[int], width: int) -> np.ndarray:
    """The trainer's mask: response positions up to and including the first end-of-sequence."""
    mask = np.zeros(width, dtype=np.float32)
    eos = trainer.tokenizer.eos_token_id
    stop = response.index(eos) + 1 if eos in response else len(response)
    mask[prompt_len - 1 : prompt_len - 1 + stop] = 1.0
    return mask


def candidates(trainer, item, groups: int, key):
    """Distinct responses sampled for one prompt, and its reference answer. Each is a token row
    as the trainer would score it, so its log-probability is the probability of that response."""
    cfg = trainer.config
    eos = trainer.tokenizer.eos_token_id
    prompt_ids = trainer._chat(item["prompt"])
    found: dict[tuple[int, ...], int] = {}
    for _ in range(groups):
        sequences, _, _, key = trainer.rollout([item], key)
        for row in np.array(sequences[0])[:, prompt_ids.shape[0]:]:
            ids = [int(v) for v in row]
            if eos in ids:
                ids = ids[: ids.index(eos) + 1]
            found[tuple(ids)] = found.get(tuple(ids), 0) + 1
    reference = [int(v) for v in trainer.tokenizer.encode(item["answer"])] + [eos]
    found.setdefault(tuple(reference[: cfg.max_tokens]), 0)
    rows = []
    for ids, count in found.items():
        padded = list(ids) + [eos] * (cfg.max_tokens - len(ids))
        rows.append({
            "response": padded,
            "sampled": count,
            "reward": exact_match(trainer._trim(padded), item["answer"]),
        })
    return prompt_ids, rows, key


def log_probability_and_score(trainer, prompt_ids, response: list[int], names: list[str]):
    """log pi(response | prompt) and its gradient over the parameters in `names`."""
    full = mx.concatenate([prompt_ids, mx.array(response)])[None]
    mask = mx.array(response_mask(trainer, prompt_ids.shape[0], response, full.shape[1] - 1))

    def log_probability(model):
        logits = model(full[:, :-1]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, full[:, 1:], reduction="none")
        return -(losses[0] * mask).sum()

    value, grads = nn.value_and_grad(trainer.model, log_probability)(trainer.model)
    table = dict(tree_flatten(grads))
    flat = mx.concatenate([table[name].reshape(-1) for name in names])
    mx.eval(value, flat)
    return float(value), np.array(flat, dtype=np.float32)


def tangent_model(blocks: list[dict], scores: np.ndarray, rank_floor: float) -> TangentModel:
    """Pack per-prompt candidates into one model, in coordinates where the Gram matrix of the
    scores is the identity's image: K = L L', one row of L per candidate. Inner products between
    scores are all an update or a probability ever uses, so nothing is lost by the change."""
    gram = scores.astype(np.float64) @ scores.astype(np.float64).T
    values, vectors = np.linalg.eigh(gram)
    keep = values > rank_floor * values.max()
    rows = vectors[:, keep] * np.sqrt(values[keep])
    width = max(len(b["rows"]) for b in blocks) + 1
    base = np.full((len(blocks), width), -np.inf)
    rewards = np.zeros((len(blocks), width))
    features = np.zeros((len(blocks), width, rows.shape[1]))
    for i, block in enumerate(blocks):
        n = len(block["rows"])
        base[i, :n] = [r["log_probability"] for r in block["rows"]]
        rewards[i, :n] = [r["reward"] for r in block["rows"]]
        features[i, :n] = rows[block["offset"] : block["offset"] + n]
        # what the candidates do not cover keeps its probability and cannot be moved
        uncovered = 1.0 - float(np.exp(base[i, :n]).sum())
        if uncovered > 1e-9:
            base[i, n] = np.log(uncovered)
    return TangentModel(base=base, features=features, rewards=rewards)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="an s11 record run with --record-outputs")
    ap.add_argument("--groups", type=int, default=4,
                    help="rollout groups per prompt that the candidates are collected from")
    ap.add_argument("--simulated-seeds", type=int, default=2000)
    ap.add_argument("--rank-floor", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    source = record(args.label)
    cfg, result = source["manifest"]["config"], source["result"]
    if cfg["optimiser"] != "sgd":
        raise SystemExit("the tangent model is simulated under plain ascent only")

    pool = json.loads(Path(cfg["prompts_file"]).read_text())
    order = np.random.default_rng(cfg["seed"]).permutation(len(pool))
    held_out = [pool[i] for i in order[: cfg["eval_prompts"]]]
    corpus = [pool[i] for i in order[cfg["eval_prompts"] :]]
    items = corpus + held_out
    train_pool = np.arange(len(corpus))
    eval_pool = np.arange(len(corpus), len(items))

    config = RealRLConfig(
        group_size=cfg["group_size"], prompts=cfg["prompts"], blocks=8,
        max_tokens=cfg["max_tokens"], lora_layers=cfg["lora_layers"], steps=cfg["steps"],
        learning_rate=cfg["learning_rate"], optimiser=cfg["optimiser"], seed=cfg["seed"],
    )
    model, tokenizer = load(cfg["model"])
    trainer = RealRLTrainer(model, tokenizer, config)
    names = [name for name, _ in tree_flatten(trainer.model.trainable_parameters())]

    started = time.time()
    key = mx.random.key(31_337 + args.seed)
    blocks, scores = [], []
    for item in items:
        prompt_ids, rows, key = candidates(trainer, item, args.groups, key)
        for row in rows:
            row["log_probability"], score = log_probability_and_score(
                trainer, prompt_ids, row["response"], names
            )
            scores.append(score)
        blocks.append({"rows": rows, "offset": len(scores) - len(rows)})
    scores = np.stack(scores)
    moving = np.abs(scores).max(axis=0) > 0
    print(f"{len(items)} prompts, {len(scores)} candidates, {int(moving.sum())} of "
          f"{scores.shape[1]} parameters with a non-zero score, "
          f"{(time.time() - started) / 60:.1f} min", flush=True)

    tangent = tangent_model(blocks, scores[:, moving], args.rank_floor)
    start = tangent.pass_rates(np.zeros((1, tangent.dim)), eval_pool)[0]
    covered = np.array([np.exp([r["log_probability"] for r in b["rows"]]).sum() for b in blocks])
    print(f"tangent model of rank {tangent.dim}; candidates cover a median "
          f"{np.median(covered):.3f} of the probability (least {covered.min():.3f}); "
          f"held-out pass rate at the start {start.mean():.4f} against "
          f"{result['base_pass_rate']:.4f} sampled", flush=True)

    recipe = Recipe(
        cfg["learning_rate"], cfg["prompts"], cfg["group_size"], cfg["steps"],
        weight_table("rloo", cfg["group_size"]), replace=False,
    )
    seeds = simulate(tangent, recipe, train_pool, eval_pool, args.simulated_seeds,
                     np.random.default_rng(args.seed), chunk=32)

    real = np.array(result["per_prompt"])
    # the record's evaluation draws a fixed number of samples per prompt; drawing the same number
    # from the simulated pass rates makes the share of prompts scoring exactly 0 or 1 comparable
    samples = cfg["eval_samples"]
    drawn = np.random.default_rng(args.seed + 1).binomial(samples, seeds.pass_rates) / samples
    summary = {
        "label": args.label,
        "tangent_std": seeds.std,
        "measured_std": result["resolved_std"],
        "measured_interval": result["resolved_interval"],
        "linear_forecast_std": result["predicted_std"],
        "ratio": seeds.std / result["resolved_std"],
        "tangent_mean": float(seeds.scores.mean()),
        "measured_mean": float(np.mean(result["held_out_scores"])),
        "start": float(start.mean()),
        "base_pass_rate": result["base_pass_rate"],
        "tangent_settled_share": float(((drawn == 0) | (drawn == 1)).mean()),
        "measured_settled_share": float(((real == 0) | (real == 1)).mean()),
        "per_prompt_mean_correlation": float(
            np.corrcoef(seeds.pass_rates.mean(axis=0), real.mean(axis=0))[0, 1]
        ),
        "tangent_per_prompt_mean": seeds.pass_rates.mean(axis=0).tolist(),
        "tangent_per_prompt_std": seeds.pass_rates.std(axis=0, ddof=1).tolist(),
        "candidates": int(len(scores)),
        "rank": int(tangent.dim),
        "coverage_median": float(np.median(covered)),
        "coverage_min": float(covered.min()),
        "simulated_seeds": args.simulated_seeds,
        "minutes": (time.time() - started) / 60,
    }
    print(f"simulated spread {seeds.std:.4f} against measured {result['resolved_std']:.4f} "
          f"[{result['resolved_interval']['lo']:.4f}, {result['resolved_interval']['hi']:.4f}]: "
          f"{summary['ratio']:.2f}x (the linear forecast was {result['predicted_std']:.4f})")
    print(f"mean held-out pass rate {summary['tangent_mean']:.3f} against "
          f"{summary['measured_mean']:.3f}; prompts scoring exactly 0 or 1: "
          f"{summary['tangent_settled_share']:.2f} against "
          f"{summary['measured_settled_share']:.2f}; "
          f"per-prompt means correlate at {summary['per_prompt_mean_correlation']:.2f}")
    path = io.save("s20_tangent_real", vars(args), summary)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
