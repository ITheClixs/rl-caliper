"""A2: the second-order progress model against exactly computed improvement and drift.

For an enumerable policy both the objective and the KL between the pre- and post-update policies
are computed exactly, so the predictions of equations (9)-(11) and (16)-(17) can be checked
without any surrogate.
"""

from __future__ import annotations

import argparse

import numpy as np

from caliper.estimators.simulate import group_gradient, sample_group
from caliper.exact.aggregate import ExactBatchModel
from caliper.exact.policy import (
    TabularPolicy,
    accept_set_for_pass_rate,
    fisher,
    make_prompt,
)
from caliper.objectives.advantages import weight_table
from caliper.runtime import io


def build(vocab, length, n_prompts, seed):
    rng = np.random.default_rng(seed)
    policy = TabularPolicy.random(vocab, length, rng, scale=0.8)
    probs = policy.sequence_probs()
    accepts, prompts = [], []
    while len(prompts) < n_prompts:
        accept = accept_set_for_pass_rate(probs, rng.uniform(0.1, 0.9), rng)
        prompt = make_prompt(policy, accept)
        if 1e-6 < prompt.pass_rate < 1 - 1e-6:
            accepts.append(accept)
            prompts.append(prompt)
    return policy, accepts, prompts


def objective(policy: TabularPolicy, accepts) -> float:
    probs = policy.sequence_probs()
    return float(np.mean([probs @ a for a in accepts]))


def exact_kl(old: np.ndarray, new: np.ndarray) -> float:
    mask = new > 0
    return float(np.sum(new[mask] * (np.log(new[mask]) - np.log(old[mask]))))


def batch_gradient(prompts, w, n_prompts, group_size, rng):
    dim = prompts[0].scores.shape[1]
    acc = np.zeros(dim)
    chosen = rng.choice(len(prompts), size=n_prompts, replace=True)
    for i in chosen:
        idx = sample_group(prompts[i], group_size, rng)
        acc += group_gradient(prompts[i], idx, w)
    return acc / n_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--prompts-pool", type=int, default=32)
    ap.add_argument("--batch-prompts", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=600)
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    policy, accepts, prompts = build(args.vocab, args.length, args.prompts_pool, args.seed)
    model = ExactBatchModel([p.moments() for p in prompts], [fisher(p) for p in prompts])
    terms = model.noise_terms(args.estimator, args.group_size, curvature="fisher")
    w = weight_table(args.estimator, args.group_size)

    base_j = objective(policy, accepts)
    base_probs = policy.sequence_probs()
    predicted_eta = terms.optimal_step_size(args.batch_prompts)
    etas = predicted_eta * np.array([0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

    rng = np.random.default_rng(args.seed + 1000)
    grads = [
        batch_gradient(prompts, w, args.batch_prompts, args.group_size, rng)
        for _ in range(args.batches)
    ]

    rows = []
    for eta in etas:
        d_j, kls = [], []
        for g in grads:
            moved = policy.perturbed(eta * g)
            d_j.append(objective(moved, accepts) - base_j)
            kls.append(exact_kl(base_probs, moved.sequence_probs()))
        rows.append(
            {
                "eta": float(eta),
                "eta_over_predicted": float(eta / predicted_eta),
                "delta_j": float(np.mean(d_j)),
                "delta_j_se": float(np.std(d_j, ddof=1) / np.sqrt(len(d_j))),
                "kl": float(np.mean(kls)),
                "kl_predicted": float(
                    0.5 * eta**2 * (terms.signal + (terms.tau_b + terms.tau_w) / args.batch_prompts)
                ),
                "progress_predicted": float(
                    eta * terms.alignment
                    - 0.5
                    * eta**2
                    * (terms.signal + (terms.tau_b + terms.tau_w) / args.batch_prompts)
                ),
            }
        )

    best = max(rows, key=lambda r: r["delta_j"])
    efficiency = terms.efficiency(args.batch_prompts)
    signal_fraction = terms.signal / (
        terms.signal + (terms.tau_b + terms.tau_w) / args.batch_prompts
    )

    header = f"{'eta/eta*':>9s} {'exact dJ':>12s} {'predicted':>12s}"
    print(header + f" {'exact KL':>11s} {'pred KL':>11s}")
    for r in rows:
        print(
            f"{r['eta_over_predicted']:9.3f} {r['delta_j']:12.3e} {r['progress_predicted']:12.3e} "
            f"{r['kl']:11.3e} {r['kl_predicted']:11.3e}"
        )
    print(f"\nexact argmax eta / predicted eta* = {best['eta_over_predicted']:.3f}")
    print(
        f"efficiency 1/(1+Bcrit/P) = {efficiency:.4f}   "
        f"drift signal fraction = {signal_fraction:.4f}"
    )
    print(f"Bcrit = {terms.critical_batch():.1f} prompts,  P = {args.batch_prompts}")

    io.save(
        "a2_progress_model",
        vars(args),
        {
            "rows": rows,
            "efficiency": efficiency,
            "signal_fraction": signal_fraction,
            "critical_batch": terms.critical_batch(),
            "predicted_eta": float(predicted_eta),
            "argmax_ratio": best["eta_over_predicted"],
        },
    )


if __name__ == "__main__":
    main()
