"""P5: does the recursion still hold when the optimiser carries state?

Adam's update depends on two moment estimates as well as the parameters, so a gradient
perturbation outlives the step that produced it. The prediction is made three ways and compared
against independently trained runs:

  lifted        transfer and injection taken on z = (theta, m, v)
  momentum      the same, with v held at its mean trajectory rather than propagated
  parameters    the parameter block of the same operators, as though the optimiser state did not
                carry a perturbation forward
  accumulation  no contraction at all

If the optimiser state matters, the second of these should be wrong in a specific direction.

The moments are warmed on mean gradients before the recursion starts. Adam's map is singular at
v = 0 and cannot be linearised there; every run shares the warmed moments, so the covariance
still starts at zero.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.exact.lifted import AdamSettings, advance_state, initial_state, transfer_and_injection
from caliper.exact.policy import fisher, make_prompt
from caliper.exact.pool import build
from caliper.objectives.advantages import weight_table
from caliper.runtime import io

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}


def sample_gradient(policy, accepts, weights, n_prompts, group_size, rng):
    probs, scores = policy.enumerate()
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    prompt_ids = rng.integers(0, accepts.shape[0], size=n_prompts)
    outcomes = np.searchsorted(cdf, rng.random((n_prompts, group_size)))
    rewards = accepts[prompt_ids[:, None], outcomes].astype(int)
    counts = rewards.sum(axis=1)
    adv = weights[counts[:, None], rewards]
    return np.einsum("pg,pgd->d", adv, scores[outcomes]) / (n_prompts * group_size)


def monte_carlo(policy, accepts, weights, cfg, settings, seeds, rng_seed, start):
    """Independent Adam runs from a shared warmed state; pairwise divergence at the end."""
    finals = []
    for s in range(seeds):
        rng = np.random.default_rng(rng_seed + 7919 * s)
        state = start
        for t in range(start.step, start.step + cfg["steps"]):
            grad = sample_gradient(
                state.policy, accepts, weights, cfg["prompts"], cfg["group_size"], rng
            )
            moment = settings.beta1 * state.moment + (1 - settings.beta1) * grad
            second = settings.beta2 * state.second + (1 - settings.beta2) * grad**2
            hat_m = moment / (1 - settings.beta1 ** (t + 1))
            hat_v = second / (1 - settings.beta2 ** (t + 1))
            move = settings.step_size * hat_m / (np.sqrt(hat_v) + settings.epsilon)
            state = type(state)(state.policy.perturbed(move), moment, second, t + 1)
        finals.append(state.policy)
    metric = np.mean([fisher(make_prompt(finals[0], a)) for a in accepts], axis=0)
    gaps = []
    for a in range(seeds):
        for b in range(a + 1, seeds):
            delta = (finals[a].logits - finals[b].logits).reshape(-1)
            gaps.append(float(0.5 * delta @ metric @ delta))
    return float(np.mean(gaps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--steps", nargs="+", type=int, default=[10, 25])
    ap.add_argument("--prompts", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--step-size", nargs="+", type=float, default=[3e-2, 1e-1])
    ap.add_argument("--bands", nargs="+", default=["easy", "mixed", "hard"])
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weights = weight_table(args.estimator, args.group_size)
    cells = []
    print(f"{'band':>6s} {'T':>4s} {'P':>4s} {'eta':>6s} {'measured':>10s} "
          f"{'lifted':>10s} {'momentum':>10s} {'params':>10s} {'accum':>10s}")
    for band, steps, prompts, eta in itertools.product(
        args.bands, args.steps, args.prompts, args.step_size
    ):
        policy, accepts = build(args.vocab, args.length, args.pool, BANDS[band], 1.0, args.seed)
        settings = AdamSettings(step_size=eta)
        cfg = {"steps": steps, "prompts": prompts, "group_size": args.group_size}
        dim = policy.dim

        # warm the moments on mean gradients: every run shares this state
        state = initial_state(policy)
        for _ in range(args.warmup):
            state = advance_state(state, accepts, weights, settings, prompts)
        start = state
        size = 3 * dim
        covariance = np.zeros((size, size))
        transfers, injections = [], []
        # the same recursion on theta alone, which is what ignoring the optimiser state gives
        momentum_cov = np.zeros((size, size))
        flat = np.zeros((dim, dim))
        walk = np.zeros((dim, dim))
        for _ in range(steps):
            transfer, injection = transfer_and_injection(
                state, accepts, weights, prompts, settings
            )
            covariance = transfer @ covariance @ transfer.T + injection
            transfers.append(transfer)
            injections.append(injection)
            slow_transfer, slow_injection = transfer_and_injection(
                state, accepts, weights, prompts, settings, carry_second_moment=False
            )
            momentum_cov = (
                slow_transfer @ momentum_cov @ slow_transfer.T + slow_injection
            )

            block = transfer[:dim, :dim]
            unit = injection[:dim, :dim]
            flat = block @ flat @ block.T + unit
            walk = walk + unit
            state = advance_state(state, accepts, weights, settings, prompts)

        metric = np.mean([fisher(make_prompt(state.policy, a)) for a in accepts], axis=0)
        lifted_kl = float(np.trace(metric @ covariance[:dim, :dim]))
        momentum_kl = float(np.trace(metric @ momentum_cov[:dim, :dim]))
        flat_kl = float(np.trace(metric @ flat))
        walk_kl = float(np.trace(metric @ walk))
        measured = monte_carlo(
            policy, accepts, weights, cfg, settings, args.seeds, args.seed + 5, start
        )

        cells.append({
            "band": band, "steps": steps, "prompts": prompts, "step_size": eta,
            "warmup": args.warmup,
            "measured_kl": measured, "lifted_kl": lifted_kl, "momentum_kl": momentum_kl,
            "final_metric": float(np.mean(accepts @ state.policy.sequence_probs())),
            "parameters_only_kl": flat_kl, "walk_kl": walk_kl,
        })
        print(f"{band:>6s} {steps:4d} {prompts:4d} {eta:6.3f} {measured:10.3e} "
              f"{lifted_kl:10.3e} {momentum_kl:10.3e} {flat_kl:10.3e} {walk_kl:10.3e}",
              flush=True)

    print()
    measured = np.array([c["measured_kl"] for c in cells])
    for key, label in [("lifted_kl", "lifted"), ("momentum_kl", "momentum only"),
                       ("parameters_only_kl", "parameters only"),
                       ("walk_kl", "accumulation")]:
        ratio = np.array([c[key] for c in cells]) / measured
        error = np.abs(np.log(ratio))
        print(f"{label:>16s}: median {np.exp(np.median(error)):7.2f}x  "
              f"worst {np.exp(error.max()):8.2f}x  range {ratio.min():.2f}--{ratio.max():.2f}")
    print("\nby band, lifted only:")
    for band in args.bands:
        rows = [c for c in cells if c["band"] == band]
        ratio = np.array([c["lifted_kl"] / c["measured_kl"] for c in rows])
        error = np.abs(np.log(ratio))
        rate = np.mean([c["final_metric"] for c in rows])
        print(f"  {band:>6s} (mean pass rate {rate:.2f}): median {np.exp(np.median(error)):6.2f}x  "
              f"worst {np.exp(error.max()):8.2f}x")
    io.save("p5_adam_lift", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
