"""P3: how far back a run remembers its own randomness, and which randomness it remembers.

Two questions the adjoint kernel answers directly and nothing else does.

The kernel k_t is the share of the final spread of the reported number that was injected at
update t. Its tail gives a memory horizon: the number of final updates holding most of the
uncertainty. A short horizon would mean reproducing a result requires reproducing only its end.
That is not what we find, and the horizon is reported here because the question is worth asking
and the answer is worth knowing.

Restricting the injected covariance to one source of randomness -- which prompts were drawn, or
which responses were sampled -- splits the same number by origin. The two shares are exact and
sum to one, because the two terms partition the batch covariance.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.exact.adjoint import forecast
from caliper.exact.pool import build
from caliper.exact.propagation import seed_memory
from caliper.runtime import io

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--steps", nargs="+", type=int, default=[10, 20, 40, 80])
    ap.add_argument("--prompts", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--group-size", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--step-size", nargs="+", type=float, default=[0.5, 2.0, 8.0])
    ap.add_argument("--bands", nargs="+", default=["easy", "mixed", "hard"])
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = []
    print(f"{'band':>6s} {'T':>4s} {'P':>4s} {'G':>3s} {'eta':>5s} {'H95':>5s} {'H95/T':>7s} "
          f"{'decay':>7s} {'prompts':>8s} {'rollouts':>9s}")
    for band, steps, prompts, group, eta in itertools.product(
        args.bands, args.steps, args.prompts, args.group_size, args.step_size
    ):
        policy, accepts = build(args.vocab, args.length, args.pool, BANDS[band], 1.0, args.seed)
        common = (policy, accepts, args.estimator, group, prompts, eta, steps)
        whole = forecast(*common)
        by_prompt = forecast(*common, source="prompts")
        by_rollout = forecast(*common, source="rollouts")

        horizon = seed_memory(whole.kernel, 0.95)
        share_prompt = by_prompt.variance / whole.variance
        # how much the metric direction is contracted over the whole run, from the adjoint norms
        decay = whole.adjoint_norm[0] / max(whole.adjoint_norm[-1], 1e-30)
        cells.append(
            {
                "band": band,
                "steps": steps,
                "prompts": prompts,
                "group_size": group,
                "step_size": eta,
                "adjoint_decay": float(decay),
                "variance": whole.variance,
                "std": whole.std,
                "horizon_95": horizon,
                "horizon_50": seed_memory(whole.kernel, 0.50),
                "prompt_share": float(share_prompt),
                "rollout_share": float(by_rollout.variance / whole.variance),
                "kernel": whole.kernel,
                "adjoint_norm": whole.adjoint_norm,
                "final_metric": whole.trajectory[-1],
            }
        )
        print(f"{band:>6s} {steps:4d} {prompts:4d} {group:3d} {eta:5.2f} {horizon:5d} "
              f"{horizon / steps:7.2f} {decay:7.3f} {share_prompt:8.1%} "
              f"{by_rollout.variance / whole.variance:9.1%}", flush=True)

    fraction = np.array([c["horizon_95"] / c["steps"] for c in cells])
    shares = np.array([c["rollout_share"] for c in cells])
    print(f"\nH95/T: median {np.median(fraction):.2f}  "
          f"range {fraction.min():.2f}--{fraction.max():.2f}")
    print(f"rollout share: median {np.median(shares):.1%}  "
          f"range {shares.min():.1%}--{shares.max():.1%}")
    by_eta = {}
    for c in cells:
        by_eta.setdefault(c["step_size"], []).append(
            (c["horizon_95"] / c["steps"], c["adjoint_decay"])
        )
    for e in sorted(by_eta):
        frac = np.median([v[0] for v in by_eta[e]])
        dec = np.median([v[1] for v in by_eta[e]])
        print(f"  eta={e:5.2f}: H95/T median {frac:.2f}, adjoint decay over the run {dec:.3f}")
    by_group = {}
    for c in cells:
        by_group.setdefault(c["group_size"], []).append(c["rollout_share"])
    for g in sorted(by_group):
        print(f"  G={g:3d}: rollout share {np.median(by_group[g]):.1%}")
    io.save("p3_memory_sources", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
