"""P6: does the filter change shape as training proceeds?

The transfer operator A_t = I + eta J_t is not fixed: J_t moves as the policy improves and the
pass-rate distribution shifts under it. This measures its spectrum at points along a run and
reports what a single number would have missed -- the spectral radius, the share of directions the
update contracts, and the spread between the fastest and slowest of them.
"""

from __future__ import annotations

import argparse

import numpy as np

from caliper.exact.adjoint import mean_trajectory, metric_value
from caliper.exact.pool import build
from caliper.exact.propagation import update_jacobian
from caliper.objectives.advantages import weight_table
from caliper.runtime import io

BANDS = {"easy": (0.35, 0.95), "mixed": (0.05, 0.95), "hard": (0.02, 0.45)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--probes", nargs="+", type=int, default=[0, 25, 50, 100, 200, 400, 600])
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--step-size", type=float, default=1.0)
    ap.add_argument("--bands", nargs="+", default=["easy", "mixed", "hard"])
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weights = weight_table(args.estimator, args.group_size)
    cells = []
    for band in args.bands:
        policy, accepts = build(args.vocab, args.length, args.pool, BANDS[band], 1.0, args.seed)
        states = mean_trajectory(policy, accepts, weights, args.step_size, args.steps)
        print(f"\n{band}")
        print(f"{'update':>7s} {'pass':>6s} {'radius':>7s} {'contracted':>11s} "
              f"{'slowest':>8s} {'fastest':>8s}")
        for probe in args.probes:
            if probe > args.steps:
                continue
            state = states[probe]
            jacobian = update_jacobian(state, accepts, weights)
            transfer = np.eye(policy.dim) + args.step_size * jacobian
            values = np.abs(np.linalg.eigvals(transfer))
            inside = values[values < 1.0 - 1e-12]
            radius = float(values.max())
            contracted = float(np.mean(values < 1.0 - 1e-12))
            slowest = float(inside.max()) if inside.size else float("nan")
            fastest = float(inside.min()) if inside.size else float("nan")
            rate = metric_value(state, accepts)
            cells.append({
                "band": band, "update": probe, "pass_rate": rate, "radius": radius,
                "contracted_share": contracted, "slowest": slowest, "fastest": fastest,
                "eigenvalues": values.tolist(),
            })
            print(f"{probe:7d} {rate:6.3f} {radius:7.4f} {contracted:11.1%} "
                  f"{slowest:8.4f} {fastest:8.4f}", flush=True)

    radii = np.array([c["radius"] for c in cells])
    shares = np.array([c["contracted_share"] for c in cells])
    print(f"\nspectral radius over the whole sweep: {radii.min():.4f}--{radii.max():.4f}")
    print(f"share of contracted directions: {shares.min():.1%}--{shares.max():.1%}")
    print("a single timescale would have to stand in for the whole of that range")
    io.save("p6_spectrum", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
