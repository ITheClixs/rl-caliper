"""A5: accuracy of the crossed split estimator against exactly known noise terms.

Also compares it against the subtractive alternative, which is what a trainer would reach for if
it estimated the total variance and the within-prompt variance separately.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.estimators.simulate import simulate_split_batch
from caliper.estimators.splits import average
from caliper.exact.aggregate import ExactBatchModel
from caliper.exact.policy import fisher, make_prompt
from caliper.exact.pool import build
from caliper.runtime import io

BANDS = {"broad": (0.05, 0.95), "hard": (0.05, 0.35), "easy": (0.65, 0.95)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=2000)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    header = f"{'band':6s} {'div':>5s} {'exact G*':>9s} {'crossed':>9s}"
    print(header + f" {'err %':>7s} {'subtractive':>12s}")
    for band, diversity in itertools.product(BANDS, [0.03, 0.08, 0.2, 0.5, 1.0]):
        seed = args.seed + hash_seed(band, diversity)
        policy, accepts = build(3, 3, args.pool, BANDS[band], diversity, seed)
        prompts = [make_prompt(policy, a) for a in accepts]
        fishers = [fisher(p) for p in prompts]
        matrix = np.mean(fishers, axis=0)
        exact = ExactBatchModel([p.moments() for p in prompts], fishers).noise_terms(
            args.estimator, args.group_size, curvature="fisher"
        )
        exact_tau_w = exact.tau_w * (args.group_size - 1)
        exact_g = 1 + np.sqrt(exact_tau_w / exact.tau_b)

        rng = np.random.default_rng(seed + 11)
        est = average(
            [
                simulate_split_batch(
                    prompts,
                    args.estimator,
                    args.prompts,
                    args.group_size,
                    rng,
                    matrix=matrix,
                    blocks=args.blocks,
                )
                for _ in range(args.batches)
            ]
        )
        crossed_g = est.optimal_group_size()
        subtractive_g = est.subtractive_group_size()
        rows.append(
            {
                "band": band,
                "diversity": diversity,
                "exact_g_star": float(exact_g),
                "crossed_g_star": float(crossed_g),
                "exact_tau_b": float(exact.tau_b),
                "crossed_tau_b": float(est.tau_b),
                "exact_tau_w_scaled": float(exact_tau_w),
                "crossed_tau_w_scaled": float(est.tau_w_scaled),
                "exact_ratio": float(exact_tau_w / exact.tau_b),
            }
        )
        print(
            f"{band:6s} {diversity:5.2f} {exact_g:9.2f} {crossed_g:9.2f} "
            f"{100 * abs(crossed_g - exact_g) / exact_g:7.1f} {subtractive_g:12.2f}"
        )

    err_g = np.array(
        [abs(r["crossed_g_star"] - r["exact_g_star"]) / r["exact_g_star"] for r in rows]
    )
    err_b = np.array([abs(r["crossed_tau_b"] - r["exact_tau_b"]) / r["exact_tau_b"] for r in rows])
    err_w = np.array(
        [
            abs(r["crossed_tau_w_scaled"] - r["exact_tau_w_scaled"]) / r["exact_tau_w_scaled"]
            for r in rows
        ]
    )
    ratios = np.array([r["exact_ratio"] for r in rows])
    gs = np.array([r["exact_g_star"] for r in rows])
    print(
        f"\nG* covered {gs.min():.2f}..{gs.max():.2f} "
        f"(noise ratio spans {ratios.max() / ratios.min():.1f}x)\n"
        f"median |error|: G* {100 * np.median(err_g):.1f}%, "
        f"tau_b {100 * np.median(err_b):.1f}%, tau_w {100 * np.median(err_w):.1f}%\n"
        f"worst |error|:  G* {100 * err_g.max():.1f}%, "
        f"tau_b {100 * err_b.max():.1f}%, tau_w {100 * err_w.max():.1f}%"
    )
    io.save("a5_estimator", vars(args), {"rows": rows})


def hash_seed(band: str, diversity: float) -> int:
    import zlib

    return zlib.crc32(f"{band}:{diversity}".encode()) % 9973


if __name__ == "__main__":
    main()
