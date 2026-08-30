"""Build corpora at pre-specified live-group probabilities.

Section 6 compares one corpus whose groups collapse against one whose groups do not, and finds
the forecast error moving from twenty-five to two. That comparison changes many things at once.
This builds several corpora that differ, as far as a prompt pool can, only in the stochastic
support they carry, so the forecast error can be measured against it rather than at two points.

A prompt with pass rate p contributes to a group of G with probability
`live(p, G) = 1 - p^G - (1-p)^G`. Selecting prompts to hit a target mean of that quantity is
therefore a matter of mixing prompts near the middle with prompts near the ends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caliper.real.support import live_probability, support_profile


def build_band(pool, target: float, size: int, group_size: int, rng) -> list[dict]:
    """A corpus of `size` prompts whose mean live probability is close to `target`.

    Prompts are sorted by their own live probability and drawn from a window whose centre is
    moved until the mean lands on the target. Sampling within the window rather than taking the
    top `size` keeps the corpus from being a single difficulty.
    """
    live = live_probability([p["base_rate"] for p in pool], group_size)
    order = np.argsort(live)
    ranked = [pool[i] for i in order]
    ranked_live = live[order]

    best, best_gap = None, float("inf")
    for start in range(0, max(len(ranked) - size, 0) + 1):
        window = ranked_live[start : start + size]
        gap = abs(float(window.mean()) - target)
        if gap < best_gap:
            best, best_gap = start, gap
    if best is None:
        raise SystemExit(f"pool of {len(pool)} is smaller than the requested {size}")
    return [ranked[i] for i in range(best, best + size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surveyed", default="runs/s10_difficulty/qwen2.5-0.5b-surveyed.json")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.15, 0.35, 0.60, 0.85])
    ap.add_argument("--size", type=int, default=255)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/s12_support_sweep")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = json.loads(Path(args.surveyed).read_text())
    rng = np.random.default_rng(args.seed)
    print(f"pool of {len(pool)} surveyed prompts")
    print(f"{'target':>8s} {'achieved':>9s} {'mean pass':>10s} {'n':>5s}  file")
    for target in args.targets:
        band = build_band(pool, target, args.size, args.group_size, rng)
        profile = support_profile([p["base_rate"] for p in band], args.group_size)
        path = out / f"band-{target:.2f}.json"
        path.write_text(json.dumps(band, indent=2))
        print(f"{target:8.2f} {profile['mean_live']:9.3f} {profile['mean_pass']:10.3f} "
              f"{profile['n']:5d}  {path}")


if __name__ == "__main__":
    main()
