"""S16: where the seed spread lives on a pretrained model, from the records alone.

The two large real-model misses (Qwen2.5-0.5B under plain ascent, Qwen2.5-7B under Adam) store,
for every seed, the held-out pass rate of each of 48 prompts estimated from 16 samples. Row 0 is
the run the forecast was made from; the other rows are seeds trained afterwards. That is enough
to ask two questions without training anything.

  Is the spread shared or per prompt? The variance of the mean score across seeds is the sum of
  a prompt-by-prompt covariance matrix. Its diagonal is the part that would remain if prompts
  flipped independently between seeds; the off-diagonal is the part a seed-level difference in
  skill would produce.

  What did the forecast see? A first-order forecast is the sensitivity of the reported score to
  small parameter changes at the stored endpoint. On a prompt the endpoint answers 16/16 or 0/16
  the policy has saturated and that sensitivity is close to zero, whatever later seeds do there.
"""

from __future__ import annotations

import json

import numpy as np

from caliper.runtime import io

RECORDS = {
    "small": ("s7_real_forecast", "1d3f31eda6cd"),  # Qwen2.5-0.5B, plain ascent, the 25x miss
    "large": ("s7_real_forecast", "107b7137ba01"),  # Qwen2.5-7B under Adam
}


def decompose(rates: np.ndarray, samples: int) -> dict:
    """Split the across-seed variance of the mean score into per-prompt and shared parts."""
    seeds, prompts = rates.shape
    covariance = np.cov(rates, rowvar=False, ddof=1)
    total = float(covariance.sum() / prompts**2)
    diagonal = float(np.trace(covariance) / prompts**2)
    # the evaluation's own binomial noise sits inside the diagonal; report it so it can be
    # seen to be small rather than assumed so
    binomial = float((rates * (1 - rates)).mean(0).sum() / (samples - 1) / prompts**2)
    ordered = np.argsort(rates.mean(1))
    low, high = rates[ordered[: seeds // 2]].mean(0), rates[ordered[seeds // 2:]].mean(0)
    return {
        "seeds": seeds,
        "prompts": prompts,
        "total_std": float(np.sqrt(total)),
        "per_prompt_share": diagonal / total,
        "shared_share": 1.0 - diagonal / total,
        "binomial_share": binomial / total,
        "high_half_better": int((high - low > 0.25).sum()),
        "high_half_worse": int((low - high > 0.25).sum()),
    }


def saturation(rates: np.ndarray) -> dict:
    """How collapsed the stored endpoint and each later seed are, prompt by prompt."""
    saturated = (rates == 0.0) | (rates == 1.0)
    anchor, later = rates[0], rates[1:]
    anchor_saturated = saturated[0]
    moved = (later != anchor) & anchor_saturated
    spread = later.var(0, ddof=1)
    return {
        "anchor_saturated": int(anchor_saturated.sum()),
        "later_saturated_share": [float(x) for x in saturated[1:].mean(1)],
        "anchor_saturated_moved": int(moved.any(0).sum()),
        "variance_on_anchor_saturated": float(spread[anchor_saturated].sum() / spread.sum()),
    }


def main() -> None:
    out = {}
    for tag, (store, name) in RECORDS.items():
        record = json.loads((io.RUNS / store / f"{name}.json").read_text())
        rates = np.asarray(record["result"]["per_prompt"], dtype=float)
        samples = record["manifest"]["config"]["eval_samples"]
        out[tag] = {
            "record": name,
            "model": record["manifest"]["config"]["model"],
            "forecast_std": record["result"]["predicted_std"],
            "seed_std": record["result"]["resolved_std"],
            **decompose(rates, samples),
            **saturation(rates),
        }
        row = out[tag]
        print(f"{tag}: {row['model']}")
        print(f"  seed sd {row['seed_std']:.4f} against a forecast of {row['forecast_std']:.4f}")
        print(f"  per-prompt share {100 * row['per_prompt_share']:.0f}%, shared "
              f"{100 * row['shared_share']:.0f}%, evaluation noise "
              f"{100 * row['binomial_share']:.1f}%")
        print(f"  higher-scoring half better on {row['high_half_better']} prompts, worse on "
              f"{row['high_half_worse']}")
        print(f"  stored endpoint saturated on {row['anchor_saturated']} of {row['prompts']}; "
              f"later seeds differ there on {row['anchor_saturated_moved']}")
        print(f"  later seeds saturated on "
              f"{', '.join(f'{100 * s:.0f}%' for s in sorted(row['later_saturated_share']))}")
    path = io.save("s16_saturation", {"records": RECORDS}, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
