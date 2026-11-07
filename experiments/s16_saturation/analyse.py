"""S16: what the pretrained runs collapse to, from the records alone.

The real-model records on the `count_in_string` task store, for every seed, the held-out pass
rate of each of 48 prompts estimated from 16 samples; row 0 is the run the forecast was made
from. The held-out prompts are regenerated from the same seed the experiment used, so each
prompt's correct answer is known. That is enough to ask what a run has learned without
training anything.

A run is classed as answering one constant `v` when it passes at least 90% of the prompts whose
answer is `v` and at most 10% of the others: it says `v` whatever it is asked, and scores the
share of prompts whose answer happens to be `v`. The question is then how much of the
across-seed spread of the score is which constant a run fell into, against the alternative that
seeds scatter around one policy.
"""

from __future__ import annotations

import json

import numpy as np

from caliper.real.tasks import FAMILIES
from caliper.runtime import io

HELD_OUT_SEED = 9001  # experiments/s7_real_forecast/run.py draws the held-out prompts with this
RECORDS = {
    "small": "1d3f31eda6cd",  # Qwen2.5-0.5B, plain ascent, the record the paper's 0.5B miss is from
    "large": "107b7137ba01",  # Qwen2.5-7B under Adam
    # two earlier 0.5B launches of the same configuration; their forecasts predate the sign fix
    # in 0f4c098 and are not used, but the seeds' training outcomes are
    "small_rerun": "098b39058393",
    "small_traces": "6fd81c84f706",
}


def modes(rates: np.ndarray, answers: np.ndarray) -> list[str]:
    """For each seed, the constant it answers, or "mixed"."""
    out = []
    for row in rates:
        label = "mixed"
        for value in np.unique(answers):
            hit = answers == value
            if row[hit].mean() >= 0.9 and row[~hit].mean() <= 0.1:
                label = str(value)
        out.append(label)
    return out


def block_correlation(rates: np.ndarray, answers: np.ndarray) -> tuple[float, float]:
    """Mean across-seed correlation of prompt pairs sharing an answer, and of pairs that do not."""
    varying = rates.std(0) > 0
    corr = np.corrcoef(rates[:, varying], rowvar=False)
    same = answers[varying][:, None] == answers[varying][None, :]
    upper = np.triu(np.ones_like(corr, dtype=bool), 1)
    return float(np.nanmean(corr[same & upper])), float(np.nanmean(corr[~same & upper]))


def mode_share(scores: np.ndarray, labels: list[str]) -> float:
    """Share of the across-seed variance of the score explained by the mode label."""
    total = float(np.sum((scores - scores.mean()) ** 2))
    within = 0.0
    for label in set(labels):
        members = scores[[i for i, x in enumerate(labels) if x == label]]
        within += float(np.sum((members - members.mean()) ** 2))
    return 1.0 - within / total if total > 0 else float("nan")


def main() -> None:
    out = {}
    for tag, name in RECORDS.items():
        record = json.loads((io.RUNS / "s7_real_forecast" / f"{name}.json").read_text())
        config, result = record["manifest"]["config"], record["result"]
        items = FAMILIES[config["family"]].sample(
            config["eval_prompts"], np.random.default_rng(HELD_OUT_SEED))
        answers = np.array([item["answer"] for item in items])
        rates = np.asarray(result["per_prompt"], dtype=float)
        scores = rates.mean(1)
        labels = modes(rates, answers)
        within, between = block_correlation(rates, answers)
        counts = {value: int((answers == value).sum()) for value in np.unique(answers)}
        repeats = [float(x) for x in result.get("repeat_stds", []) if x and np.isfinite(x)]
        out[tag] = {
            "record": name,
            "model": config["model"],
            "seeds": len(labels),
            "modes": labels,
            "anchor_mode": labels[0],
            "constant_seeds": sum(label != "mixed" for label in labels),
            "answer_counts": counts,
            "scores": [float(s) for s in scores],
            "mode_share": mode_share(scores, labels),
            "same_answer_correlation": within,
            "other_answer_correlation": between,
            "forecast_repeats": repeats,
        }
        row = out[tag]
        print(f"{tag} ({name}): {row['model']}")
        print(f"  modes by seed (row 0 is the anchor): {labels}")
        print(f"  {row['constant_seeds']} of {row['seeds']} seeds answer one constant; "
              f"the mode label explains {100 * row['mode_share']:.0f}% of the score variance")
        print(f"  prompts sharing an answer correlate {within:.2f} across seeds, "
              f"others {between:.2f}; answer counts {counts}")
        if len(repeats) > 1:
            print(f"  forecast repeats from the same run: {repeats}")
    path = io.save("s16_saturation", {"records": RECORDS, "held_out_seed": HELD_OUT_SEED}, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
