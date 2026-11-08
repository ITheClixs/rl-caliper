"""S16, sweep: do runs collapse onto one answer where the forecast worked, too?

On the counting task most pretrained runs end up giving one answer to every prompt, and the
forecast misses there by a factor of 25 (`analyse.py`). The support sweep found one corpus where
the frozen forecast was within 1.5x and others where it was not. This reads the reruns of that
sweep made with `s11_prospective/run.py --record-outputs`, which store each run's most frequent
answer per held-out prompt, and asks of every run how much of the held-out set it answers with
its single most common output. If collapse is what defeats the forecast, the band where the
forecast held should show less of it than the band where it failed.
"""

from __future__ import annotations

import numpy as np

from caliper.runtime import io

LABELS = ("modes-band-0.85", "modes-band-0.60")


def collapse(outputs: list[dict]) -> tuple[str, float]:
    """A run's most common answer across prompts, and the share of prompts it is the mode of."""
    modes = [o["mode"] for o in outputs]
    values, counts = np.unique(modes, return_counts=True)
    return str(values[counts.argmax()]), float(counts.max() / len(modes))


def explained(scores: np.ndarray, shares: np.ndarray) -> float:
    """Squared correlation between a run's collapse share and its held-out score."""
    if scores.size < 3 or np.std(shares) == 0 or np.std(scores) == 0:
        return float("nan")
    return float(np.corrcoef(scores, shares)[0, 1] ** 2)


def main() -> None:
    records = {r["result"]["label"]: r for r in io.load_all("s11_prospective")
               if r["result"].get("label") in LABELS}
    out = {}
    for label in LABELS:
        if label not in records:
            print(f"{label}: not run yet")
            continue
        result = records[label]["result"]
        anchor_answer, anchor_share = collapse(result["anchor_outputs"])
        later = [collapse(o) for o in result["seed_outputs"]]
        shares = np.array([s for _, s in later])
        scores = np.array(result["held_out_scores"])
        answers = np.array(result["held_out_answers"])
        out[label] = {
            "record": records[label]["manifest"]["config_hash"],
            "ratio": result["ratio"],
            "error": max(result["ratio"], 1 / result["ratio"]),
            "anchor_answer": anchor_answer,
            "anchor_share": anchor_share,
            "anchor_score": float(np.mean(result["anchor_per_prompt"])),
            "seed_answers": [a for a, _ in later],
            "seed_shares": [float(s) for s in shares],
            "median_seed_share": float(np.median(shares)),
            "collapsed_seeds": int((shares >= 0.5).sum()),
            "seeds": int(shares.size),
            "share_explains_score": explained(scores, shares),
            "commonest_answer_share": float(
                np.max(np.unique(answers, return_counts=True)[1]) / answers.size),
        }
        row = out[label]
        print(f"{label}: forecast error {row['error']:.2f}x")
        print(f"  anchor answers {row['anchor_answer']!r} on {100 * row['anchor_share']:.0f}% "
              f"of prompts (score {row['anchor_score']:.3f})")
        print(f"  later seeds: median share {100 * row['median_seed_share']:.0f}%, "
              f"{row['collapsed_seeds']} of {row['seeds']} above half; answers "
              f"{row['seed_answers']}")
        print(f"  share explains {100 * row['share_explains_score']:.0f}% of the score variance; "
              f"the commonest correct answer covers {100 * row['commonest_answer_share']:.0f}% "
              f"of prompts")
    if out:
        path = io.save("s16_sweep", {"labels": list(LABELS)}, out)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
