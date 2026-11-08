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

import argparse

import numpy as np

from caliper.runtime import io

# the first test (docs/predictions.md); the second passes all four bands with --labels
LABELS = ("modes-band-0.85", "modes-band-0.60")


def collapse(outputs: list[dict]) -> tuple[str, float]:
    """A run's most common answer across prompts, and the share of prompts it is the mode of."""
    modes = [o["mode"] for o in outputs]
    values, counts = np.unique(modes, return_counts=True)
    return str(values[counts.argmax()]), float(counts.max() / len(modes))


def split_prompts(runs: list[list[dict]], firm: float = 0.9) -> int:
    """Prompts on which at least two runs answer firmly, with different answers.

    This is the criterion committed in docs/predictions.md before the band-0.60 result: a run
    is firm on a prompt when one string is at least `firm` of its samples.
    """
    count = 0
    for j in range(len(runs[0])):
        answers = {run[j]["mode"] for run in runs if run[j]["share"] >= firm}
        count += len(answers) >= 2
    return count


def shared_prompts(runs: list[list[dict]], firm: float = 0.9) -> int:
    """Prompts on which every run answers firmly with the same string.

    Not part of the committed criterion: added after the band-0.60 forecast turned out to be too
    large rather than too small. Runs absorbed into one shared answer have no spread to show on
    that prompt, whatever a linearised forecast carried forward for it.
    """
    count = 0
    for j in range(len(runs[0])):
        firm_answers = [run[j]["mode"] for run in runs if run[j]["share"] >= firm]
        count += len(firm_answers) == len(runs) and len(set(firm_answers)) == 1
    return count


def explained(scores: np.ndarray, shares: np.ndarray) -> float:
    """Squared correlation between a run's collapse share and its held-out score."""
    if scores.size < 3 or np.std(shares) == 0 or np.std(scores) == 0:
        return float("nan")
    return float(np.corrcoef(scores, shares)[0, 1] ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=list(LABELS))
    labels = ap.parse_args().labels
    records = {r["result"]["label"]: r for r in io.load_all("s11_prospective")
               if r["result"].get("label") in labels}
    out = {}
    for label in labels:
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
            "split_prompts": split_prompts([result["anchor_outputs"]] + result["seed_outputs"]),
            "shared_prompts": shared_prompts([result["anchor_outputs"]] + result["seed_outputs"]),
            "ratio_direction": "too large" if result["ratio"] > 1 else "too small",
            "prompts": len(result["anchor_outputs"]),
            "commonest_answer_share": float(
                np.max(np.unique(answers, return_counts=True)[1]) / answers.size),
        }
        row = out[label]
        print(f"{label}: forecast error {row['error']:.2f}x ({row['ratio_direction']})")
        print(f"  shared prompts (post hoc): {row['shared_prompts']} of {row['prompts']}")
        print(f"  anchor answers {row['anchor_answer']!r} on {100 * row['anchor_share']:.0f}% "
              f"of prompts (score {row['anchor_score']:.3f})")
        print(f"  later seeds: median share {100 * row['median_seed_share']:.0f}%, "
              f"{row['collapsed_seeds']} of {row['seeds']} above half; answers "
              f"{row['seed_answers']}")
        print(f"  split prompts: {row['split_prompts']} of {row['prompts']}")
        print(f"  share explains {100 * row['share_explains_score']:.0f}% of the score variance; "
              f"the commonest correct answer covers {100 * row['commonest_answer_share']:.0f}% "
              f"of prompts")
    # a record named by its labels must hold all of them, or a dry run would stand in for the test
    if len(out) == len(labels):
        path = io.save("s16_sweep", {"labels": list(labels)}, out)
        print(f"wrote {path}")
    else:
        print("not saved: some bands have not run")


if __name__ == "__main__":
    main()
