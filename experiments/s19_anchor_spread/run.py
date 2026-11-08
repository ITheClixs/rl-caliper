"""S19: would another run have frozen the same forecast?

The prospective test (s11) freezes its forecast from one anchor run. Its half-sample check splits
the prompts the injected term is estimated from, so it measures one source of error and no other:
the anchor's trajectory, and the metric gradient estimated where it ended, are shared by both
halves. On the support sweep the half-samples agree to within 2% at band 0.60 while the forecast is
2.80x the measured spread, so the check says nothing about whether that miss belongs to the corpus
or to the run that happened to be the anchor.

Every s11 run is reproducible from its index: all start from the same adapters and differ only in
their sampling keys. Retraining validation seed k with its states kept therefore reproduces it, and
the forecast the protocol would have frozen, had seed k been the anchor, can be computed after the
fact. It is graded against the spread over the other twelve runs, so no forecast is scored against
its own run. The anchor's own index is rerun too, as a check that this script reproduces the frozen
forecast. A few more forecasts from the anchor with other estimation keys separate the part of the
variation that comes from the estimate from the part that comes from the trajectory.

Reads the s11 record named by --label, which must have been run with --record-outputs so that the
anchor's per-prompt rates are stored.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from caliper.analysis.uncertainty import resolved_spread_interval
from caliper.real.adjoint import forecast
from caliper.real.train import RealRLConfig, RealRLTrainer
from caliper.runtime import io

S11 = Path(__file__).resolve().parents[1] / "s11_prospective" / "run.py"


def s11():
    """The prospective test's own training and evaluation code, so nothing here re-implements it."""
    spec = importlib.util.spec_from_file_location("s11_run", S11)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(label: str) -> dict:
    hits = [r for r in io.load_all("s11_prospective") if r["result"].get("label") == label]
    if len(hits) != 1:
        raise SystemExit(f"{label}: {len(hits)} s11 records, not one")
    if "anchor_per_prompt" not in hits[0]["result"]:
        raise SystemExit(f"{label} was not run with --record-outputs")
    return hits[0]


def eval_key(run: int) -> mx.array:
    """The evaluation key s11 gave run `run`: 555 for the anchor, 555 + 101 s for seed s."""
    return mx.random.key(555 + 101 * run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="an s11 record run with --record-outputs")
    ap.add_argument("--anchors", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--extra-keys", type=int, default=2,
                    help="further forecasts from run 0 with other estimation keys")
    args = ap.parse_args()

    source = record(args.label)
    cfg, result = source["manifest"]["config"], source["result"]
    if cfg["optimiser"] != "sgd":
        raise SystemExit("only the plain-ascent form is reproduced here")
    # every run's per-prompt held-out rates, run 0 being the original anchor
    per_prompt = np.array([result["anchor_per_prompt"]] + result["per_prompt"])
    runs = per_prompt.shape[0]
    if any(k < 0 or k >= runs for k in args.anchors):
        raise SystemExit(f"anchors must be run indices below {runs}")

    helpers = s11()
    pool = json.loads(Path(cfg["prompts_file"]).read_text())
    order = np.random.default_rng(cfg["seed"]).permutation(len(pool))
    held_out = [pool[i] for i in order[: cfg["eval_prompts"]]]
    corpus = [pool[i] for i in order[cfg["eval_prompts"]:]]
    config = RealRLConfig(
        group_size=cfg["group_size"], prompts=cfg["prompts"], blocks=8,
        max_tokens=cfg["max_tokens"], lora_layers=cfg["lora_layers"], steps=cfg["steps"],
        learning_rate=cfg["learning_rate"], optimiser=cfg["optimiser"], seed=cfg["seed"],
    )
    model, tokenizer = load(cfg["model"])
    initial = helpers.adapter_state(RealRLTrainer(model, tokenizer, config).model)
    del model

    def frozen_forecast(trainer, states, key: int, seed: int) -> dict:
        prediction = forecast(
            trainer, states, corpus, held_out, cfg["learning_rate"], cfg["prompts"],
            mx.random.key(key), seed=seed, transport=False,
            variance_prompts=cfg["variance_prompts"], check_agreement=False,
            optimiser_states=None, cross_fit=cfg["cross_fit"],
        )
        halves = [float(np.sqrt(max(v, 0.0))) for v in prediction.split_variance]
        return {"std": float(prediction.std), "half_samples": halves}

    anchors, estimates = [], []
    for k in args.anchors:
        started = time.time()
        trainer, states, _, live, _, _ = helpers.train_run(
            cfg["model"], config, corpus, initial, k, keep=True)
        rates, _ = helpers.evaluate(trainer, held_out, cfg["eval_samples"], eval_key(k))
        drift = float(np.abs(rates - per_prompt[k]).max())
        made = frozen_forecast(trainer, states, 4242, cfg["seed"])
        # graded against every run but its own
        measured = resolved_spread_interval(
            np.delete(per_prompt, k, axis=0), cfg["eval_samples"], seed=cfg["seed"])
        row = {"run": k, "reproduced": drift == 0.0, "score_drift": drift, "live_share": live,
               "forecast": made["std"], "half_samples": made["half_samples"],
               "measured": measured, "ratio": made["std"] / measured["resolved"]}
        anchors.append(row)
        print(f"run {k:2d}: forecast {row['forecast']:.4f} (halves {made['half_samples'][0]:.4f}"
              f" / {made['half_samples'][1]:.4f}) against {measured['resolved']:.4f} over the "
              f"other {runs - 1}, ratio {row['ratio']:.2f}; scores reproduced: "
              f"{row['reproduced']} ({(time.time() - started) / 60:.1f} min)", flush=True)
        if k == 0:
            for j in range(1, args.extra_keys + 1):
                again = frozen_forecast(trainer, states, 4242 + j, cfg["seed"] + j)
                estimates.append({"key": 4242 + j, **again})
                print(f"  run 0, estimation key {4242 + j}: forecast {again['std']:.4f}",
                      flush=True)
        del trainer, states

    forecasts = np.array([a["forecast"] for a in anchors])
    summary = {
        "frozen_forecast": result["predicted_std"],
        "anchors": anchors,
        "estimation_repeats": estimates,
        "forecast_range": float(forecasts.max() / forecasts.min()),
        "median_ratio": float(np.median([a["ratio"] for a in anchors])),
    }
    print(f"forecasts span {summary['forecast_range']:.2f}x over {len(anchors)} anchors; "
          f"median ratio {summary['median_ratio']:.2f}", flush=True)
    path = io.save("s19_anchor_spread",
                   {"label": args.label, "anchors": args.anchors,
                    "extra_keys": args.extra_keys}, summary)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
