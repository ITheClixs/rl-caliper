"""D1: the prefill-to-decode cost ratio that sets the compute-optimal group size.

Equation (16) inflates the statistically optimal group size by sqrt(1 + c_pre/c_dec), where c_pre
is the cost paid once per prompt and c_dec the cost paid once per response. Both are properties of
the serving stack, so they are measured here rather than assumed: prefill of a shared prompt, and
decode of a batch of streams, on this machine.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from caliper.runtime import io


def timed(fn, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repeats):
        mx.eval(fn())
    return (time.perf_counter() - start) / repeats


def prefill_seconds(model, length: int, batch: int, repeats: int) -> float:
    tokens = mx.array(np.random.randint(0, 1000, size=(batch, length)))

    def run():
        cache = make_prompt_cache(model)
        return model(tokens, cache=cache)

    return timed(run, repeats)


def decode_seconds(model, context: int, batch: int, repeats: int) -> float:
    """Time for one decode step over `batch` streams that already hold `context` tokens.

    The cache is built once at batch one and broadcast, which is both cheaper to set up and the
    situation the cost model describes: responses in a group share their prompt's prefill.
    """
    prompt = mx.array(np.random.randint(0, 1000, size=(1, context)))
    cache = make_prompt_cache(model)
    mx.eval(model(prompt, cache=cache))
    for layer in cache:
        keys, values = layer.state
        layer.keys = mx.repeat(keys, batch, axis=0)
        layer.values = mx.repeat(values, batch, axis=0)
        layer.offset = keys.shape[2]
    step = mx.array(np.random.randint(0, 1000, size=(batch, 1)))

    def run():
        return model(step, cache=cache)

    return timed(run, repeats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--prompt-lengths", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--response-lengths", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()

    model, _ = load(args.model)
    rows = []

    print(f"{'prompt':>7s} {'streams':>8s} {'prefill ms':>11s} {'decode ms/tok':>14s}")
    for length in args.prompt_lengths:
        for streams in args.group_sizes:
            batch = args.prompts * streams
            prefill = prefill_seconds(model, length, args.prompts, args.repeats)
            decode = decode_seconds(model, length, batch, args.repeats)
            rows.append(
                {
                    "prompt_length": length,
                    "group_size": streams,
                    "streams": batch,
                    "prefill_seconds": prefill,
                    "decode_seconds_per_token": decode,
                }
            )
            print(f"{length:7d} {batch:8d} {1000 * prefill:11.2f} {1000 * decode:14.2f}")

    print(f"\n{'prompt':>7s} {'response':>9s} {'G':>4s} {'c_pre/c_dec':>12s} {'inflation':>10s}")
    ratios = []
    for length in args.prompt_lengths:
        for response in args.response_lengths:
            for streams in args.group_sizes:
                row = next(
                    r
                    for r in rows
                    if r["prompt_length"] == length and r["group_size"] == streams
                )
                # cost paid once per prompt, against cost paid once per response
                c_pre = row["prefill_seconds"] / args.prompts
                c_dec = response * row["decode_seconds_per_token"] / row["streams"]
                ratio = c_pre / c_dec
                ratios.append(
                    {
                        "prompt_length": length,
                        "response_length": response,
                        "group_size": streams,
                        "cost_ratio": ratio,
                        "inflation": float(np.sqrt(1 + ratio)),
                    }
                )
                if streams in (1, 8, 32):
                    print(
                        f"{length:7d} {response:9d} {streams:4d} {ratio:12.3f} "
                        f"{np.sqrt(1 + ratio):10.2f}"
                    )

    io.save("d1_cost_model", vars(args), {"timings": rows, "ratios": ratios})


if __name__ == "__main__":
    main()
