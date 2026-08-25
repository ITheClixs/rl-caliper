"""P1: does propagated covariance predict seed divergence where a single timescale does not?

The enumerable policy admits three things at once: the exact linearised covariance recursion, the
exact Fisher information, and as many independent stochastic trajectories as we care to run. That
makes it possible to compare candidate models of run-to-run divergence against ground truth rather
than against each other.

Models compared:
  propagated   S_{t+1} = A_t S_t A_t^T + Q_t, scored as tr(F S_T)
  scalar       tau_c * D_noise, the single-timescale approximation
  random walk  noise accumulated with no contraction
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from caliper.exact.policy import fisher, make_prompt
from caliper.exact.pool import build
from caliper.exact.propagation import injected_covariance, propagate, seed_memory
from caliper.exact.train import ExactTrainer, RunConfig
from caliper.objectives.advantages import weight_table
from caliper.runtime import io


def single_timescale(trace: list[float], injected: float) -> float:
    """The scalar approximation: one exponential fitted to the divergence trace.

    This is what a single persistence time buys, and it is the comparator the propagated
    recursion has to beat.
    """
    from scipy.optimize import curve_fit

    steps = np.arange(1, len(trace) + 1, dtype=float)
    values = np.array(trace)

    def approach(step, level, tau):
        return level * (1.0 - np.exp(-2.0 * step / tau))

    try:
        (_, tau), _ = curve_fit(
            approach, steps, values, p0=[values[-1], len(trace) / 3],
            bounds=([0.0, 0.2], [np.inf, 1e4]), maxfev=40000,
        )
    except Exception:
        return float("nan")
    return float(tau * injected)


def exact_kl(reference: np.ndarray, other: np.ndarray) -> float:
    mask = other > 0
    return float(np.sum(other[mask] * (np.log(other[mask]) - np.log(reference[mask]))))


def monte_carlo(policy, accepts, estimator, group_size, prompts, step_size, steps, runs, seed):
    """Independent trajectories differing only in sampling randomness."""
    finals = []
    for index in range(runs):
        config = RunConfig(
            estimator=estimator,
            n_prompts=prompts,
            group_size=group_size,
            steps=steps,
            initial_step_size=step_size,
            controller_gain=0.0,      # freeze the step size: the theory is stated at fixed eta
            measure_every=10**9,
        )
        trainer = ExactTrainer(policy, accepts, config)
        trainer.run(np.random.default_rng(seed + 104729 * index))
        finals.append(trainer.policy)
    return finals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=3)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--estimator", default="rloo")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--prompt-counts", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--step-sizes", type=float, nargs="+", default=[0.3, 1.0])
    ap.add_argument("--diversities", type=float, nargs="+", default=[0.15, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = []
    header = f"{'div':>5s} {'G':>3s} {'P':>4s} {'eta':>5s} {'measured':>10s} {'propagated':>11s}"
    print(header + f" {'scalar':>10s} {'walk':>10s} {'H95':>4s}", flush=True)
    for diversity, group_size, prompts, step_size in itertools.product(
        args.diversities, args.group_sizes, args.prompt_counts, args.step_sizes
    ):
        policy, accepts = build(
            args.vocab, args.length, args.pool, (0.05, 0.95), diversity, args.seed
        )
        weights = weight_table(args.estimator, group_size)

        result = propagate(
            policy, accepts, args.estimator, group_size, prompts, step_size, args.steps
        )

        finals = monte_carlo(
            policy, accepts, args.estimator, group_size, prompts, step_size,
            args.steps, args.runs, args.seed + 1,
        )
        probabilities = [p.sequence_probs() for p in finals]
        measured = float(
            np.mean(
                [
                    exact_kl(probabilities[i], probabilities[j])
                    for i in range(len(finals))
                    for j in range(len(finals))
                    if i != j
                ]
            )
        )

        # random walk: noise accumulated with no contraction at all
        metric = np.mean([fisher(make_prompt(policy, a)) for a in accepts], axis=0)
        injected = step_size**2 * np.trace(
            metric @ injected_covariance(policy, accepts, weights, prompts)
        )
        walk = args.steps * injected

        # the single-timescale estimator this work previously relied on: fit one exponential to
        # the divergence trace, then multiply the injected drift by the fitted persistence time
        scalar = single_timescale(result.trace, injected)

        cells.append(
            {
                "diversity": diversity, "group_size": group_size, "prompts": prompts,
                "step_size": step_size, "measured_kl": measured,
                "propagated_kl": result.predicted_kl, "scalar_kl": scalar,
                "walk_kl": walk, "kernel": result.kernel, "trace": result.trace,
                "seed_memory_95": seed_memory(result.kernel, 0.95),
                "seed_memory_50": seed_memory(result.kernel, 0.50),
            }
        )
        print(
            f"{diversity:5.2f} {group_size:3d} {prompts:4d} {step_size:5.2f} {measured:10.3e} "
            f"{result.predicted_kl:11.3e} {scalar:10.3e} {walk:10.3e} "
            f"{cells[-1]['seed_memory_95']:4d}",
            flush=True,
        )

    measured = np.array([c["measured_kl"] for c in cells])
    for name in ("propagated_kl", "scalar_kl", "walk_kl"):
        predicted = np.array([c[name] for c in cells])
        ratio = predicted / measured
        error = np.abs(np.log(ratio))
        print(
            f"\n{name:14s} median |log ratio| {np.median(error):.3f}  "
            f"({np.exp(np.median(error)):.2f}x)   range {ratio.min():.2f}-{ratio.max():.2f}"
        )
    io.save("p1_propagation", vars(args), {"cells": cells})


if __name__ == "__main__":
    main()
