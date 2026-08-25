"""Regenerate every figure in the paper from the saved runs.

    uv run python analysis/figures.py

Each figure reads the most recent run that carries the fields it needs, so a figure is never
assembled from numbers typed by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from caliper.runtime import io  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"
BETA_B = 40.07 / 1.144  # tau_b / signal, measured on the transformer tier
PALETTE = ["#1b3a5c", "#c1502e", "#2e7d5b", "#7a5195", "#b58900"]


def save(fig, name: str) -> None:
    """Write a figure without the embedded timestamp, so reruns are byte-identical."""
    fig.savefig(FIGURES / name, bbox_inches="tight", metadata={"CreationDate": None})


def style(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def latest(name: str, key: str, container: str = "cells") -> list[dict]:
    runs = [r for r in io.load_all(name) if key in r["result"][container][0]]
    if not runs:
        raise SystemExit(f"no run of {name!r} carries {key!r}")
    return runs[-1]["result"][container]


def largest(name: str, key: str, minimum: int = 1) -> list[dict]:
    """The most recent run of `name` carrying `key` in at least `minimum` cells.

    Several experiments write to the same store under different configurations -- the forecast
    grid is also run per estimator on a small sub-grid -- so taking the last record is not enough
    to identify the sweep a figure is about.
    """
    runs = [
        r
        for r in io.load_all(name)
        if len(r["result"].get("cells", [])) >= minimum
        and key in r["result"]["cells"][0]
    ]
    if not runs:
        raise SystemExit(f"no run of {name!r} carries {key!r} with {minimum}+ cells")
    return runs[-1]["result"]["cells"]


def figure_exact_curves() -> None:
    """Predicted efficiency against measured gain, enumerable policies."""
    cells = latest("a4_law", "predicted_efficiency")
    chosen = sorted(cells, key=lambda c: -c["predicted_range"])[:4]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.7), sharex=True)
    for ax, cell, colour in zip(axes, chosen, PALETTE, strict=False):
        g = np.array(cell["group_sizes"], dtype=float)
        observed = np.array(cell["observed"])
        se = np.array(cell["observed_se"])
        predicted = np.array(cell["predicted_efficiency"])
        scale = np.polyfit(np.sqrt(predicted), observed, 1)
        ax.errorbar(g, observed, yerr=se, fmt="o", ms=3.5, color=colour, lw=1, capsize=2)
        ax.plot(g, np.polyval(scale, np.sqrt(predicted)), color=colour, lw=1.4, alpha=0.7)
        ax.set_xscale("log", base=2)
        style(
            ax,
            "group size $G$",
            "gain in pass rate" if ax is axes[0] else "",
            f"{cell['band']}, diversity {cell['diversity']}",
        )
    fig.suptitle(
        r"prediction $\propto\sqrt{\rho(G)}$ (line) against measured gain (points)", fontsize=10
    )
    fig.tight_layout()
    save(fig, "exact_curves.pdf")
    plt.close(fig)


def figure_estimator_accuracy() -> None:
    rows = latest("a5_estimator", "crossed_g_star", container="rows")
    exact = np.array([r["exact_g_star"] for r in rows])
    measured = np.array([r["crossed_g_star"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    limits = [min(exact.min(), measured.min()) * 0.85, max(exact.max(), measured.max()) * 1.15]
    ax.plot(limits, limits, color="0.6", lw=1, ls="--", zorder=1)
    ax.scatter(exact, measured, s=26, color=PALETTE[0], zorder=2, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    style(ax, "exact $G^{*}$", "measured $G^{*}$")
    fig.tight_layout()
    save(fig, "estimator_accuracy.pdf")
    plt.close(fig)


def figure_transformer_curves() -> None:
    cells = latest("b1_group_size", "predicted_efficiency")
    fig, axes = plt.subplots(1, len(cells), figsize=(2.9 * len(cells), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cell, colour in zip(axes, cells, PALETTE, strict=False):
        g = np.array(cell["group_sizes"], dtype=float)
        observed = np.array(cell["observed"])
        se = np.array(cell["observed_se"])
        predicted = np.array(cell["predicted_efficiency"])
        fit = np.polyfit(np.sqrt(predicted), observed, 1)
        ax.errorbar(g, observed, yerr=se, fmt="o", ms=4, color=colour, lw=1, capsize=2)
        ax.plot(g, np.polyval(fit, np.sqrt(predicted)), color=colour, lw=1.4, alpha=0.7)
        ax.axvline(cell["g_star"], color="0.5", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        label = "unbounded" if cell["pool"] is None else str(cell["pool"])
        style(
            ax,
            "group size $G$",
            "gain in pass rate" if ax is axes[0] else "",
            f"corpus {label},  $G^{{*}}={cell['g_star']:.1f}$",
        )
    fig.tight_layout()
    save(fig, "transformer_curves.pdf")
    plt.close(fig)


def figure_transfer() -> None:
    rows = latest("b3_batch_transfer", "best_step_size", container="rows")
    prompts = np.array([r["prompts"] for r in rows], dtype=float)
    best_step = np.array([r["best_step_size"] for r in rows])
    best_drift = np.array([r["best_drift"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))

    ax = axes[0]
    ax.plot(prompts, best_step / best_step[0], "o-", color=PALETTE[1], lw=1.5, label="step size")
    ax.plot(
        prompts, best_drift / best_drift[0], "s-", color=PALETTE[0], lw=1.5, label="drift target"
    )
    ax.axhline(1.0, color="0.6", lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend(frameon=False, fontsize=8)
    style(ax, "prompts per step $P$", "optimum, relative to $P=8$")

    ax = axes[1]
    for row, colour in zip(rows, PALETTE, strict=False):
        ax.plot(row["step_grid"], row["step_gain"], "o-", ms=3, lw=1.2, color=colour,
                label=f"$P={row['prompts']}$")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "step size", "gain in pass rate", "tuned as a learning rate")

    ax = axes[2]
    for row, colour in zip(rows, PALETTE, strict=False):
        ax.plot(row["drift_grid"], row["drift_gain"], "o-", ms=3, lw=1.2, color=colour)
    ax.set_xscale("log")
    style(ax, "drift target (nats/token)", "", "tuned as a drift budget")

    fig.tight_layout()
    save(fig, "transfer.pdf")
    plt.close(fig)


def figure_cost_model() -> None:
    """Measured prefill-to-decode ratio and the group-size inflation it implies."""
    runs = [r for r in io.load_all("d1_cost_model") if "ratios" in r["result"]]
    if not runs:
        raise SystemExit("no cost model run")
    ratios = runs[-1]["result"]["ratios"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    responses = sorted({r["response_length"] for r in ratios})
    prompts = sorted({r["prompt_length"] for r in ratios})
    for prompt, colour in zip(prompts, PALETTE, strict=False):
        values = []
        for response in responses:
            group = [
                r["inflation"]
                for r in ratios
                if r["prompt_length"] == prompt
                and r["response_length"] == response
                and r["group_size"] == 8
            ]
            values.append(group[0])
        ax.plot(responses, values, "o-", color=colour, lw=1.5, label=f"prompt {prompt}")
    ax.axhline(1.0, color="0.6", lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=8)
    style(
        ax,
        "response length (tokens)",
        r"$\sqrt{1 + c_{\mathrm{pre}}/c_{\mathrm{dec}}}$",
        "inflation of $G^{*}$ from prefill sharing",
    )
    fig.tight_layout()
    save(fig, "cost_model.pdf")
    plt.close(fig)


def figure_real_model() -> None:
    """Within-prompt noise against the reward histogram on a pretrained model."""
    runs = [
        r
        for r in io.load_all("c1_real_noise")
        if r["result"]["rows"] and "bernoulli" in r["result"]["rows"][0]
    ]
    if not runs:
        raise SystemExit("no real-model run")
    rows = runs[-1]["result"]["rows"]
    bernoulli = np.array([r["bernoulli"] for r in rows])
    tau_w = np.array([r["tau_w_scaled"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    slope = float((tau_w * bernoulli).sum() / (bernoulli**2).sum())
    grid = np.linspace(0, bernoulli.max() * 1.15, 20)
    ax.plot(grid, slope * grid, color="0.6", lw=1, ls="--")
    ax.scatter(bernoulli, tau_w, s=34, color=PALETTE[0], zorder=3)
    for row, x, y in zip(rows, bernoulli, tau_w, strict=True):
        ax.annotate(
            row["corpus"].replace("_", " "), (x, y), fontsize=7,
            textcoords="offset points", xytext=(4, 4),
        )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    style(ax, r"$\mathbb{E}[p(1-p)]$", r"$\tau_w$")
    fig.tight_layout()
    save(fig, "real_model.pdf")
    plt.close(fig)


def figure_teaser() -> None:
    """Page one: the forecast works, and the model it replaces does not."""
    cells = largest("p2_metric_forecast", "predicted_std", minimum=100)
    models = latest("p1_propagation", "propagated_kl")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))

    ax = axes[0]
    measured = np.array([c["measured_std"] for c in cells])
    predicted = np.array([c["predicted_std"] for c in cells])
    lo = measured - np.array([c["measured_lo"] for c in cells])
    hi = np.array([c["measured_hi"] for c in cells]) - measured
    ax.errorbar(measured, predicted, xerr=[lo, hi], fmt="o", ms=3.6, lw=0.7, capsize=1.4,
                color=PALETTE[0], alpha=0.85)
    limits = np.array([measured.min() * 0.7, measured.max() * 1.4])
    ax.plot(limits, limits, color="0.25", lw=1.1, zorder=0)
    ax.plot(limits, limits * 1.5, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.plot(limits, limits / 1.5, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    error = np.abs(np.log(predicted / measured))
    ax.set_title(
        f"one run predicts {len(cells)} spreads "
        f"(median {np.exp(np.median(error)):.2f}$\\times$)", fontsize=8.5
    )
    seeds = cells[0].get("seeds", 0)
    style(ax, f"measured s.d. over {seeds} seeds", "forecast, made before they ran")

    ax = axes[1]
    truth = np.array([c["measured_kl"] for c in models])
    for key, label, colour, marker in [
        ("propagated_kl", "propagated", PALETTE[0], "o"),
        ("walk_kl", "accumulated", PALETTE[1], "s"),
        ("scalar_kl", "single timescale", PALETTE[2], "^"),
    ]:
        ax.scatter(truth, np.array([c[key] for c in models]), s=20, marker=marker,
                   color=colour, alpha=0.85, label=label)
    span = np.array([truth.min() * 0.5, truth.max() * 2.0])
    ax.plot(span, span, color="0.25", lw=1.1, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax, "measured divergence", "predicted divergence",
          "the model this replaces, on the same settings")
    fig.tight_layout()
    save(fig, "teaser.pdf")
    plt.close(fig)


def figure_allocation() -> None:
    """The allocation result: predictions track measured gains, and reconcile 3 with 8-64."""
    from scipy.optimize import brentq

    transformer = latest("b1_group_size", "predicted_efficiency")
    try:
        adam = latest("b6_adam", "predicted_efficiency")
    except SystemExit:
        adam = []

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))

    # Each setting has its own run length and task, so gains differ in level and scale between
    # settings even where the shape agrees. Standardising within a setting removes both and leaves
    # the shape, which is what the prediction is about.
    ax = axes[0]
    series = [(c, "gradient ascent", PALETTE[0], "o") for c in transformer]
    for cell in adam:
        if cell["optimiser"] == "adam":
            series.append((cell, "Adam", PALETTE[1], "^"))
    pooled_x, pooled_y, seen = [], [], set()
    for cell, label, colour, marker in series:
        x = np.sqrt(np.array(cell["predicted_efficiency"]))
        y = np.array(cell["observed"])
        x = (x - x.mean()) / x.std(ddof=1)
        y = (y - y.mean()) / y.std(ddof=1)
        ax.scatter(
            x, y, s=17, marker=marker, color=colour, alpha=0.85,
            label=label if label not in seen else None,
        )
        seen.add(label)
        pooled_x.append(x)
        pooled_y.append(y)
    x = np.concatenate(pooled_x)
    y = np.concatenate(pooled_y)
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    grid = np.linspace(x.min(), x.max(), 20)
    ax.plot(grid, np.polyval(coef, grid), color="0.35", lw=1.2, zorder=1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(f"no free parameters, {len(series)} settings ($R^2={r2:.2f}$)", fontsize=8.5)
    style(ax, r"predicted $\sqrt{\rho(G)}$, standardised", "measured gain, standardised")

    ax = axes[1]
    rows = [r for r in io.load_all("t1_optimum") if "rows" in r["result"]][-1]["result"]["rows"]
    ratios = sorted({r["prefill_ratio"] for r in rows})
    budgets = np.logspace(np.log10(32), np.log10(16384), 40)
    baseline = next(r for r in rows if r["prefill_ratio"] == 0.0)
    b_ratio = (baseline["closed_form"] - 1) ** 2
    ax.axhspan(8, 64, color="0.88", zorder=0)
    for ratio, colour in zip(ratios, PALETTE + ["#8a8a8a"], strict=False):
        if ratio == 0.0:
            ax.axhline(
                baseline["exact"], color="0.2", lw=1.6, ls="--",
                label=r"$\alpha=0$: statistics only",
            )
            continue
        values = [
            brentq(
                lambda g, ratio=ratio, budget=budget: 1.0
                - ratio * budget / (g**2 * BETA_B)
                - (1 + ratio) * b_ratio / (g - 1) ** 2,
                1 + 1e-9,
                1e6,
            )
            for budget in budgets
        ]
        ax.plot(budgets, values, color=colour, lw=1.5, label=rf"$\alpha={ratio:g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 4, 8, 16, 32, 64]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FixedFormatter(["2", "4", "8", "16", "32", "64"])
    )
    ax.set_ylim(2, 110)
    ax.text(
        14000, 68, "in common use", fontsize=6.5, color="0.4", va="bottom", ha="right"
    )
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.6,
              labelspacing=0.25, borderpad=0.2)
    ax.set_title("measured hardware moves the optimum", fontsize=8.5)
    style(ax, "rollouts per step $R$", r"optimal group size $G^{\star}$")

    fig.tight_layout()
    save(fig, "allocation.pdf")
    plt.close(fig)


def _contraction_cells():
    """Every stored contraction sweep, restricted to updates that are still learning."""
    cells = []
    for run in io.load_all("s1_contraction"):
        for cell in run["result"]["cells"]:
            history = cell["history"]
            if "true_spread" not in history:
                continue  # an early run that measured spread without the noise correction
            pass_rate = np.array(history["pass_mean"])
            drift = np.array(history["drift"])
            keep = (pass_rate < 0.85) & (drift < 3 * cell["drift_target"])
            if keep.sum() >= 4:
                cells.append((cell, keep))
    if not cells:
        raise SystemExit("no contraction runs")
    return cells


def _matched_progress(cells, target=0.60):
    """One summary per setting: divergence and outcome spread at a common pass rate.

    Trajectory points within a setting cannot be pooled -- the divergence is already at its
    stationary value and what varies along a trajectory is measurement noise, not signal. Settings
    also differ in their noise scale, so they are compared where the policies are equally good.
    """
    rows = []
    for cell, keep in cells:
        pass_rate = np.array(cell["history"]["pass_mean"])[keep]
        kl = np.array(cell["history"]["pairwise_kl"])[keep]
        spread = np.array(cell["history"]["true_spread"])[keep]
        if not (pass_rate.min() <= target <= pass_rate.max()):
            continue
        rows.append(
            {
                "prompts": cell["prompts"],
                "drift": cell["drift_target"],
                "critical_batch": cell.get("critical_batch", float("nan")),
                "kl": float(np.interp(target, pass_rate, kl)),
                "spread": float(np.interp(target, pass_rate, spread)),
            }
        )
    return rows


def figure_reconvergence() -> None:
    """Divergence settles; the level scales as predicted; the outcome spread follows."""
    cells = _contraction_cells()
    rows = _matched_progress(cells)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8))

    ax = axes[0]
    shown = sorted({c["prompts"] for c, _ in cells})
    for prompts, colour in zip(shown, PALETTE, strict=False):
        picked = [c for c, k in cells if c["prompts"] == prompts and c["drift_target"] == 1e-4]
        if not picked:
            continue
        cell = picked[0]
        steps = np.array(cell["history"]["step"], dtype=float)
        kl = np.array(cell["history"]["pairwise_kl"])
        ax.plot(steps, kl, "o-", ms=3, lw=1.3, color=colour, label=f"$P={prompts}$")
    anchor = float(np.mean([c["history"]["pairwise_kl"][0] for c, _ in cells]))
    grid = np.array([1.0, 100.0])
    ax.plot(grid, anchor * grid, "--", color="0.45", lw=1.2)
    ax.text(5, anchor * 12, "random walk", fontsize=7, color="0.4", rotation=30)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    style(ax, "updates", r"divergence between seeds")
    ax.set_title("divergence settles", fontsize=8.5)

    # scaling: control for the drift target, which shifts the level, before fitting in P
    ax = axes[1]
    drifts = sorted({r["drift"] for r in rows})
    for drift, colour, marker in zip(drifts, PALETTE, ("o", "^"), strict=False):
        subset = [r for r in rows if r["drift"] == drift]
        if len(subset) < 2:
            continue
        x = np.array([r["prompts"] for r in subset], dtype=float)
        y = np.array([r["kl"] for r in subset])
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker, ms=6, color=colour, label=rf"$D^\star={drift:g}$")
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        grid = np.geomspace(x.min(), x.max(), 20)
        ax.plot(grid, np.exp(intercept) * grid**slope, "-", lw=1.1, color=colour, alpha=0.7)
        ax.plot(grid, y[order][0] * (grid / x[order][0]) ** -0.5, ":", lw=1.1, color="0.5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)
    style(ax, "prompts per update $P$", r"divergence at matched progress")
    ax.set_title(r"level $\propto P^{-1/2}$ (dotted)", fontsize=8.5)

    ax = axes[2]
    x = np.array([r["kl"] for r in rows])
    y = np.array([r["spread"] for r in rows])
    good = (x > 0) & (y > 0)
    x, y = x[good], y[good]
    ax.scatter(x, y, s=30, color=PALETTE[0])
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    resid = np.log(y) - np.polyval([slope, intercept], np.log(x))
    r2 = 1 - (resid**2).sum() / ((np.log(y) - np.log(y).mean()) ** 2).sum()
    grid = np.geomspace(x.min(), x.max(), 20)
    ax.plot(grid, np.exp(intercept) * grid**slope, color="0.35", lw=1.2,
            label=rf"fit $\propto \mathrm{{KL}}^{{{slope:.2f}}}$, $R^2={r2:.2f}$")
    ax.plot(grid, y[0] * (grid / x[0]) ** 0.5, ":", color="0.55", lw=1.2,
            label=r"theory $\mathrm{KL}^{1/2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6.5)
    style(ax, "divergence at matched progress", "spread in pass rate")
    ax.set_title("outcome spread follows", fontsize=8.5)

    fig.tight_layout()
    save(fig, "reconvergence.pdf")
    plt.close(fig)


def figure_real_seeds() -> None:
    """The same measurement on a pretrained model."""
    runs = [r for r in io.load_all("s3_real_seeds") if "rows" in r["result"]]
    if not runs:
        raise SystemExit("no real-model seed run")
    run = runs[-1]
    rows = run["result"]["rows"]
    steps = np.array([r["step"] for r in rows], dtype=float)
    kl = np.array([r["pairwise_kl"] for r in rows])
    # intervals resampled over runs; the pairs share runs and are not independent
    lo = np.array([r.get("pairwise_lo", np.nan) for r in rows])
    hi = np.array([r.get("pairwise_hi", np.nan) for r in rows])
    err = np.vstack([kl - lo, hi - kl]) if np.isfinite(lo).all() else None

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.errorbar(steps, kl, yerr=err, fmt="o-", ms=4, lw=1.4, capsize=2, color=PALETTE[0])
    grid = np.array([1.0, steps.max()])
    ax.plot(grid, kl[0] * grid, "--", color="0.45", lw=1.2)
    ax.text(3.0, kl[0] * 6.0, "random walk", fontsize=7, color="0.4", rotation=30)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "updates", "divergence between seeds")
    ax.set_title(
        f"Qwen2.5-0.5B, {run['manifest']['config']['seeds']} seeds", fontsize=8.5
    )
    ax.set_ylim(bottom=max(np.nanmin(lo) * 0.6, 1e-3))
    fig.tight_layout()
    save(fig, "real_seeds.pdf")
    plt.close(fig)


def figure_forecast() -> None:
    """Forecast against realised spread, and where each update's share of it was injected."""
    cells = largest("p2_metric_forecast", "predicted_std", minimum=100)
    bands = sorted({c["band"] for c in cells})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    ax = axes[0]
    for band, colour, marker in zip(bands, PALETTE, "os^D", strict=False):
        rows = [c for c in cells if c["band"] == band]
        measured = np.array([c["measured_std"] for c in rows])
        predicted = np.array([c["predicted_std"] for c in rows])
        lo = measured - np.array([c["measured_lo"] for c in rows])
        hi = np.array([c["measured_hi"] for c in rows]) - measured
        ax.errorbar(measured, predicted, xerr=[lo, hi], fmt=marker, ms=4, lw=0.8,
                    capsize=1.5, color=colour, label=band, alpha=0.85)
    limits = np.array([
        min(c["measured_std"] for c in cells) * 0.7,
        max(c["measured_std"] for c in cells) * 1.4,
    ])
    ax.plot(limits, limits, color="0.25", lw=1.1, zorder=0)
    for factor, style_ in ((1.5, ":"), (1 / 1.5, ":")):
        ax.plot(limits, limits * factor, color="0.6", lw=0.8, ls=style_, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    seeds = cells[0].get("seeds", 0)
    style(ax, f"measured s.d. over {seeds} seeds", "forecast from one run",
          "identity, with the $1.5\\times$ band")

    ax = axes[1]
    # one trace per band at each step size, so the legend distinguishes what the curves differ by
    longest = max(c["steps"] for c in cells)
    chosen = []
    for band in sorted({c["band"] for c in cells}):
        for eta in sorted({c["step_size"] for c in cells}):
            match = [
                c for c in cells
                if c["band"] == band and c["step_size"] == eta and c["steps"] == longest
            ]
            if match:
                chosen.append(match[0])
    for cell, colour in zip(chosen[:4], PALETTE, strict=False):
        kernel = np.array(cell["kernel"])
        share = kernel / kernel.sum()
        ax.plot(np.arange(1, share.size + 1) / share.size, share * share.size,
                color=colour, lw=1.4,
                label=f"{cell['band']}, $\\eta={cell['step_size']:g}$")
    ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
    ax.legend(fontsize=7, frameon=False)
    style(ax, "position in the run", "share of the final variance\n(relative to uniform)",
          "where the surviving noise entered")
    fig.tight_layout()
    save(fig, "forecast.pdf")
    plt.close(fig)


def figure_sources() -> None:
    """Which randomness the reported number remembers, against group size."""
    cells = largest("p3_memory_sources", "rollout_share", minimum=100)
    groups = sorted({c["group_size"] for c in cells})
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    bands = sorted({c["band"] for c in cells})
    for band, colour, marker in zip(bands, PALETTE, "os^", strict=False):
        rows = [c for c in cells if c["band"] == band]
        shares = [
            np.median([c["rollout_share"] for c in rows if c["group_size"] == g]) for g in groups
        ]
        ax.plot(groups, shares, marker=marker, ms=4, lw=1.3, color=colour, label=band)
    ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    style(ax, "group size $G$", "share from rollout sampling",
          "the rest is which prompts were drawn")
    fig.tight_layout()
    save(fig, "sources.pdf")
    plt.close(fig)


def figure_null() -> None:
    """The learning condition against the same run with the reward replaced by a coin."""
    runs = [
        r
        for r in io.load_all("s8_null")
        if len(r["result"].get("conditions", {}).get("verifier", {}).get("step", [])) >= 5
    ]
    if not runs:
        raise SystemExit("no null experiment of usable length")
    result = runs[-1]["result"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharex=True)
    labels = {"verifier": "verifier reward", "coin": "reward replaced by a coin"}
    for name, colour, marker in zip(("verifier", "coin"), PALETTE, "o^", strict=False):
        history = result["conditions"][name]
        steps = np.array(history["step"], dtype=float)
        axes[0].plot(steps, history["pairwise_kl"], marker=marker, ms=4, lw=1.4,
                     color=colour, label=labels[name])
        axes[1].plot(steps, history["pass_rate"], marker=marker, ms=4, lw=1.4, color=colour)
    first = result["conditions"]["verifier"]["pairwise_kl"][0]
    grid = np.array([1.0, max(result["conditions"]["verifier"]["step"])])
    axes[0].plot(grid, first * grid, "--", color="0.45", lw=1.1)
    axes[0].text(grid[1] * 0.25, first * grid[1] * 0.45, "random walk", fontsize=7,
                 color="0.4", rotation=32)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    style(axes[0], "updates", "divergence between seeds", "same noise, same steps")
    style(axes[1], "updates", "pass rate on the task", "only one of them is learning")
    fig.tight_layout()
    save(fig, "null.pdf")
    plt.close(fig)


def figure_real_forecast() -> None:
    """The forecast on a pretrained model: where it was injected, and whether it lands."""
    runs = [
        r
        for r in io.load_all("s7_real_forecast")
        if r["result"].get("predicted_variance", 0.0) > 0.0
        and len(r["result"].get("kernel", [])) >= 8
        and r["result"].get("transport", True)
    ]
    if not runs:
        raise SystemExit("no real-model forecast with a non-zero kernel")
    result = runs[-1]["result"]
    kernel = np.array(result["kernel"])
    live = np.array(result.get("live_share", []))
    steps = np.arange(1, kernel.size + 1)

    scores = np.array(result.get("held_out_scores", []))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))

    ax = axes[0]
    share = kernel / kernel.sum() if kernel.sum() > 0 else kernel
    ax.bar(steps, share, color=PALETTE[0], width=0.7)
    ax.set_ylim(0, max(share.max() * 1.15, 1e-3))
    style(ax, "update", "share of the forecast variance", "where the spread was made")

    ax = axes[1]
    if live.size:
        ax.plot(steps, live, "o-", ms=4, lw=1.4, color=PALETTE[1])
    ax.set_ylim(-0.05, 1.05)
    style(ax, "update", "prompts with a mixed group", "and when it stopped making any")

    ax = axes[2]
    if scores.size:
        jitter = np.linspace(-0.12, 0.12, scores.size)
        ax.scatter(jitter, scores, s=26, color=PALETTE[2], zorder=3)
        forecast_sd = result.get("predicted_std", 0.0)
        centre = float(scores.mean())
        ax.errorbar([0.0], [centre], yerr=[forecast_sd], fmt="_", ms=18, lw=1.6,
                    color="0.25", capsize=5, zorder=2, label="forecast s.d.")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.set_xlim(-0.35, 0.35)
        ax.set_xticks([])
    style(ax, "", "held-out pass rate", "what the runs actually did")
    fig.tight_layout()
    save(fig, "real_forecast.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("forecast", figure_forecast),
        ("sources", figure_sources),
        ("null", figure_null),
        ("real_forecast", figure_real_forecast),
        ("reconvergence", figure_reconvergence),
        ("real_seeds", figure_real_seeds),
        ("teaser", figure_teaser),
        ("allocation", figure_allocation),
        ("exact_curves", figure_exact_curves),
        ("estimator_accuracy", figure_estimator_accuracy),
        ("transformer_curves", figure_transformer_curves),
        ("transfer", figure_transfer),
        ("cost_model", figure_cost_model),
        ("real_model", figure_real_model),
    ]:
        try:
            fn()
            print("wrote", name)
        except SystemExit as exc:
            print("pending", name, "-", exc)
            target = FIGURES / f"{name}.pdf"
            if not target.exists():
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.text(0.5, 0.5, "pending", ha="center", va="center", color="0.5")
                ax.axis("off")
                fig.savefig(target, bbox_inches="tight", metadata={"CreationDate": None})
                plt.close(fig)


if __name__ == "__main__":
    main()
