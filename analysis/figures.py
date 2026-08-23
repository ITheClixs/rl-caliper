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
PALETTE = ["#1b3a5c", "#c1502e", "#2e7d5b", "#7a5195", "#b58900"]


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
    fig.savefig(FIGURES / "exact_curves.pdf", bbox_inches="tight")
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
    fig.savefig(FIGURES / "estimator_accuracy.pdf", bbox_inches="tight")
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
    fig.savefig(FIGURES / "transformer_curves.pdf", bbox_inches="tight")
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
    fig.savefig(FIGURES / "transfer.pdf", bbox_inches="tight")
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
    fig.savefig(FIGURES / "cost_model.pdf", bbox_inches="tight")
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
    fig.savefig(FIGURES / "real_model.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, fn in [
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
            print("skipped", name, "-", exc)


if __name__ == "__main__":
    main()
