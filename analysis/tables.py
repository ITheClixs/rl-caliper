"""Emit the paper's tables as LaTeX, straight from the saved runs.

    uv run python analysis/tables.py

Nothing in paper/tables is edited by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from caliper.runtime import io

TABLES = Path(__file__).resolve().parents[1] / "paper" / "tables"
BINS = [(1.00, 1.15), (1.15, 1.35), (1.35, 2.00), (2.00, 8.00)]


def latest(name: str, key: str, container: str = "cells") -> list[dict]:
    runs = [r for r in io.load_all(name) if key in r["result"][container][0]]
    if not runs:
        raise SystemExit(f"no run of {name!r} carries {key!r}")
    return runs[-1]["result"][container]


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(body)
    print("wrote", name)


def table_exact_agreement() -> None:
    cells = latest("a4_law", "r2")
    r2 = np.array([c["r2"] for c in cells])
    spread = np.array([c["predicted_range"] for c in cells])
    rho = np.array([c["spearman"] for c in cells])
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"predicted spread $\max\rho/\min\rho$ & pools & median $R^2$ & median Spearman \\",
        r"\midrule",
    ]
    for lo, hi in BINS:
        mask = (spread >= lo) & (spread < hi)
        if mask.sum():
            lines.append(
                f"${lo:.2f}$--${hi:.2f}$ & {mask.sum()} & "
                f"{np.median(r2[mask]):.3f} & {np.median(rho[mask]):+.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("exact_agreement", "\n".join(lines))


def table_transfer() -> None:
    rows = latest("b3_batch_transfer", "best_step_size", container="rows")
    reference_step = int(np.argmax(rows[0]["step_gain"]))
    reference_drift = int(np.argmax(rows[0]["drift_gain"]))
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"prompts per step $P$ & 8 & 16 & 32 & 64 \\",
        r"\midrule",
    ]

    def row(label, values, fmt="{:.3f}"):
        return label + " & " + " & ".join(fmt.format(v) for v in values) + r" \\"

    lines.append(row("best step size $\\eta^{*}$", [r["best_step_size"] for r in rows], "{:.4f}"))
    lines.append(
        row("best drift target $D^{*}$", [r["best_drift"] for r in rows], "{:.2e}")
    )
    penalty_step, penalty_drift = [], []
    for r in rows:
        best = max(r["step_gain"])
        penalty_step.append(100 * (r["step_gain"][reference_step] - best) / best)
        best_d = max(r["drift_gain"])
        penalty_drift.append(100 * (r["drift_gain"][reference_drift] - best_d) / best_d)
    lines.append(row(r"cost of transferring $\eta$ (\%)", penalty_step, "{:+.1f}"))
    lines.append(row(r"cost of transferring $D^{*}$ (\%)", penalty_drift, "{:+.1f}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("transfer", "\n".join(lines))


def table_transformer() -> None:
    cells = latest("b1_group_size", "predicted_efficiency")
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"corpus $N$ & $\tau_b$ & $\tau_w$ & predicted $G^{*}$ & "
        r"best measured $G$ & Spearman & $R^2$ \\",
        r"\midrule",
    ]
    for cell in cells:
        observed = np.array(cell["observed"])
        predicted = np.array(cell["predicted_efficiency"])
        order = np.argsort(predicted)
        rank_obs = np.argsort(np.argsort(observed))
        rank_pred = np.argsort(np.argsort(predicted))
        spearman = float(np.corrcoef(rank_obs, rank_pred)[0, 1])
        design = np.vstack([np.ones_like(predicted), np.sqrt(predicted)]).T
        coef, *_ = np.linalg.lstsq(design, observed, rcond=None)
        resid = observed - design @ coef
        r2 = float(1 - (resid**2).sum() / ((observed - observed.mean()) ** 2).sum())
        label = r"unbounded" if cell["pool"] is None else str(cell["pool"])
        del order
        lines.append(
            f"{label} & {cell['tau_b']:.1f} & {cell['tau_w_scaled']:.1f} & "
            f"{cell['g_star']:.2f} & {cell['group_sizes'][int(np.argmax(observed))]} & "
            f"{spearman:+.3f} & {r2:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("transformer", "\n".join(lines))


def table_real() -> None:
    rows = latest("c1_real_noise", "g_star", container="rows")
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"corpus & pass rate & $\E[p(1-p)]$ & $\tau_b$ & $\tau_w$ & $G^{*}$ & "
        r"$\tau_w/\E[p(1-p)]$ \\",
        r"\midrule",
    ]
    ratios = np.array([r["tau_w_scaled"] / r["bernoulli"] for r in rows])
    for r, ratio in zip(rows, ratios / ratios.mean(), strict=True):
        lines.append(
            f"{r['corpus'].replace('_', ' ')} & {r['mean_pass_rate']:.3f} & "
            f"{r['bernoulli']:.3f} & {r['tau_b']:.2e} & {r['tau_w_scaled']:.2e} & "
            f"{r['g_star']:.2f} & {ratio:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("real", "\n".join(lines))


PLACEHOLDER = r"\emph{pending: this table is generated from a run that has not been stored yet.}"


def table_difficulty() -> None:
    runs = [
        r
        for r in io.load_all("c2_difficulty")
        if r["result"]["rows"] and "bernoulli" in r["result"]["rows"][0]
    ]
    if not runs:
        raise SystemExit("no difficulty-bucket run")
    rows = runs[-1]["result"]["rows"]
    ratios = np.array([r["tau_w_scaled"] / max(r["bernoulli"], 1e-12) for r in rows])
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"bucket & screened $p$ & measured $p$ & $\E[p(1-p)]$ & $\tau_w$ & "
        r"$\tau_w/\E[p(1-p)]$, relative \\",
        r"\midrule",
    ]
    for row, ratio in zip(rows, ratios / ratios.mean(), strict=True):
        lines.append(
            f"{row['bucket']} & {row['screen_pass_rate']:.3f} & {row['probe_pass_rate']:.3f} & "
            f"{row['bernoulli']:.3f} & {row['tau_w_scaled']:.2e} & {ratio:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("difficulty", "\n".join(lines))


def main() -> None:
    for name, fn in (
        ("exact_agreement", table_exact_agreement),
        ("transfer", table_transfer),
        ("transformer", table_transformer),
        ("real", table_real),
        ("difficulty", table_difficulty),
    ):
        try:
            fn()
        except SystemExit as exc:
            if not (TABLES / f"{name}.tex").exists():
                write(name, PLACEHOLDER)
            print("pending:", exc)


if __name__ == "__main__":
    main()
