"""Emit the paper's tables as LaTeX, straight from the saved runs.

    uv run python analysis/tables.py

Nothing in paper/tables is edited by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from caliper.analysis import grids
from caliper.analysis.efficiency import optimal_group_size
from caliper.runtime import io

TABLES = Path(__file__).resolve().parents[1] / "paper" / "tables"
BINS = [(1.00, 1.15), (1.15, 1.35), (1.35, 2.00), (2.00, 8.00)]


def latest(name: str, key: str, container: str = "cells") -> list[dict]:
    runs = [r for r in io.load_all(name) if key in r["result"][container][0]]
    if not runs:
        raise SystemExit(f"no run of {name!r} carries {key!r}")
    return runs[-1]["result"][container]


def only(store: str, **config) -> dict:
    """The one record of `store` whose configuration has every given value.

    Stores accumulate reruns and variants; quoting the newest one means a later run silently
    replaces the number in the paper. This is the alternative: name the configuration, and fail
    if it does not pick out exactly one record.
    """
    hits = [
        r for r in io.load_all(store)
        if all(r["manifest"]["config"].get(k) == v for k, v in config.items())
    ]
    if len(hits) != 1:
        raise SystemExit(f"{store}: {config} matches {len(hits)} records, not one")
    return hits[0]


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
    # 24 probe repeats; the 8-repeat run is the earlier, noisier version of the same sweep
    rows = only("b3_batch_transfer", probe_repeats=24)["result"]["rows"]
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
    records = [
        r for r in io.load_all("b1_group_size")
        if r["result"].get("cells") and "predicted_efficiency" in r["result"]["cells"][0]
    ]
    if not records:
        raise SystemExit("no transformer group-size sweep")
    chosen = max(records, key=lambda r: len(r["result"]["cells"]))
    cells = chosen["result"]["cells"]
    budget = chosen["manifest"]["config"]["rollouts_per_step"]
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"corpus $N$ & $\tau_b$ & $\tau_w$ & predicted $G^{\star}$ & "
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
        # recomputed here rather than read from the record: runs made before the fixed-budget
        # correction stored an optimum that evaluated the corpus factor at one reference split
        g_star = optimal_group_size(
            cell["tau_b"], cell["tau_w_scaled"], rollouts=budget, corpus=cell["pool"]
        )
        lines.append(
            f"{label} & {cell['tau_b']:.1f} & {cell['tau_w_scaled']:.1f} & "
            f"{g_star:.2f} & {cell['group_sizes'][int(np.argmax(observed))]} & "
            f"{spearman:+.3f} & {r2:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("transformer", "\n".join(lines))


def table_real() -> None:
    rows = latest("c1_real_noise", "g_star", container="rows")
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"corpus & pass rate & $\E[p(1-p)]$ & $\tau_b$ & $\tau_w$ & $G^{\star}$ & "
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


def table_diagnosis() -> None:
    """The 2x2 at the held-out setting: group size against the rule that sets the step."""
    runs = [r for r in io.load_all("b5_diagnosis") if "rows" in r["result"]]
    if not runs:
        raise SystemExit("no diagnosis run")
    result = runs[-1]["result"]
    rows = result["rows"]
    groups = sorted({r["group_size"] for r in rows})
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"best achievable gain & " + " & ".join(f"$G = {g}$" for g in groups) + r" \\",
        r"\midrule",
    ]
    modes = (("step_size", "tuning a learning rate"), ("drift", "tuning a drift target"))
    for mode, label in modes:
        values = []
        for g in groups:
            row = next(r for r in rows if r["group_size"] == g and r["mode"] == mode)
            mark = r"$^{\ast}$" if row["at_edge"] else ""
            values.append(f"{row['best_gain']:+.4f}{mark}")
        lines.append(label + " & " + " & ".join(values) + r" \\")
    predicted = []
    for g in groups:
        bcrit = (result["tau_b"] + result["tau_w_scaled"] / (g - 1)) / result["signal"]
        prompts = runs[-1]["manifest"]["config"]["rollouts"] // g
        predicted.append(f"{1 / (1 + bcrit / prompts):.3f}")
    lines.append(r"predicted efficiency $\rho$ & " + " & ".join(predicted) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("diagnosis", "\n".join(lines))


def table_adam() -> None:
    # the rerun with primed batches and a measurement floor, the version the paper describes
    runs = [only("b6_adam", prime_batches=16, measurement_floor=0.01)]
    if not runs:
        raise SystemExit("no optimiser comparison run")
    cells = runs[-1]["result"]["cells"]
    groups = cells[0]["group_sizes"]
    lines = [
        r"\begin{tabular}{l" + "c" * len(groups) + r"cc}",
        r"\toprule",
        r"gain by group size & " + " & ".join(str(g) for g in groups)
        + r" & measured $G^{\star}$ & $R^2$ \\",
        r"\midrule",
    ]
    for cell in cells:
        name = {"sgd": "gradient ascent", "adam": "Adam"}.get(
            cell["optimiser"], cell["optimiser"]
        )
        values = " & ".join(f"{v:+.3f}" for v in cell["observed"])
        lines.append(
            f"{name} & {values} & {cell['g_star']:.2f} & {cell['r2']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("adam", "\n".join(lines))


def table_optimum() -> None:
    """Theorem 1 against exact numerical optimisation of the cost-aware objective."""
    runs = [r for r in io.load_all("t1_optimum") if "rows" in r["result"]]
    if not runs:
        raise SystemExit("no optimum verification run")
    rows = runs[-1]["result"]["rows"]
    budgets = sorted({r["rollouts"] for r in rows})
    ratios = sorted({r["prefill_ratio"] for r in rows})
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$c_{\mathrm{pre}}/c_{\mathrm{dec}}$ & "
        + " & ".join(f"$R={b}$" for b in budgets)
        + r" \\",
        r"\midrule",
    ]
    for ratio in ratios:
        cells = []
        for budget in budgets:
            row = next(
                r for r in rows if r["prefill_ratio"] == ratio and r["rollouts"] == budget
            )
            cells.append(f"{row['exact']:.1f}")
        lines.append(f"{ratio:.2f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    approx = {}
    for ratio in ratios:
        approx[ratio] = next(r for r in rows if r["prefill_ratio"] == ratio)["closed_form"]
    summary = ", ".join(f"$\\alpha={k:.2f}$: {v:.1f}" for k, v in approx.items())
    lines.append(
        r"\multicolumn{5}{l}{\footnotesize small-batch limit \eqref{eq:gstar-cost} --- "
        + summary
        + r"} \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("optimum", "\n".join(lines))


_GRADED: dict[int, dict] = {}


def _graded(pool: int) -> dict:
    """A grid with its measured spreads over the later seeds only, computed once per run."""
    if pool not in _GRADED:
        _GRADED[pool] = grids.exclude_anchor(grids.forecast_grid(pool))
    return _GRADED[pool]


def headline_grid():
    """The prospective forecast grid the paper's headline numbers come from. See `grids`."""
    try:
        return _graded(grids.HEADLINE_POOL)
    except LookupError as error:
        raise SystemExit(str(error)) from error


def replicate_grid():
    """The same grid on a smaller prompt pool, quoted as a replicate of the headline."""
    try:
        return _graded(grids.REPLICATE_POOL)
    except LookupError as error:
        raise SystemExit(str(error)) from error


def table_forecast() -> None:
    """Forecast accuracy, broken out by the run length and the batch it was made at."""
    cells = headline_grid()["result"]["cells"]
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"grouped by & settings & median & worst & inside 95\% \\",
        r"\midrule",
    ]

    def block(label, rows):
        ratio = np.array([c["predicted_std"] / c["measured_std"] for c in rows])
        error = np.abs(np.log(ratio))
        inside = sum(c["measured_lo"] <= c["predicted_std"] <= c["measured_hi"] for c in rows)
        lines.append(
            f"{label} & {len(rows)} & {np.exp(np.median(error)):.2f}$\\times$ & "
            f"{np.exp(error.max()):.2f}$\\times$ & {inside}/{len(rows)} \\\\"
        )

    block("all settings", cells)
    if any("vocab" in c for c in cells):
        for shape in sorted({(c["vocab"], c["length"]) for c in cells if "vocab" in c}):
            rows = [c for c in cells if (c.get("vocab"), c.get("length")) == shape]
            block(f"\\quad policy ${shape[0]}^{{{shape[1]}}}$", rows)
    for steps in sorted({c["steps"] for c in cells}):
        block(f"\\quad $T = {steps}$", [c for c in cells if c["steps"] == steps])
    for prompts in sorted({c["prompts"] for c in cells}):
        block(f"\\quad $P = {prompts}$", [c for c in cells if c["prompts"] == prompts])
    for eta in sorted({c["step_size"] for c in cells}):
        block(f"\\quad $\\eta = {eta:g}$", [c for c in cells if c["step_size"] == eta])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("forecast", "\n".join(lines))


def adam_lift_record(warmup: int, bias_correction: bool) -> dict:
    """The full exact-tier Adam sweep run with a given warm-up and bias correction.

    The validated sweep warms the moments for five updates and corrects for bias; the cold
    variants match what the pretrained runs did. Selected by identity, so adding a variant
    cannot replace the table the paper quotes.
    """
    hits = [
        r for r in io.load_all("p5_adam_lift")
        if len(r["result"].get("cells", [])) >= 8
        and r["manifest"]["config"].get("warmup") == warmup
        and (not r["manifest"]["config"].get("no_bias_correction", False)) == bias_correction
    ]
    if len(hits) != 1:
        raise SystemExit(f"expected one Adam sweep at warmup {warmup}, bias correction "
                         f"{bias_correction}; found {len(hits)}")
    return hits[0]


def adam_cold_macros() -> dict[str, str]:
    """The lifted forecast's error when Adam starts cold, with and without bias correction."""
    macros: dict[str, str] = {}
    for tag, warmup, corrected in (("Warm", 5, True), ("Cold", 0, True), ("ColdRaw", 0, False)):
        cells = adam_lift_record(warmup, corrected)["result"]["cells"]
        error = np.array([
            max(c["lifted_kl"] / c["measured_kl"], c["measured_kl"] / c["lifted_kl"])
            for c in cells
        ])
        macros |= {
            f"AdamLift{tag}Median": f"{np.median(error):.2f}",
            f"AdamLift{tag}Worst": f"{error.max():.{0 if error.max() >= 100 else 2}f}",
        }
    # the warmed sweep's other models and its bands, on the table's geometric-median convention
    cells = adam_lift_record(5, True)["result"]["cells"]

    def log_error(subset: list[dict], key: str) -> np.ndarray:
        return np.abs(np.log(np.array([c[key] / c["measured_kl"] for c in subset])))

    for tag, key, places in (("Frozen", "momentum_kl", 2), ("Params", "parameters_only_kl", 1),
                             ("Walk", "walk_kl", 1)):
        median = np.exp(np.median(log_error(cells, key)))
        macros[f"AdamLift{tag}Median"] = f"{median:.{places}f}"
    for band in ("hard", "mixed", "easy"):
        subset = [c for c in cells if c["band"] == band]
        error = log_error(subset, "lifted_kl")
        tag = band.capitalize()
        macros |= {
            f"AdamLift{tag}Median": f"{np.exp(np.median(error)):.2f}",
            f"AdamLift{tag}Worst": f"{np.exp(error.max()):.1f}",
            f"AdamLift{tag}Top": f"{max(c['final_metric'] for c in subset):.2f}",
        }
    return macros


def table_adam_lift() -> None:
    """How well each model of Adam's seed divergence does, overall and by pool."""
    cells = adam_lift_record(5, True)["result"]["cells"]
    measured = np.array([c["measured_kl"] for c in cells])
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"model & median & worst & live signal only \\",
        r"\midrule",
    ]
    live = [c for c in cells if c["band"] == "mixed"]
    options = [("lifted_kl", r"lifted, $z = (\theta, m, v)$"),
               ("momentum_kl", "second moment frozen"),
               ("parameters_only_kl", r"$\theta$ block only"),
               ("walk_kl", "accumulation")]
    for key, label in options:
        if key not in cells[0]:
            continue
        error = np.abs(np.log(np.array([c[key] for c in cells]) / measured))
        live_error = np.abs(
            np.log(np.array([c[key] for c in live]) / np.array([c["measured_kl"] for c in live]))
        )
        lines.append(
            f"{label} & {np.exp(np.median(error)):.2f}$\\times$ & "
            f"{np.exp(error.max()):.1f}$\\times$ & "
            f"{np.exp(np.median(live_error)):.2f}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("adam_lift", "\n".join(lines))


def table_spectrum() -> None:
    """What the transfer operator's spectrum looks like at points along a run."""
    runs = [r for r in io.load_all("p6_spectrum") if len(r["result"].get("cells", [])) >= 6]
    if not runs:
        raise SystemExit("no spectrum sweep")
    cells = runs[-1]["result"]["cells"]
    bands = sorted({c["band"] for c in cells})
    updates = sorted({c["update"] for c in cells})
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l" + "c" * len(updates) + r"@{}}",
        r"\toprule",
        "update & " + " & ".join(str(u) for u in updates) + r" \\",
        r"\midrule",
    ]
    for band in bands:
        rows = {c["update"]: c for c in cells if c["band"] == band}
        radii = [
            f"{1000 * (rows[u]['radius'] - 1.0):.1f}" if u in rows else "--" for u in updates
        ]
        shares = [f"{100 * rows[u]['contracted_share']:.0f}" if u in rows else "--"
                  for u in updates]
        lines.append(rf"{band}, $10^3(\rho - 1)$ & " + " & ".join(radii) + r" \\")
        lines.append(r"\quad contracting, \% & " + " & ".join(shares) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("spectrum", "\n".join(lines))


def _half_sample(record):
    """The two half-sample forecasts a run reported, if they were recorded or recoverable."""
    stored = record["result"].get("split_variance")
    if stored and all(v == v for v in stored):
        return [float(np.sqrt(max(v, 0.0))) for v in stored]
    label = str(record["manifest"]["config"].get("label", ""))
    for path in sorted((TABLES.parents[1] / "runs" / "logs").glob("*.split.json")):
        blob = json.loads(path.read_text())
        if label.split("-")[-1] in blob["source"] or "wide" in blob["source"]:
            return [float(v) for v in blob["half_sample_std"]]
    return None


def table_sizing() -> None:
    """Runs each policy needs to hold the mistake rate at the finest target."""
    record = sizing_record()
    if record is None:
        raise SystemExit("s13_sizing has not been run over the full grid")
    result = record["result"]
    runs = result["runs_at_target"]
    ratio = result["ratio_at_finest"]
    order = [
        ("fixed", "fixed budget, no stopping rule"),
        ("empirical", "deviation from the seeds in hand"),
        ("constant", "one deviation for every recipe"),
        ("frozen", "frozen forecast, before any run"),
        ("adjoint", "adjoint forecast, from one run"),
        ("oracle", "measured deviation"),
    ]
    rows = []
    for key, label in order:
        cost = runs[key][-1]
        median, lo, hi = ratio[key]
        cell = "--" if cost is None else f"{cost:.2f}"
        rows.append(f"    {label} & {cell} & ${median:.2f}$ & $[{lo:.2f}, {hi:.2f}]$ " + r"\\")
    body = [
        r"\begin{tabular}{lccc}",
        r"  \toprule",
        r"  policy & runs & vs.\ adjoint & $95\%$ \\",
        r"  \midrule",
        *rows,
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    write("sizing", chr(10).join(body))


def real_sizing_record():
    """The pretrained sizing benchmark at its default operating point.

    The cap and the target are choices, and on four comparisons they change which baseline comes
    second. The default is recorded here and the sensitivity is reported in the text rather than
    the best of the settings being quoted.
    """
    hits = [r for r in io.load_all("s14_real_sizing")
            if r["manifest"]["config"].get("prefix") == "sizing-"
            and r["manifest"]["config"].get("cap") == 8
            and r["result"].get("recipes", 0) >= 6]
    return hits[-1] if hits else None


def tolerance_record():
    """The corruption sweep, in the form that sweeps several targets at once.

    Earlier records graded against every seed, including the ones a policy drew, which understates
    the fixed-budget baseline and so understates the break-even. They are skipped by shape.
    """
    hits = [r for r in io.load_all("s13_tolerance") if "summary" in r["result"]]
    return hits[-1] if hits else None


def latest_record(store: str):
    records = io.load_all(store)
    return records[-1] if records else None


def sizing_record(holdout: int = 32):
    """The sizing benchmark over the full grid, at a given size of reserved ordering set.

    Two choices make records incomparable rather than merely different. A threshold range too
    narrow to let a predictor reach its own operating point measures the range instead of the
    predictor. And reading the true ordering from seeds a policy is also shown correlates the
    policy's evidence with the answer, which flatters whichever policy sees the most seeds.
    Records are selected on both, by configuration rather than by date.
    """
    hits = [
        r for r in io.load_all("s13_sizing")
        if r["manifest"]["config"].get("cells") == 192
        and r["manifest"]["config"].get("ceiling", 0) >= 100
        and r["manifest"]["config"].get("holdout", 0) == holdout
        and "frozen" in r["result"].get("runs_at_target", {})
    ]
    return hits[-1] if hits else None


OTHER_ENV_SEED = 991  # the grid generated from a different environment seed


def diagnostic_macros(cells: list[dict], prefix: str, cut: float | None = None) -> dict[str, str]:
    """How well the one-run curvature residual flags a bad forecast on one grid.

    With `cut` unset the threshold is fitted on this grid, refusing the fifth of settings with
    the largest residual; given, it is applied unchanged.
    """
    from scipy.stats import spearmanr

    residual = np.array([c["curvature_residual"] for c in cells])
    ratio = np.array([
        max(c["predicted_std"] / c["measured_std"], c["measured_std"] / c["predicted_std"])
        for c in cells
    ])
    if cut is None:
        cut = float(np.quantile(residual, 0.8))
    kept = residual <= cut
    worst = int(np.argmax(ratio))
    macros = {
        f"{prefix}Cells": f"{len(cells)}",
        f"{prefix}Low": f"{residual.min():.4f}",
        f"{prefix}High": f"{residual.max():.4f}",
        f"{prefix}Threshold": f"{cut:.4f}",
        f"{prefix}Coverage": f"{100 * kept.mean():.0f}",
        f"{prefix}WorstKept": f"{ratio[kept].max():.2f}",
        f"{prefix}WorstAll": f"{ratio.max():.2f}",
        f"{prefix}Rank": f"{spearmanr(residual, ratio)[0]:.2f}",
        # where the single worst forecast sits among the residuals, as a percentile
        f"{prefix}WorstPercentile": f"{100 * (residual < residual[worst]).mean():.0f}",
    }
    for level, name in ((1.2, "Low"), (1.3, "High")):
        bad = ratio > level
        if 1 < bad.sum() < len(bad) - 1:
            auroc = float(np.mean([
                [(a > b) + 0.5 * (a == b) for b in residual[~bad]] for a in residual[bad]
            ]))
            macros[f"{prefix}Auroc{name}"] = f"{auroc:.2f}"
            macros[f"{prefix}Fails{name}"] = f"{bad.sum()}"
    return macros


HELDOUT_LABEL = {
    "fixed": "fixed budget",
    "constant": "one deviation for every recipe",
    "empirical": "deviation from spent seeds",
    "frozen": "closed form, no run",
    "calibrated": "closed form, one scale from other tasks",
    "adjoint": "adjoint forecast",
    "oracle": "measured deviation",
}


def heldout_runs(holdout: int, cap: int) -> list[dict]:
    """Held-out sizing records with a given seed split, one per truth/available draw."""
    return [
        r for r in io.load_all("s15_sizing_heldout")
        if r["manifest"]["config"]["holdout"] == holdout and r["manifest"]["config"]["cap"] == cap
    ]


def heldout_record() -> dict:
    """The held-out benchmark on the split `s13_sizing/run.py` uses (32 seeds fix the ordering,
    16 per arm available, seed 0), chosen by identity rather than by load order."""
    hits = [r for r in heldout_runs(32, 16) if r["manifest"]["config"]["seed"] == 0]
    if len(hits) != 1:
        raise SystemExit(f"expected one held-out sizing record at 32/16, seed 0; found {len(hits)}")
    return hits[0]["result"]["results"]


def split_rows(key: str) -> list[dict]:
    """The 24/24 splits, where a fixed budget can be measured below its cap."""
    return [r["result"]["results"][key] for r in heldout_runs(24, 24)]


def heldout_macros() -> dict[str, str]:
    results = heldout_record()
    macros: dict[str, str] = {}
    splits = split_rows("margin=4,target=0.001")
    if splits:
        fixed = [row["held_out"]["fixed"]["runs"] for row in splits]
        adjoint = [row["held_out"]["adjoint"]["runs"] for row in splits]
        gain = [row["held_out"]["fixed"]["ratio_to_adjoint"] for row in splits]
        frozen = [row["held_out"]["frozen"]["ratio_to_adjoint"] for row in splits]
        macros |= {
            "SplitCount": f"{len(splits)}",
            "SplitFixedRunsLow": f"{min(fixed):.1f}",
            "SplitFixedRunsHigh": f"{max(fixed):.1f}",
            "SplitAdjointRunsLow": f"{min(adjoint):.2f}",
            "SplitAdjointRunsHigh": f"{max(adjoint):.2f}",
            "SplitFixedGainLow": f"{min(gain):.1f}",
            "SplitFixedGainHigh": f"{max(gain):.1f}",
            "SplitFrozenGainLow": f"{min(frozen):.2f}",
            "SplitFrozenGainHigh": f"{max(frozen):.2f}",
            "SplitUncappedApprox": f"{splits[0]['uncapped_fixed_runs']:.0f}",
        }
    for tag, key in (("Strict", "margin=4,target=0.001"), ("Loose", "margin=4,target=0.01"),
                     ("TieLoose", "margin=2,target=0.01")):
        row = results[key]
        held, plain = row["held_out"], row["nominal"]
        adjoint = held["adjoint"]
        macros |= {
            f"Held{tag}Pairs": f"{row['pairs']}",
            f"Held{tag}AdjointRuns": f"{adjoint['runs']:.2f}",
            f"Held{tag}AdjointMistakes": f"{100 * adjoint['mistakes']:.3f}",
            f"Held{tag}UncappedFixedRuns": f"{row['uncapped_fixed_runs']:.0f}",
            f"Held{tag}UncappedGain": f"{row['uncapped_fixed_runs'] / adjoint['runs']:.1f}",
            f"Nominal{tag}AdjointRuns": f"{plain['adjoint']['runs']:.2f}",
            f"Nominal{tag}AdjointMistakes": f"{100 * plain['adjoint']['mistakes']:.3f}",
            f"Nominal{tag}ConstantMistakes": f"{100 * plain['constant']['mistakes']:.3f}",
            f"Nominal{tag}EmpiricalMistakes": f"{100 * plain['empirical']['mistakes']:.3f}",
            f"Nominal{tag}FrozenRuns": f"{plain['frozen']['runs']:.0f}",
            f"Nominal{tag}CalibratedRuns": f"{plain['calibrated']['runs']:.2f}",
            f"Nominal{tag}CalibratedMistakes": f"{100 * plain['calibrated']['mistakes']:.3f}",
            f"Nominal{tag}ConstantRuns": f"{plain['constant']['runs']:.2f}",
        }
        for name in ("fixed", "constant", "empirical", "frozen", "oracle"):
            title = name.capitalize()
            lo, hi = held[name]["ratio_interval"]
            macros |= {
                f"Held{tag}{title}Gain": f"{held[name]['ratio_to_adjoint']:.2f}",
                f"Held{tag}{title}Lo": f"{lo:.2f}",
                f"Held{tag}{title}Hi": f"{hi:.2f}",
            }
    return macros


def table_heldout() -> None:
    """Runs and realised mistakes when each policy's threshold is set before it is graded."""
    results = heldout_record()
    columns = ("margin=4,target=0.01", "margin=4,target=0.001")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{target $1\%$} & \multicolumn{2}{c}{target $0.1\%$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"threshold set on other tasks & runs & mistakes & runs & mistakes \\",
        r"\midrule",
    ]
    for name, label in HELDOUT_LABEL.items():
        cells = []
        for key in columns:
            value = results[key]["held_out"][name]
            cells += [f"{value['runs']:.2f}", f"{100 * value['mistakes']:.2f}\\%"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r"textbook threshold, no tuning & & & & \\"]
    for name in ("constant", "frozen", "calibrated", "adjoint"):
        cells = []
        for key in columns:
            value = results[key]["nominal"][name]
            cells += [f"{value['runs']:.2f}", f"{100 * value['mistakes']:.2f}\\%"]
        lines.append(f"\\quad {HELDOUT_LABEL[name]} & " + " & ".join(cells) + r" \\")
    measured = []
    for key in columns:
        runs = [row["held_out"]["fixed"]["runs"] for row in split_rows(key)]
        measured.append(f"{min(runs):.1f}--{max(runs):.1f}" if runs else "--")
    lines += [
        r"\midrule",
        f"fixed budget, 24-seed cap, {len(heldout_runs(24, 24))} splits & {measured[0]} & & "
        f"{measured[1]} & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("heldout", "\n".join(lines))


def saturation_macros() -> dict[str, str]:
    """What the pretrained runs collapse to (`s16_saturation/analyse.py`)."""
    # the mode analysis; the store also keeps the earlier, withdrawn saturation analysis
    result = only("s16_saturation", held_out_seed=9001)["result"]
    names = {"small": "Small", "large": "Large", "small_rerun": "SmallRerun",
             "small_traces": "SmallTraces"}
    macros: dict[str, str] = {}
    for tag, title in names.items():
        row = result[tag]
        macros |= {
            f"Mode{title}Seeds": f"{row['seeds']}",
            f"Mode{title}Constant": f"{row['constant_seeds']}",
            f"Mode{title}Share": f"{100 * row['mode_share']:.0f}",
            f"Mode{title}Same": f"{row['same_answer_correlation']:.2f}",
            f"Mode{title}Other": f"{row['other_answer_correlation']:.2f}",
            f"Mode{title}Anchor": row["anchor_mode"],
        }
    small = result["small"]
    ones = small["answer_counts"]["1"]
    prompts = sum(small["answer_counts"].values())
    repeats = result["large"]["forecast_repeats"]
    macros |= {
        "ModeOnePrompts": f"{ones}",
        "ModePrompts": f"{prompts}",
        "ModeOneScore": f"{ones / prompts:.3f}",
        "ModeLargeRepeatRatio": f"{max(repeats) / min(repeats):.0f}",
    }
    return macros


def shortfall_macros() -> dict[str, str]:
    """How far short the 0.5B forecasts are, cheap and transported, as the text quotes them."""
    chosen = experiment("real_05b_sgd", require=("resolved_std",))
    result = chosen["result"]
    measured = result["resolved_std"]
    variants = cheap_variants(chosen, io.load_all("s7_real_forecast"))
    cheap = variants[None]["predicted_std"]
    band = result["resolved_interval"]
    # the three highest held-out scores sit together; the text quotes them and the rest's range
    scores = sorted(result["held_out_scores"])
    cluster, rest = scores[-3:], scores[:-3]
    macros = {
        "RealShortCheap": f"{measured / cheap:.0f}",
        "RealShortTransported": f"{measured / result['predicted_std']:.0f}",
        "RealCheapForecast": f"{cheap:.4f}",
        "RealTransportedForecast": f"{result['predicted_std']:.4f}",
        "RealObservedSpread": f"{result['observed_spread']['std']:.4f}",
        "RealBinomialPart": f"{np.sqrt(result['binomial_variance']):.4f}",
        "RealSeedPart": f"{measured:.4f}",
        "RealSeedLo": f"{band['lo']:.4f}",
        "RealSeedHi": f"{band['hi']:.4f}",
        "RealClusterLow": f"{cluster[0]:.3f}",
        "RealClusterMid": f"{cluster[1]:.3f}",
        "RealClusterHigh": f"{cluster[2]:.3f}",
        "RealClusterWidth": f"{cluster[2] - cluster[0]:.3f}",
        "RealRestLow": f"{rest[0]:.3f}",
        "RealRestHigh": f"{rest[-1]:.3f}",
    }
    if 64 in variants:
        macros["RealCheapWide"] = f"{variants[64]['predicted_std']:.4f}"
    return macros


def collapse_sweep_macros() -> dict[str, str]:
    """The collapse test on two support-sweep bands (`s16_saturation/sweep.py`).

    Selected by the labels it was run on, so a later sweep over other bands cannot replace it.
    """
    labels = ["modes-band-0.85", "modes-band-0.60"]
    result = only("s16_sweep", labels=labels)["result"]
    macros = {}
    for tag, label in zip(("Held", "Missed"), labels, strict=True):
        row = result[label]
        macros |= {
            f"Collapse{tag}Band": label.rsplit("-", 1)[1],
            f"Collapse{tag}Error": f"{row['error']:.2f}",
            f"Collapse{tag}Split": f"{row['split_prompts']}",
            f"Collapse{tag}Shared": f"{row['shared_prompts']}",
            f"Collapse{tag}Constant": f"{row['collapsed_seeds']}",
            f"Collapse{tag}AnchorPrompts": f"{round(row['anchor_share'] * row['prompts'])}",
        }
    held = result[labels[0]]
    macros |= {
        "CollapsePrompts": f"{held['prompts']}",
        "CollapseSeeds": f"{held['seeds']}",
        # the refutation bound committed in docs/predictions.md: band 0.85's split share
        "CollapseSplitBound": f"{100 * held['split_prompts'] / held['prompts']:.0f}",
    }
    return macros


def estimated_macros() -> dict[str, str]:
    """The closed form with traces estimated from one run (`s13_sizing/estimated.py`)."""
    result = only("s18_sizing_estimated", holdout=32, cap=16, seed=0)["result"]
    macros = {
        "EstVsExactMedian": f"{result['estimated_vs_exact_median']:.2f}",
        "EstFloored": f"{result['floored']}",
        "EstConfigurations": f"{result['configurations']}",
    }
    for tag, key in (("Strict", "margin=4,target=0.001"), ("Loose", "margin=4,target=0.01")):
        held = result["results"][key]["held_out"]["estimated"]
        plain = result["results"][key]["nominal"]["estimated_calibrated"]
        lo, hi = held["ratio_interval"]
        macros |= {
            f"Est{tag}Gain": f"{held['ratio_to_adjoint']:.2f}",
            f"Est{tag}Lo": f"{lo:.2f}",
            f"Est{tag}Hi": f"{hi:.2f}",
            f"Est{tag}Mistakes": f"{100 * held['mistakes']:.3f}",
            f"Est{tag}NominalRuns": f"{plain['runs']:.2f}",
        }
    return macros


def hand_typed_macros() -> dict[str, str]:
    """Numbers the eight-page text used to type by hand, now computed from their records."""
    macros: dict[str, str] = {}
    cells = headline_grid()["result"]["cells"]
    shift = np.abs(np.log([c["predicted_std"] / c["untransported_std"] for c in cells]))
    macros["TransportShiftPercent"] = f"{100 * (np.exp(np.median(shift)) - 1):.0f}"
    held_over_carried = [c["untransported_std"] / c["predicted_std"] for c in cells]
    macros["TransportHeldLow"] = f"{min(held_over_carried):.2f}"
    macros["TransportHeldHigh"] = f"{max(held_over_carried):.2f}"

    rows = only("a5_estimator", batches=2500)["result"]["rows"]
    error = [abs(r["crossed_g_star"] / r["exact_g_star"] - 1) for r in rows]
    ratios = [r["exact_ratio"] for r in rows]
    # the example quoted is the most homogeneous pool in the broad band
    example = min((r for r in rows if r["band"] == "broad"), key=lambda r: r["diversity"])
    hard = min((r for r in rows if r["band"] == "hard"), key=lambda r: r["diversity"])
    g_star = [r["exact_g_star"] for r in rows]
    tau_b_error = [abs(r["crossed_tau_b"] / r["exact_tau_b"] - 1) for r in rows]
    tau_w_error = [abs(r["crossed_tau_w_scaled"] / r["exact_tau_w_scaled"] - 1) for r in rows]
    macros |= {
        "EstimatorGStarWorst": f"{100 * max(error):.1f}",
        "EstimatorTauBError": f"{100 * np.median(tau_b_error):.1f}",
        "EstimatorTauWError": f"{100 * np.median(tau_w_error):.1f}",
        "EstimatorGStarLow": f"{min(g_star):.1f}",
        "EstimatorGStarHigh": f"{max(g_star):.1f}",
        "SubtractiveHardExample": f"{hard['subtractive_g_star']:.1f}",
        "SubtractiveHardExact": f"{hard['exact_g_star']:.1f}",
        "EstimatorGStarError": f"{100 * np.median(error):.1f}",
        "EstimatorRatioRange": f"{max(ratios) / min(ratios):.0f}",
        "SubtractiveExample": f"{example['subtractive_g_star']:.1f}",
        "SubtractiveExact": f"{example['exact_g_star']:.1f}",
    }

    record = real_sizing_record()
    if record is not None:
        arms = record["result"]["arms"]
        forecast = [a["adjoint"] for a in arms]
        measured = [a["oracle"] for a in arms]
        means = [float(np.mean(a["scores"])) for a in arms]
        macros |= {
            "RealSizeTrackForecast": f"{max(forecast) / min(forecast):.1f}",
            "RealSizeTrackMeasured": f"{max(measured) / min(measured):.1f}",
            "RealSizeMeanGap": f"{max(means) - min(means):.3f}",
            "RealSizeDevLow": f"{min(measured):.3f}",
            "RealSizeDevHigh": f"{max(measured):.3f}",
        }
    return macros


def table_numbers() -> None:
    """Numbers quoted in the paper's prose, as macros, so they are not transcribed by hand."""
    macros: dict[str, str] = {}

    forecast = [headline_grid()]
    if forecast:
        cells = forecast[-1]["result"]["cells"]
        ratio = np.array([c["predicted_std"] / c["measured_std"] for c in cells])
        error = np.abs(np.log(ratio))
        spread = np.array([c["measured_std"] for c in cells])
        inside = sum(c["measured_lo"] <= c["predicted_std"] <= c["measured_hi"] for c in cells)
        macros |= {
            "ForecastSettings": f"{len(cells)}",
            "ForecastMedian": f"{np.exp(np.median(error)):.2f}",
            "ForecastWorst": f"{np.exp(error.max()):.2f}",
            "ForecastInside": f"{inside}",
            "ForecastSpreadRange": f"{spread.max() / spread.min():.0f}",
            "ForecastSeeds": f"{cells[0]['seeds']}",
        }
        held = np.abs(np.log(np.array([c["untransported_std"] / c["measured_std"] for c in cells])))
        held_inside = sum(
            c["measured_lo"] <= c["untransported_std"] <= c["measured_hi"] for c in cells)
        macros |= {
            "ForecastHeldMedian": f"{np.exp(np.median(held)):.2f}",
            "ForecastHeldWorst": f"{np.exp(held.max()):.2f}",
            "ForecastHeldInside": f"{held_inside}",
        }
    replicate = replicate_grid()["result"]["cells"]
    ratio = np.array([c["predicted_std"] / c["measured_std"] for c in replicate])
    error = np.abs(np.log(ratio))
    inside = sum(c["measured_lo"] <= c["predicted_std"] <= c["measured_hi"] for c in replicate)
    macros |= {
        "ForecastReplicatePool": f"{grids.REPLICATE_POOL}",
        "ForecastHeadlinePool": f"{grids.HEADLINE_POOL}",
        "ForecastReplicateMedian": f"{np.exp(np.median(error)):.2f}",
        "ForecastReplicateWorst": f"{np.exp(error.max()):.2f}",
        "ForecastReplicateInside": f"{inside}",
    }

    null = [
        r
        for r in io.load_all("s8_null")
        if len(r["result"].get("conditions", {}).get("verifier", {}).get("step", [])) >= 5
    ]
    if null:
        result = null[-1]["result"]
        left = result["conditions"]["verifier"]
        right = result["conditions"]["coin"]
        macros |= {
            "NullLearning": f"{result['growth']['verifier']:.2f}",
            "NullCoin": f"{result['growth']['coin']:.1f}",
            "NullGap": f"{right['pairwise_kl'][-1] / left['pairwise_kl'][-1]:.0f}",
            "NullUpdates": f"{left['step'][-1]}",
            "NullPassStart": f"{left['pass_rate'][0]:.2f}",
            "NullPassEnd": f"{left['pass_rate'][-1]:.2f}",
        }

    spectrum = [r for r in io.load_all("p6_spectrum") if len(r["result"].get("cells", [])) >= 6]
    if spectrum:
        cells = spectrum[-1]["result"]["cells"]
        radii = np.array([c["radius"] for c in cells])
        shares = np.array([c["contracted_share"] for c in cells])
        fastest = np.array([c["fastest"] for c in cells])
        macros |= {
            "SpectrumRadiusMax": f"{radii.max():.4f}",
            "SpectrumRadiusMin": f"{radii.min():.4f}",
            "SpectrumShareLow": f"{100 * shares.min():.0f}",
            "SpectrumShareHigh": f"{100 * shares.max():.0f}",
            "SpectrumFastest": f"{np.nanmin(fastest):.3f}",
        }

    memory = [
        r for r in io.load_all("p3_memory_sources") if len(r["result"].get("cells", [])) >= 100
    ]
    if memory:
        cells = memory[-1]["result"]["cells"]
        fraction = np.array([c["horizon_95"] / c["steps"] for c in cells])
        by_group: dict[int, list[float]] = {}
        for cell in cells:
            by_group.setdefault(cell["group_size"], []).append(cell["rollout_share"])
        first, tilted = [], 0
        for cell in cells:
            kernel = np.array(cell["kernel"])
            if kernel.sum() <= 0:
                continue
            share = kernel / kernel.sum()
            half = share.size // 2
            first.append(share[:half].sum())
            grid = (np.arange(share.size) - (share.size - 1) / 2) / max(share.size - 1, 1)
            tilted += int(np.polyfit(grid, share * share.size, 1)[0] < 0)
        macros |= {
            "MemorySettings": f"{len(cells)}",
            "MemoryFraction": f"{np.median(fraction):.2f}",
            "RolloutShareMedian": f"{100 * np.median([c['rollout_share'] for c in cells]):.0f}",
            "FirstHalfShare": f"{100 * np.median(first):.0f}",
            "TiltedEarly": f"{tilted}",
        }
        # LaTeX command names cannot contain digits, so the group size is spelled out
        spelled = {2: "two", 4: "four", 8: "eight", 16: "sixteen", 32: "thirtytwo"}
        for group, values in sorted(by_group.items()):
            name = spelled.get(group)
            if name:
                macros[f"RolloutShareG{name}"] = f"{100 * np.median(values):.0f}"

    # the same 54-setting grid the propagation table reports, not whichever pilot ran last
    propagation = [only("p1_propagation", steps=25)]
    if propagation:
        cells = propagation[0]["result"]["cells"]
        measured = np.array([c["measured_kl"] for c in cells])
        summary = {}
        for key, name in (("propagated_kl", "Propagated"), ("walk_kl", "Walk"),
                          ("scalar_kl", "Scalar")):
            error = np.abs(np.log(np.array([c[key] for c in cells]) / measured))
            summary[f"{name}Median"] = f"{np.exp(np.median(error)):.2f}"
            summary[f"{name}Worst"] = f"{np.exp(error.max()):.1f}"
        summary["PropagationSettings"] = f"{len(cells)}"
        summary["PropagatedPercent"] = f"{100 * (float(summary['PropagatedMedian']) - 1):.0f}"
        macros |= summary

    # the 0.5B forecast run of Section 6; the store also holds the 7B run, which has traces too and
    # had been quoted here as though it were the 0.5B one
    traced = [experiment("real_05b_sgd", require=("train_traces",))]
    if traced[0] is not None:
        outcome = traced[0]["result"]
        traces = np.array(outcome["train_traces"], dtype=float)
        scores = np.array(outcome["held_out_scores"], dtype=float)
        window = min(3, traces.shape[1])
        early = traces[:, :window].mean(axis=1)
        late = traces[:, -window:].mean(axis=1)
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(8000):
            pick = rng.integers(0, scores.size, size=scores.size)
            if np.std(early[pick]) == 0 or np.std(scores[pick]) == 0:
                continue
            draws.append(np.corrcoef(early[pick], scores[pick])[0, 1])
        low, high = np.percentile(draws, [2.5, 97.5])
        kernel = np.array(outcome.get("kernel") or [])
        if kernel.size and kernel.sum() > 0:
            # the same share as FirstHalfShare, but on the pretrained model rather than the
            # exact tier, where the kernel is far more concentrated
            share = kernel / kernel.sum()
            macros["RealFirstHalfShare"] = f"{100 * share[: share.size // 2].sum():.0f}"
        macros |= {
            "EarlyCorrelation": f"{np.corrcoef(early, scores)[0, 1]:+.2f}",
            "EarlyCorrelationLow": f"{low:+.2f}",
            "EarlyCorrelationHigh": f"{high:+.2f}",
            "LateCorrelation": f"{np.corrcoef(late, scores)[0, 1]:+.2f}",
            "TraceRuns": f"{scores.size}",
            "TraceWindow": f"{window}",
        }

    # the same correlation at seven billion parameters, from the training trace the run logged
    big = [
        r
        for r in io.load_all("s6_scale")
        if r["result"].get("train_history") and len(r["result"].get("held_out_scores", [])) >= 4
    ]
    if big:
        outcome = big[0]["result"]
        history = outcome["train_history"]
        order = sorted(history, key=lambda k: int(k))
        traces = np.array([history[k] for k in order], dtype=float)
        scores = np.array(outcome["held_out_scores"], dtype=float)[: traces.shape[0]]
        window = min(3, traces.shape[1])
        early = traces[:, :window].mean(axis=1)
        macros["EarlyCorrelationBig"] = f"{np.corrcoef(early, scores)[0, 1]:+.2f}"
        base = outcome["base_pass_rate"]
        spread = outcome["observed_spread"]["std"]
        gain = outcome["observed_spread"]["mean"] - base
        macros |= {
            "ScaleRuns": f"{scores.size}",
            "ScaleBase": f"{base:.3f}",
            "ScaleMean": f"{outcome['observed_spread']['mean']:.3f}",
            "ScaleLow": f"{min(outcome['held_out_scores']):.3f}",
            "ScaleHigh": f"{max(outcome['held_out_scores']):.3f}",
            "ScaleSpread": f"{spread:.4f}",
            "ScaleGain": f"{gain:.3f}",
            # what one run can say about the effect: the spread as a share of the mean improvement
            "ScaleShare": f"{100 * spread / gain:.0f}",
        }
        # the tail slope of divergence against update count: +1 is what accumulation requires
        rows = outcome.get("divergence") or []
        if len(rows) >= 3:
            steps = np.array([r["step"] for r in rows], dtype=float)
            values = np.array([r["mean"] for r in rows], dtype=float)
            keep = steps >= steps.max() / 2
            if keep.sum() >= 2:
                slope = np.polyfit(np.log(steps[keep]), np.log(values[keep]), 1)[0]
                macros["ScaleTailSlope"] = f"{slope:+.2f}"
            full = np.polyfit(np.log(steps), np.log(values), 1)[0]
            macros["ScaleFullSlope"] = f"{full:+.2f}"
            macros["ScaleDivergenceEnd"] = f"{values[-1]:.2f}"
            macros["ScaleDivergenceStart"] = f"{values[0]:.2f}"


    # the seven-billion forecast under Adam, against the spread it was made before seeing
    got = experiment("real_7b_adam", require=("resolved_std",))
    guess = experiment("real_7b_adam_forecast")
    stateless = experiment("real_7b_adam_stateless")
    if got and guess:
        outcome, prediction = got["result"], guess["result"]
        spread = outcome["resolved_std"]
        point = prediction["predicted_std"]
        band = outcome.get("resolved_interval") or {}
        halves = sorted(_half_sample(guess) or [])
        macros |= {
            "AdamScaleSeeds": f"{len(outcome['held_out_scores'])}",
            "AdamScaleBase": f"{outcome['base_pass_rate']:.3f}",
            "AdamScaleMean": f"{outcome['observed_spread']['mean']:.3f}",
            "AdamScaleSpread": f"{outcome['observed_spread']['std']:.4f}",
            "AdamScaleBinomial": f"{np.sqrt(outcome['binomial_variance']):.4f}",
            "AdamScaleResolved": f"{spread:.4f}",
            "AdamScaleForecast": f"{point:.4f}",
            "AdamScaleShortfall": f"{spread / point:.1f}",
            "AdamScaleWidePrompts": f"{guess['manifest']['config']['variance_prompts']}",
        }
        if band:
            macros["AdamScaleResolvedLow"] = f"{band['lo']:.4f}"
            macros["AdamScaleResolvedHigh"] = f"{band['hi']:.4f}"
        if halves:
            lo, hi = halves
            macros |= {
                "AdamScaleHalfLow": f"{lo:.4f}",
                "AdamScaleHalfHigh": f"{hi:.4f}",
                "AdamScaleHalfRatio": f"{hi / lo:.1f}",
                "AdamScaleShortfallHigh": f"{spread / lo:.0f}",
                "AdamScaleShortfallLow": f"{spread / hi:.1f}",
            }
        if stateless:
            bare = stateless["result"]["predicted_std"]
            macros |= {
                "AdamScaleStateless": f"{bare:.5f}",
                "AdamScaleStatelessShortfall": f"{spread / bare:.0f}",
                "AdamScaleStateGain": f"{point / bare:.1f}",
            }

    # the support sweep: four pre-specified regimes on one model
    bands = [
        r["result"] for r in io.load_all("s11_prospective")
        if str(r["result"].get("label", "")).startswith("band-")
        and r["result"].get("resolved_std") is not None
    ]
    if bands:
        bands.sort(key=lambda o: float(o["label"].split("-")[1]))
        errors = [max(o["ratio"], 1 / o["ratio"]) for o in bands]
        halves = [
            max(o["predicted_half_samples"]) / max(min(o["predicted_half_samples"]), 1e-12)
            for o in bands
        ]
        inside = [o for o in bands if 2 / 3 < o["ratio"] < 1.5]
        macros |= {
            "SweepBands": f"{len(bands)}",
            "SweepBest": f"{min(errors):.2f}",
            "SweepWorst": f"{max(errors):.2f}",
            "SweepInside": f"{len(inside)}",
            "SweepErrors": ", ".join(f"{e:.2f}" for e in errors),
            # forecast over measured, so the direction of each miss is visible
            "SweepRatios": ", ".join(f"{o['ratio']:.2f}" for o in bands),
            "SweepHalfLow": f"{min(halves):.2f}",
            "SweepHalfHigh": f"{max(halves):.2f}",
            "SweepRealised": ", ".join(f"{o['anchor_live_share']:.2f}" for o in bands),
        }
        # the band that learns most spends its own support along the anchor run
        fastest = max(bands, key=lambda o: o["observed_spread"]["mean"] - o["base_pass_rate"])
        macros |= {
            "SweepFastBand": fastest["label"].split("-")[1],
            "SweepFastStart": f"{fastest['anchor_trace'][0]['live_share']:.2f}",
            "SweepFastEnd": f"{fastest['anchor_trace'][-1]['live_share']:.2f}",
        }
        if inside:
            won = inside[0]
            won_halves = max(won["predicted_half_samples"]) / max(
                min(won["predicted_half_samples"]), 1e-12)
            macros |= {
                "SweepWinBand": won["label"].split("-")[1],
                "SweepWinForecast": f"{won['predicted_std']:.4f}",
                "SweepWinMeasured": f"{won['resolved_std']:.4f}",
                "SweepWinLow": f"{won['resolved_interval']['lo']:.4f}",
                "SweepWinHigh": f"{won['resolved_interval']['hi']:.4f}",
                "SweepWinRatio": f"{max(won['ratio'], 1 / won['ratio']):.2f}",
                "SweepWinResidual": f"{won['curvature_residual']:.1f}",
                "SweepWinHalfRatio": f"{won_halves:.2f}",
                "SweepWinGain": f"{won['observed_spread']['mean'] - won['base_pass_rate']:.3f}",
            }

    # the difficulty scan: where each family sits for each policy, on the live-group coordinate
    scans = [r for r in io.load_all("s10_difficulty") if r["result"].get("families")]
    # where two scans of one model cover the same family, the larger scan is the later and more
    # complete survey; merge in increasing size so it wins whatever order the records load in
    scans.sort(key=lambda r: (len(r["result"]["families"]), r["manifest"]["config_hash"]))
    if scans:
        by_model = {}
        for record in scans:
            model = record["manifest"]["config"]["model"]
            for row in record["result"]["families"]:
                by_model.setdefault(model, {})[row["family"]] = row
        small = next((v for k, v in by_model.items() if "0.5B" in k), {})
        large = {}
        for k, v in by_model.items():
            if "7B" in k:
                large |= v
        matched = set(small) & set(large)
        macros |= {
            "ScanFamilies": f"{len(set(small) | set(large))}",
            "ScanMatched": f"{len(matched)}",
            "ScanSmallUsable": f"{sum(1 for r in small.values() if r['in_band'] >= 0.2)}",
            "ScanLargeUsable": f"{sum(1 for r in large.values() if r['in_band'] >= 0.2)}",
        }
        # the rung where a graded family falls off the cliff
        rungs = [(k, v["mean"]) for k, v in large.items() if k.startswith("sort_digits_")]
        if len(rungs) >= 2:
            rungs.sort()
            macros["ScanCliffHigh"] = f"{max(m for _, m in rungs):.3f}"
            macros["ScanCliffLow"] = f"{min(m for _, m in rungs):.3f}"

    # the prospective test on a corpus of genuinely mixed prompts. The store also holds the support
    # sweep, the pretrained sizing recipes and a two-step smoke run, and taking the newest record
    # had quoted each of those here in turn; the run this paragraph describes is selected by label.
    mixed = [
        r for r in io.load_all("s11_prospective")
        if r["result"].get("label") == "qwen2.5-0.5b-mixed"
        and r["result"].get("resolved_std") is not None
    ]
    if len(mixed) > 1:
        raise SystemExit("more than one record labelled qwen2.5-0.5b-mixed")
    if mixed:
        outcome = mixed[-1]["result"]
        spread = outcome["resolved_std"]
        point = outcome["predicted_std"]
        low, high = sorted(outcome["predicted_half_samples"])
        macros |= {
            "MixedSeeds": f"{len(outcome['held_out_scores'])}",
            "MixedBase": f"{outcome['base_pass_rate']:.3f}",
            "MixedMean": f"{outcome['observed_spread']['mean']:.3f}",
            "MixedLive": f"{outcome['anchor_live_share']:.2f}",
            "MixedForecast": f"{point:.4f}",
            "MixedObserved": f"{outcome['observed_spread']['std']:.4f}",
            "MixedResolved": f"{spread:.4f}",
            "MixedRatio": f"{point / max(spread, 1e-12):.1f}",
            "MixedHalfRatio": f"{high / max(low, 1e-12):.2f}",
            "MixedResidual": f"{outcome['curvature_residual']:.1f}",
            "MixedDecision": outcome["decision"],
        }
        band = outcome.get("resolved_interval") or {}
        if band:
            macros["MixedResolvedLow"] = f"{band['lo']:.4f}"
            macros["MixedResolvedHigh"] = f"{band['hi']:.4f}"

    # the one-run diagnostic: calibrated on the headline grid, then scored on the replicate and on
    # a grid from a different environment seed with that threshold held fixed
    headline_cells = headline_grid()["result"]["cells"]
    macros |= diagnostic_macros(headline_cells, "Diag")
    cut = float(macros["DiagThreshold"])
    macros |= diagnostic_macros(replicate_grid()["result"]["cells"], "DiagReplicate", cut)
    other = [
        r for r in io.load_all("p2_metric_forecast")
        if r["manifest"]["config"].get("seed") == OTHER_ENV_SEED and r["result"].get("cells")
    ]
    if len(other) != 1:
        raise SystemExit(f"expected one grid at environment seed {OTHER_ENV_SEED}")
    macros |= diagnostic_macros(other[0]["result"]["cells"], "DiagOtherEnv", cut)

    # and what it says about the two pretrained runs, using that threshold unchanged
    for record in io.load_all("s9_abstain"):
        outcome = record["result"]
        per = np.array(outcome["per_checkpoint"], dtype=float)
        radii = np.array(outcome["radii"], dtype=float)
        live = radii > 0.0
        if not live.any():
            continue
        tag = "Small" if "0.5b" in outcome["label"] else "Large"
        macros |= {
            f"Abstain{tag}": f"{np.nanmean(per[live]):.1f}",
            f"Abstain{tag}Live": f"{int(live.sum())}",
            f"Abstain{tag}Checks": f"{len(per)}",
        }

    saturation = [r for r in io.load_all("s6_saturation") if r["result"].get("divergence")]
    if saturation:
        result = saturation[-1]["result"]
        divergences = [row["divergence"] for row in result["divergence"]]
        macros |= {
            "SatScore": f"{result['held_out_scores'][0]:.3f}",
            "SatPrompts": f"{result['prompts']}",
            "SatDisagreeing": f"{result['disagreeing_prompts']}",
            "SatDegenerate": f"{100 * result['degenerate_fraction']:.0f}",
            "SatKLlow": f"{min(divergences):.1e}".replace("e-0", "e-"),
            "SatKLhigh": f"{max(divergences):.1e}".replace("e-0", "e-"),
        }

    sizing = sizing_record()
    if sizing is not None:
        result = sizing["result"]
        finest = result["targets"][-1]
        runs = result["runs_at_target"]
        ratio = result["ratio_at_finest"]

        def band(name):
            median, lo, hi = ratio[name]
            return f"{median:.2f}", f"{lo:.2f}", f"{hi:.2f}"

        fixed, fixed_lo, fixed_hi = band("fixed")
        empirical, empirical_lo, empirical_hi = band("empirical")
        macros |= {
            "SizePairs": f"{result['pairs']}",
            "SizeArms": f"{result['arms']}",
            "SizeSeeds": f"{result['seeds']}",
            "SizeHoldout": f"{result['holdout']}",
            "SizeCap": f"{result['seeds'] - result['holdout']}",
            "SizeTarget": f"{100 * finest:.1f}",
            "SizeAdjointRuns": f"{runs['adjoint'][-1]:.2f}",
            "SizeOracleRuns": f"{runs['oracle'][-1]:.2f}",
            "SizeFixedRuns": f"{runs['fixed'][-1]:.1f}",
            "SizeFixedGain": fixed,
            "SizeFixedLo": fixed_lo,
            "SizeFixedHi": fixed_hi,
            "SizeEmpiricalRuns": f"{runs['empirical'][-1]:.2f}",
            "SizeEmpiricalGain": empirical,
            "SizeEmpiricalLo": empirical_lo,
            "SizeEmpiricalHi": empirical_hi,
            "SizeConstantGain": band("constant")[0],
            "SizeConstantLo": band("constant")[1],
            "SizeConstantHi": band("constant")[2],
            "SizeFrozenGain": band("frozen")[0],
            "SizeFrozenRuns": f"{runs['frozen'][-1]:.2f}",
            "SizeFrozenVsFixed": f"{runs['fixed'][-1] / runs['frozen'][-1]:.1f}",
            "SizeOracleGain": band("oracle")[0],
            "SizeOracleLo": band("oracle")[1],
            "SizeOracleHi": band("oracle")[2],
            "SizeForecastError": f"{result['forecast_accuracy']['median']:.2f}",
            "SizeFrozenScale": f"{result['frozen_scale']:.0f}",
            "SizeFrozenScaleSpread": f"{result['frozen_scale_spread']:.1f}",
            "SizeFrozenRelative": f"{result['frozen_relative']:.2f}",
            "SizeFrozenRelativeWorst": f"{result['frozen_relative_worst']:.2f}",
        }

    overlap = sizing_record(holdout=0)
    if overlap is not None:
        macros |= {
            "SizeOverlapFixed": f"{overlap['result']['ratio_at_finest']['fixed'][0]:.2f}",
            "SizeOverlapEmpirical": f"{overlap['result']['ratio_at_finest']['empirical'][0]:.2f}",
            "SizeOverlapRuns": f"{overlap['result']['runs_at_target']['adjoint'][-1]:.2f}",
        }

    real = real_sizing_record()
    if real is not None:
        result = real["result"]
        margin = min(result["runs_at_target"], key=float)
        runs = result["runs_at_target"][margin]
        ratio = result["ratio_at_finest"][margin]
        macros |= {
            "RealSizeRecipes": f"{result['recipes']}",
            "RealSizeSeeds": f"{result['seeds']}",
            "RealSizeRuns": f"{result['recipes'] * (result['seeds'] + 1)}",
            "RealSizePairs": f"{result['pairs'][margin]}",
            "RealSizeAllPairs": f"{result['recipes'] * (result['recipes'] - 1) // 2}",
            "RealSizeMargin": f"{float(margin):g}",
            "RealSizeError": f"{result['forecast_error']['median']:.2f}",
            "RealSizeErrorWorst": f"{result['forecast_error']['worst']:.2f}",
            "RealSizeTarget": f"{100 * result['targets'][-1]:.0f}",
            "RealSizeAdjoint": f"{runs['adjoint'][-1]:.2f}",
            "RealSizeFixed": f"{runs['fixed'][-1]:.2f}",
            "RealSizeFixedGain": f"{ratio['fixed'][0]:.2f}",
            "RealSizeFixedLo": f"{ratio['fixed'][1]:.2f}",
            "RealSizeFixedHi": f"{ratio['fixed'][2]:.2f}",
            "RealSizeEmpiricalGain": f"{ratio['empirical'][0]:.2f}",
            "RealSizeEmpiricalLo": f"{ratio['empirical'][1]:.2f}",
            "RealSizeEmpiricalHi": f"{ratio['empirical'][2]:.2f}",
            "RealSizeConstantGain": f"{ratio['constant'][0]:.2f}",
            "RealSizeOracleGain": f"{ratio['oracle'][0]:.2f}",
            "RealSizeOracleLo": f"{ratio['oracle'][1]:.2f}",
            "RealSizeOracleHi": f"{ratio['oracle'][2]:.2f}",
        }

    choice = latest_record("s13_selection")
    if choice is not None:
        result = choice["result"]
        macros |= {
            "PickTasks": f"{result['tasks']}",
            "PickConstant": f"{100 * result['kept']['constant'][0]:.0f}",
            "PickFrozen": f"{100 * result['kept']['frozen'][0]:.0f}",
            "PickAdjoint": f"{100 * result['kept']['adjoint'][0]:.0f}",
            "PickOracle": f"{100 * result['kept']['oracle'][0]:.0f}",
            "PickAdjointLow": f"{100 * result['kept']['adjoint'][1]:.0f}",
            "PickGain": f"{100 * result['gain_points']:.1f}",
            "PickGainLo": f"{100 * result['gain_low']:.1f}",
            "PickGainHi": f"{100 * result['gain_high']:.1f}",
            "PickOraclePoints": f"{100 * result['oracle_points']:.1f}",
            "PickShare": f"{result['share_of_oracle']:.2f}",
        }

    tolerance = tolerance_record()
    if tolerance is not None:
        result = tolerance["result"]
        finest = str(min(float(t) for t in result["targets"]))
        summary = result["summary"][finest]
        rows = {float(k): v for k, v in result["runs"][finest].items()}
        column = result["biases"].index(1.0)

        # A systematic scale error should be invisible: the stopping threshold divides it out.
        # Read along the row where the forecast is otherwise exact, over the columns the
        # threshold grid can actually reach.
        flat = rows[min(rows)]
        reached = [(b, v) for b, v in zip(result["biases"], flat, strict=True) if v is not None]
        macros |= {
            "TolFixed": f"{summary['fixed']:.1f}",
            "TolClean": f"{summary['clean']:.2f}",
            "TolTarget": f"{100 * float(finest):.1f}",
            "TolScaleRange": f"{max(b for b, _ in reached) / min(b for b, _ in reached):.0f}",
            "TolScaleSpread": f"{max(v for _, v in reached) / min(v for _, v in reached):.3f}",
            "TolTested": f"{max(rows):.0f}",
        }
        if summary["breakeven"] is not None:
            macros["TolBreakeven"] = f"{summary['breakeven']:.1f}"

        # The break-even moves with how reliable the answer has to be, because a fixed budget
        # scales badly with the target while a corrupted forecast does not. Report the loosest
        # target too, which is where the condition actually binds.
        loosest = str(max(float(t) for t in result["targets"]))
        loose = result["summary"][loosest]
        macros |= {
            "TolLooseTarget": f"{100 * float(loosest):.0f}",
            "TolLooseFixed": f"{loose['fixed']:.2f}",
            "TolLooseClean": f"{loose['clean']:.2f}",
        }
        if loose["breakeven"] is not None:
            macros["TolLooseBreakeven"] = f"{loose['breakeven']:.1f}"
        for factor, tag in ((1.5, "OneHalf"), (2.0, "Double"), (2.5, "TwoHalf"),
                            (4.0, "Quadruple")):
            if factor in rows and rows[factor][column] is not None:
                macros[f"TolRuns{tag}"] = f"{rows[factor][column]:.1f}"

    if not macros:
        raise SystemExit("no records to build paper macros from")
    macros |= heldout_macros()
    macros |= saturation_macros()
    macros |= adam_cold_macros()
    macros |= estimated_macros()
    macros |= shortfall_macros()
    macros |= collapse_sweep_macros()
    macros |= hand_typed_macros()
    command = chr(92) + "newcommand"
    lines = [
        command + "{" + chr(92) + "num" + name + "}{" + value + "}"
        for name, value in sorted(macros.items())
    ]
    write("numbers", chr(10).join(lines))


def _sci(value: float) -> str:
    """Scientific notation the way the paper writes it."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def table_null() -> None:
    """The learning condition against the condition with the signal removed."""
    runs = [
        r
        for r in io.load_all("s8_null")
        if len(r["result"].get("conditions", {}).get("verifier", {}).get("step", [])) >= 5
    ]
    if not runs:
        raise SystemExit("no null experiment of usable length")
    result = runs[-1]["result"]
    left, right = result["conditions"]["verifier"], result["conditions"]["coin"]
    lines = [
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{learning} & \multicolumn{2}{c}{signal removed} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"updates & diverg. & pass & diverg. & pass \\",
        r"\midrule",
    ]
    for index, step in enumerate(left["step"]):
        lines.append(
            f"{step} & {_sci(left['pairwise_kl'][index])} & {left['pass_rate'][index]:.3f} & "
            f"{_sci(right['pairwise_kl'][index])} & {right['pass_rate'][index]:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("null", "\n".join(lines))


def per_seed_evaluation(record) -> bool:
    """Did each run in this record draw its own evaluation randomness?

    Only the record itself can say. A run that predates the field cannot be dated from its
    manifest, because a long run outlives commits and the stored SHA describes the tree at save
    time. Absent an explicit statement we assume the draw was shared, which is the conservative
    reading: it reports the seed term as a bracket rather than a subtraction that may over-correct.
    """
    return bool(record["result"].get("eval_key_per_seed", False))


def table_scale() -> None:
    """The same measurement at two model sizes, side by side."""
    rows = []
    chosen = experiment("real_05b_sgd", require=("resolved_std",))
    if chosen is not None:
        rows.append(("Qwen2.5-0.5B", chosen["result"], chosen["manifest"]["config"],
                     per_seed_evaluation(chosen)))
    # the re-scoring pass is what the seven-billion numbers come from: the training loop scored
    # as it went, before each run drew its own evaluation randomness
    big = [
        r
        for store in ("s6_rescore", "s6_scale")
        for r in io.load_all(store)
        if r["result"].get("resolved_std") is not None
        and len(r["result"].get("held_out_scores", [])) >= 4
    ]
    if big:
        rows.append(("Qwen2.5-7B", big[0]["result"], big[0]["manifest"]["config"],
                     per_seed_evaluation(big[0])))
    if len(rows) < 2:
        raise SystemExit("need both model sizes measured")

    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "c" * len(rows) + r"@{}}",
        r"\toprule",
        "quantity & " + " & ".join(name for name, _, _, _ in rows) + r" \\",
        r"\midrule",
    ]

    def line(label, values):
        lines.append(f"{label} & " + " & ".join(values) + r" \\")

    line("runs", [f"{len(r['held_out_scores'])}" for _, r, _, _ in rows])
    line("updates", [f"{c.get('steps', c.get('step', '?'))}" for _, _, c, _ in rows])
    line("held-out pass rate, base", [f"{r['base_pass_rate']:.3f}" for _, r, _, _ in rows])
    line("held-out pass rate, after",
         [f"{r['observed_spread']['mean']:.3f}" for _, r, _, _ in rows])
    lines.append(r"\midrule")
    line("spread across runs", [f"{r['observed_spread']['std']:.4f}" for _, r, _, _ in rows])
    line(r"\quad evaluation part",
         [f"{np.sqrt(r['binomial_variance']):.4f}" for _, r, _, _ in rows])
    # Subtracting the binomial term assumes each run drew its own evaluation. Where that is not
    # recorded the draw was shared, the subtraction would over-correct, and all the data support
    # is the bracket of Section 5: the seed term lies between the subtraction and the spread.
    seed_part = []
    for _, outcome, _, per_seed in rows:
        observed = outcome["observed_spread"]["std"]
        if per_seed:
            seed_part.append(f"{outcome['resolved_std']:.4f}")
        else:
            lower = np.sqrt(max(observed**2 - outcome["binomial_variance"], 0.0))
            seed_part.append(f"[{lower:.4f}, {observed:.4f}]")
    line(r"\quad seed part", seed_part)
    spread = []
    for _, outcome, _, _ in rows:
        scores = np.sort(np.array(outcome["held_out_scores"]))
        gaps = np.diff(scores)
        cut = int(np.argmax(gaps))
        spread.append(f"{cut + 1} / {scores.size - cut - 1}")
    line("split at the largest gap", spread)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("scale", "\n".join(lines))


def table_power() -> None:
    """What the real-model design can resolve, simulated at the design actually used."""
    runs = [r for r in io.load_all("s7_power") if r["result"].get("rows")]
    if not runs:
        raise SystemExit("no power simulation")
    rows = runs[-1]["result"]["rows"]
    counts = sorted({row["runs"] for row in rows})[:2]  # two columns fit; three do not
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "cc" * len(counts) + r"@{}}",
        r"\toprule",
        "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n} runs}}" for n in counts) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(len(counts))),
        "true s.d. & "
        + " & ".join(["resolved & estimate"] * len(counts))
        + r" \\",
        r"\midrule",
    ]
    for true_sd in sorted({row["true_sd"] for row in rows}):
        cells = []
        for count in counts:
            match = [r for r in rows if r["true_sd"] == true_sd and r["runs"] == count]
            if match:
                cells.append(f"{100 * match[0]['nonzero_rate']:.0f}\\% & "
                             f"{match[0]['median_estimate']:.3f}")
            else:
                cells.append("-- & --")
        lines.append(f"{true_sd:.3f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("power", "\n".join(lines))


# Experiments are selected by what they are, never by which ran last. A table headed
# "Qwen2.5-0.5B" must not silently pick up a seven-billion run that happens to be newer, which is
# exactly what "latest with a resolved spread" did once a second real-model study existed.
EXPERIMENTS = {
    "real_05b_sgd": {
        "store": "s7_real_forecast",
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        "optimiser": "sgd",
        "steps": 12,
        # the launch after the transport sign fix (0f4c098): the earlier launch predates it, and
        # the traces run skipped the forecast
        "label": "qwen2.5-0.5b",
        "skip_forecast": False,
    },
    "real_7b_adam": {
        "store": "s7_real_forecast",
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "optimiser": "adam",
        "steps": 12,
        "label": "qwen2.5-7b-adam",
    },
    # the forecast that carries Adam's state backwards, and the superseded one that did not.
    # Both are kept because the difference between them is a result in its own right.
    "real_7b_adam_forecast": {
        "store": "s7_real_forecast",
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "optimiser": "adam",
        "steps": 12,
        "label": "qwen2.5-7b-adam-state",
    },
    "real_7b_adam_stateless": {
        "store": "s7_real_forecast",
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "optimiser": "adam",
        "steps": 12,
        "label": "qwen2.5-7b-adam-wide",
    },
}


def experiment(name: str, require=None):
    """The one record matching a named experiment, or None.

    `require` names result fields that must be present, so a caller asking for a measured spread
    cannot be handed a forecast-only run. When several records share the identity, the one with
    the most held-out seeds wins, and among those the most recent: `io.load_all` returns records
    oldest first and the sort is stable, so the choice does not depend on file order.
    """
    spec = EXPERIMENTS[name]
    hits = []
    for record in io.load_all(spec["store"]):
        config = record["manifest"]["config"]
        if config.get("model") != spec["model"]:
            continue
        if config.get("optimiser", "sgd") != spec["optimiser"]:
            continue
        if config.get("steps") != spec["steps"]:
            continue
        if spec.get("label") and config.get("label") != spec["label"]:
            continue
        if "skip_forecast" in spec and config.get("skip_forecast") != spec["skip_forecast"]:
            continue
        if any(record["result"].get(field) is None for field in (require or ())):
            continue
        hits.append(record)
    if not hits:
        return None
    hits.sort(key=lambda r: len(r["result"].get("held_out_scores", [])))
    return hits[-1]


def cheap_variants(chosen: dict, records: list[dict]) -> dict:
    """The untransported forecasts that can stand beside `chosen`, keyed by `variance_prompts`.

    A variant must be the same model, optimiser and run length as `chosen`, held at `grad M`,
    and must have recorded the adapter initialisation it started from. Runs made before the
    initialisation was recorded are excluded rather than silently pooled with the rest, and the
    survivors are required to agree on that checksum, so the comparison between them varies the
    one thing it means to vary. Records are taken oldest first, so where two share a key the
    later one is kept.
    """
    config = chosen["manifest"]["config"]
    identity = (config.get("model"), config.get("optimiser", "sgd"), config.get("steps"))
    variants = [
        record
        for record in records
        if (
            record["manifest"]["config"].get("model"),
            record["manifest"]["config"].get("optimiser", "sgd"),
            record["manifest"]["config"].get("steps"),
        )
        == identity
        and not record["result"].get("transport", True)
        and len(record["result"].get("kernel", [])) >= 8
        and record["result"].get("init_checksum") is not None
    ]
    if not variants:
        return {}
    origin = variants[-1]["result"]["init_checksum"]
    return {
        record["manifest"]["config"].get("variance_prompts"): record["result"]
        for record in variants
        if record["result"]["init_checksum"] == origin
    }


def table_real_forecast() -> None:
    """The forecast on a pretrained model, against the spread it was made before seeing."""
    chosen = experiment("real_05b_sgd", require=("resolved_std",))
    if chosen is None or len(chosen["result"].get("kernel", [])) < 8:
        raise SystemExit("no 0.5B forecast with a measured spread")
    result = chosen["result"]
    observed = result["observed_spread"]
    # only forecasts of the same model that started from the same adapters can be compared
    cheap = cheap_variants(chosen, io.load_all("s7_real_forecast"))
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        r"quantity & value \\",
        r"\midrule",
        f"held-out pass rate, base & {result['base_pass_rate']:.3f} \\\\",
        f"held-out pass rate, after {observed['n_runs']} runs & {observed['mean']:.3f} \\\\",
        r"\midrule",
        f"observed s.d.\\ across runs & {observed['std']:.4f} \\\\",
        f"\\quad evaluation (binomial) part & {np.sqrt(result['binomial_variance']):.4f} \\\\",
        f"\\quad seed part, by subtraction & {result['resolved_std']:.4f} \\\\",
    ]
    band = result.get("resolved_interval")
    if band:
        lines.append(
            f"\\quad its nominal $95\\%$ interval & [{band['lo']:.4f}, {band['hi']:.4f}] \\\\"
        )
    lines += [
        r"\midrule",
        f"forecast from one run & {result['predicted_std']:.4f} \\\\",
    ]
    for prompts in sorted(cheap, key=lambda v: (v is not None, v or 0)):
        label = (
            "\\quad held at $\\nabla M$"
            if prompts is None
            else f"\\quad held, $\\Sigma$ from {prompts} prompts"
        )
        lines.append(f"{label} & {cheap[prompts]['predicted_std']:.4f} \\\\")
    agreement = result.get("jvp_agreement") or []
    if agreement:
        lines.append(
            r"\midrule"
        )
        lines.append(
            f"agreement of two $Jb$ estimates & {np.nanmedian(agreement):.2f} \\\\"
        )
    live = result.get("live_share") or []
    if live:
        alive = sum(1 for value in live if value > 0)
        lines.append(f"updates with a mixed group & {alive} of {len(live)} \\\\")
    kernel = np.array(result.get("kernel") or [])
    if kernel.size and kernel.sum() > 0:
        share = kernel / kernel.sum()
        half = share.size // 2
        lines.append(
            f"share of it made in the first half & "
            f"{100 * share[:half].sum():.0f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("real_forecast", "\n".join(lines))


def table_transport() -> None:
    """What carrying the adjoint back is worth, against holding it at grad M."""
    cells = headline_grid()["result"]["cells"]
    if "untransported_std" not in cells[0]:
        raise SystemExit("the headline grid carries no transport ablation")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{median} & \multicolumn{2}{c}{worst} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"settings & carried & held & carried & held \\",
        r"\midrule",
    ]

    def row(label, rows):
        out = []
        for key in ("predicted_std", "untransported_std"):
            out.append(np.abs(np.log(np.array([c[key] / c["measured_std"] for c in rows]))))
        lines.append(
            f"{label} & {np.exp(np.median(out[0])):.2f} & {np.exp(np.median(out[1])):.2f} & "
            f"{np.exp(out[0].max()):.2f} & {np.exp(out[1].max()):.2f} \\\\"
        )

    row(f"all {len(cells)}", cells)
    for steps in sorted({c["steps"] for c in cells}):
        row(f"\\quad $T = {steps}$", [c for c in cells if c["steps"] == steps])
    for eta in sorted({c["step_size"] for c in cells}):
        row(f"\\quad $\\eta = {eta:g}$", [c for c in cells if c["step_size"] == eta])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("transport", "\n".join(lines))


def table_estimator_forecast() -> None:
    """Forecast accuracy for each estimator in the family, on a common sub-grid."""
    by_estimator: dict[str, list[dict]] = {}
    for record in io.load_all("p2_metric_forecast"):
        cells = record["result"].get("cells", [])
        config = record["manifest"]["config"]
        name = config.get("estimator", "rloo")
        if cells and name != "rloo":
            by_estimator[name] = cells
    # the other estimators' sub-grids were run before per-seed scores were stored, so their spread
    # includes the anchor; compare RLOO on the same footing rather than on the stricter one
    by_estimator["rloo"] = grids.forecast_grid()["result"]["cells"]
    if len(by_estimator) < 2:
        raise SystemExit("no cross-estimator sweep")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"estimator & settings & median & worst & inside 95\% \\",
        r"\midrule",
    ]
    label = {"rloo": "RLOO", "grpo_mean": "GRPO, mean baseline",
             "grpo_std": "GRPO, standardised"}
    for name in ("rloo", "grpo_mean", "grpo_std"):
        cells = by_estimator.get(name)
        if not cells:
            continue
        ratio = np.array([c["predicted_std"] / c["measured_std"] for c in cells])
        error = np.abs(np.log(ratio))
        inside = sum(c["measured_lo"] <= c["predicted_std"] <= c["measured_hi"] for c in cells)
        lines.append(
            f"{label.get(name, name)} & {len(cells)} & "
            f"{np.exp(np.median(error)):.2f}$\\times$ & {np.exp(error.max()):.2f}$\\times$ & "
            f"{inside}/{len(cells)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("estimator_forecast", "\n".join(lines))


def table_propagation() -> None:
    """Each model of seed divergence against Monte Carlo, broken out by pool homogeneity.

    A single median hides the result. Accumulation is adequate where the prompt pool is
    heterogeneous and the mean gradient field nearly flat, and wrong by a factor of two where the
    pool is homogeneous and the objective supplies real curvature. That is what the recursion
    predicts, so the breakdown is the finding rather than a caveat on it.
    """
    # the 54-setting grid at 25 updates; the smaller runs in the store are its pilots
    runs = [only("p1_propagation", steps=25)]
    if not runs:
        raise SystemExit("no propagation validation run")
    cells = runs[-1]["result"]["cells"]
    measured = np.array([c["measured_kl"] for c in cells])
    diversities = sorted({c["diversity"] for c in cells})
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}l" + "c" * (len(diversities) + 2) + r"@{}}",
        r"\toprule",
        "model & all & "
        + " & ".join(rf"$d = {d:g}$" for d in diversities)
        + r" & worst \\",
        r"\midrule",
    ]
    for key, label in (
        ("propagated_kl", r"propagated, \eqref{eq:unrolled}"),
        ("walk_kl", "accumulation"),
        ("scalar_kl", "single timescale"),
    ):
        error = np.abs(np.log(np.array([c[key] for c in cells]) / measured))
        cols = [f"{np.exp(np.median(error)):.2f}"]
        for value in diversities:
            mask = np.array([c["diversity"] == value for c in cells])
            cols.append(f"{np.exp(np.median(error[mask])):.2f}")
        cols.append(f"{np.exp(error.max()):.1f}")
        lines.append(f"{label} & " + " & ".join(cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("propagation", "\n".join(lines))


def main() -> None:
    for name, fn in (
        ("exact_agreement", table_exact_agreement),
        ("transfer", table_transfer),
        ("transformer", table_transformer),
        ("real", table_real),
        ("difficulty", table_difficulty),
        ("diagnosis", table_diagnosis),
        ("adam", table_adam),
        ("optimum", table_optimum),
        ("forecast", table_forecast),
        ("estimator_forecast", table_estimator_forecast),
        ("transport", table_transport),
        ("real_forecast", table_real_forecast),
        ("power", table_power),
        ("scale", table_scale),
        ("propagation", table_propagation),
        ("adam_lift", table_adam_lift),
        ("spectrum", table_spectrum),
        ("null", table_null),
        ("sizing", table_sizing),
        ("heldout", table_heldout),
        ("numbers", table_numbers),
    ):
        try:
            fn()
        except SystemExit as exc:
            if not (TABLES / f"{name}.tex").exists():
                write(name, PLACEHOLDER)
            print("pending:", exc)
    report_missing_macros()


def report_missing_macros() -> None:
    """Name the macros the paper cites that no experiment has produced yet.

    A number quoted in the text but never generated is a LaTeX error deep in a log. Saying so
    here, next to the run that would supply it, is the more useful place to find out.
    """
    paper = TABLES.parent
    sources = [paper / "main.tex", paper / "icml" / "main.tex"]
    sources += sorted((paper / "sections").glob("*.tex"))
    sources += sorted((paper / "icml" / "sections").glob("*.tex"))
    used = set()
    for path in (p for p in sources if p.exists()):
        used |= set(re.findall(r"\\num[A-Za-z]+", path.read_text()))
    defined = set(re.findall(r"\\num[A-Za-z]+", (TABLES / "numbers.tex").read_text()))
    missing = sorted(name.lstrip("\\") for name in used - defined)
    if missing:
        print("macros cited but not generated:", ", ".join(missing))


if __name__ == "__main__":
    main()
