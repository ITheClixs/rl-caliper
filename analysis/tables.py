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
    runs = [r for r in io.load_all("b6_adam") if "cells" in r["result"]]
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


def table_forecast() -> None:
    """Forecast accuracy, broken out by the run length and the batch it was made at."""
    runs = [r for r in io.load_all("p2_metric_forecast") if r["result"].get("cells")]
    if not runs:
        raise SystemExit("no forecast validation run")
    cells = runs[-1]["result"]["cells"]
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


def table_adam_lift() -> None:
    """How well each model of Adam's seed divergence does, overall and by pool."""
    runs = [
        r for r in io.load_all("p5_adam_lift") if len(r["result"].get("cells", [])) >= 8
    ]
    if not runs:
        raise SystemExit("no Adam lift sweep")
    cells = runs[-1]["result"]["cells"]
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


def table_numbers() -> None:
    """Numbers quoted in the paper's prose, as macros, so they are not transcribed by hand."""
    macros: dict[str, str] = {}

    forecast = [
        r for r in io.load_all("p2_metric_forecast") if len(r["result"].get("cells", [])) >= 100
    ]
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

    if not macros:
        raise SystemExit("no records to build paper macros from")
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


def table_real_forecast() -> None:
    """The forecast on a pretrained model, against the spread it was made before seeing."""
    runs = [
        r
        for r in io.load_all("s7_real_forecast")
        if r["result"].get("resolved_std") is not None
        and len(r["result"].get("kernel", [])) >= 8
    ]
    if not runs:
        raise SystemExit("no real-model forecast with a measured spread")
    result = runs[-1]["result"]
    observed = result["observed_spread"]
    cheap = [
        r["result"]
        for r in io.load_all("s7_real_forecast")
        if not r["result"].get("transport", True) and len(r["result"].get("kernel", [])) >= 8
    ]
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
        r"\midrule",
        f"forecast from one run & {result['predicted_std']:.4f} \\\\",
    ]
    if cheap:
        lines.append(f"\\quad without transport & {cheap[-1]['predicted_std']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("real_forecast", "\n".join(lines))


def table_transport() -> None:
    """What carrying the adjoint back is worth, against holding it at grad M."""
    runs = [
        r
        for r in io.load_all("p2_metric_forecast")
        if len(r["result"].get("cells", [])) >= 100
        and "untransported_std" in r["result"]["cells"][0]
    ]
    if not runs:
        raise SystemExit("no transport ablation")
    cells = runs[-1]["result"]["cells"]
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
        if cells:
            by_estimator[name] = cells
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
    """Candidate models of seed divergence against exact Monte Carlo ground truth."""
    runs = [r for r in io.load_all("p1_propagation") if len(r["result"]["cells"]) >= 8]
    if not runs:
        raise SystemExit("no propagation validation run")
    cells = runs[-1]["result"]["cells"]
    measured = np.array([c["measured_kl"] for c in cells])
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"model of seed divergence & median & worst & ratio \\",
        r"\midrule",
    ]
    names = [
        ("propagated_kl", r"propagated, \eqref{eq:unrolled}"),
        ("walk_kl", "accumulation"),
        ("scalar_kl", "single timescale"),
    ]
    for key, label in names:
        predicted = np.array([c[key] for c in cells])
        ratio = predicted / measured
        error = np.abs(np.log(ratio))
        lines.append(
            f"{label} & {np.exp(np.median(error)):.2f}$\\times$ & "
            f"{np.exp(error.max()):.1f}$\\times$ & "
            f"{ratio.min():.2f}--{ratio.max():.2f} \\\\"
        )
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
        ("propagation", table_propagation),
        ("adam_lift", table_adam_lift),
        ("spectrum", table_spectrum),
        ("null", table_null),
        ("numbers", table_numbers),
    ):
        try:
            fn()
        except SystemExit as exc:
            if not (TABLES / f"{name}.tex").exists():
                write(name, PLACEHOLDER)
            print("pending:", exc)


if __name__ == "__main__":
    main()
