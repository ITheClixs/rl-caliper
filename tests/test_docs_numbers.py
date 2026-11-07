"""Numbers quoted in the README and docs must match the macros generated from the run records.

`analysis/tables.py` writes every number the paper cites to `paper/tables/numbers.tex`. The
README and the notes in `docs/` are written by hand, and a headline that was regenerated without
them is how the README once quoted 1.11x while the paper said 1.10x. Each check below renders a
phrase from the macros and requires it to appear verbatim, whitespace aside.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MACRO = re.compile(r"\\newcommand\{\\num(\w+)\}\{([^}]*)\}")


def macros() -> dict[str, str]:
    text = (ROOT / "paper" / "tables" / "numbers.tex").read_text()
    return dict(MACRO.findall(text))


def flat(path: str) -> str:
    return " ".join((ROOT / path).read_text().split())


CHECKS = [
    ("README.md", "median error **{ForecastMedian}x**, worst {ForecastWorst}x, inside the measured "
                  "95% interval in {ForecastInside} of {ForecastSettings}"),
    ("README.md", "against the value measured over {ForecastSeeds} seeds trained afterwards"),
    ("README.md", "errors {SweepErrors}"),
    ("README.md", "**{AdamScaleShortfallLow}x to {AdamScaleShortfallHigh}x short**"),
    ("README.md", "**{HeldStrictAdjointRuns}** runs, against {HeldStrictUncappedFixedRuns} for an "
                  "uncapped fixed budget (extrapolated; the measured one hits its 32-run cap) and "
                  "{HeldStrictEmpiricalGain}x as many from spent seeds"),
    ("README.md", "comes within {HeldStrictFrozenGain}x of the forecast"),
    ("README.md", "its scale is off by {SizeFrozenScale}x"),
    ("README.md", "decides in {NominalLooseAdjointRuns} runs with a {NominalLooseAdjointMistakes}% "
                  "mistake rate"),
    ("README.md", "AUROC {DiagAurocHigh}"),
    ("docs/forecast.md", "Median absolute error **{ForecastMedian}x**, worst **{ForecastWorst}x**; "
                         "forecast inside the measured 95% interval in **{ForecastInside} of "
                         "{ForecastSettings}**"),
    ("docs/forecast.md", "replicate: {ForecastReplicateMedian}x, {ForecastReplicateWorst}x, "
                         "{ForecastReplicateInside}."),
    ("docs/forecast.md", "| adjoint carried back | {ForecastMedian}x | {ForecastWorst}x |"),
    ("docs/forecast.md", "| adjoint held at grad M | {ForecastHeldMedian}x | "
                         "{ForecastHeldWorst}x |"),
    ("docs/forecast.md", "interval in {ForecastInside} and {ForecastHeldInside} settings"),
    ("docs/claims.md", "forecast matches realised spread to {ForecastMedian}x median, "
                       "{ForecastWorst}x worst"),
    ("docs/claims.md", "inside the measured 95% interval in {ForecastInside} of "
                       "{ForecastSettings}"),
    ("docs/claims.md", "`p2`: {ForecastReplicateMedian}x, {ForecastReplicateWorst}x, "
                       "{ForecastReplicateInside} of {ForecastSettings} inside"),
    ("docs/claims.md", "AUROC {DiagAurocHigh} for errors past 1.3x on the 192-setting "
                       "headline grid, {DiagReplicateAurocHigh} on the pool-16 replicate"),
    ("docs/claims.md", "the worst setting sits at the {DiagWorstPercentile}th residual percentile"),
    ("docs/claims.md", "the worst error is unchanged ({DiagWorstAll}x)"),
    ("docs/claims.md", "thresholds read in sample, the forecast needs {SizeFixedGain}x fewer runs"),
    ("docs/claims.md", "the forecast needs {HeldStrictAdjointRuns} runs at 0.1%"),
    ("docs/claims.md", "that is {HeldStrictUncappedGain}x fewer than an uncapped fixed budget"),
    ("docs/claims.md", "and {HeldStrictEmpiricalGain}x fewer than estimating the deviation"),
    ("docs/claims.md", "within {HeldStrictFrozenGain}x of the forecast once its threshold is "
                       "learned | `s15`: [{HeldStrictFrozenLo}, {HeldStrictFrozenHi}]"),
    ("docs/claims.md", "scale off {SizeFrozenScale}x); the forecast decides in "
                       "{NominalLooseAdjointRuns} runs with {NominalLooseAdjointMistakes}% "
                       "mistakes"),
]


@pytest.mark.parametrize("path,template", CHECKS)
def test_a_quoted_number_matches_the_generated_one(path, template):
    expected = " ".join(template.format(**macros()).split())
    assert expected in flat(path), f"{path} should contain: {expected}"
