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
    ("docs/claims.md", "median {RealJvpLow} and {RealJvpHigh} in the two released records"),
    ("docs/claims.md", "give {UnseededTransported} and {RealTransportedForecast} with transport "
                       "({UnseededTransportedFactor}x apart), {UnseededCheap} and "
                       "{RealCheapForecast} without ({UnseededCheapFactor}x)"),
    ("docs/forecast.md", "the transported forecast gives {UnseededTransported} and "
                         "{RealTransportedForecast} ({UnseededTransportedFactor}x apart) and the "
                         "cheap form {UnseededCheap} and {RealCheapForecast} "
                         "({UnseededCheapFactor}x)"),
    ("docs/claims.md", "`p4_linearity`, rank correlation {LinearityRank} | measured |"),
    ("docs/claims.md", "the residual does bound the worst case ({LinearityWorstBelow}x vs "
                       "{LinearityWorstAbove}x)"),
    ("docs/forecast.md", "from {LinearitySmallStep} to {LinearityLargeStep} as the step size "
                         "grows eightfold"),
    ("docs/forecast.md", "the rank correlation between them is {LinearityRank}. The residual "
                         "bounds the worst case ({LinearityWorstBelow}x below 5%, about "
                         "{LinearityWorstAbove}x above)"),
    ("README.md", "where it is {CollapseMissedError}x too large (band {CollapseMissedBand}) they "
                  "barely collapse"),
    ("README.md", "({CollapseMissedSplit} of {CollapsePrompts} against {CollapseHeldSplit};"),
    ("docs/claims.md", "band {CollapseMissedBand} splits on {CollapseMissedSplit} of "
                       "{CollapsePrompts} prompts against {CollapseHeldSplit} at band "
                       "{CollapseHeldBand}"),
    ("docs/claims.md", "`s11`: forecast over measured {SweepRatios} | measured |"),
    ("docs/predictions.md", "Band {CollapseMissedBand} has {CollapseMissedSplit} split prompts "
                            "of {CollapsePrompts}"),
    ("docs/predictions.md", "Only {CollapseMissedShared} prompt gets the same firm answer from "
                            "every run, against {CollapseHeldShared} at band {CollapseHeldBand}."),
    ("docs/forecast.md", "| measured spread across runs | {RealObservedSpread} |"),
    ("docs/forecast.md", "| evaluation (binomial) part | {RealBinomialPart} |"),
    ("docs/forecast.md", "| seed part, by subtraction | {RealSeedPart}, interval [{RealSeedLo}, "
                         "{RealSeedHi}] |"),
    ("docs/forecast.md", "| forecast, cheap form, `Sigma` from 64 prompts | {RealCheapWide} |"),
    ("docs/forecast.md", "| forecast, with transport | {RealTransportedForecast} |"),
    ("docs/forecast.md", "Three runs land at {RealClusterLow}, {RealClusterMid} and "
                         "{RealClusterHigh}, within {RealClusterWidth} of each other, while the "
                         "remaining five spread from {RealRestLow} to {RealRestHigh}."),
    ("docs/claims.md", "cheap form {RealCheapForecast} against a measured seed term of "
                       "{RealSeedPart} [{RealSeedLo}, {RealSeedHi}]"),
    ("README.md", "median error **{ForecastMedian}x**, worst {ForecastWorst}x, inside the measured "
                  "95% interval in {ForecastInside} of {ForecastSettings}"),
    ("README.md", "against the value measured over {ForecastSeeds} seeds trained afterwards"),
    ("README.md", "errors {SweepErrors}"),
    ("README.md", "**{AdamScaleShortfallLow}x to {AdamScaleShortfallHigh}x short**"),
    ("README.md", "**{HeldStrictAdjointRuns}** runs; a fixed budget measured below its cap needs "
                  "{SplitFixedGainLow}x to {SplitFixedGainHigh}x as many over {SplitCount} seed "
                  "splits, and estimating the spread from spent seeds {HeldStrictEmpiricalGain}x"),
    ("README.md", "the closed form with one scale learned elsewhere decides in "
                  "{NominalLooseCalibratedRuns} runs, and the forecast in "
                  "{NominalLooseAdjointRuns}"),
    ("README.md", "comes within {HeldStrictFrozenGain}x of the forecast"),
    ("README.md", "({EstStrictGain}x with its noise traces estimated from one run"),
    ("docs/claims.md", "with traces estimated from one run it is within {EstStrictGain}x | "
                       "`s18_sizing_estimated`: [{EstStrictLo}, {EstStrictHi}] at 0.1%, "
                       "{EstLooseGain}x at 1%; {EstFloored} of {EstConfigurations}"),
    ("README.md", "its scale is off by {SizeFrozenScale}x"),
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
    ("docs/claims.md", "needs {SplitFixedGainLow}x to {SplitFixedGainHigh}x as many | `s15` with "
                       "24/24 seed splits, {SplitCount} splits: {SplitFixedRunsLow} to "
                       "{SplitFixedRunsHigh} runs against {SplitAdjointRunsLow} to "
                       "{SplitAdjointRunsHigh}"),
    ("docs/claims.md", "and {HeldStrictEmpiricalGain}x fewer than estimating the deviation"),
    ("docs/claims.md", "within {HeldStrictFrozenGain}x of the forecast once its threshold is "
                       "learned | `s15`: [{HeldStrictFrozenLo}, {HeldStrictFrozenHi}]"),
    ("docs/claims.md", "scale off {SizeFrozenScale}x); with one scale learned on other tasks it "
                       "decides in {NominalLooseCalibratedRuns} runs, the forecast in "
                       "{NominalLooseAdjointRuns} runs with {NominalLooseAdjointMistakes}% "
                       "mistakes"),
]


@pytest.mark.parametrize("path,template", CHECKS)
def test_a_quoted_number_matches_the_generated_one(path, template):
    expected = " ".join(template.format(**macros()).split())
    assert expected in flat(path), f"{path} should contain: {expected}"
