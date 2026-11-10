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
    ("README.md", "runs gaining at most {HeadlineGainMax} in pass rate"),
    ("README.md", "({RegradeLearningCount} settings gaining at least {LearnGainCut}), "
                  "{RegradeSeeds} seeds each"),
    ("README.md", "more than 3x off in **{RegradeCarriedOverThree}%** of settings, up to "
                  "{RegradeCarriedWorst}x short"),
    ("README.md", "within 1.5x in **{RegradeTangentWithin}%** (the deviation over ten real seeds: "
                  "{RegradeTenWithin}%), beyond 3x in {RegradeTangentOverThree}%"),
    ("README.md", "against up to {SweepGainHigh} for the pretrained sweep"),
    ("README.md", "within 1.5x in {RegradeTangentWithin}% of those settings, about what the "
                  "deviation over ten real seeds achieves, and beyond 3x in "
                  "{RegradeTangentOverThree}%"),
    ("README.md", "it is beyond 3x in {RegradeCarriedOverThree}% of settings"),
    ("docs/claims.md", "gain a median of {HeadlineGainMedian} in pass rate and at most "
                       "{HeadlineGainMax} ({ReplicateGainMax} on the replicate)"),
    ("docs/claims.md", "beyond 3x in {RegradeCarriedOverThree}% of {RegradeLearningCount} "
                       "learning settings (too small in {RegradeCarriedShort}, too large in "
                       "{RegradeCarriedLong}), worst {RegradeCarriedWorst}x; cheap form median "
                       "{RegradeHeldMedian}x"),
    ("docs/claims.md", "median error {RegradeTangentMedian}x, within 1.5x in "
                       "{RegradeTangentWithin}% and beyond 3x in {RegradeTangentOverThree}% of the "
                       "same settings; the deviation over ten real seeds is within 1.5x in "
                       "{RegradeTenWithin}% and beyond 3x in {RegradeTenOverThree}%"),
    ("docs/claims.md", "{LearnTangentOverThreeCount} of {LearnLearningCount}, then "
                       "{RegradeTangentOverThreeCount} of {RegradeLearningCount}; "
                       "{LearnCarriedMedian}x; {RegradeFortySevenBelowHalf}%"),
    ("docs/predictions.md", "has a median error of {LearnCarriedMedian}x, not the 2x predicted"),
    ("docs/predictions.md", "median error of {LearnTangentMedian}x and are within 1.5x in "
                            "78% of the learning settings, but {LearnTangentOverThreeCount} of "
                            "the {LearnLearningCount} are beyond 3x"),
    ("docs/predictions.md", "median error of {RegradeTangentMedian}x against the 500-seed "
                            "deviation (1.3x allowed), are beyond 3x in "
                            "{RegradeTangentOverThreeCount} settings, {RegradeTangentOverThree}%"),
    ("docs/predictions.md", "within 1.5x in {RegradeTangentWithinCount} of "
                            "{RegradeLearningCount}, which is {RegradeTangentWithin}%"),
    ("docs/predictions.md", "beyond 3x in {RegradeCarriedOverThreeCount} of "
                            "{RegradeLearningCount} settings, {RegradeCarriedOverThree}%. It is "
                            "too small in {RegradeCarriedShort} of them and too large in "
                            "{RegradeCarriedLong}"),
    ("docs/predictions.md", "| deviation over 47 real seeds | 47 | {RegradeFortySevenWithin}% | "
                            "{RegradeFortySevenOverThree}% |"),
    ("docs/predictions.md", "| seeds simulated in the tangent model | 0 | {RegradeTangentWithin}% "
                            "| {RegradeTangentOverThree}% |"),
    ("docs/predictions.md", "| deviation over 10 real seeds | 10 | {RegradeTenWithin}% | "
                            "{RegradeTenOverThree}% |"),
    ("docs/predictions.md", "| deviation over 5 real seeds | 5 | {RegradeFiveWithin}% | "
                            "{RegradeFiveOverThree}% |"),
    ("docs/predictions.md", "| forecast of Theorem 2 | 1 | {RegradeCarriedWithin}% | "
                            "{RegradeCarriedOverThree}% |"),
    ("docs/predictions.md", "| deviation over 3 real seeds | 3 | {RegradeThreeWithin}% | "
                            "{RegradeThreeOverThree}% |"),
    ("README.md", "fitted exponent `{SpreadExponent}` against a predicted `+0.5`"),
    ("docs/results-seeds.md", "| tail slope of divergence against updates | "
                              "{ReconvergenceSlopeLow} to {ReconvergenceSlopeHigh}, median "
                              "{ReconvergenceSlopeMedian} |"),
    ("docs/results-seeds.md", "exponent in P {LevelBatch}, 95% CI [{LevelBatchLow}, "
                              "{LevelBatchHigh}]"),
    ("docs/results-seeds.md", "exponent in drift {LevelDrift}, 95% CI [{LevelDriftLow}, "
                              "{LevelDriftHigh}]"),
    ("docs/results-seeds.md", "fitted exponent {SpreadExponent}, 95% CI [{SpreadExponentLow}, "
                              "{SpreadExponentHigh}]"),
    ("docs/theory-seeds.md", "(fitted exponent {SpreadExponent}, 95% CI [{SpreadExponentLow}, "
                             "{SpreadExponentHigh}] against a predicted +0.5)"),
    ("docs/theory-seeds.md", "Tail slopes from {ReconvergenceSlopeLow} to "
                             "{ReconvergenceSlopeHigh} reject accumulation."),
    ("docs/theory-seeds.md", "give exponent {LevelBatch} in `P` (95% CI [{LevelBatchLow}, "
                             "{LevelBatchHigh}]) and {LevelDrift} in `D` (95% CI "
                             "[{LevelDriftLow}, {LevelDriftHigh}])"),
    ("docs/claims.md", "tail slopes {ReconvergenceSlopeLow} to {ReconvergenceSlopeHigh} "
                       "against +1"),
    ("docs/claims.md", "fitted exponent {SpreadExponent}, CI [{SpreadExponentLow}, "
                       "{SpreadExponentHigh}] against +0.5"),
    ("docs/claims.md", "exponents with intervals: {LevelBatch} [{LevelBatchLow}, "
                       "{LevelBatchHigh}], {LevelDrift} [{LevelDriftLow}, {LevelDriftHigh}]"),
    ("README.md", "held at band {AbsorbOverBand} ({AbsorbOverShared} shared prompt) and failed at "
                  "band {AbsorbUnderBand} ({AbsorbUnderShared}, where it needed "
                  "{AbsorbUnderBound})"),
    ("docs/claims.md", "band {AbsorbOverBand} has {AbsorbOverShared} shared prompt (at most "
                       "{AbsorbOverBound} predicted), band {AbsorbUnderBand} has "
                       "{AbsorbUnderShared} (at least {AbsorbUnderBound} predicted); the four "
                       "bands have {AbsorbShared}"),
    ("docs/predictions.md", "Band {AbsorbOverBand} has {AbsorbOverShared} shared prompt of "
                            "{CollapsePrompts}, within the bound of {AbsorbOverBound}."),
    ("docs/predictions.md", "Band {AbsorbUnderBand} has {AbsorbUnderShared} shared prompts of "
                            "{CollapsePrompts}, short of the {AbsorbUnderBound} it needed."),
    ("docs/predictions.md", "(ratios {AbsorbOverRatio} and {AbsorbUnderRatio})"),
    ("docs/predictions.md", "the shared counts are {AbsorbShared}"),
    ("docs/predictions.md", "it is {AbsorbUnderForecast} against a measured seed term whose "
                            "interval is [{AbsorbUnderLow}, {AbsorbUnderHigh}]"),
    ("docs/predictions.md", "the commonest correct answer covers {AbsorbTopCommonest}% of band "
                            "0.85's prompts and {AbsorbUnderCommonest}% of band 0.15's"),
    ("docs/claims.md", "median {AdamLiftWarmMedian}x with the curvature blocks, "
                       "{AdamLiftBareMedian}x without (worst {AdamLiftWarmWorst}x and "
                       "{AdamLiftBareWorst}x)"),
    ("docs/claims.md", "worth {AdamLiftParamsMedian}x in variance on the exact tier with theta "
                       "alone, {AdamLiftWalkMedian}x with no transport at all"),
    ("docs/theory-seeds.md", "median {AdamLiftBareMedian}x without them, {AdamLiftWarmMedian}x "
                             "with"),
    ("docs/forecast.md", "the two released launches that did give {EarlyCorrelation} and "
                         "{EarlyCorrelationRepeat}, the bootstrap interval at eight runs spans "
                         "[{EarlyCorrelationLow}, {EarlyCorrelationHigh}], and at 7B the "
                         "correlation is {EarlyCorrelationBig}."),
    ("docs/claims.md", "give {EarlyCorrelation} and {EarlyCorrelationRepeat} (a third"),
    ("docs/claims.md", "interval [{EarlyCorrelationLow}, {EarlyCorrelationHigh}]; "
                       "{EarlyCorrelationBig} at 7B"),
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
    ("README.md", "failed ({CollapseMissedSplit} of {CollapsePrompts} against "
                  "{CollapseHeldSplit})"),
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
    ("README.md", "median error **{ForecastMedian}x** ({ForecastHeldMedian}x in the cheap form "
                  "used on real models), worst {ForecastWorst}x, inside the measured 95% interval in "
                  "{ForecastInside} of {ForecastSettings}"),
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
