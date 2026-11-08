# Predictions recorded before the result

Each entry is committed before the run that tests it finishes, so the commit history shows the
prediction preceded the evidence. An entry is kept whether it held or not.

## 2026-11-08: seed-dependent collapse defeats the forecast, uniform collapse does not

**Observed so far.** On the counting task of Section 6 most pretrained runs answer one constant to
every prompt, but different runs choose different constants ("1", "2", "4"), and the forecast misses
by 25x (`experiments/s16_saturation/analyse.py`). On the support-sweep corpus at live probability
0.85, rerun with every run's answers recorded (`s11_prospective --record-outputs`, label
`modes-band-0.85`), all thirteen runs answer "1" to all 38 counting prompts. That shared collapse
carries almost no spread (per-seed s.d. 0.004 on those prompts); the spread is on the other 26, and
there the forecast is within 1.24x.

**Hypothesis.** Collapse onto an absorbing answer defeats a linearised forecast only when the answer
a run is absorbed into depends on its seed.

**Criterion.** A prompt is *split* when at least two runs answer it with one string at least 90% of
the time and those strings differ. Band 0.85 has 5 split prompts of 64 (8%).

**Prediction.** The rerun of the band at live probability 0.60 (label `modes-band-0.60`), where the
original forecast missed by 2.80x, has a larger share of split prompts than band 0.85.

**What would refute it.** A split share at or below 8% on band 0.60, with the forecast still missing
by more than 1.5x.

**Result (2026-11-08, after the run). Refuted.** Band 0.60 has 3 split prompts of 64 (5%), fewer
than band 0.85's 5, and the forecast still misses by 2.80x (`runs/s16_sweep/6933b62b1f87.json`,
from `experiments/s16_saturation/sweep.py`). The rerun reproduces the original band-0.60 scores
run for run, so this is the same miss the sweep reported.

**What the prediction got wrong before the run.** At band 0.60 the forecast is too large (ratio
of forecast to measured spread 2.80), not too small. Seed-dependent collapse adds spread that a
linearisation cannot see, so it can only make a forecast too small. It could never have explained
this miss, and checking the sign of the ratio before writing the prediction would have shown that.

**Looked at after the result, not predicted.** Band 0.60 barely collapses at all. Its anchor answers
"1" on 29 of 64 prompts and no later run gives one answer to more than half of them. Only 1 prompt
gets the same firm answer from every run, against 26 at band 0.85. The forecast was accurate where
runs were absorbed into a shared answer and too large where they were not. That is one pair of
bands, observed after the fact. It is not a finding until a band that was not used to suggest it
is tested.

**What stands.** Constant-answer collapse explains the 25x shortfall on the counting task of
Section 6 and nothing in the support sweep. The 2.80x and 2.13x overpredictions in the two middle
bands remain unexplained.

## 2026-11-08: the forecast is too large where runs are not absorbed into shared answers

**Observed so far.** The pattern comes from two bands, seen after the refutation above. A prompt is
*shared* when every run (anchor and later seeds) answers it with one string at least 90% of the time
and all those strings agree (`shared_prompts` in `experiments/s16_saturation/sweep.py`). Band 0.85
has 26 shared prompts of 64, and its forecast is about right (forecast over measured 0.81). Band
0.60 has 1 of 64, and its forecast is too large (2.80).

**Hypothesis.** Whether the forecast is too large tracks how many prompts the runs are absorbed
into together. Few shared prompts go with a forecast that is too large; many go with one that is
not too large. This is an empirical pattern with no mechanism behind it. The sign is part of the
claim, because the last entry failed on exactly that.

**What is already known and what is not.** Bands 0.15 and 0.35 were run in the original sweep, so
their ratios are fixed. Band 0.15 is 0.60, too small. Band 0.35 is 2.13, too large. The reruns are
deterministic: band 0.85's and band 0.60's reproduced their scores run for run. The original runs
stored no answers, so neither band's shared count has been seen.

**Prediction.** Rerun with `--record-outputs` (labels `modes-band-0.15` and `modes-band-0.35`):
1. band 0.35, where the forecast is too large, has at most 13 shared prompts of 64;
2. band 0.15, where it is not too large, has at least 14.

The cut is halfway between the two bands already seen (26 and 1).

**What would refute it.** Either part failing. Each part is reported on its own, and the
hypothesis holds only if both do.

**Result (2026-11-08, after the runs). Refuted.** Both reruns reproduce their original held-out
scores run for run, and the forecasts are unchanged (ratios 2.13 and 0.60).
1. Band 0.35 has 1 shared prompt of 64, within the bound of 13. This part holds.
2. Band 0.15 has 2 shared prompts of 64, short of the 14 it needed. This part fails.

Across the four bands in order of live probability (0.15, 0.35, 0.60, 0.85) the shared counts are
2, 1, 1, 26 (`runs/s16_sweep/59f833094a47.json`, from `experiments/s16_saturation/sweep.py`). Band
0.15's forecast is not too large: it is 0.028 against a measured seed term whose interval is
[0.010, 0.070], so that band does not even show a resolved miss. Yet its runs share almost no
answers. Shared answers separate band 0.85 from the other three. They do not separate the
forecasts that are too large from those that are not.

**What the prediction got wrong before the run.** The sign was stated and checked this time. The
pattern came from two bands that differed in more than how much their runs shared: band 0.85 is the
easiest corpus and band 0.60 a harder one. A cut halfway between two points cannot tell the variable
it names from whatever else separates them. Runs are absorbed into one shared answer where one
answer is right often enough to be rewarded: the commonest correct answer covers 36% of band 0.85's
prompts and 3% of band 0.15's. Those shares are fixed by how the corpora were built, so they were
knowable before the run, and they are the reason part 2 should have been expected to fail.

**What stands.** Two predictions about collapse have now failed on the support sweep. Collapse
explains the 25x shortfall on the counting task of Section 6. It explains no error in the sweep,
in either direction. The overpredictions at bands 0.35 and 0.60 remain unexplained, and we have no
third candidate we would commit to.
