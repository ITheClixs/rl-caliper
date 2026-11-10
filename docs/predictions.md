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

## 2026-11-08: the sweep's misses belong to the corpus, not to the run that was the anchor

**Observed so far.** Each band's forecast was frozen from one anchor run (run 0). Its only check on
its own precision is the half-sample split of the prompts the injected term is estimated from, and
both halves share the anchor's trajectory and the metric gradient estimated at its end. At band 0.60
the halves agree to within 2% (0.110 and 0.109) while the forecast is 2.80x the measured spread.
Three corpora in the middle of the support range (bands 0.35 and 0.60, and the mixed corpus of
Section 6) were all too large, each from its own anchor. Against that, on band 0.85's corpus the
s11 sizing runs froze 0.0438 at learning rate 2.5e-4 and 0.0205 at 5e-4, where the measured spread
halves (0.0135 and 0.0247) as a linear forecast says it should. Something in the forecast moves a
lot between nearby trajectories, and nothing so far measures how much.

**Test.** Every s11 run is reproducible from its index, so validation seed k can be retrained with
its states kept and the forecast the protocol would have frozen from it computed, with the same
estimation keys (`experiments/s19_anchor_spread/run.py`). Each is graded against the resolved spread
over the other twelve runs. Runs 0 to 3 on bands 0.60 and 0.85. Those leave-one-out spreads are
already known from the records: 0.0391, 0.0390, 0.0371, 0.0381 at band 0.60; 0.0253, 0.0274,
0.0276, 0.0268 at band 0.85.

**Validity check, not part of the prediction.** Run 0 must reproduce its held-out scores and the
frozen forecast (0.1095 and 0.0205). If it does not, the script is wrong and nothing below is graded.

**Hypothesis.** The sweep's pattern is a property of the corpora: another anchor on the same corpus
would have missed the same way.

**Prediction.**
1. Band 0.60: at least three of the four forecasts exceed 1.5x their leave-one-out spread.
2. Band 0.85: at least three of the four forecasts are within 1.5x of theirs.

**What would refute it.** Either part failing. Each part is reported on its own. Not predicted, but
reported whatever it shows: the ratio of the largest to the smallest forecast on each band, and two
more forecasts from run 0 with other estimation keys, which hold the trajectory fixed.

**Confounds checked before the run.** The anchors differ only in their sampling keys, and the
estimation keys are held fixed, so a difference between anchors comes from the trajectory, or from
the same keys drawing differently on different states. The leave-one-out spreads overlap in eleven
of their twelve runs, so the four gradings within a band are not independent.

**Result.** Pending: recorded when both bands finish.

## 2026-11-10: the enumerable tier fails the same way once its runs learn, and simulated seeds do not

**Observed so far.** The 192 settings behind the headline (median error 1.10x) barely learn. The
run each forecast was made from gains a median of 0.009 in pass rate and at most 0.036 (0.069 on
the replicate grid). The pretrained sweep gains between 0.03 and 0.27. So the tier where the
forecast works and the tier where it does not differ in how far the policy moves, as well as in
everything else, and nothing so far separates the two.

An exploratory run on one environment (shape 3x3, pool 24, environment seed 0, group size 8, 32
later seeds, step sizes 1.5 to 60, 12 and 50 updates; not stored) raised the step size until runs
gained up to 0.13. With every input still exact, the forecast of Theorem 2 had a median error of
2.4x over those 48 settings, was more than 25x off in a tenth of them, and was up to 500x short
where the policy had collapsed onto one sequence. The same two signatures as on pretrained models:
too large by 2x to 3x at moderate movement, short by orders of magnitude after collapse.

The same exploratory run tried a different estimator. Linearise the model once, at the starting
parameters, instead of linearising the training dynamics about one run: the surrogate policy is
`pi'(a) ∝ pi_0(a) exp(s_a . theta')` with `s_a` the score of response `a` at `theta_0`. Training
this surrogate needs no forward or backward pass through the model, so thousands of seeds can be
simulated with the sampling, the estimator's weights and the softmax all kept exact. Its spread over
simulated seeds had a median error of 1.15x on those 48 settings and a worst case of 2.1x. It used
no training run at all.

**Test.** `experiments/p7_learning_regime/run.py` on settings the exploratory run did not touch:
shapes 3x3 and 4x2, pool 16, diversities 1.0 and 0.3, environment seed 991, group sizes 4 and 8,
prompts 8 and 32, 12 and 50 updates, step sizes 1.5, 6, 20 and 60, three difficulty bands: 384
settings, each graded against 47 seeds trained after the forecasts are made, the anchor excluded.
A setting is *learning* if its anchor run gains at least 0.05 in pass rate and *stationary* if it
gains less than 0.02. With 47 seeds a perfect forecast would score a median of about 1.07x.

**Hypothesis.** The failures on pretrained models are failures of linear response in the learning
regime, which the enumerable tier reproduces with exact inputs; and they are failures of the
linearisation of the dynamics, not of the model, so a surrogate that linearises only the model
survives them.

**Prediction.**
1. Learning settings, forecast of Theorem 2 (adjoint carried back): median error at least 2x, and
   at least a quarter of them off by more than 5x.
2. Learning settings, simulated seeds in the tangent model at `theta_0`: median error at most 1.3x,
   at least three quarters within 1.5x, none beyond 3x.
3. Stationary settings: the simulated seeds cost nothing where the forecast already works. Their
   median error is at most 0.10 above the forecast's.

**What would refute it.** Any part failing; each is reported on its own. If part 1 fails the
exploratory failure was particular to one environment. If part 2 fails the surrogate's exploratory
accuracy was.

**Confounds checked before the run.** On this tier the surrogate is favoured in a way it will not
be on a network: a tabular tree policy and its tangent model span the same family of distributions,
so the only error is that the features `s_a` are frozen while the true scores move. On a
transformer the log-probabilities are not linear in the parameters and the surrogate has a second
source of error that this test cannot see. A pass here licenses the test on a pretrained model; it
does not replace it. For a policy of length one the two coincide exactly, which is a unit test and
not evidence.

**Result.** Mixed: two of the three parts fail on a clause each, and the reason the second failed
turned out to be the measurement (`runs/p7_learning_regime/d9ac6f7bcf36.json`, 384 settings, 354 of
them learning and only 7 stationary, because this environment learns faster than the exploratory
one).

1. Failed on its first clause, held on its second. On the learning settings the forecast of
   Theorem 2 has a median error of 1.59x, not the 2x predicted. It is within 1.5x in 48% of them,
   more than 5x off in 30% (a quarter was predicted), and 653x off at worst. Holding the adjoint
   fixed is worse at the median (2.63x).
2. Held on its first two clauses, failed on its third. The simulated seeds have a median error of
   1.24x and are within 1.5x in 78% of the learning settings, but 25 of the 354 are beyond 3x, where
   none was predicted, the worst at 51x.
3. Held, on seven settings: median 1.10x for the simulated seeds against 1.20x for the forecast.

**What the 25 misses are.** In every one of them the simulated seeds spread *more* than the 47
real ones, in settings where the 47 nearly agree (a measured deviation between 0.0001 and 0.005).
That pattern has two readings, and they were told apart after the fact by training 1500 real seeds
in each of the 25 settings (not stored as a record; `far` is the share of seeds ending more than
0.01 from the median). In 23 of them the 1500 seeds contain a few runs, between 0.1% and 12%, that
end somewhere else, and the deviation over 1500 seeds is 3 to 37 times the deviation over 47. The
simulated seeds are within 2.3x of the 1500-seed deviation in those 23. In the other two, 1500
seeds show no such runs and the simulated spread is 3x and 17x too large. So most of what part 2
counted as the surrogate's error was the yardstick's: a deviation measured from 47 seeds misses
outcomes that occur in one run in a hundred, and those outcomes are most of the variance. The
forecast of Theorem 2 agrees with the 47-seed number in those settings for the same reason the
47-seed number is wrong. It describes the majority outcome only.

This was found by looking at failures, on settings chosen because they failed, so it is not a
confirmation of anything. The entry below tests it.

## 2026-11-10: against 500 seeds, the simulated seeds hold and a 47-seed deviation does not

**Observed so far.** The entry above. Its grading used 47 seeds per setting, and in 23 of the 25
settings where the simulated seeds looked worst, 1500 real seeds sided with the simulation.

**Test.** The same script and grid on an environment neither earlier run touched (environment seed
2718), with 500 seeds trained after the anchor instead of 47. Everything is graded against the
deviation over those 500. The deviation over the first 47 of them is graded the same way, as the
yardstick the entry above used.

**Hypothesis.** In the learning regime the distribution of a run's score over seeds has rare
outcomes that carry most of its variance. Seeds simulated in the tangent model reproduce them;
a few dozen real seeds, and a linearisation about one run, do not.

**Prediction.** On the learning settings (anchor gain at least 0.05):
1. Simulated seeds: median error at most 1.3x, at least 75% within 1.5x, at most 5% beyond 3x.
2. The forecast of Theorem 2: beyond 3x in at least 35% of them. Against 47 seeds it was 34%; the
   larger sample should count more of its misses, not fewer.
3. The first 47 seeds: their deviation is less than half the 500-seed deviation in at least 10% of
   the learning settings.

**What would refute it.** Any part failing. Part 3 failing would mean the rare outcomes were
particular to the settings picked out above, and the 23 of 25 a selection effect.

**Confounds checked before the run.** The 500-seed deviation is itself an estimate, and a heavy
tail makes it a noisy one: an outcome that occurs once in a thousand runs is missed by 500 seeds
more often than not. Part 1 allows for that with its 5%. The two settings above where 1500 seeds
showed nothing say the surrogate also invents rare outcomes sometimes, so part 1 is not expected
to hold with room to spare.

**Result.** Pending.
