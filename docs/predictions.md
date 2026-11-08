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

**Result.** Pending: recorded when the run finishes.
