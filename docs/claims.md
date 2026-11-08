# Claim index

Every claim the paper makes, what supports it, and how strongly. Written so that a reader who
distrusts a particular number can find the experiment that produced it without reading the code.

Legend for strength:
* **exact** — checked against enumeration or a closed form, to machine precision
* **measured** — a designed experiment with intervals, on the tier named
* **observed** — seen consistently, but not with enough replicates to put an interval on
* **negative** — we looked and did not find it

## The forecast

| claim | evidence | strength |
|---|---|---|
| `Var[M] = sum_t b'Q_t b` with `b_s = A_s' b_{s+1}` | Theorem 2; `tests/test_adjoint.py` builds `S_T` explicitly and compares | exact |
| forecast matches realised spread to 1.10x median, 1.56x worst | `p2_metric_forecast`, 192 settings at a pool of 24, graded on the 47 seeds trained after the anchor, prospective | measured |
| the same on a replicate grid at a pool of 16 | `p2`: 1.10x, 1.50x, 154 of 192 inside | measured |
| inside the measured 95% interval in 162 of 192 | same | measured |
| holds across two policy shapes (1.10x each) | same, shape as an axis | measured |
| holds across the estimator family | `p2` re-run under `grpo_mean` (1.10x) and `grpo_std` (1.10x), 12 settings each | measured |
| the transport is worth ~2% at these run lengths | `p2` ablation: adjoint held at `grad M` | measured |
| the cheap form costs one extra pass per update | arithmetic on the algorithm | exact |

## What the forecast is for: sizing a comparison

| claim | evidence | strength |
|---|---|---|
| with thresholds read in sample, the forecast needs 9.57x fewer runs than a fixed budget at 0.1% | `s13_sizing`; the fixed budget sits at its 32-run cap, which inflates the ratio | measured |
| with thresholds chosen on other tasks, the forecast needs 3.14 runs at 0.1% | `s15_sizing_heldout`, leave-one-task-out over 12 tasks, intervals resample tasks | measured |
| a fixed budget measured below its cap needs 5.6x to 8.2x as many | `s15` with 24/24 seed splits, 3 splits: 15.3 to 23.3 runs against 2.75 to 2.86 | measured |
| ~~that is 4.5x fewer than an uncapped fixed budget~~ | withdrawn: a normal approximation on selection-inflated gaps; it said 10 where the measured budget needs 15.3 to 23.3 | negative |
| and 2.12x fewer than estimating the deviation from spent seeds | `s15` | measured |
| the free closed form is within 1.11x of the forecast once its threshold is learned | `s15`: [0.97, 1.28], an interval containing one | measured |
| the stored run's scale replaces one learned factor | `s15`: at the textbook threshold the closed form never stops (scale off 84x); with one scale learned on other tasks it decides in 4.64 runs, the forecast in 4.37 runs with 0.001% mistakes at a 1% target | measured |
| no policy settles near-ties within the seeds available | `s15` at a 2 SE margin: every policy hits the cap at 0.1%, and at 1% all end near a 1% floor | measured |
| ~~a forecast used for sizing does not need to be calibrated~~ | withdrawn: true only where the threshold is tuned on comparisons with known outcomes | negative |
| ~~the break-even is wider than every pretrained error~~ | withdrawn: the unanimous corpus (25x) and 7B under Adam (3.1-17x) are outside it | negative |

## The mechanism

| claim | evidence | strength |
|---|---|---|
| `S_{t+1} = A_t S_t A_t' + Q_t` | Theorem 1 | exact |
| propagated covariance predicts divergence to 3% median, 1.2x worst | `p1_propagation`, 54 settings against Monte Carlo | measured |
| accumulation is wrong by 2.06x on homogeneous pools, 1.03x on heterogeneous ones | same, split by diversity | measured |
| a single timescale is wrong by 58x median, 791x worst | same | measured |
| the mean update is a gradient field, so `J` is symmetric | Proposition 3; measured asymmetry 3e-10 across three estimators | exact |
| `A_t` is not a contraction: radius > 1 throughout | `p6_spectrum`, three difficulty bands, 600 updates | measured |
| 31-51% of directions contract, share grows during the run | same | measured |
| the filtering requires a learning signal | `s8_null`: 0.14x with a verifier against 90.5x with a coin, same steps | measured |
| divergence does not accumulate | `s1`, `s5`: tail slopes -1.15 to +0.26 against +1 | measured |

## Where the uncertainty comes from

| claim | evidence | strength |
|---|---|---|
| over the grid as a whole, rollout sampling outweighs prompt selection (median share 78%) | `p3_memory_sources`, exact decomposition of `Sigma_t` | measured |
| ~~rollout sampling dominates at every group size~~ | withdrawn: the share is 46% at `G=16`, so the majority changes hands between `G=8` and `G=16` | negative |
| the share is monotone in `G` (93% at 2, 46% at 16) | same, 288 settings | measured |
| the prompt count cannot change the mix | both terms carry `1/P` | exact |
| there is no short memory horizon | `p3`: 95% of variance spans a median 1.00 of the run | negative |
| the kernel tilts toward the start | `p3`: 254 of 288 settings, median first-half share 55% | measured |

## Boundaries

| claim | evidence | strength |
|---|---|---|
| linearisation residual grows with the step but does not predict the error | `p4_linearity`, rank correlation 0.17 | measured |
| the residual does bound the worst case (1.21x vs 1.57x) | same | measured |
| lifting to Adam's state halves the error of ignoring it | `p5_adam_lift`, 24 settings | measured |
| the Adam lift is unreliable where the pass rate saturates | same, split by band: 1.48x live against 8.0x easy | measured |
| a collapsed policy injects nothing | count-based advantage is zero on a unanimous group | exact |
| RLVR collapses sampling within a few updates on a real model | `s7`: live share reaches zero at update five and stays under 5% after, at 8 and at 64 prompts | measured |
| a saturated metric cannot see a seed difference | `s6_saturation`: 0 of 32 prompts disagree, policies differ by up to 8e-2 | observed |
| the real-model directional derivative is noise-dominated | `s7`: cosine between independent estimates, median 0.07-0.5 over four runs | observed |
| the forecast underpredicts on a real model by ~25x | `s7`: cheap form 0.0025 against a measured seed term of 0.0631 [0.0350, 0.0735] | measured |
| the transport is not reproducible on a real model | `s7`: two initialisations give 0.0225 and 0.0040 with transport (5.7x apart), 0.0056 and 0.0025 without (2.2x) | measured |
| real RLVR runs end in groups, not scattered | `s7` at 0.5B: 3 runs within 0.008 of each other near 0.35, 5 spread over 0.204-0.300 | observed |
| on the counting task most pretrained runs answer one constant to every prompt | `s16`: at 0.5B 5 of 8 runs (6 and 5 of 8 in two earlier launches), at 7B 4 of 10; every anchor answers "1" | measured |
| which constant a run falls into is most of the 0.5B seed spread | `s16`: the mode label explains 94% of the score variance (99% and 81% in the earlier launches); 40% at 7B | measured |
| ~~at 0.5B the spread is mostly prompts flipping independently~~ | withdrawn: prompts sharing an answer correlate 0.90 across seeds and others -0.33; the per-prompt share was the residual of those blocks | negative |
| ~~the stored endpoint is locally flat, so the forecast sees nothing~~ | withdrawn: 16/16 from 16 samples allows p near 0.84, and the stored metric-gradient norm (0.45) is not near zero | negative |
| the 7B forecast is not reproducible from its own run | `s7`: two repeats from the same run give 0.00066 and 0.451, a factor of 681 | measured |
| ~~which mode a run lands in is decided early~~ | withdrawn: three reruns gave 0.88, 0.65, 0.44; interval [-0.19, +0.96]; +0.11 at 7B | negative |
| the sqrt(KL) route to outcome spread does not hold | fitted exponent +0.10, CI [-0.03, +0.45] against +0.5 | negative |
| at 7B under Adam the spread is 56% of the effect | `s7`: 10 runs, 0.203 -> 0.362, resolved seed sd 0.0892 [0.0398, 0.1194] | measured |
| ~~finite corpus raises G*~~ | withdrawn: the budget fixes R = PG, so the correction is a function of G; it lowers G* by sqrt((N-1)/N) | negative |
| the fixed-budget finite-corpus optimum is 1 + sqrt((N-1)tau_w/(N tau_b)) | `b1`: matches brute force over feasible G to 0.02 | exact |
| one prospective pretrained forecast is inside 1.5x | `s11` band 0.85: frozen 0.0205 against measured 0.0253, error 1.24x, 12 held-out seeds | measured |
| ~~more stochastic support gives a better forecast~~ | withdrawn: four pre-specified bands give 1.66, 2.13, 2.80, 1.24 with no trend | negative |
| ~~the overprediction is metric-gradient estimation bias~~ | withdrawn: cross-fitting two independent gradients moves 0.0604 to 0.0616 | negative |
| the diagnostic refuses the accurate run too | `s11` band 0.85 residual 8.0 against a threshold of 0.0300 | measured |
| a one-run residual ranks the settings where the forecast fails above chance | `p2`: AUROC 0.78 for errors past 1.3x on the 192-setting headline grid, 0.85 on the pool-16 replicate | measured |
| ~~refusing the worst fifth caps the error at 1.29x~~ | withdrawn: true on the first 96-setting grid only. On the 192-setting headline grid the worst setting sits at the 75th residual percentile and is accepted, so the worst error is unchanged (1.56x); the pool-16 replicate behaves the same (1.50x) | negative |
| the same threshold refuses both pretrained runs | `s9`: residual 2.0 at 0.5B and 160.5 at 7B against 0.0300 | measured |
| no pretrained regime the diagnostic accepts | not found; both real runs are refused | negative |
| the forecast is short by 3.1x to 17x at 7B under Adam | `s7`: 0.0207 against 0.0892, half-samples 0.0052/0.0287 | measured |
| ~~short by 38x at 7B~~ | withdrawn: that forecast carried no adjoint between updates, worth 16x in variance on the exact tier | negative |
| carrying Adam's state is what the lift buys, not the curvature | exact tier: 1.10x with and without the curvature blocks, 16.1x with neither | measured |
| the forecast needs its own error bar to be usable | `s7`: two estimates from one run 684x apart at 8 prompts, 2.4x at 32 | measured |
| the diagnostic costs 70% of running the seeds | `s7`: 181 min for one 32-prompt backward pass against 260 min for 10 seeds | measured |
| at 7B the spread is 84% of the effect | `s6`: 8 runs, 0.203 -> 0.313, spread 0.0925, evaluation part 0.0055 | measured |
| divergence does not accumulate at 7B either | `s6`: tail slope +0.09 against +1, over updates 10 to 20 | measured |
| the shortfall is not an estimation problem | `s7`: same initialisation, 8 vs 64 prompts moves the forecast 0.0025 -> 0.0026 | measured |

## Withdrawn or superseded

| former claim | what replaced it |
|---|---|
| `KL_inf = tau_c * D_noise` | the covariance recursion; the scalar law is off by 195x |
| outcome spread follows `sqrt(KL)` | Theorem 2, which propagates against the metric directly |
| `KL_inf ~ P^-0.45 D^0.35` at R^2 = 0.94 | exponents with intervals: -0.61 [-1.42, +0.24], +0.59 [+0.27, +1.04] |
| divergence is stationary | bounded over the horizons measured; the spectral radius exceeds one |

## Not claimed

* an error bar for a frontier-scale run: the largest model measured is 7B, on short-answer tasks
* that the forecast is usable on a real model as it stands: at a step size that produces learning
  the runs leave the linear regime, and at one small enough to stay in it the seed term is below
  what eight runs can resolve
* that the cheap estimator is safe over long runs: it is exact only where `Phi ~ I` along `grad M`
* that the bootstrap interval on the real-model spread has nominal coverage: simulated coverage is
  68-88% (see `experiments/s7_real_forecast/power.py`)
