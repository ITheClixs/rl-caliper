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
| forecast matches realised spread to 1.11x median, 1.58x worst | `p2_metric_forecast`, 192 settings, 48 seeds each, prospective | measured |
| inside the measured 95% interval in 162 of 192 | same | measured |
| holds across two policy shapes (1.11x vs 1.10x) | same, shape as an axis | measured |
| holds across the estimator family | `p2` re-run under `grpo_mean` (1.10x) and `grpo_std` (1.10x), 12 settings each | measured |
| the transport is worth ~2% at these run lengths | `p2` ablation: adjoint held at `grad M` | measured |
| the cheap form costs one extra pass per update | arithmetic on the algorithm | exact |

## The mechanism

| claim | evidence | strength |
|---|---|---|
| `S_{t+1} = A_t S_t A_t' + Q_t` | Theorem 1 | exact |
| propagated covariance predicts divergence to 9% median | `p1_propagation` against Monte Carlo | measured |
| accumulation is wrong by up to 3.9x | same | measured |
| a single timescale is wrong by two orders of magnitude | same | measured |
| the mean update is a gradient field, so `J` is symmetric | Proposition 3; measured asymmetry 3e-10 across three estimators | exact |
| `A_t` is not a contraction: radius > 1 throughout | `p6_spectrum`, three difficulty bands, 600 updates | measured |
| 31-51% of directions contract, share grows during the run | same | measured |
| the filtering requires a learning signal | `s8_null`: 0.14x with a verifier against 90.5x with a coin, same steps | measured |
| divergence does not accumulate | `s1`, `s5`: tail slopes -1.15 to +0.26 against +1 | measured |

## Where the uncertainty comes from

| claim | evidence | strength |
|---|---|---|
| rollout sampling dominates prompt selection | `p3_memory_sources`, exact decomposition of `Sigma_t` | exact |
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
| RLVR collapses sampling within a few updates on a real model | `s7`: live share to zero by update five, twice, at two step sizes | observed |
| a saturated metric cannot see a seed difference | `s6_saturation`: 0 of 32 prompts disagree, policies differ by up to 8e-2 | observed |
| the real-model directional derivative is noise-dominated | `s7`: cosine between independent estimates, median 0.07-0.5 | observed |
| the sqrt(KL) route to outcome spread does not hold | fitted exponent +0.10, CI [-0.03, +0.45] against +0.5 | negative |

## Withdrawn or superseded

| former claim | what replaced it |
|---|---|
| `KL_inf = tau_c * D_noise` | the covariance recursion; the scalar law is off by 195x |
| outcome spread follows `sqrt(KL)` | Theorem 2, which propagates against the metric directly |
| `KL_inf ~ P^-0.45 D^0.35` at R^2 = 0.94 | exponents with intervals: -0.61 [-1.42, +0.24], +0.59 [+0.27, +1.04] |
| divergence is stationary | bounded over the horizons measured; the spectral radius exceeds one |

## Not claimed

* an error bar for a frontier-scale run: the largest model measured is 7B, on short-answer tasks
* that the cheap estimator is safe over long runs: it is exact only where `Phi ~ I` along `grad M`
* that the bootstrap interval on the real-model spread has nominal coverage: simulated coverage is
  68-88% (see `experiments/s7_real_forecast/power.py`)
