# Tier A results: the exactly enumerable testbed

All numbers below come from policies small enough to enumerate, so the objective, the Fisher
information, the drift and the noise decomposition are computed rather than estimated. Sampling
enters only where a real trainer would have it.

## What was checked

| Claim | Check | Outcome |
|---|---|---|
| Result 1, exact expectation `E[z] = lambda(p,G) h` | brute-force enumeration over all `|Y|^G` group outcomes | agreement to 1e-12 for all five estimators, G = 2,3,4 |
| Result 2, exact covariance | same | agreement to 1e-12 |
| RLOO is unbiased, mean-baseline GRPO is RLOO times (G-1)/G | closed form | exact, for every p and G |
| Result 3, Bernoulli identity | closed form | `p Q_1 + (1-p) Q_0 = p(1-p) G/(G-1)` exactly |
| `(G-1) tr(Sigma_w)` is G-independent | sweep G = 2..32 | varies by 6% over a 16x range in G |
| Constant rescaling of advantages changes nothing | RLOO vs mean-baseline GRPO | identical critical batch size and identical progress; step size absorbs the factor exactly |
| Drift model, `D = (1/2) eta^2 E[g^T F g]` | exact KL after a real update | within 3% over three orders of magnitude in eta |
| Split estimator recovers the noise terms | 32 pools spanning `G* = 3.9` to `31.6` | median error in `G*` of 3.2%, worst case 16.6% |
| Result 4b, `G* = 1 + sqrt(tau_w/tau_b)` | training at every split of a fixed rollout budget | see below |

## The curvature model does not hold; the drift model does

The obvious route to a critical batch size is a second-order model of the objective. It fails here:
`J = E[r]` is not a log-likelihood, its Hessian is not the negative Fisher, and a log-linear policy
keeps improving far past the step size a quadratic model calls optimal. Measured against exactly
computed improvement, the curvature model underestimates the useful step size by more than 4x.

The drift model is accurate to 3% over the same range. The theory was rebuilt on it: progress at a
fixed drift budget goes as `sqrt(rho)`, and the batch-composition question becomes a question about
`rho` alone.

## Result 4b end to end

For each prompt pool the predicted efficiency curve `rho(G)` is computed with no free parameters and
compared against the improvement obtained by training at every split of a fixed rollout budget,
holding total rollouts, total drift and step count fixed.

| predicted spread `max rho / min rho` | cells | median R^2 of observed gain against `sqrt(rho)` | median Spearman |
|---|---|---|---|
| 1.00 - 1.15 | 8 | 0.55 | +0.62 |
| 1.15 - 1.35 | 12 | 0.78 | +0.76 |
| 1.35 - 2.00 | 6 | 0.97 | +0.96 |
| 2.00 - 8.00 | 6 | 0.98 | +0.98 |

The prediction is confirmed wherever it discriminates. Where the theory predicts that the split does
not matter -- pools of near-identical prompts, where `tau_b` is small and `rho(G)` is nearly flat --
the measured curves are flat to within their standard errors, which is the same prediction.

Sensitivity to the split is itself predicted: it grows as the rollouts per step fall, because the
G-dependent term enters as `G Bcrit(G) / R`. At R = 240 the predicted spread reaches 7.5x on
heterogeneous pools; at R = 48 on homogeneous pools it is 1.05x and nothing is measurable.
