# Forecasting the spread of a reported number

Companion to `theory-seeds.md`, which supplies the covariance recursion this rests on.

## 1. The deliverable

A run reports a score `M(theta_T)`: pass rate, exact match, win rate, mean reward. The question is
what standard deviation that score would have under reseeding, answered from the run that was
already going to be trained.

## 2. The identity

Linearising the metric about the endpoint,

    Var_seed[M] = grad M^T S_T grad M                                              (F1)

and substituting the unrolled covariance `S_T = sum_t Phi Q_t Phi^T` turns the quadratic form into
a sum of scalars:

    b_T = grad M(theta_T),   b_s = A_s^T b_{s+1}
    Var[M] = sum_t b_{t+1}^T Q_t b_{t+1} = (eta^2 / P) sum_t Var(b_{t+1} . g_hat_t)   (F2)

`S_T` is never formed. Each term is the variance across prompts of one batch gradient projected
onto one direction.

## 3. Why the transpose is free

The per-prompt mean of a count-based estimator is `lambda(p_x, G) grad p_x`, and `lambda` depends on
theta only through `p_x`. So with `Lambda' = lambda`,

    g_bar = grad_theta E_x[Lambda(p_x)]                                            (F3)

a gradient field. Its Jacobian is a Hessian, hence symmetric, so `A_s^T b = A_s b = b + eta J_s b`
is a Hessian-vector product: two gradient evaluations by finite differences, or one double backward.
Measured asymmetry of `J` is 3e-10 across `rloo`, `grpo_mean` and `grpo_std`, which is the
finite-difference floor.

For RLOO, where `lambda = 1`, the potential is the pass rate itself. The mean update ascends exactly
the quantity being reported.

## 4. The protocol

Retrospective agreement is cheap and unconvincing. Everything reported here is prospective:

1. train one run, storing its states;
2. compute the forecast from those states alone;
3. record it;
4. only then train the remaining seeds;
5. compare.

## 5. What was measured

* **192 settings**, enumerable policies, 48 seeds each. Two policy shapes (vocab 3 over 3
  positions, vocab 4 over 2), two prompt-pool diversities, three difficulty bands, two run lengths,
  two prompt counts, two group sizes, two step sizes. Realised spreads span a factor of 47.
* Median absolute error **1.11x**, worst **1.58x**; forecast inside the measured 95% interval in
  **162 of 192**.
* The two policy shapes agree (1.11x against 1.10x), so the result is not a property of one
  tabular family.
* Coverage tracks the step size: 90/96 at eta=0.5 against 72/96 at eta=1.5. The median barely
  moves (1.09x against 1.13x); it is the tail that the linearisation costs.

## 5b. What the backward pass is worth

Holding `b` at `grad M(theta_T)` instead of carrying it back keeps the injected term and discards
the transport. Over the same 192 settings:

| | median error | worst |
|---|---|---|
| adjoint carried back | 1.11x | 1.58x |
| adjoint held at grad M | 1.11x | 1.75x |

The two forecasts agree to a median of 2% (extremes 0.72x to 1.05x) and land inside the measured
interval in the same 162 settings. Carrying it back buys the tail, and only where the run is long
and the step large: at T=25, eta=1.5 the worst case is 1.58x carried against 1.75x held.

Why: `theory-seeds.md` section 3b measures most of the spectrum of `A_t` within 1e-4 of one, and
`grad M` points into that part of it, so the transfer operator is close to the identity along the
one direction the forecast uses.

**Recommended default:** the cheap form,

    Var[M] ~= sum_t (eta_t^2 / P) Var_i( grad M . g_hat_{t,i} )

which is one extra pass per update and no Hessian-vector products. Use the full recursion when the
spectrum of `A_t` moves away from one, or when the run is long enough that a per-update discrepancy
of a percent compounds. The recursion is what tells you which regime you are in.

## 6. Attribution and memory

Restricting `Sigma_t` to `Sigma_b` or `Sigma_w` splits the forecast by origin; the shares are exact
and sum to one.

| G | rollout sampling | prompt selection |
|---|---|---|
| 2 | 93% | 7% |
| 4 | 82% | 18% |
| 8 | 65% | 35% |
| 16 | 46% | 54% |

Both terms carry `1/P`, so the number of prompts scales them equally and cannot change the mix.

The per-update kernel was checked for a memory horizon and does not have one: across 288 settings
the number of final updates holding 95% of the variance is a median of 1.00 of the run, against 0.95
for an exactly uniform kernel. Raising the step size sixteenfold does not move it, and neither does
starting from a policy already trained for 1200 updates.

Where the kernel is not flat it tilts toward the **start**. In 254 of 288 settings more of the final
variance was injected in the first half of the run than the second, median first-half share 55%.
With the transfer operator close to the identity along `grad M` (section 5b), `k_t ~ eta_t^2
Sigma_t / P`, and `Sigma_t` is largest early because `E[p(1-p)]` is largest before the policy has
learned. Early noise is both bigger and no more forgotten than late noise, so the beginning of a run
is where a larger batch buys the most.

## 6b. On a real model

`experiments/s7_real_forecast`, Qwen2.5-0.5B, twelve updates of plain gradient ascent, eight seeds,
on a task the base model passes at 0.16.

| quantity | value |
|---|---|
| measured spread across runs | 0.0694 |
| evaluation (binomial) part | 0.0077 |
| seed part, by subtraction | 0.0689, interval [0.0033, 0.0737] |
| forecast, cheap form | **0.0056** |
| forecast, cheap form, `Sigma` from 64 prompts | 0.0052 |
| forecast, with transport | 0.0225 |
| cosine between two `J b` estimates | 0.21 |
| share of the forecast made in the first half | 92% |

The cheap form is short by twelve. The transported version is four times larger, and that is noise
rather than curvature: the transport is worth 2% where it can be computed exactly, and here it is
built from vectors that barely correlate with an independent estimate of themselves.

Two comfortable explanations are ruled out. Estimating the injected term from 64 prompts instead of
8 moves the forecast by 7% and in the wrong direction, so it is not biased low by a small batch. The
live share measured on those 64 prompts still falls to zero by the fifth update, so the collapse is
not a small-sample artefact. The shortfall is not an estimation problem.

The reason is the outcome distribution. Three runs land at 0.350 and five at 0.217, within-group
spread under 0.01 against a between-group gap of 0.133. A linearised model predicts the spread of a
unimodal perturbation about one trajectory; a standard deviation is the wrong summary of this
outcome for anyone.

What survives is the mechanism. A run's first three batches correlate with its final score at 0.88
(last three: -0.24), and the kernel puts 92% of the variance in the first half of the run, 73% in
update four alone. The uncertainty is made early and then amplified, which is what a spectral radius
above one predicts. The amplification is not linear, and that is what the forecast misses.

## 7. Boundaries

* **Linearity.** The residual `||g_bar(B) - g_bar(A) - J(A)(B-A)|| / ||g_bar(B) - g_bar(A)||` rises
  from 0.028 to 0.165 as the step size grows eightfold, while the median forecast error stays near
  1.1x; the rank correlation between them is 0.17. The residual bounds the worst case (1.21x below
  5%, about 1.57x above) rather than setting the accuracy. Compute it anyway: it is one extra
  gradient evaluation on a checkpoint a run already has.
* **Adam.** Lifting to `z = (theta, m, v)` halves the error of ignoring the optimiser state but is
  unreliable where the pass rate saturates. See `theory-seeds.md`.
* **Real models.** Every ingredient becomes an estimate. `src/caliper/real/adjoint.py` implements the
  backward pass with common random numbers so the finite difference is a directional derivative
  rather than a difference of two noise draws.
