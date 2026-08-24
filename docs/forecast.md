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

* **48 settings**, enumerable policy, 64 seeds each, varying difficulty band, run length, prompt
  count and step size; realised spreads span a factor of 30.
* Median absolute error **1.08x**, worst **1.67x**; forecast inside the measured 95% interval in
  **36 of 48**. Coverage below the nominal rate is expected: with 64 seeds the measured interval is
  narrow enough that a systematic error of tens of percent registers as a miss.
* Accuracy is flat in run length (1.07x at T=10, 1.09x at T=25) and degrades slightly with step
  size (1.07x at eta=0.5, 1.13x at eta=1.5).

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
starting from a policy already trained for 1200 updates. Filtering lowers the level; it does not
localise where the spread was made.

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
