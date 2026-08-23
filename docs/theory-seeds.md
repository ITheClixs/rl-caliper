# Seed dynamics: why run-to-run divergence settles

Companion to `theory.md`, which supplies the noise decomposition this rests on.

## 1. The question

Two RLVR runs start from the same policy and differ only in the randomness of their rollouts. How
far apart do they end up, and does the gap grow with the length of the run?

The default mental model is a random walk: gradient noise perturbs each update, perturbations
accumulate, and the divergence between two runs grows linearly in the number of steps. Under that
model the only defence is a bigger batch, and a longer run is always less reproducible than a short
one.

## 2. The recursion

Let `D_t = theta_t^A - theta_t^B`. With updates `theta <- theta + eta g_hat`,

    D_{t+1} = D_t + eta [ g_bar(theta^A) - g_bar(theta^B) ] + eta [ xi^A_t - xi^B_t ]

where `xi` is the sampling noise of one batch, with `Cov(xi) = Sigma / P` from `theory.md` (7).
Linearising the mean gradient field around the shared trajectory, with `H := d g_bar / d theta`,

    D_{t+1} = (I + eta H) D_t + eta nu_t,     Cov(nu) = 2 Sigma / P                       (S1)

This is a linear stochastic recursion. Everything follows from the spectrum of `I + eta H`.

* Where the objective is locally concave, `H` is negative definite, `|I + eta H| < 1`, and the
  homogeneous part **contracts**. Divergence injected at step `t` decays.
* Where the objective is flat, `H ~ 0`, the recursion is a random walk and divergence accumulates.

RLVR is optimising a shared objective, and both runs are pulled toward the same attractor. The
question is whether the restoring term or the noise term dominates.

## 3. Stationary divergence

Take a single eigendirection with `eta H` eigenvalue `-eta lambda`, `lambda > 0`, and per-step
noise variance `2 eta^2 sigma^2 / P`. The stationary variance solves

    c = (1 - eta lambda)^2 c + 2 eta^2 sigma^2 / P

so `c (2 eta lambda - eta^2 lambda^2) = 2 eta^2 sigma^2 / P`, and for `eta lambda << 1`

    c = eta sigma^2 / (lambda P)                                                          (S2)

In function space the divergence between the two policies is `KL ~= (1/2) D^T F D`, so

    E[KL_inf] = (eta / (2 P)) * sum_i (F-weighted sigma_i^2 / lambda_i)
              = D_noise / (eta lambda_bar)                                                (S3)

using `D_noise = (1/2) eta^2 tr(F Sigma) / P`, the diffusive part of the drift from `theory.md`
(9), and `lambda_bar` for the curvature-weighted mean. Writing `tau_c := 1 / (eta lambda_bar)` for
the number of updates over which an injected divergence persists,

    E[KL_inf] = tau_c * D_noise                                                           (S4)

**The stationary divergence between two seeds is the diffusive drift injected per step, multiplied
by how long a perturbation survives.** Both factors are measurable: `D_noise` from the split
estimator at no extra cost, `tau_c` from a short fork of the run.

## 4. What this predicts

1. **Divergence does not accumulate.** `E[KL_t]` approaches `E[KL_inf]` and then stays there. A
   log-log slope of divergence against step count is `+1` under the random-walk model and `0` here.
2. **Run length is irrelevant.** Nothing in (S4) depends on `T`.
3. **Scaling.** Holding the drift target `D` fixed, `eta = sqrt(2D / (Gcal + N/P))`. Below the
   critical batch size the noise term dominates, `eta ~ sqrt(D P / N)`, and (S3) gives

       KL_inf  ∝  P^(-1/2) * D^(+1/2)                                                     (S5)

   Bigger batches and smaller steps both reduce it, and neither the number of updates nor the
   total movement of the policy appears.
4. **Outcome variance.** A displacement-linear reading of the objective gives a spread in final
   task performance proportional to `sqrt(KL_inf)`.

## 5. Where it breaks

(S1) linearises the gradient field and (S3) assumes `eta lambda << 1`. Both fail when the step
grows relative to the curvature. This is not hypothetical: a controller holding drift at a fixed
target raises `eta` as the signal `Gcal` decays, and once a task saturates `Gcal` collapses, `eta`
grows without bound, and the runs separate rather than settling. Measured runs cross from one
regime to the other, which bounds where the law should be applied and is a caution about drift
targeting rather than about the law.
