# Seed dynamics: how training noise reaches the end of a run

Companion to `theory.md`, which supplies the noise decomposition this rests on.

## 1. The question

Two RLVR runs start from the same policy and differ only in the randomness of their rollouts. How
far apart do they end up, and does the gap grow with the length of the run?

The default mental model is a random walk: gradient noise perturbs each update, perturbations
accumulate, and divergence grows linearly in the number of steps.

## 2. The recursion

Let `D_t = theta_t^A - theta_t^B`. With updates `theta <- theta + eta g_hat` and
`J_t = d g_bar / d theta`,

    D_{t+1} = A_t D_t + eta nu_t,      A_t = I + eta J_t,   Cov(nu) = 2 Sigma / P        (S1)

Where the objective is locally concave `A_t` contracts, so a perturbation injected at one update
decays over the ones that follow. The covariance of a run about its mean therefore satisfies

    S_{t+1} = A_t S_t A_t^T + Q_t,     Q_t = eta^2 Sigma_t / P,   S_0 = 0                (S2)

and unrolling,

    S_T = sum_t  Phi(T,t+1) Q_t Phi(T,t+1)^T,   Phi(T,s) = A_{T-1} ... A_s               (S3)

Two independent seeds have `Cov(theta^A - theta^B) = 2 S_T`, so their expected policy divergence is
`tr(F_T S_T)`. Per-update contributions `k_t = tr(F_T Phi Q_t Phi^T)` say where in training the
surviving noise was injected.

## 3. What replaced what

An earlier version of these notes compressed the spectrum of `A_t` into one effective eigenvalue and
claimed `KL_inf = tau_c * D_noise`. Measured against exact Monte Carlo in the enumerable policy, that
scalar law is wrong by a median factor of **195x**. The recursion (S3) is accurate to **1.09x** at
the median and 1.17x at worst over twelve settings; ignoring contraction entirely is wrong by up to
3.9x. The scalar law is kept only as a comparator.

The failure is what (S3) predicts when a spectrum of decay rates is fitted with one exponential:
fast directions equilibrate within a few updates and slow ones set the final level, so a single
fitted timescale is pinned by the early rise.

## 3b. The operator is not a contraction

Measuring the spectrum of `A_t = I + eta J_t` along a 600-update mean trajectory, at three pool
difficulties:

* the spectral radius exceeds one everywhere -- `10^3(rho - 1)` runs from about 20 early to 0.1 late;
* only 31%-51% of directions contract at all, and that share rises as the run proceeds;
* among the contracting directions the fastest has eigenvalue about 0.98 and the slowest is 1.0000
  to four places -- decay timescales from roughly 45 updates to effectively unbounded.

Three consequences. Divergence stays bounded over the horizons measured because expansion is slow
relative to run length (1.02^25 is about 1.6), not because perturbations are destroyed. One
effective timescale cannot stand in for a spectrum this wide, which is the 195x failure in section
3. And the slow end of the spectrum has forgotten nothing after eighty updates, which is why the
per-update kernel has no horizon (see `forecast.md`).

The word used throughout is *filtered*, not *contracted*, and this is what it means.

## 4. The filtering comes from learning

Removing the learning signal while leaving the noise intact -- every reward replaced by an
independent coin flip -- turns a divergence trace that ends at 0.14x of where it started into one
that grows 90.5x over eighty updates, a gap of 671x at the end. Same rollouts, same batches, same
step sizes. The learning condition's mean drift per update is *larger* (6.10e-3 against 5.17e-3),
so what brings those runs back together is not smaller steps. See `experiments/s8_null`.

The tabular policy at small drift is the other side of the same coin: its gradient field is nearly
flat, `J ~ 0`, and there divergence accumulates exactly as (S1) says it should.

## 5. What is not established

* ~~**No one-run error bar on a benchmark number.**~~ Superseded. Carrying (S3) to outcome spread
  via `sqrt(KL)` failed (fitted exponent +0.10, 95% CI [-0.03, +0.45] against a predicted +0.5) and
  the claim was withdrawn; propagating against the gradient of the target metric directly does work,
  and is written up in `forecast.md`.
* **Bounded, not stationary.** Tail slopes from -1.15 to +0.26 reject accumulation. They do not
  establish strict stationarity, since `J_t`, `Sigma_t` and `eta_t` all move during training, and
  the spectral radius above one means the bound is a statement about horizons, not a fixed point.
* **Scaling is weakly pinned.** Nine settings give exponent -0.61 in `P` (95% CI [-1.42, +0.24]) and
  +0.59 in `D` (95% CI [+0.27, +1.04]). Both intervals contain the predicted -1/2 and +1/2; the `P`
  interval also contains zero.
* **Transfer operators are exact only where the policy is enumerable.** On a real model `A_t` would
  need Jacobian-vector products along a stored trajectory.
* **Scale.** The largest model measured is 0.5B, six seeds, forty updates.

## 6. Where the linearisation breaks

(S1) needs the runs close enough for `g_bar` to be locally linear between them and `eta ||J|| << 1`.
A controller holding drift at a fixed target raises `eta` as the signal decays, and
`E[p(1-p)] -> 0` as a task is solved takes the signal with it. Runs cross out of the regime exactly
this way; the observable that announces it is the realised drift departing from its target.
