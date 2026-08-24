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

## 4. The contraction comes from learning

Removing the learning signal while leaving the noise intact -- every reward replaced by an
independent coin flip -- turns a flat divergence trace into one that grows 10.7x over eighty
updates. Same batches, same step sizes, same gradient magnitudes. This is what separates the result
from a claim that seeds happen to end up nearby.

The tabular policy at small drift is the other side of the same coin: its gradient field is nearly
flat, `J ~ 0`, and there divergence accumulates exactly as (S1) says it should.

## 5. What is not established

* **No one-run error bar on a benchmark number.** (S3) predicts policy divergence. Carrying that to
  outcome spread via `sqrt(KL)` failed: fitted exponent +0.10, 95% CI [-0.03, +0.45] against a
  predicted +0.5, interval containing zero. The claim is withdrawn. The route that should work is to
  propagate the covariance against the gradient of the target metric directly.
* **Bounded, not stationary.** Tail slopes from -1.15 to +0.26 reject accumulation. They do not
  establish strict stationarity, since `J_t`, `Sigma_t` and `eta_t` all move during training.
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
