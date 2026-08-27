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

`A_t` reweights a perturbation by the local curvature of the estimator potential rather than
passing it through. That reweighting is not a contraction (see section 3b), but it is enough to
stop noise accumulating. The covariance of a run about its mean satisfies

    S_{t+1} = A_t S_t A_t^T + Q_t,     Q_t = eta^2 Sigma_t / P,   S_0 = 0                (S2)

and unrolling,

    S_T = sum_t  Phi(T,t+1) Q_t Phi(T,t+1)^T,   Phi(T,s) = A_{T-1} ... A_s               (S3)

Two independent seeds have `Cov(theta^A - theta^B) = 2 S_T`, so their expected policy divergence is
`tr(F_T S_T)`. Per-update contributions `k_t = tr(F_T Phi Q_t Phi^T)` say where in training the
surviving noise was injected.

## 3. What replaced what

An earlier version of these notes compressed the spectrum of `A_t` into one effective eigenvalue and
claimed `KL_inf = tau_c * D_noise`. Measured against exact Monte Carlo over 54 settings, that scalar
law is wrong by a median factor of **58x** and up to 791x. The recursion (S3) is accurate to
**1.03x** at the median and 1.2x at worst.

Whether ignoring the filter matters depends on the prompt pool, and this is the more useful finding:

| pool diversity | propagated | accumulated | worst accumulated |
|---|---|---|---|
| 0.15 (similar prompts) | 1.04x | **2.06x** | 3.82x |
| 1.0 | 1.02x | 1.03x | 1.33x |
| 2.5 | 1.02x | 1.03x | 1.33x |

Filtering is done by the curvature of the estimator potential, and a pool of similar prompts is
where that curvature is consistent enough to matter. Where prompts disagree with each other the mean
field is nearly flat, `A_t ~ I`, and a run genuinely does accumulate its noise. On a narrow task
distribution the recursion is necessary; on a broad one accumulation is fine.

The failure is what (S3) predicts when a spectrum of decay rates is fitted with one exponential:
fast directions equilibrate within a few updates and slow ones set the final level, so a single
fitted timescale is pinned by the early rise.

## 3b. The operator is not a contraction

Measuring the spectrum of `A_t = I + eta J_t` along a 600-update mean trajectory, at three pool
difficulties:

* the spectral radius exceeds one everywhere, with `10^3(rho - 1)` running from about 20 early to 0.1 late;
* only 31%-51% of directions contract at all, and that share rises as the run proceeds;
* among the contracting directions the fastest has eigenvalue about 0.98 and the slowest is 1.0000
  to four places, giving decay timescales from roughly fifty updates to effectively unbounded.

Three consequences. Divergence stays bounded over the horizons measured because expansion is slow
relative to run length (1.02^25 is about 1.6), not because perturbations are destroyed. One
effective timescale cannot stand in for a spectrum this wide, which is the 195x failure in section
3. And the slow end of the spectrum has forgotten nothing after eighty updates, which is why the
per-update kernel has no horizon (see `forecast.md`).

The word used throughout is *filtered*, not *contracted*, and this is what it means.

## 4. The filtering comes from learning

Removing the learning signal while leaving the noise intact, by replacing every reward with an
independent coin flip, turns a divergence trace that ends at 0.14x of where it started into one
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

## 8. Adam, and the limits of lifting

Adam's state is `z = (theta, m, v)`, not theta alone: a gradient perturbation survives in the
moments after the step that produced it. `caliper/exact/lifted.py` takes the transfer operator
`dPhi/dz` and the injection `B Cov(g_hat) B^T` with `B = dPhi/dg_hat` by central differences on the
exact mean map, so nothing about the derivation is taken on faith.

Two structural facts. The map is singular at `v = 0`, because the derivative of
`1/(sqrt(v) + eps)` diverges there, so the moments must be warmed before the covariance is
propagated; every seed shares the
warmed state, so `S_0 = 0` still holds. And the difference steps must be taken relative to each
coordinate, since `v` is of order `g^2` and an absolute epsilon is meaningless there.

Measured against 48 independent Adam runs over 24 settings (`experiments/p5_adam_lift`):

| model of seed divergence | median | worst |
|---|---|---|
| lifted, `z = (theta, m, v)` | **1.10x** | 1.8x |
| second moment frozen | 1.21x | 1.6x |
| theta block only | 24.60x | 73.7x |
| accumulation | 24.22x | 73.8x |

Readings. The optimiser state is nearly the whole problem: a recursion on `theta` alone is barely
better than assuming noise accumulates unfiltered. Dropping the blocks that need the Jacobian of
the mean update costs nothing measurable, 1.10x either way, which is what makes the lift affordable
on a real model where that Jacobian cannot be estimated.

An earlier version of this table reported 2.61x for the lift and 8x on the saturating band, and
attributed the latter to Adam's normalisation being a nonlinearity a linear recursion cannot hold.
That was a bug: the mean map squared the mean gradient where the mean of the stochastic map carries
`E[g^2] = gbar^2 + diag Cov(g)`, discarding a variance term worth six to fifteen times the term
kept. The old reading is preserved here because the failure looked exactly like a property of
saturation, which is why it survived as long as it did.

With the mean map corrected the lift holds on every band, including the saturating one, at 1.10x
against a worst case of 1.8x. There is no longer a saturation condition attached to it here. What
does still bound the whole approach is `E[p(1-p)]`: when it reaches zero no noise is injected and
no metric difference is detectable, which is a statement about the run rather than about Adam.
