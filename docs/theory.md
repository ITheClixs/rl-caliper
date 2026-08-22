# Theory notes

Source of truth for the analytic results implemented in `caliper`. Every numbered result here has a
corresponding exactness test in `tests/` that checks it against brute-force enumeration.

## 1. Setup

Policy `pi_theta(y|x)` over token sequences, prompt distribution `D`, verifiable binary reward
`r(x,y) in {0,1}`. Objective

    J(theta) = E_{x~D} E_{y~pi} [ r(x,y) ] = E_x [ p(x) ],     p(x) := Pr[r = 1 | x]

Score `s(y) := grad_theta log pi(y|x)`, so `E_{y~pi}[s] = 0` and the per-prompt gradient is

    h(x) := E_y [ r s ] = p(x) * u1(x)

where `u1(x) := E[s | r = 1, x]`. Writing `u0 := E[s | r = 0, x]`, the zero-mean property of the score
gives the identity used throughout:

    p * u1 + (1 - p) * u0 = 0    =>    u0 = -(p / (1 - p)) * u1                        (1)

Let `S_r := Cov(s | r)` and `M_r := E[s s^T | r] = S_r + u_r u_r^T`.

**Everything below depends on the policy only through `(p, u1, S_0, S_1)` per prompt.** This is a
consequence of the reward being binary, and it is what makes an exact finite-`G` theory possible.

## 2. The count-based group estimator family

A batch is `P` prompts, `G` rollouts each. Write `k = sum_j r_j` for the number of successes in a group.
Consider any advantage of the form

    A_j = w(k, r_j)                                                                     (2)

i.e. the advantage of a rollout depends only on its own reward and on the group's success count. This
covers the estimators in use:

| name | `w(k, r)` |
|---|---|
| RLOO | `r - (k - r)/(G - 1)` |
| GRPO, mean baseline (Dr. GRPO) | `r - k/G` |
| GRPO, standardised | `(r - k/G) / sqrt((k/G)(1 - k/G))`, zero when `k in {0, G}` |
| pass-rate centred | `r - p_hat`, external `p_hat` |

The per-prompt estimator is `z := (1/G) sum_j w(k, r_j) s_j` and the batch estimator is
`g_hat = (1/P) sum_i z_i`.

Define, with `m ~ Binomial(G-1, p)` and `m' ~ Binomial(G-2, p)`:

    W_r  := E_m [ w(r + m, r) ]
    Q_r  := E_m [ w(r + m, r)^2 ]
    C_ab := E_m'[ w(a + b + m', a) * w(a + b + m', b) ]        (symmetric in a, b)      (3)

### Result 1 (exact expectation: the zoo is a difficulty reweighting)

    E[z | x] = lambda(p, G) * h(x),     lambda(p, G) = W_1(p, G) - W_0(p, G)            (4)

*Proof.* Condition on `r_j`; the remaining `G-1` rewards are iid `Bernoulli(p)` and independent of
`s_j` given `r_j`, so `E[w(k, r_j) s_j] = sum_r Pr[r] W_r u_r`. Summing over `j` and applying (1)
gives `E[z] = p (W_1 - W_0) u1 = lambda h`. QED

Consequences, all exact:

* RLOO: `W_1 = 1 - p`, `W_0 = -p`, so `lambda = 1`. Unbiased for every `p` and `G`.
* GRPO mean baseline: `lambda = (G - 1)/G`, independent of `p`. It is RLOO times a scalar.
* GRPO standardised: `lambda(p, G)` is a nontrivial function of `p`, computed by the binomial sum (3).

So for binary rewards **no member of this family is biased in direction**: each computes the true
per-prompt gradient `h(x)`, reweighted by a scalar `lambda(p(x), G)`. The family optimises

    J_lambda := gradient field  E_x [ lambda(p(x), G) h(x) ]                            (5)

and its members differ only through the difficulty weight `lambda`. Standardisation is therefore not a
variance-reduction device but a change of objective: it up-weights prompts whose pass rate is near 0 or 1
relative to prompts of intermediate difficulty.

### Result 2 (exact per-prompt covariance)

With `U := u1 u1^T`,

    Cov(z | x) = (1/G) [ p Q_1 S_1 + (1-p) Q_0 S_0 ]
               + c_U(p, G) * U                                                          (6)

    c_U = (1/G) ( p Q_1 + p^2 Q_0 / (1 - p) )
        + p^2 [ ((G-1)/G) (C_11 - 2 C_10 + C_00) - lambda^2 ]

*Proof sketch.* Split `E[z z^T]` into the `j = l` and `j != l` parts. The diagonal part contributes
`(1/G)[p Q_1 M_1 + (1-p) Q_0 M_0]`. For `j != l`, condition on `(r_j, r_l)` and the count `m'` among the
other `G-2` rollouts; the two scores are conditionally independent, giving
`((G-1)/G) sum_{a,b} Pr[a] Pr[b] C_ab u_a u_b^T`, which collapses to
`((G-1)/G) p^2 (C_11 - 2 C_10 + C_00) U` by (1). Subtract `E[z] E[z]^T = lambda^2 p^2 U` and substitute
`M_r = S_r + u_r u_r^T`. QED

## 3. Hierarchical noise decomposition

Prompts are iid, so with `g_bar := E_x[lambda h]`,

    Cov(g_hat) = (1/P) [ Sigma_b + Sigma_w ]                                            (7)
    Sigma_b := Cov_x( lambda(p(x),G) h(x) )      (between-prompt, does not shrink with G)
    Sigma_w := E_x[ Cov(z | x) ]                 (within-prompt, shrinks with G)

`Sigma_b` is the irreducible variance of prompt sampling; `Sigma_w` is the reducible variance of rollout
sampling. Increasing `G` attacks only the second term. This is the structural fact the rest of the paper
rests on, and it has no analogue in pretraining, where there is a single level of sampling.

### Result 3 (the reward histogram predicts the rollout-level noise)

Assume score isotropy across the two reward classes, `tr(S_0) ~= tr(S_1) ~= sigma_s^2`. Then for RLOO

    p Q_1 + (1-p) Q_0 = p (1-p)     (exactly, for every G)

so

    tr(Sigma_w) ~= (sigma_s^2 / G) * E_x[ p(x) (1 - p(x)) ]                             (8)

The rollout-level noise is, up to one scalar, the **mean Bernoulli variance of the pass-rate
distribution** — a quantity every RLVR trainer already logs for free. `sigma_s^2` is a slowly varying
property of the model, not of the data, so a single calibration measurement converts the reward
histogram into a noise estimate at zero cost. Isotropy is an assumption and is measured, not asserted,
in the exact testbed.

## 4. Progress, efficiency and the critical batch size

Second-order model of one gradient-ascent step of size `eta` with curvature `H`:

    E[dJ] = eta |g_bar|^2 - (1/2) eta^2 ( g_bar^T H g_bar + tr(H Cov(g_hat)) )          (9)

    eta*   = |g_bar|^2 / ( g_bar^T H g_bar + tr(H Cov(g_hat)) )
    E[dJ]* = (1/2) |g_bar|^4 / ( g_bar^T H g_bar + tr(H Cov(g_hat)) )                   (10)

Substituting (7) and writing `Gcal := g_bar^T H g_bar`:

    E[dJ]* = dJ_max / (1 + Bcrit(G) / P),      dJ_max := (1/2) |g_bar|^4 / Gcal         (11)

    Bcrit(G) := [ tr(H Sigma_b) + tr(H Sigma_w(G)) ] / Gcal        (units: prompts)     (12)

`Bcrit` is the RL analogue of the gradient noise scale, and it is a *two-level* quantity: a floor set by
prompt diversity plus a term that `G` divides down. The steps/examples Pareto frontier retains the
familiar form `S/S_min = 1 + Bcrit/P`, `E/E_min = 1 + P/Bcrit`.

### Result 4 (the statistically optimal group size is the smallest one)

At a fixed number of rollouts per step `R = P G`, using `tr(H Sigma_w(G)) = tau_w / G + O(1/G^2)`,

    progress per rollout  ∝  |g_bar(G)|^4 / [ Gcal(G) R + G tau_b + tau_w ]             (13)

For any estimator with `lambda` independent of `p` (RLOO, GRPO mean baseline) the numerator and
`tau_b` do not depend on `G`, so (13) is strictly decreasing in `G`: **per rollout spent, the smallest
admissible group is optimal.** For standardised GRPO, `lambda(p, G)` varies with `G` and the optimum is
interior. This is a sharp prediction and it is consistent with the reported but unexplained observation
that very small groups converge faster per sample than large ones.

### Result 5 (compute-optimal group size on real hardware)

Rollout cost is not uniform in `G`. When the group shares the prompt prefix, prefill is paid once per
prompt and decode once per rollout:

    cost per step  =  P ( c_pre + G c_dec )                                             (14)

Maximising progress per unit *cost* rather than per rollout gives an interior optimum

    G*  =  argmax_G  |g_bar(G)|^4 / [ (c_pre + G c_dec) ( Gcal(G) + (tau_b G + tau_w)/R ) ]   (15)

so the compute-optimal group size is set jointly by the pass-rate distribution and by the measured
prefill/decode cost ratio of the serving stack. `c_pre` and `c_dec` are measured, not assumed.

### Result 6 (schedule)

`E_x[p(1-p)]` falls as a policy improves on a fixed prompt set, so by (8) `tau_w` falls while `tau_b`
does not. By (13) and (15) the optimal `G` therefore *decreases* over a run and the optimal `P`
increases at fixed budget. Fixed-`(P, G)` recipes are mis-allocated at one end of training or the other,
by an amount the theory quantifies.

## 5. Drift and the step size

The realised behavioural drift of one update, in nats per token, is

    D := E_x[ KL( pi_{t+1}(.|x) || pi_t(.|x) ) ] ~= (1/2) eta^2 * E[ g_hat^T F g_hat ]
       = (1/2) eta^2 ( g_bar^T F g_bar + tr(F Cov(g_hat)) )                             (16)

with `F` the Fisher information of the policy. Drift splits into a signal part and a diffusive part that
buys nothing. Their ratio is

    rho := g_bar^T F g_bar / ( g_bar^T F g_bar + tr(F Cov(g_hat)) ) = 1 / (1 + Bcrit^F / P)   (17)

### Result 7 (drift decomposition identity)

Comparing (11) and (17): when curvature and Fisher agree (the Gauss-Newton correspondence that holds for
policy gradients near on-policy), **the batch-size efficiency equals the signal fraction of the measured
drift**. Efficiency is therefore observable from the KL a trainer already logs, with no extra gradient
work: a run whose logged drift is mostly diffusive is running below its critical batch size, and the
deficit is quantified without a sweep.

## 6. Estimation from a single batch

Split the batch two ways and use, for any PSD `M`, the unbiased pair

    g_bar^T M g_bar  =  E[ g_A^T M g_B ]
    tr(M Cov)        =  (1/4) E[ (g_A - g_B)^T M (g_A - g_B) ]                          (18)

for independent half-batch estimates `g_A, g_B`.

* Split by **prompt** (disjoint halves of the `P` prompts): yields `tr(M(Sigma_b + Sigma_w))/P`.
* Split by **rollout** (within each prompt, half the group against the other half): the prompt-level
  component is shared and cancels, yielding `tr(M Sigma_w)` alone.
* Difference: `tr(M Sigma_b)`.

Both splits reuse gradients that are computed anyway, so the full hierarchical decomposition costs no
extra rollouts and no extra backward passes beyond accumulating into two buffers instead of one.
`M = I` gives the practical noise scale; `M = F` is obtained from measured KL rather than from
materialising `F`.

## 7. Scale transfer (conjecture under test)

`Bcrit` is measured in prompts, `D` in nats per token: both are function-space quantities, invariant to
the parameterisation. The conjecture is that a recipe specified as a target drift `D*` and a target
efficiency `rho*` transfers across model width, depth and task, while the raw `(eta, clip, KL
coefficient, P, G)` that realise them do not. The transfer experiment fits the map on a subset of widths
and tasks and tests it on held-out ones; a negative result here is reported as such and does not affect
Results 1-7.
