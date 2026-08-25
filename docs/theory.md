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
sampling.

**Finite corpora.** Real runs draw prompts without replacement from a corpus of `N`, not i.i.d. from
an infinite distribution. The standard finite-population correction applies to the prompt level only:

    Cov(g_hat) = (1/P) [ (1 - f) Sigma_b + Sigma_w ],      f := (P - 1) / (N - 1)          (7b)

When the batch covers the whole corpus, `f = 1` and prompt sampling contributes no variance at all.
Everything downstream inherits the factor: `Bcrit` uses `(1-f) tau_b`, and

    G* = 1 + sqrt( tau_w / ((1 - f) tau_b) )                                              (14b)

so the optimal group size *grows* as the corpus shrinks toward the batch size. This is a knob on
`G*` that does not require changing the task, and it is the one the transformer experiments use.
It also explains why the same estimator must be corrected when it is run: the different-block inner
product has expectation `|g_bar|^2 - tau_b/(N-1)` rather than `|g_bar|^2`. Increasing `G` attacks only the second term. This is the structural fact the rest of the paper
rests on, and it has no analogue in pretraining, where there is a single level of sampling.

### Result 3 (the reward histogram predicts the rollout-level noise)

Assume score isotropy across the two reward classes, `tr(S_0) ~= tr(S_1) ~= sigma_s^2`. Then for RLOO

    p Q_1 + (1-p) Q_0 = p (1-p) * G / (G - 1)     (exactly, for every G)

so, dividing by the G in (6),

    tr(Sigma_w) ~= sigma_s^2 * E_x[ p(x) (1 - p(x)) ] / (G - 1)                         (8)

The rollout-level noise is, up to one scalar, the **mean Bernoulli variance of the pass-rate
distribution** — a quantity every RLVR trainer already logs for free. `sigma_s^2` is a slowly varying
property of the model, not of the data, so a single calibration measurement converts the reward
histogram into a noise estimate at zero cost. Isotropy is an assumption and is measured, not asserted,
in the exact testbed.

## 4. Drift, efficiency and the critical batch size

A curvature model of the objective is tempting but wrong here: `J = E[r]` is not a log-likelihood,
its Hessian is not the negative Fisher, and along a policy-gradient direction a log-linear policy
keeps improving far past the point a quadratic model predicts. This was measured, not assumed
(experiment A2). What *is* accurate is the model of how far the policy moves.

### The drift model

One update of size `eta` moves the policy by

    D(eta) := E_x[ KL( pi_{t+1}(.|x) || pi_t(.|x) ) ]
            = (1/2) eta^2 E[ g_hat^T F g_hat ] + O(eta^3)
            = (1/2) eta^2 ( Gcal + N / P ),                                              (9)

    Gcal := g_bar^T F g_bar,        N := tr(F Sigma_b) + tr(F Sigma_w(G))

with `F` the Fisher information of the policy. In the exactly enumerable testbed this predicts the
true KL to within 3% over three orders of magnitude in `eta`.

Drift splits into a **signal** part `(1/2) eta^2 Gcal`, which moves the policy along the mean update,
and a **diffusive** part `(1/2) eta^2 N / P`, which is a random walk in function space. Their ratio is
the efficiency

    rho := Gcal / (Gcal + N/P) = 1 / (1 + Bcrit / P),      Bcrit := N / Gcal             (10)

`Bcrit` is measured in prompts and is the RL analogue of the gradient noise scale. It is a *two-level*
quantity: a floor set by prompt diversity plus a term that `G` divides down.

### Result 4a (progress at a fixed drift budget)

To first order the expected improvement of an update is `eta * a` with `a := grad J^T g_bar`. Solving
(9) for `eta` at a given drift `D`,

    E[dJ] = a * sqrt( 2D / (Gcal + N/P) ) = sqrt(2D) * (a / sqrt(Gcal)) * sqrt(rho)      (11)

**Progress at a fixed drift budget scales as the square root of the efficiency.** The estimator enters
only through `a / sqrt(Gcal)`, the alignment of its mean with the true gradient measured in Fisher
units; the batch composition enters only through `rho`.

Because KL is quadratic in the step, a fixed *total* drift `D_tot` spread over `N` steps buys path
length `sqrt(2 N D_tot / (Gcal + N/P))`: many small steps traverse further than one large step of the
same total drift. Total progress over a run is therefore

    total dJ  ∝  sqrt( N * rho ),          N = number of updates                          (12)

### Result 4b (closed-form optimal group size)

Write `tau_b := tr(F Sigma_b)` and `tau_w := (G-1) tr(F Sigma_w(G))`, the latter `G`-independent to
leading order by (8), verified to 6% over G = 2..32 in the exact testbed. Hardware fixes the
rollouts available per step, `R = P G`; the question is how to split them. Maximising (12) means
maximising `N rho = R_tot / (G (P + Bcrit(G)))`, i.e. minimising

    G * ( P + Bcrit(G) ),     Bcrit(G) = ( tau_b + tau_w/(G-1) ) / Gcal                  (13)

At fixed `R = P G` this is convex in `G` with the interior minimum

    G* = 1 + sqrt( tau_w / tau_b )                                                       (14)

**The optimal group size is one plus the square root of the ratio of within-prompt to
between-prompt gradient noise**, independent of `R`. Both traces are measurable from a single batch
by the two-split estimator of Section 6, so `G*` is an observable, not a hyperparameter. Holding `P`
fixed rather than `R` gives `G* = 1 + sqrt(tau_w / (P Gcal + tau_b))`, which recovers (14) when the
run is below its critical batch size. Estimators whose `lambda` depends on `p` also shift `G*`
through `a/sqrt(Gcal)`; that contribution is computed exactly rather than approximated.

### Result 5 (compute-optimal group size on real hardware)

Rollout cost is not uniform in `G`. When a group shares the prompt prefix, prefill is paid once per
prompt and decode once per response, so a step costs `P (c_pre + G c_dec)`. Writing
`alpha = c_pre/c_dec`, the objective to minimise at fixed `R = P G` is

    f(G) = (alpha + G) ( R/G + beta_b + beta_w/(G-1) )                                   (15)

`f` is strictly convex on `G > 1` (its second derivative is
`2 alpha R / G^3 + 2 (1+alpha) beta_w / (G-1)^3 > 0`), `f'` runs from `-inf` at `G -> 1` to
`beta_b > 0` at infinity, so there is a unique interior minimum, at the root of

    beta_b = alpha R / G^2 + (1 + alpha) beta_w / (G-1)^2                                 (16)

Two consequences matter in practice.

* With `alpha = 0` the first term vanishes and (16) collapses to `G* = 1 + sqrt(tau_w/tau_b)`,
  exactly, with `R` dropping out. This is Result 4b.
* With `alpha > 0`, `G*` is strictly increasing in both `alpha` and `R`. The convenient form
  `G* = 1 + sqrt((1+alpha) tau_w/tau_b)` is only the `alpha R / G^2 -> 0` limit and understates the
  optimum badly at large `R`: at the noise ratio measured on our transformers it gives 7.6 where
  the true optimum is 51.0 (`alpha = 10.9`, `R = 8192`).

An earlier version of these notes stated the small-batch form as if it held generally. It does not;
`experiments/t1_optimum` checks the corrected statement against direct numerical minimisation and
agrees to one part in 10^7.

### Result 6 (schedule)

`E_x[p(1-p)]` falls as a policy improves on a fixed prompt set, so by (8) `tau_w` falls while `tau_b`
does not. By (14) the optimal `G` therefore *decreases* over a run, as the square root of a quantity
read straight off the reward histogram, and the optimal `P` rises at fixed budget.

## 5. The step size is a drift target

Equation (9) inverts to give the step size that realises a chosen drift:

    eta(D) = sqrt( 2D / (Gcal + N/P) )                                                   (17)

so a run can be specified by a drift target `D*` instead of a learning rate. Learning rate, clip
range and KL coefficient are three knobs acting on the single quantity `D`, which is why they
interact and why none of them transfers between scales or tasks. `D` is in nats per prompt and is
a property of the policy's behaviour, not of its parameterisation, which is the basis of the
transfer conjecture in Section 7.

## 6. Estimation from a single batch

Index the microbatches by prompt block `a = 1..K` and rollout sub-group `b = 1,2`, and accumulate

    g[a][b] = mean gradient over prompt block a using sub-group b

Blocks hold disjoint prompts and, given a prompt, the sub-groups are independent, so

    signal = mean over a != a' of  g[a][0] . g[a'][1]
    tau_b  = (P/K) * ( mean over a of g[a][0] . g[a][1]  -  signal )
    tau_w  = (P/(2K)) * mean over a of | g[a][0] - g[a][1] |^2

Every term is an inner product of independent quantities rather than a difference of two large
variances. Raising `K` averages the block statistics over `K(K-1)` pairs instead of one and shrinks
the `P/K` multiplier on the noisiest term, so more microbatches means a better estimate at no extra
cost. `M = I` gives the practical form; `M = F` comes from measured KL.

Three biases have to be handled, all found by checking against exactly known values:

1. Estimating `tau_b` as (total variance - within-prompt variance) is badly conditioned; it returns
   a negative `tau_b` on homogeneous pools and compresses the answer elsewhere.
2. A prompt sampled twice in one batch lands in two blocks, so the different-block product picks up
   a same-prompt term and `tau_b` is biased down. Draw distinct prompts.
3. Drawing without replacement from a corpus of `N` makes blocks anticorrelated; the exact
   correction `+ tau_b/(N-1)` restores the signal estimate. See (7b).

With those in place the estimator recovers `G*` to a median error of 1.3% over pools spanning a
factor of 26 in the noise ratio.

## 7. Scale transfer (conjecture under test)

`Bcrit` is measured in prompts, `D` in nats per token: both are function-space quantities, invariant to
the parameterisation. The conjecture is that a recipe specified as a target drift `D*` and a target
efficiency `rho*` transfers across model width, depth and task, while the raw `(eta, clip, KL
coefficient, P, G)` that realise them do not. The transfer experiment fits the map on a subset of widths
and tasks and tests it on held-out ones; a negative result here is reported as such and does not affect
Results 1-7.

## The estimator potential

Result 1 says the per-prompt mean of a count-based estimator is `lambda(p_x, G) grad p_x`, and
`lambda` depends on theta only through `p_x`. Define `Lambda` with `Lambda' = lambda`. Then

    g_bar(theta) = grad_theta E_x[ Lambda(p_x) ] =: grad Psi(theta)                    (P1)

so the mean update is a gradient field, its Jacobian `grad^2 Psi` is symmetric, and the transfer
operator `A = I + eta grad^2 Psi` is self-adjoint. Consequences:

* the backward pass of `forecast.md` needs no transpose, since a Hessian-vector product suffices, which
  finite differences give from two gradient evaluations;
* `Psi` is not the objective `J` unless `lambda == 1`; the gap is exactly the difficulty
  reweighting of Result 1;
* for RLOO, `lambda == 1` and `Psi == J`: the mean update ascends the pass rate itself.

Measured asymmetry of the Jacobian is 3e-10 across `rloo`, `grpo_mean` and `grpo_std`, which is the
finite-difference floor. `tests/test_adjoint.py` checks it.

Note that `Psi` being a potential does not make `A` a contraction. Its spectral radius exceeds one
throughout a run; see `theory-seeds.md`.
