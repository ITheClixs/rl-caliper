# caliper

Forecasting how far the result of an RL post-training run would move under reseeding, from the run
itself.

A reported score is the output of a stochastic procedure, and the only instrument the field has for
its spread is replication. This repository implements a cheaper one. Gradient noise injected at one
update is filtered by the updates that follow rather than accumulated, so the covariance of a run
about its mean obeys a recursion; carrying the gradient of the reported metric backwards along a
single stored trajectory turns that recursion into an error bar, as a sum of scalars with no
covariance matrix formed.

## What is here

```
src/caliper/
  objectives/   the estimator family as a (G+1) x 2 weight table
  exact/        enumerable policies: exact scores, Fisher, moments, and a trainer
  estimators/   the K x 2 split estimator and its sampling harness
  population/   transformers trained as a batched population, and the RLVR trainer
  control/      step size as a feedback loop on realised drift
  real/         the same measurement on a pretrained model through MLX
  analysis/     peak fitting and interval estimation
  runtime/      run manifests
experiments/    one directory per experiment, config-driven
analysis/       figure and table generation for the paper
paper/          LaTeX sources
docs/           theory notes and result notes
```

## What it does

`caliper.forecast` is framework-agnostic. Supply one object per update that can answer two
questions about the trajectory you stored, and it accumulates the error bar:

```python
from caliper.forecast import run_backward

class MyUpdate:                  # step_size, n_prompts as attributes
    def projected_variance(self, b):   # Var over prompts of b . per-prompt gradient
        ...
    def jacobian_vector(self, b):      # J b, one Hessian-vector product
        ...

result = run_backward(metric_gradient, updates)
result.std                # forecast s.d. of the reported score across seeds
result.interval(0.784)    # a normal interval around the number you are reporting
result.kernel             # each update's share of the variance
result.memory()           # how many final updates hold 95% of it
```

Two implementations of that protocol ship here: `caliper.exact.adjoint` for enumerable policies,
where every term is exact, and `caliper.real.adjoint` for a pretrained model through MLX -- two
gradient evaluations per stored update, with common random numbers so the finite difference is a
directional derivative rather than a difference of two noise draws.

## Results in one paragraph

**The forecast.** Tested prospectively -- one run trained and frozen, the prediction recorded, and
only then 63 more seeds trained -- the forecast of the across-seed standard deviation of the
reported pass rate has a median absolute error of `1.11x` over 192 settings -- two policy shapes, two prompt-pool
diversities, and a 47x range of spreads -- a worst case of `1.58x`, and lands inside the measured
95% interval in 162 of them.

**Why it works.** Run-to-run variance in RLVR does not accumulate. Across nine settings the log-log
slope of policy divergence against update count runs from `-1.15` to `+0.26`, where a random walk
requires `+1`, and the same holds on Qwen2.5-0.5B. The cause is the learning signal: replacing every
reward with an independent coin flip, leaving the rollouts, batches and step sizes otherwise
untouched, turns a trace that ends at `0.14x` of where it started into one that grows `90.5x` over
eighty updates -- a gap of `671x`. The learning condition moves *further* per update, so what brings
those runs back together is not smaller steps.

**Why it is cheap.** The mean update of a count-based estimator is the gradient of a scalar
potential `E_x[Lambda(p_x)]`, so its Jacobian is symmetric (measured asymmetry 3e-10) and the
backward pass is a Hessian-vector product. For RLOO the potential is the pass rate itself.

**Where the spread comes from.** Restricting the injected covariance to one term attributes the
forecast by origin. Rollout sampling dominates prompt selection at every group size measured --
93% at `G=2`, 65% at `G=8`, 46% at `G=16` -- and the prompt count scales both equally, so it cannot
change the mix.

The level is predicted by propagating the update covariance,
`S_{t+1} = A_t S_t A_t' + eta^2 Sigma_t / P` with `A_t = I + eta J_t`. Against exact Monte Carlo
over twelve settings this is accurate to 1.09x at the median; ignoring contraction is wrong by up
to 3.9x, and compressing the spectrum of `A_t` into a single timescale by up to 844x. Scaling
exponents are `-0.61` in `P` (95% CI `[-1.42, +0.24]`) and `+0.59` in drift (`[+0.27, +1.04]`),
both compatible with the predicted `-1/2` and `+1/2` and neither tightly pinned.

Underneath sits an exact finite-`G` covariance for the advantage estimators in current use, which
makes the injected noise measurable, shows those estimators differ only by a weight on prompt
difficulty, and fixes the optimal group size at `G* = 1 + sqrt(tau_w/tau_b)`.

What is *not* established: an earlier route from policy divergence to outcome spread via `sqrt(KL)`
fails (fitted exponent `+0.10`, CI `[-0.03, +0.45]` against a predicted `+0.5`) and is withdrawn --
Theorem 2 replaces it. We looked for a short memory horizon and did not find one: 95% of the
forecast variance spans a median of 1.00 of the run, against 0.95 for a uniform kernel. Lifting the
recursion to Adam's optimiser state halves the error of ignoring that state but is unreliable where
the pass rate saturates. See `docs/forecast.md` and `docs/theory-seeds.md` for the full list of
boundaries.

## Running it

```
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest                                  # includes the exactness tests
uv run python experiments/s1_contraction/run.py   # seed divergence over training
uv run python experiments/p1_propagation/run.py    # covariance recursion vs Monte Carlo
uv run python experiments/p2_metric_forecast/run.py  # the forecast, tested prospectively
uv run python experiments/p3_memory_sources/run.py   # attribution and memory horizon
uv run python experiments/a4_law/run.py           # enumerable policies
uv run python experiments/b1_group_size/run.py    # transformers from scratch
uv run python analysis/figures.py                 # regenerate every figure
```

The real-model experiments additionally need `uv pip install -e ".[mlx]"` and run on Apple Silicon.

Every run writes a manifest recording its configuration, the commit it ran at, and the platform.
No number in the paper is transcribed by hand; figures and tables are generated from those records.
