# caliper

Measurement tools for the batch-size and step-size scaling of reinforcement learning post-training
of language models.

Two RL post-training runs that start from the same policy and differ only in the randomness of
their rollouts do not drift apart. Gradient noise injected at one update is filtered by the
updates that follow rather than accumulated, so divergence stays bounded over a run. This
repository holds the measurement, the covariance recursion that predicts it, and the noise
decomposition both rest on.

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

## Results in one paragraph

Run-to-run variance in RLVR does not accumulate. Across nine settings the log-log slope of policy
divergence against update count runs from `-1.15` to `+0.26`, where a random walk requires `+1`,
and the same holds on Qwen2.5-0.5B. The cause is the learning signal: replacing every reward with
an independent coin flip, leaving the noise otherwise untouched, turns a flat trace into one that
grows 10.7x over eighty updates.

The level is predicted by propagating the update covariance,
`S_{t+1} = A_t S_t A_t' + eta^2 Sigma_t / P` with `A_t = I + eta J_t`. Against exact Monte Carlo
over twelve settings this is accurate to 1.09x at the median; ignoring contraction is wrong by up
to 3.9x, and compressing the spectrum of `A_t` into a single timescale by up to 844x. Scaling
exponents are `-0.61` in `P` (95% CI `[-1.42, +0.24]`) and `+0.59` in drift (`[+0.27, +1.04]`),
both compatible with the predicted `-1/2` and `+1/2` and neither tightly pinned.

Underneath sits an exact finite-`G` covariance for the advantage estimators in current use, which
makes the injected noise measurable, shows those estimators differ only by a weight on prompt
difficulty, and fixes the optimal group size at `G* = 1 + sqrt(tau_w/tau_b)`.

What is *not* established: carrying the covariance to spread in a benchmark number via `sqrt(KL)`
fails (fitted exponent `+0.10`, CI `[-0.03, +0.45]` against a predicted `+0.5`), so the paper
withdraws that claim. See `docs/theory-seeds.md` for the full list of boundaries.

## Running it

```
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest                                  # includes the exactness tests
uv run python experiments/s1_contraction/run.py   # seed divergence over training
uv run python experiments/p1_propagation/run.py   # covariance recursion vs Monte Carlo
uv run python experiments/a4_law/run.py           # enumerable policies
uv run python experiments/b1_group_size/run.py    # transformers from scratch
uv run python analysis/figures.py                 # regenerate every figure
```

The real-model experiments additionally need `uv pip install -e ".[mlx]"` and run on Apple Silicon.

Every run writes a manifest recording its configuration, the commit it ran at, and the platform.
No number in the paper is transcribed by hand; figures and tables are generated from those records.
