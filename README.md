# caliper

Measurement tools for the batch-size and step-size scaling of reinforcement learning post-training
of language models.

Two RL post-training runs that start from the same policy and differ only in the randomness of
their rollouts do not drift apart. Their divergence rises for a few tens of updates, settles, and
stays settled. This repository holds the measurement, the theory that predicts its level, and the
noise decomposition both rest on.

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
and the same holds on Qwen2.5-0.5B. Gradient noise pushes two runs apart and the curvature of the
shared objective pulls them back; the fixed point scales as `P^-0.45 D^0.35` against a predicted
`P^-1/2 D^1/2`. Training longer therefore costs nothing in reproducibility and batch composition is
the dominant lever. Underneath sits an exact finite-`G` covariance for the advantage estimators in
current use, which makes the injected noise measurable, and which separately shows those estimators
differ only by a weight on prompt difficulty and fixes the optimal group size at
`G* = 1 + sqrt(tau_w/tau_b)`.

## Running it

```
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest                                  # includes the exactness tests
uv run python experiments/s1_contraction/run.py   # seed divergence over training
uv run python experiments/a4_law/run.py           # enumerable policies
uv run python experiments/b1_group_size/run.py    # transformers from scratch
uv run python analysis/figures.py                 # regenerate every figure
```

The real-model experiments additionally need `uv pip install -e ".[mlx]"` and run on Apple Silicon.

Every run writes a manifest recording its configuration, the commit it ran at, and the platform.
No number in the paper is transcribed by hand; figures and tables are generated from those records.
