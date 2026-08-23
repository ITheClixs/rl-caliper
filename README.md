# caliper

Measurement tools for the batch-size and step-size scaling of reinforcement learning post-training
of language models.

RLVR samples at two levels — prompts, and responses per prompt — and a run has to decide how to
split a fixed rollout budget between them, and how large a step the result supports. Both are
normally tuned. This repository is the argument that both can be measured, together with the code
that does the measuring and the experiments that check it.

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

For binary verifiable rewards the estimators in current use — RLOO, GRPO with a mean baseline,
standardised GRPO — all compute the true per-prompt gradient scaled by a weight `lambda(p, G)` on
prompt difficulty, so those with a `p`-independent weight are the same estimator up to a step size.
The gradient noise splits into a between-prompt term that responses cannot reduce and a
within-prompt term that shrinks as `1/(G-1)`. The optimal group size is
`G* = 1 + sqrt(tau_w / tau_b)`, inflated by `sqrt(1 + c_pre/c_dec)` once prefill sharing is priced
in, and both traces are estimable from the microbatch gradients a trainer already accumulates. A
run specified by a target drift transfers across batch size; a run specified by a learning rate
does not.

## Running it

```
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest                                  # includes the exactness tests
uv run python experiments/a4_law/run.py        # enumerable policies
uv run python experiments/b1_group_size/run.py # transformers from scratch
uv run python analysis/figures.py              # regenerate every figure
```

The real-model experiments additionally need `uv pip install -e ".[mlx]"` and run on Apple Silicon.

Every run writes a manifest recording its configuration, the commit it ran at, and the platform.
No number in the paper is transcribed by hand; figures and tables are generated from those records.
