# caliper

How much would an RL post-training score move if you reran it with a different seed, and can one
run tell you before you pay for the others?

A reported score is the output of a stochastic procedure. The usual way to learn its spread is to
train the recipe several times. This repository implements a cheaper estimate and measures where it
works and where it does not. Noise injected at one update is reweighted by the updates that follow
rather than simply accumulated, so the covariance of a run about its mean obeys a recursion.
Carrying the gradient of the reported metric backwards along a single stored trajectory turns that
recursion into an error bar: a sum of one scalar per update, with no covariance matrix formed.

![Forecast against measured spread](docs/teaser.png)

*Left: on enumerable policies, the across-seed standard deviation forecast from one frozen run
against the value measured over 47 seeds trained afterwards (1.0x is perfect; dotted lines are
1.5x). Right: the simpler models this replaces. This is the tier where the method works; the
table below includes the tiers where it does not.*

**Paper:** [full version (PDF)](paper/pdf/caliper.pdf) ·
[eight-page version (PDF)](paper/pdf/caliper-icml.pdf) · LaTeX sources in `paper/`

## Results at a glance

"Error" is the ratio between forecast and measured standard deviation, taken in whichever direction
is above one, so 1.0x is perfect and 2x means off by a factor of two either way.

| setting | what was compared | result |
|---|---|---|
| enumerable policies, exact inputs, at most 25 updates | forecast frozen before 47 further seeds were trained, 192 settings | median error **1.10x**, worst 1.56x, inside the measured 95% interval in 162 of 192 |
| Qwen2.5-0.5B, four pre-specified corpora, prospective | frozen forecast against 12 held-out seeds | errors 1.66, 2.13, 2.80, 1.24 |
| Qwen2.5-0.5B, corpus whose reward groups go unanimous | cheap forecast against 8 seeds | **25x short** |
| Qwen2.5-7B (4-bit, LoRA) under Adam | lifted forecast against 10 seeds | **3.1x to 17x short** |
| sizing a comparison (enumerable tier, 1020 comparisons) | runs to reach a 0.1% mistake rate, thresholds chosen on other tasks | **3.14** runs, against 14 for an uncapped fixed budget and 2.12x as many from spent seeds |

Two things the sizing benchmark shows that the headline does not. A closed form that needs no
stored run comes within 1.11x of the forecast once its stopping threshold has been learned on other
tasks, so on this tier the backward pass adds little to *ranking* configurations. What the stored
run adds is an *absolute* scale: with the textbook threshold and no calibration the closed form never
stops (its scale is off by 84x), while the forecast decides in 4.37 runs with a 0.001% mistake rate
for a 1% target.

Why it fails on pretrained models, as far as the records show: in both large misses the run the
forecast was made from answers every held-out prompt 16/16 or 0/16, where the score is locally flat,
while later seeds differ in which way each prompt collapses. At 0.5B two thirds of the seed spread is
prompts flipping independently between seeds; at 7B 60% is shared. A first-order forecast around
one collapsed trajectory has no term for that. This is from two runs and is stated as an
observation, not a result.

`docs/claims.md` indexes every claim in the paper against the experiment that supports it and how
strongly, including the claims that were withdrawn and why.

## What is not established

* Anything at frontier scale. Every experiment runs on one laptop; the largest model is a 4-bit 7B
  with LoRA, and every pretrained forecast spans at most 12 updates.
* That the forecast is accurate on pretrained models in general. It is inside 1.5x in one of four
  pre-specified regimes, and misses by 25x and by 3-17x in the two other pretrained settings.
* A coverage guarantee for the stopping rule. The textbook threshold realised its target on our grid;
  that is an observation on one grid.
* That the one-run curvature diagnostic bounds the worst case. It ranks failures above chance (AUROC
  0.78) but on the 192-setting grid it accepts the single worst forecast; an earlier claim that it
  capped the tail did not survive doubling the grid and is withdrawn.
* A route from policy divergence to outcome spread via `sqrt(KL)` (fitted exponent `+0.10` against a
  predicted `+0.5`), and a short memory horizon. Both were looked for and not found.

## What is here

```
src/caliper/
  forecast.py   the backward pass, as a protocol any trainer can implement
  objectives/   the estimator family as a (G+1) x 2 weight table
  exact/        enumerable policies: exact scores, Fisher, moments, a trainer,
                the covariance recursion, the adjoint, and Adam's lifted state
  estimators/   the K x 2 split estimator and its sampling harness
  population/   transformers trained as a batched population, and the RLVR trainer
  control/      step size as a feedback loop on realised drift
  real/         the same measurement on a pretrained model through MLX
  analysis/     efficiency curves, interval estimation, sizing, and grid selection
  runtime/      run manifests
experiments/    one directory per experiment, config-driven
runs/           the stored record of every run the paper quotes
analysis/       figure and table generation for the paper
paper/          LaTeX sources and built PDFs
docs/           theory notes, result notes, and the claim index
```

## Using the forecast

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

`caliper.exact.adjoint` runs its forecast through this protocol (`ExactUpdate`). The pretrained-model
version in `caliper.real.adjoint` implements the same recursion inline, because it also carries
Adam's moments backwards and cross-fits the injected term, which the protocol does not express yet.

The full backward pass costs three passes per stored update. Holding `b` at `grad M` instead of
carrying it back changes the forecast by a median of 2% at these run lengths, so the cheap form,
one extra pass per update and no Hessian-vector products, is the recommended default.

## Running it

```
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest                                      # includes the exactness tests
uv run python experiments/p2_metric_forecast/run.py  # the forecast, tested prospectively
uv run python experiments/s13_sizing/run.py          # sizing, thresholds read in sample
uv run python experiments/s13_sizing/heldout.py      # sizing, thresholds set before grading
uv run python experiments/s16_saturation/analyse.py  # where the pretrained seed spread is
uv run python analysis/tables.py                   # regenerate every table and paper macro
uv run python analysis/figures.py                  # regenerate every figure
```

The pretrained-model experiments additionally need `uv pip install -e ".[mlx]"` and Apple Silicon.

Every run writes a manifest with its configuration, the commit it ran at, whether the tree was
dirty, and the platform. Numbers in tables and in the paper's macros are generated from those
records, and `tests/test_docs_numbers.py` fails if a number quoted in this README or in `docs/`
disagrees with them. Some numbers in the paper's prose are still typed by hand.

## How this was built

This is a single-author project. I chose the research question and its framing, decided which
experiments to run, judged the results, and decided which claims to keep, correct or withdraw; I
also derived and checked parts of the theory and wrote or debugged parts of the code. AI coding
agents (Claude Code) wrote most of the code, the LaTeX and the documentation, and ran the sweeps
under that direction, which is why the commit history is dense. A revision pass in November 2026
was also done with Claude; its commits carry a `Co-Authored-By` trailer. It fixed a table that
printed 7B numbers in a 0.5B table, restored the headline grid after an accidental switch, withdrew
the diagnostic's tail claim, regraded the sizing benchmark with thresholds set before grading, and
located the pretrained seed spread from the stored records.
