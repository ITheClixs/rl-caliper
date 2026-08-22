"""Batch-level invariants the theory demands."""

import numpy as np
import pytest

from caliper.exact.aggregate import ExactBatchModel, predicted_optimal_group_size
from caliper.exact.policy import TabularPolicy, accept_set_for_pass_rate, fisher, make_prompt


def population(seed: int, n_prompts: int = 32, vocab: int = 3, length: int = 3):
    rng = np.random.default_rng(seed)
    policy = TabularPolicy.random(vocab, length, rng)
    probs, _ = policy.enumerate()
    prompts, fishers = [], []
    while len(prompts) < n_prompts:
        accept = accept_set_for_pass_rate(probs, rng.uniform(0.05, 0.95), rng)
        prompt = make_prompt(policy, accept)
        if 1e-6 < prompt.pass_rate < 1 - 1e-6:
            prompts.append(prompt.moments())
            fishers.append(fisher(prompt))
    return ExactBatchModel(prompts, fishers)


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_constant_rescaling_does_not_change_progress(group_size):
    """GRPO with a mean baseline is RLOO times (G-1)/G; the step size absorbs the difference."""
    model = population(0)
    a = model.noise_terms("rloo", group_size)
    b = model.noise_terms("grpo_mean", group_size)
    assert b.critical_batch() == pytest.approx(a.critical_batch())
    assert b.progress(16) == pytest.approx(a.progress(16))
    assert b.optimal_step_size(16) == pytest.approx(
        a.optimal_step_size(16) * group_size / (group_size - 1)
    )


def test_rloo_signal_terms_are_group_size_independent():
    model = population(1)
    ref = model.noise_terms("rloo", 2)
    for g in [3, 7, 16]:
        terms = model.noise_terms("rloo", g)
        assert terms.tau_b == pytest.approx(ref.tau_b)
        assert terms.alignment == pytest.approx(ref.alignment)
        assert terms.signal == pytest.approx(ref.signal)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_closed_form_group_size_matches_the_exact_optimum(seed):
    model = population(seed)
    rows = [(g, model.progress_per_rollout_budget("rloo", g, 4096)) for g in range(2, 41)]
    exact = max(rows, key=lambda r: r[1])[0]
    ref = model.noise_terms("rloo", 8)
    predicted = predicted_optimal_group_size(ref.tau_b, ref.tau_w * 7)
    assert abs(predicted - exact) <= 1.0


def test_prefill_amortisation_raises_the_optimal_group_size():
    model = population(2)
    ref = model.noise_terms("rloo", 8)
    plain = predicted_optimal_group_size(ref.tau_b, ref.tau_w * 7, prefill_ratio=0.0)
    shared = predicted_optimal_group_size(ref.tau_b, ref.tau_w * 7, prefill_ratio=8.0)
    assert shared > plain
    assert shared - 1 == pytest.approx((plain - 1) * 3.0)
