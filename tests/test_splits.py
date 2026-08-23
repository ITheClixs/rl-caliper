"""The four-buffer estimator must recover the exact hierarchical noise terms."""

import numpy as np
import pytest

from caliper.estimators.simulate import simulate_split_batch
from caliper.estimators.splits import average
from caliper.exact.aggregate import ExactBatchModel
from caliper.exact.policy import TabularPolicy, accept_set_for_pass_rate, fisher, make_prompt


@pytest.fixture(scope="module")
def population():
    # the pool has to be large relative to the batch: drawing a sizeable fraction of a small pool
    # without replacement makes the blocks negatively correlated, which is a finite-population
    # effect the iid theory does not model
    rng = np.random.default_rng(7)
    policy = TabularPolicy.random(3, 3, rng)
    probs, _ = policy.enumerate()
    prompts, moments, fishers = [], [], []
    while len(prompts) < 160:
        accept = accept_set_for_pass_rate(probs, rng.uniform(0.1, 0.9), rng)
        prompt = make_prompt(policy, accept)
        if 1e-6 < prompt.pass_rate < 1 - 1e-6:
            prompts.append(prompt)
            moments.append(prompt.moments())
            fishers.append(fisher(prompt))
    return prompts, ExactBatchModel(moments, fishers)


def test_split_estimator_recovers_exact_noise_terms(population):
    prompts, model = population
    group_size, n_prompts, batches = 8, 16, 1500
    exact = model.noise_terms("rloo", group_size, curvature="identity")

    rng = np.random.default_rng(11)
    avg = average(
        [
            simulate_split_batch(prompts, "rloo", n_prompts, group_size, rng, blocks=8)
            for _ in range(batches)
        ]
    )

    assert avg.tau_total() == pytest.approx(exact.tau_b + exact.tau_w, rel=0.10)
    assert avg.tau_w == pytest.approx(exact.tau_w, rel=0.10)
    assert avg.tau_b == pytest.approx(exact.tau_b, rel=0.20)
    assert avg.signal == pytest.approx(exact.signal, rel=0.30)

    exact_g = 1.0 + np.sqrt(exact.tau_w * (group_size - 1) / exact.tau_b)
    assert avg.optimal_group_size() == pytest.approx(exact_g, abs=0.6)


def test_split_estimator_rejects_odd_group_sizes(population):
    prompts, _ = population
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_split_batch(prompts, "rloo", 8, 7, rng)


@pytest.mark.parametrize("group_size", [4, 8, 16])
def test_between_and_within_terms_scale_as_predicted(population, group_size):
    """tau_b must not depend on G; the scaled within-prompt term must not either."""
    prompts, model = population
    exact = model.noise_terms("rloo", group_size, curvature="identity")
    rng = np.random.default_rng(3)
    avg = average(
        [
            simulate_split_batch(prompts, "rloo", 16, group_size, rng, blocks=8)
            for _ in range(1200)
        ]
    )
    assert avg.tau_b == pytest.approx(exact.tau_b, rel=0.25)
    assert avg.tau_w == pytest.approx(exact.tau_w, rel=0.20)


def test_prompt_repetition_biases_the_between_prompt_term(population):
    """Sampling prompts with replacement puts one prompt in two blocks and shrinks tau_b."""
    prompts, model = population
    exact = model.noise_terms("rloo", 8, curvature="identity")

    def estimate(replace):
        rng = np.random.default_rng(5)
        return average(
            [
                simulate_split_batch(
                    prompts, "rloo", 16, 8, rng, blocks=8, replace=replace
                )
                for _ in range(800)
            ]
        ).tau_b

    with_replacement = estimate(True)
    without = estimate(False)
    assert with_replacement < without
    assert without == pytest.approx(exact.tau_b, rel=0.15)


def test_finite_corpus_correction_recovers_the_signal(population):
    """Without replacement the blocks are negatively correlated; the correction removes it."""
    prompts, model = population
    exact = model.noise_terms("rloo", 8, curvature="identity")

    def estimate(correct):
        rng = np.random.default_rng(11)
        return average(
            [
                simulate_split_batch(
                    prompts, "rloo", 16, 8, rng, blocks=8, correct_finite_corpus=correct
                )
                for _ in range(1500)
            ]
        ).signal

    raw = estimate(False)
    corrected = estimate(True)
    assert raw < corrected
    assert abs(corrected - exact.signal) < abs(raw - exact.signal)
