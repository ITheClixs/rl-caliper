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


def test_sampling_with_replacement_is_the_iid_model(population):
    """Drawing prompts with replacement needs no correction; drawing distinct ones does.

    Two blocks of an with-replacement batch are independent draws, so the different-block inner
    product is unbiased for the squared mean gradient and tau_b comes out right. It is drawing
    without replacement that biases it, upward by N/(N-1), because distinct blocks are then
    negatively correlated.
    """
    prompts, model = population
    exact = model.noise_terms("rloo", 8, curvature="identity")

    def estimate(replace, corpus):
        rng = np.random.default_rng(5)
        return average(
            [
                simulate_split_batch(
                    prompts, "rloo", 16, 8, rng, blocks=8, replace=replace,
                    correct_finite_corpus=corpus,
                )
                for _ in range(1200)
            ]
        ).tau_b

    assert estimate(True, False) == pytest.approx(exact.tau_b, rel=0.06)
    assert estimate(False, True) == pytest.approx(exact.tau_b, rel=0.06)


@pytest.fixture(scope="module")
def small_population():
    """A corpus small enough that the finite-population factor is worth more than the noise."""
    rng = np.random.default_rng(3)
    policy = TabularPolicy.random(3, 3, rng)
    probs, _ = policy.enumerate()
    prompts, moments, fishers = [], [], []
    while len(prompts) < 24:
        accept = accept_set_for_pass_rate(probs, rng.uniform(0.1, 0.9), rng)
        prompt = make_prompt(policy, accept)
        if 1e-6 < prompt.pass_rate < 1 - 1e-6:
            prompts.append(prompt)
            moments.append(prompt.moments())
            fishers.append(fisher(prompt))
    return prompts, ExactBatchModel(moments, fishers)


def test_the_finite_corpus_correction_applies_to_the_between_prompt_term(small_population):
    """Without replacement from a corpus of N, tau_b is biased up by N/(N-1) until corrected.

    The uncorrected estimator returns `n (W - A)`, while the block covariance under sampling
    without replacement makes `W - A = N B / (n (N-1))`, so the unbiased form carries the extra
    factor `(N-1)/N`.
    """
    prompts, model = small_population
    exact = model.noise_terms("rloo", 8, curvature="identity")

    def estimate(corpus):
        rng = np.random.default_rng(17)
        return average(
            [
                simulate_split_batch(
                    prompts, "rloo", 8, 8, rng, blocks=4, replace=False,
                    correct_finite_corpus=corpus,
                )
                for _ in range(4000)
            ]
        ).tau_b

    raw, corrected = estimate(False), estimate(True)
    assert raw > corrected, "the uncorrected estimator should be the larger one"
    assert corrected == pytest.approx(exact.tau_b, rel=0.05)


def _retired_test_prompt_repetition_biases_the_between_prompt_term(population):
    """Retired: the premise was wrong. See the test above."""
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
