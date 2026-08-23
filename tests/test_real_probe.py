"""End-to-end check of the pretrained-model probe, on the smallest model available."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

MODEL = "mlx-community/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def loaded():
    from mlx_lm import load

    return load(MODEL)


@pytest.mark.mlx
def test_group_sampling_shares_the_prompt_and_returns_a_group(loaded):
    from caliper.real.generate import sample_group

    model, tokenizer = loaded
    ids = mx.array(tokenizer.encode("Answer with one word: what colour is the sky?"))
    out = sample_group(model, ids, 4, 6, 1.0, mx.random.key(0), eos_id=tokenizer.eos_token_id)
    mx.eval(out)
    assert out.shape == (4, 6)


@pytest.mark.mlx
def test_probe_returns_finite_noise_terms(loaded):
    from caliper.real.probe import ProbeConfig, RealNoiseProbe
    from caliper.real.tasks import FAMILIES

    model, tokenizer = loaded
    rng = np.random.default_rng(0)
    corpus = FAMILIES["add_two"].sample(16, rng)
    config = ProbeConfig(group_size=4, prompts=4, blocks=2, max_tokens=8, lora_layers=2)
    probe = RealNoiseProbe(model, tokenizer, config)
    estimates, rates = probe.measure(corpus, 2, rng)
    assert len(estimates) == 2
    assert rates.shape == (8,)
    assert np.all((rates >= 0) & (rates <= 1))
    for estimate in estimates:
        assert np.isfinite(estimate.tau_w)
        assert np.isfinite(estimate.tau_b)
        assert np.isfinite(estimate.signal)
        assert estimate.tau_w >= 0
    # a corpus the model never solves has no reward variance, so the measured noise is exactly zero
    if float(rates.sum()) == 0.0:
        assert all(e.tau_w == 0.0 and e.tau_b == 0.0 for e in estimates)
