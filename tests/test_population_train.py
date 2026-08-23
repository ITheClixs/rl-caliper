"""Behaviour of the population RLVR trainer that the measurements depend on."""

import numpy as np
import pytest
import torch

from caliper.envs.modsum import ModSum
from caliper.population.model import ModelConfig, PopulationTransformer
from caliper.population.train import (
    RLConfig,
    RLVRTrainer,
    SFTConfig,
    draw_prompts,
    make_pool,
    measure_noise,
    pass_rate,
    restore,
    snapshot,
    supervised_warmup,
)


@pytest.fixture(scope="module")
def setup():
    task = ModSum(modulus=5, chain=3)
    generator = torch.Generator().manual_seed(0)
    config = ModelConfig(
        vocab=task.vocab, context=task.total_len, width=32, depth=2, heads=2, population=4
    )
    model = PopulationTransformer(config, generator)
    return task, model, generator


def test_pool_prompts_are_distinct_within_a_batch(setup):
    """A prompt appearing twice in a batch would land in two blocks and bias tau_b down."""
    task, model, generator = setup
    unique = torch.arange(64).reshape(1, 64, 1).repeat(4, 1, task.prompt_len)
    prompts = draw_prompts(task, unique, 4, 16, generator)
    for member in range(4):
        rows = {tuple(r.tolist()) for r in prompts[member]}
        assert len(rows) == 16


def test_drawing_more_prompts_than_the_pool_holds_is_rejected(setup):
    task, model, generator = setup
    pool = make_pool(task, 4, 8, generator)
    with pytest.raises(ValueError):
        draw_prompts(task, pool, 4, 16, generator)


def test_snapshot_and_restore_round_trip(setup):
    task, model, generator = setup
    state = snapshot(model)
    with torch.no_grad():
        model.token_embedding += 1.0
    assert not torch.allclose(model.token_embedding, state["token_embedding"])
    restore(model, state)
    torch.testing.assert_close(model.token_embedding, state["token_embedding"])


def test_advantages_follow_the_weight_table(setup):
    task, model, generator = setup
    config = RLConfig(prompts=4, group_size=4, blocks=2)
    trainer = RLVRTrainer(model, task, config, generator)
    rewards = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    advantage = trainer.advantages(rewards, trainer.weights)
    # RLOO with G = 4 and two successes: a success gets 1 - 1/3, a failure gets -2/3
    torch.testing.assert_close(
        advantage[0, 0], torch.tensor([1 - 1 / 3, -2 / 3, 1 - 1 / 3, -2 / 3])
    )


def test_instrumentation_grid_has_the_expected_shape(setup):
    task, model, generator = setup
    config = RLConfig(prompts=8, group_size=4, blocks=4)
    trainer = RLVRTrainer(model, task, config, generator)
    tokens, rewards = trainer.rollout()
    cells = trainer._instrument(tokens, rewards)
    flat = sum(p.numel() for p in model.parameters()) // model.config.population
    assert cells.shape == (4, 2, model.config.population, flat)


def test_block_count_falls_back_to_a_divisor(setup):
    task, model, generator = setup
    config = RLConfig(prompts=6, group_size=4, blocks=8)
    trainer = RLVRTrainer(model, task, config, generator)
    assert trainer.blocks == 3


def test_instrumentation_is_disabled_when_the_group_is_too_small(setup):
    task, model, generator = setup
    config = RLConfig(prompts=8, group_size=2, blocks=2)
    trainer = RLVRTrainer(model, task, config, generator)
    assert trainer.sub_weights is None
    with pytest.raises(ValueError):
        trainer._instrument(*trainer.rollout())


def test_a_policy_that_never_succeeds_does_not_move(setup):
    """Every group is degenerate, every advantage is zero, so the update is exactly zero."""
    task, model, generator = setup
    state = snapshot(model)
    config = RLConfig(prompts=8, group_size=4, blocks=2, initial_step_size=0.05)
    trainer = RLVRTrainer(model, task, config, generator)
    _, rewards = trainer.rollout()
    assert float(rewards.sum()) == 0.0
    info = trainer.step(instrument=False)
    assert all(d == 0.0 for d in info["drift"])
    torch.testing.assert_close(model.token_embedding, state["token_embedding"])
    restore(model, state)


def test_a_step_moves_a_warmed_policy_and_reports_its_drift():
    task = ModSum(modulus=5, chain=2)
    generator = torch.Generator().manual_seed(5)
    config = ModelConfig(
        vocab=task.vocab, context=task.total_len, width=32, depth=2, heads=2, population=2
    )
    model = PopulationTransformer(config, generator)
    supervised_warmup(
        model, task, SFTConfig(steps=200, batch=32, target_pass_rate=None), generator
    )
    state = snapshot(model)
    trainer = RLVRTrainer(
        model,
        task,
        RLConfig(prompts=8, group_size=4, blocks=2, initial_step_size=0.05),
        generator,
    )
    info = trainer.step(instrument=False)
    assert len(info["drift"]) == model.config.population
    assert all(d >= 0 for d in info["drift"])
    assert not torch.allclose(model.token_embedding, state["token_embedding"])


def test_measure_noise_leaves_the_policy_unchanged(setup):
    task, model, generator = setup
    state = snapshot(model)
    config = RLConfig(prompts=8, group_size=4, blocks=2)
    terms = measure_noise(model, task, config, generator, batches=2)
    assert set(terms) == {"signal", "tau_b", "tau_w", "tau_w_scaled"}
    torch.testing.assert_close(model.token_embedding, state["token_embedding"])


def test_warmup_raises_the_pass_rate(setup):
    task, _, _ = setup
    generator = torch.Generator().manual_seed(3)
    config = ModelConfig(
        vocab=task.vocab, context=task.total_len, width=32, depth=2, heads=2, population=2
    )
    model = PopulationTransformer(config, generator)
    before = pass_rate(model, task, 128, generator).mean()
    supervised_warmup(
        model, task, SFTConfig(steps=120, batch=32, target_pass_rate=None), generator
    )
    after = pass_rate(model, task, 128, generator).mean()
    assert after > before


def test_pass_rate_covers_a_small_pool_by_repeating(setup):
    task, model, generator = setup
    pool = make_pool(task, 4, 8, generator)
    rate = pass_rate(model, task, 32, generator, pool)
    assert rate.shape == (4,)
    assert np.all((rate.numpy() >= 0) & (rate.numpy() <= 1))


def test_priming_moves_the_preconditioner_but_not_the_policy(setup):
    """Adam's metric does not exist until its second moment has seen gradients."""
    task, model, generator = setup
    state = snapshot(model)
    config = RLConfig(prompts=8, group_size=4, blocks=2, optimiser="adam")
    measure_noise(model, task, config, generator, batches=1, prime_batches=2)
    torch.testing.assert_close(model.token_embedding, state["token_embedding"])


def test_priming_changes_what_the_probe_reports_under_adam():
    task = ModSum(modulus=5, chain=2)
    generator = torch.Generator().manual_seed(11)
    config = ModelConfig(
        vocab=task.vocab, context=task.total_len, width=32, depth=2, heads=2, population=2
    )
    model = PopulationTransformer(config, generator)
    supervised_warmup(
        model, task, SFTConfig(steps=200, batch=32, target_pass_rate=None), generator
    )
    settings = RLConfig(prompts=8, group_size=4, blocks=2, optimiser="adam")
    cold = measure_noise(model, task, settings, generator, batches=3, prime_batches=0)
    warm = measure_noise(model, task, settings, generator, batches=3, prime_batches=4)
    assert not torch.allclose(cold["tau_w"], warm["tau_w"])
