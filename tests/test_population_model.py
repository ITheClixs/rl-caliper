"""Structural guarantees the population trainer relies on."""

import torch

from caliper.envs.modsum import ModSum
from caliper.population.model import ModelConfig, PopulationTransformer


def build(population=4, width=32):
    task = ModSum(modulus=5, chain=3)
    gen = torch.Generator().manual_seed(0)
    config = ModelConfig(
        vocab=task.vocab,
        context=task.total_len,
        width=width,
        depth=2,
        heads=2,
        population=population,
    )
    return task, gen, PopulationTransformer(config, gen)


def test_every_parameter_has_population_on_axis_zero():
    _, _, model = build()
    for name, param in model.named_parameters():
        assert param.shape[0] == model.config.population, name


def test_members_are_independent():
    """A loss that touches only one member must leave every other member's gradient at zero."""
    task, gen, model = build()
    prompts = task.sample_prompts(8, gen).unsqueeze(0).expand(4, -1, -1).contiguous()
    tokens = torch.cat([prompts, task.targets(prompts)], dim=-1)
    logits = model(tokens[:, :, :-1])
    logits[1].sum().backward()
    for name, param in model.named_parameters():
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        for member in range(4):
            has_grad = grad[member].abs().sum() > 0
            assert has_grad == (member == 1), (name, member)


def test_causality():
    """Changing a later token must not change the logits at an earlier position."""
    task, gen, model = build()
    prompts = task.sample_prompts(4, gen).unsqueeze(0).expand(4, -1, -1).contiguous()
    tokens = torch.cat([prompts, task.targets(prompts)], dim=-1)
    baseline = model(tokens)
    altered = tokens.clone()
    altered[:, :, -1] = (altered[:, :, -1] + 1) % task.modulus
    changed = model(altered)
    torch.testing.assert_close(baseline[:, :, :-1], changed[:, :, :-1])


def test_generation_leaves_the_prompt_untouched():
    task, gen, model = build()
    prompts = task.sample_prompts(6, gen).unsqueeze(0).expand(4, -1, -1).contiguous()
    out = model.generate(prompts, task.response_len, gen)
    assert out.shape[-1] == task.total_len
    torch.testing.assert_close(out[:, :, : task.prompt_len], prompts)
