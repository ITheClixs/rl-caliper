import torch

from caliper.envs.modsum import ModSum


def test_targets_are_running_sums():
    task = ModSum(modulus=5, chain=3)
    prompts = torch.tensor([[1, 2, 3, task.sep], [4, 4, 4, task.sep]])
    want = torch.tensor([[1, 3, 1], [4, 3, 2]])
    assert torch.equal(task.targets(prompts), want)


def test_reward_is_all_or_nothing():
    task = ModSum(modulus=5, chain=3)
    prompts = torch.tensor([[1, 2, 3, task.sep]])
    assert task.reward(prompts, torch.tensor([[1, 3, 1]])).item() == 1.0
    assert task.reward(prompts, torch.tensor([[1, 3, 2]])).item() == 0.0


def test_sampled_prompts_have_the_right_shape():
    task = ModSum(modulus=7, chain=4)
    gen = torch.Generator().manual_seed(0)
    prompts = task.sample_prompts(16, gen)
    assert prompts.shape == (16, task.prompt_len)
    assert (prompts[:, -1] == task.sep).all()
    assert prompts[:, :-1].max() < task.modulus
