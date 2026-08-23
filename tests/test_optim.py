"""Per-member step sizes and the preconditioner the estimator has to read."""

import pytest
import torch

from caliper.population.optim import PopulationAdam, PopulationSGD


def parameters(population=3, shape=(4, 5)):
    params = [torch.zeros(population, *shape, requires_grad=True) for _ in range(2)]
    for index, p in enumerate(params):
        p.grad = torch.full_like(p, float(index + 1))
    return params


def test_sgd_applies_a_different_step_to_each_member():
    params = parameters()
    opt = PopulationSGD(params, population=3)
    steps = torch.tensor([0.1, 0.2, 0.4])
    before = [p.detach().clone() for p in params]
    opt.step(steps)
    for p, start, scale in [(params[0], before[0], 1.0), (params[1], before[1], 2.0)]:
        for member in range(3):
            expected = start[member] - steps[member] * scale
            torch.testing.assert_close(p[member].detach(), expected)


def test_adam_is_scale_invariant_in_the_gradient():
    """Rescaling the gradient must not rescale an Adam update; only the step size may."""
    steps = torch.tensor([0.1, 0.1, 0.1])
    moves = []
    for scale in (1.0, 100.0):
        params = parameters()
        for p in params:
            p.grad = p.grad * scale
        opt = PopulationAdam(params, population=3)
        before = params[0].detach().clone()
        opt.step(steps)
        moves.append((params[0].detach() - before).clone())
    torch.testing.assert_close(moves[0], moves[1], rtol=1e-4, atol=1e-7)


def test_adam_preconditioner_matches_the_update_direction():
    params = parameters()
    opt = PopulationAdam(params, population=3)
    opt.step(torch.tensor([0.1, 0.1, 0.1]))
    flat = torch.cat([p.grad.reshape(3, -1) for p in params], dim=1)
    preconditioned = opt.precondition(flat)
    assert preconditioned.shape == flat.shape
    # after one step the second moment equals (1 - b2) g^2 debiased to g^2, so |g / sqrt(v)| ~ 1
    assert preconditioned.abs().max() < 1.01
    assert preconditioned.abs().min() > 0.99


def test_sgd_preconditioner_is_the_identity():
    params = parameters()
    opt = PopulationSGD(params, population=3)
    flat = torch.cat([p.grad.reshape(3, -1) for p in params], dim=1)
    torch.testing.assert_close(opt.precondition(flat), flat)


def test_unknown_optimiser_is_rejected():
    with pytest.raises(NotImplementedError):
        from caliper.population.optim import PopulationOptimiser

        PopulationOptimiser(parameters(), population=3).step(torch.ones(3))
