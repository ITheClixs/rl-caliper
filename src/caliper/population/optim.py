"""Optimisers with a per-member step size for a stacked population of models.

Adam is scale invariant, so a per-member step size cannot be smuggled in by rescaling gradients;
it has to be applied to the update. Both rules also expose the preconditioner, because the noise
estimates have to be taken in the metric the update actually moves in.
"""

from __future__ import annotations

import torch


class PopulationOptimiser:
    def __init__(self, parameters, population: int):
        self.params = list(parameters)
        self.population = population

    def _shape(self, step_sizes: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        return step_sizes.reshape((-1,) + (1,) * (param.dim() - 1))

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def precondition(self, flat_grad: torch.Tensor) -> torch.Tensor:
        return flat_grad

    def step(self, step_sizes: torch.Tensor) -> None:
        raise NotImplementedError


class PopulationSGD(PopulationOptimiser):
    @torch.no_grad()
    def step(self, step_sizes: torch.Tensor) -> None:
        for p in self.params:
            if p.grad is not None:
                p -= self._shape(step_sizes, p) * p.grad


class PopulationAdam(PopulationOptimiser):
    def __init__(self, parameters, population: int, betas=(0.9, 0.95), eps: float = 1e-8):
        super().__init__(parameters, population)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, step_sizes: torch.Tensor) -> None:
        b1, b2 = self.betas
        self.t += 1
        bias1 = 1 - b1**self.t
        bias2 = 1 - b2**self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[i].mul_(b1).add_(p.grad, alpha=1 - b1)
            self.v[i].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            update = (self.m[i] / bias1) / ((self.v[i] / bias2).sqrt() + self.eps)
            p -= self._shape(step_sizes, p) * update

    @torch.no_grad()
    def precondition(self, flat_grad: torch.Tensor) -> torch.Tensor:
        """Map a gradient into the metric Adam moves in, using the current second moment."""
        if self.t == 0:
            return flat_grad
        bias2 = 1 - self.betas[1] ** self.t
        scale = torch.cat(
            [((v / bias2).sqrt() + self.eps).reshape(self.population, -1) for v in self.v], dim=1
        )
        return flat_grad / scale
