"""Running-sum-modulo task: a verifiable synthetic environment with tunable difficulty.

The prompt is a sequence of digits; the response is the sequence of running sums modulo m. The
task needs sequential computation rather than lookup, the verifier is exact, and difficulty is set
by the modulus and the chain length. Reward is terminal: the response is correct only if every
running sum is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModSum:
    modulus: int = 7
    chain: int = 4

    @property
    def sep(self) -> int:
        return self.modulus

    @property
    def eos(self) -> int:
        return self.modulus + 1

    @property
    def vocab(self) -> int:
        return self.modulus + 2

    @property
    def prompt_len(self) -> int:
        return self.chain + 1  # digits and the separator

    @property
    def response_len(self) -> int:
        return self.chain

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.response_len

    def sample_prompts(self, n: int, generator: torch.Generator) -> torch.Tensor:
        digits = torch.randint(
            0, self.modulus, (n, self.chain), generator=generator, device=generator.device
        )
        sep = torch.full((n, 1), self.sep, dtype=torch.long, device=digits.device)
        return torch.cat([digits, sep], dim=1)

    def targets(self, prompts: torch.Tensor) -> torch.Tensor:
        digits = prompts[..., : self.chain]
        return torch.cumsum(digits, dim=-1) % self.modulus

    def reward(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """1 when every running sum in the response is correct."""
        want = self.targets(prompts)
        return (responses[..., : self.chain] == want).all(dim=-1).float()
