"""A verifiable task for pretrained models: modular running sums, stated in words.

Same structure as the synthetic environment used for the from-scratch models, so the two tiers
measure the same quantity, but presented as text a pretrained instruction model can attempt and
scored by exact match on the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEMPLATE = (
    "Compute the running sums of the list below, each taken modulo {modulus}.\n"
    "List: {digits}\n"
    "Answer with the running sums separated by single spaces, and nothing else."
)


@dataclass(frozen=True)
class WordedModSum:
    modulus: int = 7
    chain: int = 4

    def sample(self, n: int, rng: np.random.Generator) -> list[dict]:
        items = []
        for _ in range(n):
            digits = rng.integers(0, self.modulus, size=self.chain)
            target = " ".join(str(int(v)) for v in np.cumsum(digits) % self.modulus)
            items.append(
                {
                    "prompt": TEMPLATE.format(
                        modulus=self.modulus,
                        digits=" ".join(str(int(d)) for d in digits),
                    ),
                    "answer": target,
                }
            )
        return items

    @staticmethod
    def reward(completion: str, answer: str) -> float:
        return float(completion.strip().split("\n")[0].strip() == answer)
