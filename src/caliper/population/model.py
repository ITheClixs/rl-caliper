"""A population of independent decoder-only transformers evaluated as one batched kernel.

Every parameter carries a leading population axis, so N independent models train simultaneously
inside single matrix multiplications. For the model sizes this study needs, running N runs this way
is the difference between a sweep that fits on a laptop and one that does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab: int
    context: int
    width: int = 64
    depth: int = 2
    heads: int = 2
    population: int = 8

    @property
    def head_dim(self) -> int:
        if self.width % self.heads:
            raise ValueError("width must divide into heads")
        return self.width // self.heads


def _normal(shape, std: float, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=generator.device) * std


class PopulationTransformer(nn.Module):
    def __init__(self, config: ModelConfig, generator: torch.Generator):
        super().__init__()
        self.config = config
        n, v, d, layers = config.population, config.vocab, config.width, config.depth
        std = d**-0.5

        # every parameter keeps the population on axis 0, so a per-member step size is a plain
        # broadcast and a flattened per-member gradient is a plain reshape
        self.token_embedding = nn.Parameter(_normal((n, v, d), std, generator))
        self.position_embedding = nn.Parameter(_normal((n, config.context, d), std, generator))
        depth_scale = (2 * layers) ** 0.5
        self.qkv = nn.ParameterList(
            [nn.Parameter(_normal((n, d, 3 * d), std, generator)) for _ in range(layers)]
        )
        self.proj = nn.ParameterList(
            [nn.Parameter(_normal((n, d, d), std / depth_scale, generator)) for _ in range(layers)]
        )
        self.fc_in = nn.ParameterList(
            [nn.Parameter(_normal((n, d, 4 * d), std, generator)) for _ in range(layers)]
        )
        self.fc_out = nn.ParameterList(
            [
                nn.Parameter(_normal((n, 4 * d, d), (4 * d) ** -0.5 / depth_scale, generator))
                for _ in range(layers)
            ]
        )
        self.ln1_gain = nn.ParameterList(
            [nn.Parameter(torch.ones(n, d, device=generator.device)) for _ in range(layers)]
        )
        self.ln2_gain = nn.ParameterList(
            [nn.Parameter(torch.ones(n, d, device=generator.device)) for _ in range(layers)]
        )
        self.ln_final = nn.Parameter(torch.ones(n, d, device=generator.device))
        self.unembedding = nn.Parameter(_normal((n, d, v), std, generator))

    @staticmethod
    def _norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        return x * gain[:, None, None, :]

    def _attention(self, x: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        cfg = self.config
        n, b, t, d = x.shape
        qkv = torch.einsum("nbtd,nde->nbte", x, self.qkv[layer])
        q, k, v = qkv.split(d, dim=-1)
        shape = (n, b, t, cfg.heads, cfg.head_dim)
        q = q.reshape(shape).permute(0, 1, 3, 2, 4)
        k = k.reshape(shape).permute(0, 1, 3, 2, 4)
        v = v.reshape(shape).permute(0, 1, 3, 2, 4)

        scores = torch.einsum("nbhtd,nbhsd->nbhts", q, k) * cfg.head_dim**-0.5
        rows = torch.arange(t, device=x.device)[:, None] + offset
        cols = torch.arange(t, device=x.device)[None, :]
        scores = scores.masked_fill(cols > rows, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("nbhts,nbhsd->nbhtd", weights, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(n, b, t, d)
        return torch.einsum("nbtd,nde->nbte", out, self.proj[layer])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (population, batch, time) -> logits (population, batch, time, vocab)."""
        n, b, t = tokens.shape
        index = torch.arange(n, device=tokens.device)[:, None, None]
        x = self.token_embedding[index, tokens] + self.position_embedding[:, None, :t, :]
        for layer in range(self.config.depth):
            x = x + self._attention(self._norm(x, self.ln1_gain[layer]), layer, offset=0)
            h = self._norm(x, self.ln2_gain[layer])
            h = torch.einsum("nbtd,nde->nbte", h, self.fc_in[layer])
            h = F.gelu(h)
            x = x + torch.einsum("nbte,ned->nbtd", h, self.fc_out[layer])
        x = self._norm(x, self.ln_final)
        return torch.einsum("nbtd,ndv->nbtv", x, self.unembedding)

    @torch.no_grad()
    def generate(
        self,
        prompts: torch.Tensor,
        steps: int,
        generator: torch.Generator,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample `steps` tokens after each prompt. prompts: (population, batch, prompt_len)."""
        tokens = prompts
        for _ in range(steps):
            logits = self.forward(tokens)[:, :, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            n, b, v = probs.shape
            flat = torch.multinomial(probs.reshape(-1, v), 1, generator=generator)
            tokens = torch.cat([tokens, flat.reshape(n, b, 1)], dim=-1)
        return tokens

    def log_probs(self, tokens: torch.Tensor, start: int) -> torch.Tensor:
        """Per-token log probabilities of the tokens from position `start` onwards."""
        logits = self.forward(tokens[:, :, :-1])[:, :, start - 1 :, :]
        targets = tokens[:, :, start:]
        return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
