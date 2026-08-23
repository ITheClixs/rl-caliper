"""Measuring the two noise scales on a pretrained model.

The K x 2 cells partition the batch, so a cell's gradient only involves that cell's sequences and
the whole grid costs one forward and one backward pass over the batch -- the same work a trainer
does when it accumulates over microbatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.tuner import linear_to_lora_layers

from caliper.estimators.splits import decompose
from caliper.objectives.advantages import weight_table
from caliper.real.generate import sample_group
from caliper.real.tasks import reward as exact_match


@dataclass
class ProbeConfig:
    group_size: int = 8
    prompts: int = 16
    blocks: int = 8
    max_tokens: int = 20
    temperature: float = 1.0
    estimator: str = "rloo"
    lora_layers: int = 8
    lora_rank: int = 8
    lora_scale: float = 20.0
    seed: int = 0
    keys: dict = field(default_factory=dict)


def attach_lora(model, config: ProbeConfig):
    model.freeze()
    linear_to_lora_layers(
        model,
        config.lora_layers,
        {"rank": config.lora_rank, "scale": config.lora_scale, "dropout": 0.0},
    )
    return [p for _, p in tree_flatten_trainable(model)]


def tree_flatten_trainable(model):
    from mlx.utils import tree_flatten

    return tree_flatten(model.trainable_parameters())


def flat_gradient(grads) -> mx.array:
    from mlx.utils import tree_flatten

    return mx.concatenate([g.reshape(-1) for _, g in tree_flatten(grads)])


class RealNoiseProbe:
    def __init__(self, model, tokenizer, config: ProbeConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.weights = weight_table(config.estimator, config.group_size // 2)
        attach_lora(model, config)

    def _chat(self, prompt: str) -> mx.array:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return mx.array(self.tokenizer.encode(text))

    def _trim(self, row: list[int]) -> str:
        eos = self.tokenizer.eos_token_id
        if eos in row:
            row = row[: row.index(eos)]
        return self.tokenizer.decode(row)

    def rollout(self, items: list[dict], key: mx.array):
        """Sample a group per prompt. Returns per-prompt sequences, response masks and rewards."""
        cfg = self.config
        sequences, masks, rewards = [], [], []
        for item in items:
            key, subkey = mx.random.split(key, 2)
            prompt_ids = self._chat(item["prompt"])
            responses = sample_group(
                self.model,
                prompt_ids,
                cfg.group_size,
                cfg.max_tokens,
                cfg.temperature,
                subkey,
                eos_id=self.tokenizer.eos_token_id,
            )
            mx.eval(responses)
            rows = np.array(responses)
            group_reward = np.array(
                [exact_match(self._trim(list(r)), item["answer"]) for r in rows]
            )
            prompt_block = mx.repeat(prompt_ids[None], cfg.group_size, axis=0)
            full = mx.concatenate([prompt_block, responses], axis=1)
            mask = np.zeros((cfg.group_size, full.shape[1] - 1), dtype=np.float32)
            mask[:, prompt_ids.shape[0] - 1 :] = 1.0
            for row_index, row in enumerate(rows):
                ids = list(row)
                if self.tokenizer.eos_token_id in ids:
                    stop = ids.index(self.tokenizer.eos_token_id) + 1
                    mask[row_index, prompt_ids.shape[0] - 1 + stop :] = 0.0
            sequences.append(full)
            masks.append(mx.array(mask))
            rewards.append(group_reward)
        return sequences, masks, np.stack(rewards), key

    def _cell_gradient(self, sequences, masks, advantages) -> mx.array:
        """Gradient of the policy-gradient loss restricted to one cell of the grid."""

        def loss_fn(model):
            total = mx.zeros(())
            count = 0.0
            for seq, mask, adv in zip(sequences, masks, advantages, strict=True):
                logits = model(seq[:, :-1])
                logprobs = nn.losses.cross_entropy(
                    logits.astype(mx.float32), seq[:, 1:], reduction="none"
                )
                per_sequence = -(logprobs * mask).sum(axis=1)
                total = total + (mx.array(adv) * per_sequence).sum()
                count += adv.shape[0]
            return -total / max(count, 1.0)

        _, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        return flat_gradient(grads)

    def measure_batch(self, items: list[dict], key: mx.array):
        cfg = self.config
        sequences, masks, rewards, key = self.rollout(items, key)
        sub = cfg.group_size // 2
        per_block = len(items) // cfg.blocks

        counts = rewards[:, :sub].sum(axis=1).astype(int), rewards[:, sub:].sum(axis=1).astype(int)
        cells = []
        for block in range(cfg.blocks):
            rows = range(block * per_block, (block + 1) * per_block)
            row_cells = []
            for piece in (0, 1):
                columns = slice(0, sub) if piece == 0 else slice(sub, cfg.group_size)
                seqs = [sequences[i][columns] for i in rows]
                cell_masks = [masks[i][columns] for i in rows]
                adv = [
                    self.weights[counts[piece][i], rewards[i, columns].astype(int)] for i in rows
                ]
                row_cells.append(self._cell_gradient(seqs, cell_masks, adv))
            cells.append(mx.stack(row_cells))
        grid = mx.stack(cells)
        mx.eval(grid)
        return np.array(grid, dtype=np.float64), rewards, key

    def measure(self, corpus: list[dict], batches: int, rng: np.random.Generator, corpus_size=None):
        cfg = self.config
        key = mx.random.key(cfg.seed)
        estimates, pass_rates = [], []
        for _ in range(batches):
            picks = rng.choice(len(corpus), size=cfg.prompts, replace=False)
            items = [corpus[i] for i in picks]
            grid, rewards, key = self.measure_batch(items, key)
            estimates.append(
                decompose(
                    grid,
                    cfg.prompts,
                    cfg.group_size,
                    corpus_size=corpus_size or len(corpus),
                )
            )
            pass_rates.append(rewards.mean(axis=1))
        return estimates, np.concatenate(pass_rates)
