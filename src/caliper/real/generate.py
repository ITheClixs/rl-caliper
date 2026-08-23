"""Batched sampling for an MLX model with a shared prompt cache.

Responses to the same prompt share that prompt's prefill: the cache is built once and then
broadcast across the group, which is the cost structure equation (16) prices.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def _broadcast_cache(cache, group_size: int) -> None:
    for layer in cache:
        keys, values = layer.state
        layer.update_and_fetch  # noqa: B018  (kept for API clarity)
        layer.keys = mx.repeat(keys, group_size, axis=0)
        layer.values = mx.repeat(values, group_size, axis=0)
        layer.offset = keys.shape[2]


def sample_group(
    model,
    prompt_tokens: mx.array,
    group_size: int,
    max_tokens: int,
    temperature: float,
    key: mx.array,
    eos_id: int | None = None,
) -> mx.array:
    """Draw `group_size` responses to one prompt. Returns (group_size, <= max_tokens)."""
    cache = make_prompt_cache(model)
    logits = model(prompt_tokens[None], cache=cache)[:, -1, :]
    _broadcast_cache(cache, group_size)

    logits = mx.repeat(logits, group_size, axis=0)
    tokens = []
    finished = mx.zeros((group_size,), dtype=mx.bool_)
    for step in range(max_tokens):
        keys = mx.random.split(key, 2)
        key, subkey = keys[0], keys[1]
        token = mx.random.categorical(logits / temperature, key=subkey)
        if eos_id is not None:
            token = mx.where(finished, mx.array(eos_id), token)
            finished = finished | (token == eos_id)
        tokens.append(token)
        if step + 1 == max_tokens:
            break
        logits = model(token[:, None], cache=cache)[:, -1, :]
    return mx.stack(tokens, axis=1)
