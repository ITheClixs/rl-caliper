"""RLVR training for a pretrained model through MLX, with LoRA adapters as the trainable set.

Only what the seed-divergence experiments need: an update step, and a way to compare the policies
two runs arrive at. LoRA adapters start at zero, so two runs begin from exactly the same policy and
differ only in the randomness of their rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from caliper.objectives.advantages import weight_table
from caliper.real.probe import ProbeConfig, RealNoiseProbe


@dataclass
class RealRLConfig(ProbeConfig):
    steps: int = 40
    learning_rate: float = 1e-5
    optimiser: str = "adam"


class RealRLTrainer(RealNoiseProbe):
    def __init__(self, model, tokenizer, config: RealRLConfig):
        super().__init__(model, tokenizer, config)
        if config.optimiser == "adam":
            self.optimiser = optim.Adam(learning_rate=config.learning_rate)
        elif config.optimiser == "sgd":
            # plain ascent is the update the propagation theory is written for
            self.optimiser = optim.SGD(learning_rate=config.learning_rate)
        else:
            raise KeyError(config.optimiser)
        self.full_weights = weight_table(config.estimator, config.group_size)

    def _gradient(self, sequences, masks, advantages):
        """Sum of per-prompt gradients, accumulated one prompt at a time.

        Holding the whole batch in one graph is what pushes a 7B model past the memory of this
        machine; the sum is the same either way.
        """
        count = float(sum(adv.shape[0] for adv in advantages))
        scale = -1.0 / max(count, 1.0)
        total = None
        loss = 0.0
        for seq, mask, adv in zip(sequences, masks, advantages, strict=True):
            weight = mx.array(adv)

            def loss_fn(model, seq=seq, mask=mask, weight=weight):
                logits = model(seq[:, :-1])
                logprobs = nn.losses.cross_entropy(
                    logits.astype(mx.float32), seq[:, 1:], reduction="none"
                )
                per_sequence = -(logprobs * mask).sum(axis=1)
                return scale * (weight * per_sequence).sum()

            value, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
            total = grads if total is None else tree_map(mx.add, total, grads)
            mx.eval(total)
            loss += float(value)
        return loss, total

    def step(self, corpus, rng: np.random.Generator, key: mx.array):
        cfg = self.config
        picks = rng.choice(len(corpus), size=cfg.prompts, replace=False)
        items = [corpus[i] for i in picks]
        sequences, masks, rewards, key = self.rollout(items, key)

        counts = rewards.sum(axis=1).astype(int)
        advantages = [
            self.full_weights[counts[i], rewards[i].astype(int)] for i in range(len(items))
        ]
        _, grads = self._gradient(sequences, masks, advantages)
        self.optimiser.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimiser.state)
        # a unanimous group has an identically zero count-based advantage and injects nothing,
        # so the share of live groups is what says whether this update carried any seed noise
        live = float(np.mean([bool(np.any(a != 0.0)) for a in advantages]))
        # the stochastic support of the objective at this update: a unanimous group has an
        # identically zero count-based advantage, so P_eff = P * live is the number of prompts
        # actually carrying gradient noise, whatever the nominal batch size says
        per_prompt = rewards.mean(axis=1)
        return {
            "pass_rate": float(rewards.mean()),
            "live_share": live,
            "effective_prompts": live * len(items),
            "reward_variance": float(np.mean(per_prompt * (1.0 - per_prompt))),
            "gradient_norm": float(np.linalg.norm(
                np.concatenate([np.asarray(v).reshape(-1) for _, v in tree_flatten(grads)])
            )),
        }, key


def response_log_probs(model, sequences: mx.array) -> mx.array:
    """Log-probabilities over the vocabulary at every next-token position of a fixed batch.

    Prompts differ in length, so the response positions are selected by the mask that accompanies
    the batch rather than by an offset; returning all positions keeps the two aligned.
    """
    logits = model(sequences[:, :-1]).astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def policy_divergence(logp_a: mx.array, logp_b: mx.array, mask: mx.array) -> float:
    """Mean KL(a || b) per response token over a batch of sequences both runs are scored on."""
    kl = mx.sum(mx.exp(logp_a) * (logp_a - logp_b), axis=-1)
    return float(mx.sum(kl * mask) / mx.maximum(mx.sum(mask), 1.0))
