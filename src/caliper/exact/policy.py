"""A small autoregressive policy whose sequence distribution can be enumerated exactly.

Logits are stored per (prefix) context, so the policy is autoregressive and its score has the
block structure of a real language model, while remaining small enough that every sequence,
every score vector and every conditional moment can be computed without sampling.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def _context_index(prefix: tuple[int, ...], vocab: int) -> int:
    """Index contexts in order of increasing prefix length: (), (a), (a b), ..."""
    idx = 0
    for t in range(len(prefix)):
        idx += vocab**t
    base = idx
    offset = 0
    for tok in prefix:
        offset = offset * vocab + tok
    return base + offset


@dataclass
class TabularPolicy:
    vocab: int
    length: int
    logits: np.ndarray  # (n_contexts, vocab)

    @property
    def n_contexts(self) -> int:
        return (self.vocab**self.length - 1) // (self.vocab - 1)

    @property
    def dim(self) -> int:
        return self.n_contexts * self.vocab

    @classmethod
    def random(cls, vocab: int, length: int, rng: np.random.Generator, scale: float = 1.0):
        n_contexts = (vocab**length - 1) // (vocab - 1)
        return cls(vocab, length, rng.normal(scale=scale, size=(n_contexts, vocab)))

    def sequences(self) -> np.ndarray:
        return np.array(list(itertools.product(range(self.vocab), repeat=self.length)), dtype=int)

    def _softmax(self) -> np.ndarray:
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def sequence_probs(self) -> np.ndarray:
        """Probabilities of every sequence, without the scores."""
        probs_tab = self._softmax()
        seqs = self.sequences()
        probs = np.ones(seqs.shape[0])
        for i, seq in enumerate(seqs):
            for t in range(self.length):
                ctx = _context_index(tuple(seq[:t]), self.vocab)
                probs[i] *= probs_tab[ctx, seq[t]]
        return probs

    def perturbed(self, delta: np.ndarray) -> TabularPolicy:
        shifted = self.logits + delta.reshape(self.logits.shape)
        return TabularPolicy(self.vocab, self.length, shifted)

    def enumerate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (probs, scores) over all vocab**length sequences."""
        probs_tab = self._softmax()
        seqs = self.sequences()
        n_seq = seqs.shape[0]
        probs = np.ones(n_seq)
        scores = np.zeros((n_seq, self.n_contexts, self.vocab))
        for i, seq in enumerate(seqs):
            for t in range(self.length):
                ctx = _context_index(tuple(seq[:t]), self.vocab)
                probs[i] *= probs_tab[ctx, seq[t]]
                scores[i, ctx, :] -= probs_tab[ctx, :]
                scores[i, ctx, seq[t]] += 1.0
        return probs, scores.reshape(n_seq, self.dim)


@dataclass
class ExactPrompt:
    """Per-prompt quantities the theory needs: p, u1, and the conditional score covariances."""

    probs: np.ndarray
    rewards: np.ndarray
    scores: np.ndarray

    @property
    def pass_rate(self) -> float:
        return float(self.probs @ self.rewards)

    def _conditional(self, target: int) -> tuple[np.ndarray, np.ndarray]:
        mask = self.rewards == target
        weight = self.probs[mask]
        total = weight.sum()
        if total <= 0:
            dim = self.scores.shape[1]
            return np.zeros(dim), np.zeros((dim, dim))
        weight = weight / total
        sub = self.scores[mask]
        mean = weight @ sub
        centred = sub - mean
        cov = (centred * weight[:, None]).T @ centred
        return mean, cov

    def moments(self) -> dict[str, np.ndarray | float]:
        u1, s1 = self._conditional(1)
        u0, s0 = self._conditional(0)
        p = self.pass_rate
        return {"p": p, "u1": u1, "u0": u0, "s1": s1, "s0": s0, "h": p * u1}


def make_prompt(policy: TabularPolicy, accept: np.ndarray) -> ExactPrompt:
    probs, scores = policy.enumerate()
    return ExactPrompt(probs=probs, rewards=accept.astype(float), scores=scores)


def accept_set_for_pass_rate(
    probs: np.ndarray, target: float, rng: np.random.Generator
) -> np.ndarray:
    """Choose a random accept set whose probability mass is close to `target`."""
    order = rng.permutation(probs.shape[0])
    accept = np.zeros(probs.shape[0], dtype=int)
    total = 0.0
    for i in order:
        if total >= target:
            break
        accept[i] = 1
        total += probs[i]
    return accept


def fisher(prompt: ExactPrompt) -> np.ndarray:
    """Exact Fisher information of the policy on this prompt, E[s s^T]."""
    return (prompt.scores * prompt.probs[:, None]).T @ prompt.scores
