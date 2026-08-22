"""Batch-level exact quantities: noise scales, critical batch size, optimal group size.

Given per-prompt moments (from `caliper.exact.policy`) this assembles the hierarchical covariance
of the batch estimator and the derived scaling quantities of Section 4 of docs/theory.md. Nothing
here samples; it is the ground truth the sampled estimators are checked against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caliper.exact import moments
from caliper.objectives.advantages import weight_table


@dataclass
class NoiseTerms:
    signal: float  # g_bar^T M g_bar
    tau_b: float  # tr(M Sigma_b)
    tau_w: float  # tr(M Sigma_w(G)), at the G it was computed for
    alignment: float  # grad J^T g_bar, the only part of the update that makes progress
    group_size: int

    def critical_batch(self) -> float:
        """Bcrit in units of prompts, equation (12)."""
        return (self.tau_b + self.tau_w) / self.signal

    def efficiency(self, n_prompts: int) -> float:
        """Fraction of the ideal per-step progress achieved at this batch size, equation (11)."""
        return 1.0 / (1.0 + self.critical_batch() / n_prompts)

    def optimal_step_size(self, n_prompts: int) -> float:
        return self.alignment / (self.signal * (1.0 + self.critical_batch() / n_prompts))

    def ideal_progress(self) -> float:
        """Progress in the zero-noise limit; depends on the estimator only through alignment."""
        return 0.5 * self.alignment**2 / self.signal

    def progress(self, n_prompts: int) -> float:
        """E[dJ] at the optimal step size, equation (10)."""
        return self.ideal_progress() * self.efficiency(n_prompts)


class ExactBatchModel:
    """Exact model of one RLVR update over a fixed prompt population."""

    def __init__(self, prompt_moments: list[dict], fishers: list[np.ndarray] | None = None):
        self.prompts = prompt_moments
        self.fishers = fishers

    @property
    def dim(self) -> int:
        return self.prompts[0]["u1"].shape[0]

    def curvature(self, kind: str = "fisher") -> np.ndarray:
        if kind == "identity":
            return np.eye(self.dim)
        if kind == "fisher":
            if self.fishers is None:
                raise ValueError("no Fisher matrices supplied")
            return np.mean(self.fishers, axis=0)
        raise KeyError(kind)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Mean of the estimator: E_x[lambda(p, G) h(x)]."""
        per = [moments.difficulty_weight(w, m["p"]) * m["h"] for m in self.prompts]
        return np.mean(per, axis=0)

    def true_gradient(self) -> np.ndarray:
        """grad J = E_x[h(x)]. The estimator family targets reweighted versions of this."""
        return np.mean([m["h"] for m in self.prompts], axis=0)

    def sigma_b(self, w: np.ndarray) -> np.ndarray:
        per = np.array([moments.difficulty_weight(w, m["p"]) * m["h"] for m in self.prompts])
        mean = per.mean(axis=0)
        centred = per - mean
        return centred.T @ centred / per.shape[0]

    def sigma_w(self, w: np.ndarray) -> np.ndarray:
        covs = [
            moments.covariance(w, m["p"], m["s0"], m["s1"], m["u1"]) for m in self.prompts
        ]
        return np.mean(covs, axis=0)

    def noise_terms(self, estimator: str, group_size: int, curvature: str = "fisher") -> NoiseTerms:
        w = weight_table(estimator, group_size)
        matrix = self.curvature(curvature)
        grad = self.gradient(w)
        return NoiseTerms(
            signal=float(grad @ matrix @ grad),
            tau_b=float(np.trace(matrix @ self.sigma_b(w))),
            tau_w=float(np.trace(matrix @ self.sigma_w(w))),
            alignment=float(self.true_gradient() @ grad),
            group_size=group_size,
        )

    def progress_per_rollout_budget(
        self, estimator: str, group_size: int, rollouts_per_step: int, curvature: str = "fisher"
    ) -> float:
        """Per-step progress when R rollouts are split as P = R/G prompts of G rollouts."""
        n_prompts = rollouts_per_step / group_size
        return self.noise_terms(estimator, group_size, curvature).progress(n_prompts)


def predicted_optimal_group_size(
    tau_b: float, tau_w_scaled: float, prefill_ratio: float = 0.0
) -> float:
    """G* = 1 + sqrt((1 + c_pre/c_dec) * tau_w / tau_b), equations (14) and (15).

    `tau_w_scaled` is (G-1) * tr(M Sigma_w(G)), which is G-independent to leading order.
    """
    return 1.0 + np.sqrt((1.0 + prefill_ratio) * tau_w_scaled / tau_b)
