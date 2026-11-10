"""The tangent model of an enumerable policy, for `caliper.tangent`.

Every sequence is a candidate, so nothing is truncated: the reference log-probabilities and scores
are exact, and the only approximation left in a simulated seed is that the scores are frozen where
the reference was taken.
"""

from __future__ import annotations

import numpy as np

from caliper.exact.policy import TabularPolicy
from caliper.tangent import TangentModel


def reference_model(policy: TabularPolicy, accepts: np.ndarray) -> TangentModel:
    """Linearise `policy` where it stands. One policy answers every prompt on this tier, so the
    prompts share their candidates and scores and differ only in which sequences they accept."""
    probs, scores = policy.enumerate()
    n_prompts = accepts.shape[0]
    with np.errstate(divide="ignore"):
        base = np.log(probs)
    return TangentModel(
        base=np.tile(base, (n_prompts, 1)),
        features=np.broadcast_to(scores, (n_prompts, *scores.shape)),
        rewards=accepts.astype(float),
    )
