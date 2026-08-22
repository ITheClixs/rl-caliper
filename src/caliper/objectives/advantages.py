"""Advantage weights for count-based group estimators.

Every estimator in the GRPO/RLOO family assigns a rollout an advantage that depends only on its
own binary reward and on the number of successes in its group. Such an estimator is therefore
fully described by a (G+1, 2) table w[k, r]. Representing it this way turns all of the moment
computations in `caliper.exact` into contractions against a binomial distribution.
"""

from __future__ import annotations

import numpy as np

_REGISTRY: dict[str, callable] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def weight_table(name: str, group_size: int, **kwargs) -> np.ndarray:
    """Return w[k, r] for k = 0..G and r in {0, 1}."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown estimator {name!r}; have {sorted(_REGISTRY)}")
    w = _REGISTRY[name](group_size, **kwargs)
    return np.ascontiguousarray(w, dtype=np.float64)


def available() -> list[str]:
    return sorted(_REGISTRY)


def _grid(group_size: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(group_size + 1, dtype=np.float64)[:, None]
    r = np.array([0.0, 1.0])[None, :]
    return k, r


@register("rloo")
def _rloo(group_size: int) -> np.ndarray:
    if group_size < 2:
        raise ValueError("RLOO needs G >= 2")
    k, r = _grid(group_size)
    return r - (k - r) / (group_size - 1)


@register("grpo_mean")
def _grpo_mean(group_size: int) -> np.ndarray:
    k, r = _grid(group_size)
    return r - k / group_size


@register("grpo_std")
def _grpo_std(group_size: int, eps: float = 0.0) -> np.ndarray:
    """GRPO with the group standard deviation in the denominator.

    Degenerate groups (k = 0 or k = G) have zero spread and contribute no gradient. With eps = 0
    they are set to zero exactly, which is what implementations do; a positive eps reproduces the
    `+ eps` variant, where degenerate groups leak a large advantage instead.
    """
    k, r = _grid(group_size)
    frac = k / group_size
    sd = np.sqrt(frac * (1.0 - frac))
    num = r - frac
    if eps > 0:
        return num / (sd + eps)
    out = np.zeros_like(num)
    live = sd > 0
    np.divide(num, sd, out=out, where=live)
    return out


@register("grpo_std_unbiased")
def _grpo_std_unbiased(group_size: int, eps: float = 0.0) -> np.ndarray:
    """Standardisation with the sample (Bessel-corrected) standard deviation."""
    if group_size < 2:
        raise ValueError("needs G >= 2")
    k, r = _grid(group_size)
    frac = k / group_size
    var = frac * (1.0 - frac) * group_size / (group_size - 1)
    sd = np.sqrt(var)
    num = r - frac
    if eps > 0:
        return num / (sd + eps)
    out = np.zeros_like(num)
    live = sd > 0
    np.divide(num, sd, out=out, where=live)
    return out


@register("centred")
def _centred(group_size: int, baseline: float = 0.5) -> np.ndarray:
    """Fixed external baseline; the group plays no role."""
    _, r = _grid(group_size)
    return np.repeat(r - baseline, group_size + 1, axis=0)
